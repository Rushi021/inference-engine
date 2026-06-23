import os                                                       # env + paths
import sys                                                       # import path + interpreter
import csv                                                       # CSV output
import json                                                      # JSON output
import time                                                      # timestamps + waiting
import argparse                                                  # CLI
import asyncio                                                   # async load driver
import subprocess                                                # spawn the server
from pathlib import Path                                         # paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import benchmark/engine directly

import httpx                                                     # async HTTP client

from benchmark.load_test import run_load                         # the load generator
from benchmark.gpu import GpuSampler, has_nvidia_smi             # GPU utilization sampling

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"  # where outputs land


def _spawn_server(port):
    env = dict(os.environ)                                      # inherit HF_HOME etc.
    proc = subprocess.Popen(                                    # start uvicorn in the background
        [sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(port), "--log-level", "warning"],
        env=env,
    )
    return proc


async def _wait_healthy(client, timeout=120):
    deadline = time.time() + timeout                           # give the model time to load
    while time.time() < deadline:
        try:
            r = await client.get("/health")                    # poll liveness
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def _run(args):
    RESULTS_DIR.mkdir(exist_ok=True)                           # ensure output dir
    engines = [e.strip() for e in args.engines.split(",")]    # e.g. naive,batched
    levels = [int(x) for x in args.concurrency.split(",")]    # e.g. 1,4,16,64

    proc = None
    base_url = args.url                                        # use existing server if given
    if base_url is None:                                      # otherwise spawn our own
        proc = _spawn_server(args.port)
        base_url = f"http://127.0.0.1:{args.port}"

    results = []
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=600) as client:
            if not await _wait_healthy(client):               # block until the model is ready
                raise RuntimeError("server did not become healthy in time")
            print(f"server ready at {base_url}; GPU sampling: {'on' if has_nvidia_smi() else 'off (CPU/MPS)'}")
            for engine in engines:                            # one row per (engine, concurrency)
                for c in levels:
                    sampler = GpuSampler().start()            # sample GPU during this cell
                    summary = await run_load(client, engine, concurrency=c,
                                             num_requests=max(args.requests, c),
                                             max_tokens=args.max_tokens)
                    summary["gpu"] = sampler.stop()           # attach GPU stats (None on CPU/MPS)
                    results.append(summary)
                    print(f"  {engine:>7} c={c:<3} -> {summary['throughput_tok_s']:>7.1f} tok/s  "
                          f"p50={summary['latency_p50_s']:.2f}s p99={summary['latency_p99_s']:.2f}s "
                          f"ttft={summary['ttft_mean_s']:.3f}s")
    finally:
        if proc is not None:                                  # always shut the spawned server down
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    stamp = time.strftime("%Y%m%d-%H%M%S")                     # unique run id
    json_path = RESULTS_DIR / f"bench-{stamp}.json"
    csv_path = RESULTS_DIR / f"bench-{stamp}.csv"
    json_path.write_text(json.dumps(results, indent=2))       # full structured results
    _write_csv(csv_path, results)                             # flat table for spreadsheets
    latest = RESULTS_DIR / "latest.json"                      # stable path for the plotter
    latest.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {json_path}\nwrote {csv_path}\nwrote {latest}")


def _write_csv(path, results):
    cols = ["engine", "concurrency", "num_requests", "max_tokens", "completed", "wall_s",
            "throughput_tok_s", "req_per_s", "latency_p50_s", "latency_p90_s",
            "latency_p99_s", "ttft_mean_s", "total_tokens"]   # flat metric columns
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in cols})           # drop the nested gpu dict for CSV


def main():
    ap = argparse.ArgumentParser(description="Benchmark engine variants under a concurrency sweep")
    ap.add_argument("--engines", default="naive,batched")     # which engines to test
    ap.add_argument("--concurrency", default="1,2,4")          # CPU-safe default; GPU: 1,4,16,64
    ap.add_argument("--requests", type=int, default=8)         # requests per (engine, concurrency) cell
    ap.add_argument("--max-tokens", type=int, default=24)      # output length per request
    ap.add_argument("--port", type=int, default=8077)          # spawned server port (8000 is taken)
    ap.add_argument("--url", default=None)                     # use an existing server instead of spawning
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
