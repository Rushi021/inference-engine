import sys                                                       # CLI args
import json                                                      # read results
from pathlib import Path                                         # paths
from collections import defaultdict                              # group by engine

import matplotlib                                                # plotting
matplotlib.use("Agg")                                            # headless: write files, no display
import matplotlib.pyplot as plt                                  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"  # input + output dir


def _group_by_engine(results):
    by_engine = defaultdict(list)                              # engine -> rows
    for r in results:
        by_engine[r["engine"]].append(r)
    for rows in by_engine.values():
        rows.sort(key=lambda r: r["concurrency"])             # x-axis order
    return by_engine


def _line_chart(by_engine, ykey, title, ylabel, out_path):
    plt.figure(figsize=(7, 4.5))                              # one metric per chart
    for engine, rows in by_engine.items():
        xs = [r["concurrency"] for r in rows]                 # concurrency on x
        ys = [r[ykey] for r in rows]                          # chosen metric on y
        plt.plot(xs, ys, marker="o", label=engine)           # one line per engine
    plt.xlabel("concurrency (in-flight requests)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)                            # PNG for the README/dashboard
    plt.close()
    print(f"wrote {out_path}")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "latest.json"  # default to latest run
    results = json.loads(Path(src).read_text())
    by_engine = _group_by_engine(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    _line_chart(by_engine, "throughput_tok_s", "Throughput vs concurrency",
                "throughput (tokens/sec)", RESULTS_DIR / "throughput.png")
    _line_chart(by_engine, "latency_p99_s", "p99 latency vs concurrency",
                "p99 latency (s)", RESULTS_DIR / "latency_p99.png")
    _line_chart(by_engine, "latency_p50_s", "p50 latency vs concurrency",
                "p50 latency (s)", RESULTS_DIR / "latency_p50.png")
    _line_chart(by_engine, "ttft_mean_s", "Time-to-first-token vs concurrency",
                "mean TTFT (s)", RESULTS_DIR / "ttft.png")


if __name__ == "__main__":
    main()
