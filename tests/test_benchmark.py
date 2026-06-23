import asyncio                                                   # drive async load test
import httpx                                                     # ASGI in-process client

from benchmark.stats import percentile, summarize                # pure stats
from benchmark.load_test import run_load                         # async load generator
from server.app import app                                       # the real FastAPI app


def test_percentile_basic():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                     # known distribution
    assert percentile(vals, 50) == 5.5                        # median by interpolation
    assert percentile(vals, 0) == 1.0                         # min
    assert percentile(vals, 100) == 10.0                     # max
    assert percentile([], 99) == 0.0                          # empty is safe
    assert percentile([42], 99) == 42.0                      # single value


def test_summarize_throughput_and_percentiles():
    records = [{"latency_s": 1.0, "ttft_s": 0.1, "tokens": 10},
               {"latency_s": 2.0, "ttft_s": 0.2, "tokens": 20}]  # 30 tokens total
    s = summarize(records, wall_s=2.0)
    assert s["completed"] == 2
    assert s["throughput_tok_s"] == 15.0                      # 30 tokens / 2.0s
    assert s["latency_p50_s"] == 1.5                          # midpoint of 1.0 and 2.0
    assert s["total_tokens"] == 30


def test_load_test_against_app_in_process():
    async def go():
        transport = httpx.ASGITransport(app=app)              # drive the app without a real port
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=120) as client:
            return await run_load(client, engine="naive", concurrency=2,
                                  num_requests=4, max_tokens=8)
    s = asyncio.run(go())
    assert s["completed"] == 4                                # all requests finished
    assert s["throughput_tok_s"] > 0                          # produced tokens
    assert s["engine"] == "naive" and s["concurrency"] == 2  # tagged correctly
