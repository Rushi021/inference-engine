import sys                                                     # CLI args
import time                                                     # duration control
import asyncio                                                   # concurrency
import httpx                                                     # async client

# Sustained closed-loop load against a deployed server, to trigger HPA autoscaling.
# usage: python scripts/gke_load.py http://EXTERNAL_IP <seconds> <concurrency> <max_tokens>
URL = sys.argv[1]
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONCURRENCY = int(sys.argv[3]) if len(sys.argv) > 3 else 24
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 128

PROMPT = "Write a detailed multi-sentence paragraph about the ocean and its ecosystems."


async def worker(client, stop_at, counter):
    while time.perf_counter() < stop_at:                       # keep firing until time is up
        try:
            await client.post("/generate", json={              # CPU-heavy generation
                "prompt": PROMPT, "max_tokens": MAX_TOKENS, "engine": "batched"}, timeout=180)
            counter[0] += 1
        except Exception:
            pass                                              # ignore transient errors during scaling


async def main():
    stop_at = time.perf_counter() + DURATION                  # run window
    counter = [0]                                             # completed request count
    async with httpx.AsyncClient(base_url=URL) as client:
        await asyncio.gather(*[worker(client, stop_at, counter) for _ in range(CONCURRENCY)])
    print(f"completed {counter[0]} requests over {DURATION}s at concurrency {CONCURRENCY}")


if __name__ == "__main__":
    asyncio.run(main())
