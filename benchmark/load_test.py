import time                                                     # per-request timing
import asyncio                                                   # concurrency

from benchmark.stats import summarize                            # aggregate the records

PROMPTS = [                                                      # a small mixed-length prompt pool
    "Write two sentences about the ocean.",
    "Explain what a CPU does, briefly.",
    "List three programming languages.",
    "What is the capital of France?",
]


async def _one_request(client, engine, prompt, max_tokens, sem, records):
    async with sem:                                            # cap in-flight requests at `concurrency`
        t0 = time.perf_counter()                               # client-side clock (captures queueing)
        resp = await client.post("/generate", json={           # hit the server like a real client
            "prompt": prompt, "max_tokens": max_tokens, "engine": engine})
        latency = time.perf_counter() - t0                     # end-to-end latency for this request
        resp.raise_for_status()                                # surface server errors
        body = resp.json()
        records.append({                                       # one record per completed request
            "latency_s": latency,                              # measured client-side
            "ttft_s": body["time_to_first_token_s"],           # reported by the server
            "tokens": body["tokens_generated"],                # output length
        })


async def run_load(client, engine, concurrency, num_requests, max_tokens, prompts=None):
    prompts = prompts or PROMPTS                               # default prompt pool
    sem = asyncio.Semaphore(concurrency)                       # closed-loop concurrency limit
    records = []                                               # collected per-request data
    t0 = time.perf_counter()                                   # run wall clock
    tasks = [                                                  # launch all requests; semaphore throttles
        _one_request(client, engine, prompts[i % len(prompts)], max_tokens, sem, records)
        for i in range(num_requests)
    ]
    await asyncio.gather(*tasks)                               # wait for the whole run
    wall = time.perf_counter() - t0
    summary = summarize(records, wall)                         # throughput + latency percentiles
    summary.update({                                           # tag with the run parameters
        "engine": engine, "concurrency": concurrency,
        "num_requests": num_requests, "max_tokens": max_tokens,
    })
    return summary
