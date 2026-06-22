import time                                                     # throughput timing
import threading                                                 # drive concurrent requests

from engine.batched import batched_generate, get_engine          # continuous-batching engine
from engine.naive import naive_generate                          # baseline for comparison
from engine.manual_decode import manual_generate                 # single-sequence reference
from engine.model import load_model                              # pre-warm


def _concurrent(fn, prompts, max_new_tokens):
    results = {}                                                # prompt -> result dict
    lock = threading.Lock()                                     # guard the dict

    def worker(p):
        r = fn(p, max_new_tokens)
        with lock:
            results[p] = r

    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_batched_matches_single_sequence():
    prompts = ["List three colors.", "What is 2+2? Answer briefly.",
               "Name a fruit.", "Say hello in French."]          # different lengths, run together
    out = _concurrent(batched_generate, prompts, max_new_tokens=20)
    for p in prompts:                                           # each must equal lone greedy decode
        ref = manual_generate(p, max_new_tokens=20, use_cache=True)
        assert out[p]["token_ids"] == ref["token_ids"]          # correctness under batching


def test_batched_beats_naive_under_concurrency():
    load_model()                                               # pre-warm so loading doesn't skew timing
    get_engine()                                               # pre-build scheduler thread
    prompts = ["Write two sentences about the sea."] * 8        # 8 concurrent requests
    mnt = 24

    t0 = time.perf_counter()
    nb = _concurrent(naive_generate, prompts, mnt)             # serialized baseline
    naive_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    bb = _concurrent(batched_generate, prompts, mnt)           # continuous batching
    batched_wall = time.perf_counter() - t0

    naive_tokens = sum(r["tokens_generated"] for r in nb.values())
    batched_tokens = sum(r["tokens_generated"] for r in bb.values())
    assert batched_tokens / batched_wall > naive_tokens / naive_wall  # higher aggregate throughput
