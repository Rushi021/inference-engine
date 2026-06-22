import sys                                                     # path setup
import time                                                     # timing
import threading                                                 # concurrent load
from pathlib import Path                                        # repo root

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import engine directly

import transformers                                              # quiet the logs
transformers.logging.set_verbosity_error()

from engine.model import load_model                             # pre-warm
from engine.naive import naive_generate                         # baseline
from engine.batched import batched_generate, get_engine         # continuous batching


def run(fn, prompts, max_new_tokens):
    results, lock = [], threading.Lock()

    def worker(p):
        r = fn(p, max_new_tokens)
        with lock:
            results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    tokens = sum(r["tokens_generated"] for r in results)
    return wall, tokens, tokens / wall


def main():
    load_model()                                               # warm weights once
    get_engine()                                               # warm scheduler thread
    prompt = "Write two sentences about the ocean."
    mnt = 40
    for n in (1, 4, 16):                                       # sweep concurrency
        prompts = [prompt] * n
        wn, tn, thn = run(naive_generate, prompts, mnt)
        wb, tb, thb = run(batched_generate, prompts, mnt)
        print(f"concurrency={n:>3}:  naive {thn:>6.1f} tok/s   "
              f"batched {thb:>6.1f} tok/s   speedup {thb / thn:.1f}x")


if __name__ == "__main__":
    main()
