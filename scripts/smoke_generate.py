import sys                                                     # read optional CLI prompt
import time                                                     # measure load + generate latency
from pathlib import Path                                        # make project root importable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "engine" imports work when run directly

from engine.model import load_model, generate                   # our loader + generate helper
from engine.config import MODEL_ID                              # which model we load


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "In one sentence, what is a KV cache?"

    print(f"Loading model: {MODEL_ID} ...")                     # progress for slow first download
    t0 = time.perf_counter()                                    # start load timer
    _, _, device = load_model()                                 # trigger download + load
    print(f"Loaded on device='{device}' in {time.perf_counter() - t0:.1f}s")

    print(f"\nPrompt: {prompt}")
    t1 = time.perf_counter()                                    # start generate timer
    reply = generate(prompt, max_new_tokens=64)                 # run one completion
    dt = time.perf_counter() - t1                               # generation wall time
    print(f"\nReply: {reply}")
    print(f"\nGenerated in {dt:.2f}s")                          # rough feel for CPU speed


if __name__ == "__main__":
    main()
