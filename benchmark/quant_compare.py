import sys                                                       # path + args
import json                                                      # save results
import time                                                      # throughput timing
from pathlib import Path                                         # paths

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import engine/benchmark

import torch                                                     # cuda memory + generation

from engine.quant import load_quantized                          # fp16/int8/int4 loaders
from engine.model import build_prompt                            # chat template
from benchmark.quality import perplexity, token_agreement, EVAL_TEXTS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"  # output dir

# GPU-only script (int8/int4 need bitsandbytes + CUDA). Run during the focused GPU session.


@torch.inference_mode()
def _measure(model, tokenizer, gen_tokens=128):
    device = next(model.parameters()).device
    prompt = build_prompt(tokenizer, "Write a short paragraph about the sea.")
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()                                    # time a fixed-length generation
    out = model.generate(ids, max_new_tokens=gen_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_ids = out[0][ids.shape[1]:].tolist()                   # generated tokens only
    tps = len(new_ids) / dt if dt > 0 else 0.0                 # decode throughput
    return tps, new_ids


def main():
    if not torch.cuda.is_available():                          # this script is GPU-only
        raise SystemExit("quant_compare requires a CUDA GPU (int8/int4 via bitsandbytes)")
    RESULTS_DIR.mkdir(exist_ok=True)
    levels = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fp16", "int8", "int4"]

    results = []
    reference_ids = None                                       # fp16 greedy output = quality reference
    for quant in levels:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, tok = load_quantized(quant=quant)               # load at this precision
        tps, gen_ids = _measure(model, tok)                    # throughput + sample output
        ppl = perplexity(model, tok, EVAL_TEXTS)               # quality: perplexity (lower better)
        if quant == "fp16":
            reference_ids = gen_ids                            # set the agreement baseline
        agree = token_agreement(reference_ids, gen_ids) if reference_ids else 1.0
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)  # peak GPU memory
        row = {
            "quant": quant,                                    # precision level
            "throughput_tok_s": round(tps, 2),                 # decode speed
            "peak_mem_mib": round(peak_mib, 1),                # memory footprint
            "perplexity": round(ppl, 3),                       # quality (lower = better)
            "token_agreement_vs_fp16": round(agree, 3),        # how often greedy matches fp16
        }
        results.append(row)
        print(f"{quant:>5}: {row['throughput_tok_s']:>7.2f} tok/s  "
              f"{row['peak_mem_mib']:>8.1f} MiB  ppl={row['perplexity']:.3f}  "
              f"agree={row['token_agreement_vs_fp16']:.3f}")
        del model
        torch.cuda.empty_cache()

    (RESULTS_DIR / "quant.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS_DIR / 'quant.json'}")


if __name__ == "__main__":
    main()
