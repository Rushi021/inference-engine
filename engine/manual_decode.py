import time                                                     # timing the two paths
import torch                                                     # tensors + inference mode

from engine.model import load_model, build_prompt               # shared loader + chat templating


@torch.inference_mode()                                          # no autograd
def _run_cached(model, input_ids, eos_id, max_new_tokens, device):
    past = None                                                  # KV cache, grows each step
    cur = input_ids                                            # first step feeds the whole prompt
    seen = 0                                                    # tokens already in the cache
    generated = []                                             # produced token ids
    for _ in range(max_new_tokens):
        cache_position = torch.arange(seen, seen + cur.shape[1], device=device)  # write slots / RoPE positions
        out = model(input_ids=cur, past_key_values=past,        # reuse cached K/V
                    use_cache=True, cache_position=cache_position)
        past = out.past_key_values                              # updated cache for next step
        seen += cur.shape[1]                                    # advance cache length
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)  # greedy pick
        generated.append(int(next_id))
        if int(next_id) == eos_id:                             # stop at end-of-sequence
            break
        cur = next_id                                          # KEY: next step feeds ONLY the new token
    return generated


@torch.inference_mode()                                          # no autograd
def _greedy_uncached(model, input_ids, eos_id, max_new_tokens):
    cur = input_ids                                            # full sequence, regrown every step
    generated = []                                             # produced token ids
    for _ in range(max_new_tokens):
        out = model(input_ids=cur, use_cache=False)            # NAIVE: recompute K/V for every token
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)  # greedy pick
        generated.append(int(next_id))
        if int(next_id) == eos_id:                             # stop at end-of-sequence
            break
        cur = torch.cat([cur, next_id], dim=1)                 # feed the entire growing sequence again
    return generated


def manual_generate(prompt: str, max_new_tokens: int = 32, use_cache: bool = True):
    model, tokenizer, device = load_model()                    # shared cached model
    text = build_prompt(tokenizer, prompt)                     # chat template
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)  # tokenize onto device
    eos_id = tokenizer.eos_token_id                            # stop token

    t0 = time.perf_counter()                                   # start clock
    if use_cache:
        ids = _run_cached(model, input_ids, eos_id, max_new_tokens, device)  # O(n) decode
    else:
        ids = _greedy_uncached(model, input_ids, eos_id, max_new_tokens)     # O(n^2) decode
    elapsed = time.perf_counter() - t0                         # decode wall time

    completion = tokenizer.decode(ids, skip_special_tokens=True)  # reply text
    return {
        "completion": completion,                             # generated text
        "token_ids": ids,                                     # exact tokens (for correctness checks)
        "tokens_generated": len(ids),                         # output length
        "used_cache": use_cache,                              # which path ran
        "total_time_s": round(elapsed, 4),                   # decode latency
        "tokens_per_sec": round(len(ids) / elapsed, 2) if elapsed > 0 else 0.0,  # throughput
    }
