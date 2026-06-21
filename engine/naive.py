import time                                                     # high-resolution timing
import threading                                                 # serialize requests + run generate off-thread
import torch                                                     # inference mode
from transformers import TextIteratorStreamer                    # stream tokens to capture time-to-first-token

from engine.model import load_model, build_prompt               # shared loader + chat templating

_lock = threading.Lock()                                         # THE naive part: one request at a time, no batching


@torch.inference_mode()                                          # no autograd during serving
def naive_generate(prompt: str, max_new_tokens: int = 64) -> dict:
    model, tokenizer, device = load_model()                      # reuse the cached model
    text = build_prompt(tokenizer, prompt)                       # apply chat template
    inputs = tokenizer(text, return_tensors="pt").to(device)     # tokenize onto device
    prompt_len = int(inputs["input_ids"].shape[1])               # prompt length in tokens

    streamer = TextIteratorStreamer(                            # yields text as it is generated
        tokenizer, skip_prompt=True, skip_special_tokens=True    # only the new reply, decoded
    )
    gen_kwargs = dict(                                          # standard greedy decode + streaming
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        streamer=streamer, return_dict_in_generate=True,         # return_dict gives us exact token ids
    )
    holder = {}                                                 # pass generate's return value out of the thread

    def _run():
        holder["out"] = model.generate(**gen_kwargs)            # heavy compute runs here

    with _lock:                                                 # block concurrent requests — sequential baseline
        t0 = time.perf_counter()                                # start clock
        worker = threading.Thread(target=_run)                  # generate must run off-thread to read the streamer
        worker.start()
        ttft = None                                             # time-to-first-token
        pieces = []                                             # collected reply text
        for chunk in streamer:                                  # consume tokens as they arrive
            if ttft is None:                                    # first token arrived
                ttft = time.perf_counter() - t0                 # record TTFT
            pieces.append(chunk)
        worker.join()                                           # generation finished
        total = time.perf_counter() - t0                        # total wall time

    seq = holder["out"].sequences[0]                            # full token sequence (prompt + reply)
    new_ids = seq[prompt_len:]                                  # only the generated tokens
    tokens_generated = int(new_ids.shape[0])                    # exact count of decode steps
    completion = "".join(pieces)                                # assembled reply text
    tps = tokens_generated / total if total > 0 else 0.0        # throughput for this request

    return {
        "completion": completion,                              # the generated text
        "prompt_tokens": prompt_len,                            # input size in tokens
        "tokens_generated": tokens_generated,                  # output size in tokens
        "time_to_first_token_s": round(ttft if ttft else total, 4),  # latency to first token
        "total_time_s": round(total, 4),                       # end-to-end latency
        "tokens_per_sec": round(tps, 2),                       # per-request throughput
    }
