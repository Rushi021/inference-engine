import time                                                     # per-request timing
import threading                                                 # scheduler thread + queues
import queue                                                     # thread-safe request intake
import torch                                                     # tensors + forward

from engine.model import load_model, build_prompt               # shared loader + chat templating


def _left_pad_seq(t, n):
    if n == 0:                                                  # nothing to pad
        return t
    b, h, _, d = t.shape                                        # [B, H, L, D]
    pad = torch.zeros(b, h, n, d, dtype=t.dtype, device=t.device)  # zero keys/values (masked out)
    return torch.cat([pad, t], dim=2)                          # prepend on the sequence axis


def _pad_cache_to(cache, target_len):
    for layer in cache.layers:                                 # every transformer layer
        cur = layer.keys.shape[2]                              # current cached length
        if cur < target_len:                                   # shorter rows need left-pad
            layer.keys = _left_pad_seq(layer.keys, target_len - cur)
            layer.values = _left_pad_seq(layer.values, target_len - cur)


def _append_row(base, new):
    for lb, ln in zip(base.layers, new.layers):                # merge new sequence as extra batch row
        lb.keys = torch.cat([lb.keys, ln.keys], dim=0)         # concat along batch axis
        lb.values = torch.cat([lb.values, ln.values], dim=0)


class _Seq:
    def __init__(self, rid, prompt_ids, max_new_tokens, eos_id):
        self.rid = rid                                         # request id
        self.prompt_ids = prompt_ids                           # 1D prompt token tensor
        self.prompt_len = int(prompt_ids.shape[0])             # prompt length
        self.max_new_tokens = max_new_tokens                   # cap on output
        self.eos_id = eos_id                                   # stop token
        self.generated = []                                    # produced token ids
        self.real_len = 0                                      # real tokens currently in cache
        self.next_token = None                                 # token to feed on the next decode step
        self.done = False                                      # finished flag
        self.event = threading.Event()                         # set when finished
        self.t_arrive = time.perf_counter()                    # arrival time
        self.t_first = None                                    # first-token time (TTFT)
        self.t_end = None                                      # completion time


class ContinuousBatchingEngine:
    def __init__(self, max_batch_size=8):
        self.model, self.tokenizer, self.device = load_model()  # one shared model
        self.eos_id = self.tokenizer.eos_token_id              # stop token
        self.max_batch_size = max_batch_size                   # decode batch width
        self.waiting = queue.Queue()                           # admitted-but-not-running requests
        self.running = []                                      # _Seq currently in the batch
        self.cache = None                                      # the single batched DynamicCache
        self.lpad = 0                                          # current padded cache length
        self._wake = threading.Event()                         # wake the scheduler when work arrives
        self._thread = threading.Thread(target=self._loop, daemon=True)  # background scheduler
        self._thread.start()

    def submit(self, prompt: str, max_new_tokens: int = 64) -> dict:
        text = build_prompt(self.tokenizer, prompt)            # chat template
        ids = self.tokenizer(text, return_tensors="pt").input_ids[0].to(self.device)  # 1D tokens
        seq = _Seq(self._next_id(), ids, max_new_tokens, self.eos_id)  # wrap request state
        self.waiting.put(seq)                                  # hand to scheduler
        self._wake.set()                                       # nudge the loop
        seq.event.wait()                                       # block caller until finished
        completion = self.tokenizer.decode(seq.generated, skip_special_tokens=True)  # decode reply
        total = seq.t_end - seq.t_arrive                       # end-to-end latency
        ttft = (seq.t_first - seq.t_arrive) if seq.t_first else total  # time to first token
        n = len(seq.generated)                                 # output length
        return {
            "completion": completion,                          # generated text
            "token_ids": seq.generated,                        # exact tokens (for correctness checks)
            "prompt_tokens": seq.prompt_len,                   # input size
            "tokens_generated": n,                             # output size
            "time_to_first_token_s": round(ttft, 4),           # TTFT
            "total_time_s": round(total, 4),                   # end-to-end
            "tokens_per_sec": round(n / total, 2) if total > 0 else 0.0,  # per-request throughput
        }

    _counter_lock = threading.Lock()                           # protect the id counter
    _counter = 0

    def _next_id(self):
        with ContinuousBatchingEngine._counter_lock:           # atomic increment
            ContinuousBatchingEngine._counter += 1
            return ContinuousBatchingEngine._counter

    @torch.inference_mode()                                     # no autograd in the scheduler
    def _prefill(self, seq: _Seq):
        ids = seq.prompt_ids.unsqueeze(0)                      # [1, prompt_len]
        out = self.model(input_ids=ids, use_cache=True)        # populate this sequence's cache
        single_cache = out.past_key_values                     # [1, H, prompt_len, D] per layer
        next_id = int(out.logits[0, -1].argmax())              # first generated token
        seq.real_len = seq.prompt_len                          # cache now holds the prompt
        seq.next_token = next_id                               # feed this on the first decode step
        seq.generated.append(next_id)                          # count the first token
        seq.t_first = time.perf_counter()                      # TTFT lands at prefill
        if next_id == self.eos_id or len(seq.generated) >= seq.max_new_tokens:
            seq.done = True                                    # one-token replies finish immediately
        return single_cache

    def _admit(self):
        while len(self.running) < self.max_batch_size and not self.waiting.empty():
            try:
                seq = self.waiting.get_nowait()                # next queued request
            except queue.Empty:
                break
            single_cache = self._prefill(seq)                  # prefill it alone
            if self.cache is None:                             # first sequence seeds the batch
                self.cache = single_cache
                self.lpad = seq.real_len
            else:
                target = max(self.lpad, seq.real_len)          # align lengths before merging
                _pad_cache_to(self.cache, target)              # grow existing rows if needed
                _pad_cache_to(single_cache, target)            # grow the new row if needed
                self.lpad = target
                _append_row(self.cache, single_cache)          # add as a new batch row
            if seq.done:                                       # finished during prefill
                self._finish(seq, just_prefilled=True)
            else:
                self.running.append(seq)                       # joins the decode batch

    @torch.inference_mode()                                     # no autograd
    def _decode_step(self):
        b = len(self.running)                                  # current batch width
        dev = self.device
        input_ids = torch.tensor([[s.next_token] for s in self.running], device=dev)  # [B,1]
        position_ids = torch.tensor([[s.real_len] for s in self.running], device=dev)  # true RoPE positions
        mask = torch.zeros(b, self.lpad + 1, dtype=torch.long, device=dev)  # [B, L+1]
        for i, s in enumerate(self.running):                   # mark each row's valid span (right-aligned)
            valid = s.real_len + 1                              # real tokens + the new one
            mask[i, self.lpad + 1 - valid:] = 1                # left side stays padded (0)
        cache_position = torch.tensor([self.lpad], device=dev)  # all rows write at the same next slot

        out = self.model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                         past_key_values=self.cache, use_cache=True, cache_position=cache_position)
        self.lpad += 1                                         # cache grew by one position
        next_ids = out.logits[:, -1, :].argmax(dim=-1)         # [B] greedy picks

        finished = []                                          # rows to evict this step
        for i, s in enumerate(self.running):
            tok = int(next_ids[i])                             # this row's next token
            s.real_len += 1                                    # it just consumed its fed token
            s.next_token = tok                                 # feed next step
            s.generated.append(tok)                            # record output
            if tok == self.eos_id or len(s.generated) >= s.max_new_tokens:
                s.done = True                                  # hit a stop condition
                finished.append(i)
        if finished:
            self._evict(finished)                              # drop finished rows from batch + cache

    def _evict(self, finished_idx):
        keep = [i for i in range(len(self.running)) if i not in set(finished_idx)]  # surviving rows
        for i in finished_idx:                                 # deliver finished sequences
            self._finish(self.running[i])
        if keep:                                               # keep survivors in the cache
            self.cache.batch_select_indices(torch.tensor(keep, device=self.device))
        else:                                                  # batch emptied
            self.cache = None
            self.lpad = 0
        self.running = [self.running[i] for i in keep]         # compact the running list

    def _finish(self, seq: _Seq, just_prefilled=False):
        seq.t_end = time.perf_counter()                        # completion time
        seq.event.set()                                        # unblock the waiting caller

    def _loop(self):
        while True:                                            # scheduler runs forever
            if not self.running and self.waiting.empty():      # idle
                self._wake.wait()                              # sleep until a request arrives
                self._wake.clear()
            self._admit()                                      # fill free slots from the queue
            if self.running:
                self._decode_step()                            # one batched step across all rows


_engine = None                                                 # process-wide singleton
_engine_lock = threading.Lock()                                # guard lazy init


def get_engine() -> ContinuousBatchingEngine:
    global _engine
    with _engine_lock:                                         # build once
        if _engine is None:
            _engine = ContinuousBatchingEngine()
        return _engine


def batched_generate(prompt: str, max_new_tokens: int = 64) -> dict:
    return get_engine().submit(prompt, max_new_tokens)         # enqueue + wait for result
