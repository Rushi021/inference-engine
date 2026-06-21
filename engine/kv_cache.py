import math                                                     # sqrt for attention scaling
import torch                                                     # tensors
import torch.nn.functional as F                                  # softmax


def scaled_dot_product_attention(q, k, v):
    d = q.shape[-1]                                              # head dimension
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)           # [B,H,Lq,Lk] similarity, scaled
    weights = F.softmax(scores, dim=-1)                         # attention distribution over keys
    return weights @ v                                          # [B,H,Lq,D] weighted sum of values


class KVCache:                                                   # the mechanism, from scratch
    def __init__(self):
        self.k = None                                           # cached keys   [B,H,L,D]
        self.v = None                                           # cached values [B,H,L,D]

    def append(self, k_new, v_new):
        if self.k is None:                                     # first token
            self.k, self.v = k_new, v_new
        else:                                                  # reuse old, append new along seq dim
            self.k = torch.cat([self.k, k_new], dim=2)         # grow keys by one position
            self.v = torch.cat([self.v, v_new], dim=2)         # grow values by one position
        return self.k, self.v                                  # full K,V seen so far

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]        # cached sequence length


def _toy_setup(seq_len, n_heads=4, head_dim=16, seed=0):
    torch.manual_seed(seed)                                     # deterministic so both paths match
    md = n_heads * head_dim                                     # model dimension
    dt = torch.float64                                          # float64 → cached/uncached match to ~1e-12
    x = torch.randn(1, seq_len, md, dtype=dt)                  # fake token embeddings [B,N,model_dim]
    wq = torch.randn(md, md, dtype=dt)                         # query projection
    wk = torch.randn(md, md, dtype=dt)                         # key projection
    wv = torch.randn(md, md, dtype=dt)                         # value projection
    return x, wq, wk, wv, n_heads, head_dim


def _project_heads(x, w, n_heads, head_dim):
    out = x @ w                                                 # linear projection [B,L,model_dim]
    b, length, _ = out.shape
    return out.view(b, length, n_heads, head_dim).transpose(1, 2)  # -> [B,H,L,D]


def decode_uncached(seq_len):
    x, wq, wk, wv, h, d = _toy_setup(seq_len)                  # shared toy weights/inputs
    outputs, projections = [], 0                                # results + work counter
    for t in range(seq_len):                                   # generate tokens one at a time
        ctx = x[:, : t + 1, :]                                 # NAIVE: re-read all tokens so far
        q = _project_heads(x[:, t : t + 1, :], wq, h, d)       # query for the new token only
        k = _project_heads(ctx, wk, h, d)                      # recompute K for ALL tokens
        v = _project_heads(ctx, wv, h, d)                      # recompute V for ALL tokens
        projections += 2 * (t + 1)                             # k+v projected over t+1 tokens
        outputs.append(scaled_dot_product_attention(q, k, v))  # attend
    return torch.cat(outputs, dim=2), projections             # [B,H,N,D], total K/V projections


def decode_cached(seq_len):
    x, wq, wk, wv, h, d = _toy_setup(seq_len)                  # identical toy weights/inputs
    cache = KVCache()                                          # our from-scratch cache
    outputs, projections = [], 0                                # results + work counter
    for t in range(seq_len):                                   # generate tokens one at a time
        xt = x[:, t : t + 1, :]                                # only the new token
        q = _project_heads(xt, wq, h, d)                       # query for the new token
        k_new = _project_heads(xt, wk, h, d)                   # project K for ONE new token
        v_new = _project_heads(xt, wv, h, d)                   # project V for ONE new token
        projections += 2                                       # only 2 projections per step
        k, v = cache.append(k_new, v_new)                      # reuse cached K/V, append the new ones
        outputs.append(scaled_dot_product_attention(q, k, v))  # attend over full cached context
    return torch.cat(outputs, dim=2), projections             # [B,H,N,D], total K/V projections


def kv_cache_memory_bytes(num_layers, num_kv_heads, head_dim,
                          seq_len, batch_size=1, dtype_bytes=2):
    return (2 * num_layers * batch_size * num_kv_heads          # 2 = keys AND values
            * head_dim * seq_len * dtype_bytes)                 # the canonical KV-cache size formula


def model_kv_cache_report(model_config, seq_len, batch_size=1, dtype_bytes=2):
    num_layers = model_config.num_hidden_layers                # transformer blocks
    num_kv_heads = getattr(model_config, "num_key_value_heads", # GQA: fewer KV heads than query heads
                           model_config.num_attention_heads)
    head_dim = model_config.hidden_size // model_config.num_attention_heads  # per-head width
    total = kv_cache_memory_bytes(num_layers, num_kv_heads, head_dim,
                                  seq_len, batch_size, dtype_bytes)
    return {
        "num_layers": num_layers,                              # L
        "num_kv_heads": num_kv_heads,                          # H_kv (GQA)
        "head_dim": head_dim,                                  # D
        "seq_len": seq_len,                                    # S
        "batch_size": batch_size,                              # B
        "dtype_bytes": dtype_bytes,                            # bytes per element
        "bytes": total,                                        # raw size
        "mib": round(total / (1024 ** 2), 2),                 # human-readable
    }
