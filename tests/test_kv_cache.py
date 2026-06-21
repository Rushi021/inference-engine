import torch                                                     # allclose

from engine.kv_cache import (
    decode_cached, decode_uncached, kv_cache_memory_bytes, model_kv_cache_report,
)
from engine.manual_decode import manual_generate                # real-model decode loop
from engine.model import load_model


def test_toy_cached_matches_uncached():
    out_cached, _ = decode_cached(32)                           # KV-cache path
    out_uncached, _ = decode_uncached(32)                       # recompute-everything path
    assert torch.allclose(out_cached, out_uncached, atol=1e-9)  # identical math (float64)


def test_toy_cached_does_less_work():
    _, work_cached = decode_cached(32)                          # 2 projections per step
    _, work_uncached = decode_uncached(32)                      # 2*(t+1) projections per step
    assert work_cached < work_uncached                         # O(n) beats O(n^2)
    assert work_cached == 2 * 32                               # exactly 2 per token
    assert work_uncached == 2 * (32 * 33) // 2                 # 2 * triangular number


def test_real_model_cached_matches_uncached():
    cached = manual_generate("List three colors.", max_new_tokens=20, use_cache=True)
    uncached = manual_generate("List three colors.", max_new_tokens=20, use_cache=False)
    assert cached["token_ids"] == uncached["token_ids"]        # correctness: identical tokens
    assert len(cached["completion"].strip()) > 0               # produced real text


def test_real_model_cached_is_faster():
    cached = manual_generate("Write a short sentence about cats.", max_new_tokens=40, use_cache=True)
    uncached = manual_generate("Write a short sentence about cats.", max_new_tokens=40, use_cache=False)
    assert cached["total_time_s"] < uncached["total_time_s"]   # cache wins on real model


def test_memory_formula():
    bytes_1 = kv_cache_memory_bytes(num_layers=24, num_kv_heads=2, head_dim=64,
                                    seq_len=1024, batch_size=1, dtype_bytes=2)
    bytes_2 = kv_cache_memory_bytes(num_layers=24, num_kv_heads=2, head_dim=64,
                                    seq_len=2048, batch_size=1, dtype_bytes=2)
    assert bytes_2 == 2 * bytes_1                              # linear in sequence length
    report = model_kv_cache_report(load_model()[0].config, seq_len=2048)
    assert report["bytes"] > 0 and report["num_layers"] > 0    # works on the real config
