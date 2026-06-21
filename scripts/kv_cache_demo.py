import sys                                                     # path setup
from pathlib import Path                                        # locate repo root

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import engine when run directly

from engine.kv_cache import decode_cached, decode_uncached, model_kv_cache_report
from engine.manual_decode import manual_generate
from engine.model import load_model


def main():
    print("=" * 60)
    print("TOY ATTENTION — from-scratch KV cache vs recompute")
    print("=" * 60)
    for n in (16, 64, 128):                                    # show the O(n) vs O(n^2) gap grow
        _, wc = decode_cached(n)
        _, wu = decode_uncached(n)
        print(f"seq_len={n:>4}:  cached K/V projections={wc:>5}   "
              f"uncached={wu:>6}   ratio={wu / wc:.1f}x")

    print("\n" + "=" * 60)
    print("REAL MODEL — manual decode loop, cached vs uncached")
    print("=" * 60)
    prompt = "Explain the KV cache in two sentences."
    c = manual_generate(prompt, max_new_tokens=48, use_cache=True)
    u = manual_generate(prompt, max_new_tokens=48, use_cache=False)
    print(f"identical tokens: {c['token_ids'] == u['token_ids']}")
    print(f"cached  : {c['total_time_s']:.3f}s  {c['tokens_per_sec']:>6.2f} tok/s")
    print(f"uncached: {u['total_time_s']:.3f}s  {u['tokens_per_sec']:>6.2f} tok/s")
    print(f"speedup : {u['total_time_s'] / c['total_time_s']:.1f}x")

    print("\n" + "=" * 60)
    print("KV CACHE MEMORY — actual model config")
    print("=" * 60)
    cfg = load_model()[0].config
    for s in (512, 2048, 8192):                                # memory scales linearly with seq len
        r = model_kv_cache_report(cfg, seq_len=s)
        print(f"seq_len={s:>5}: {r['mib']:>7.2f} MiB  "
              f"(L={r['num_layers']}, kv_heads={r['num_kv_heads']}, head_dim={r['head_dim']})")


if __name__ == "__main__":
    main()
