import math                                                     # floor/ceil for interpolation


def percentile(values, p):
    if not values:                                             # no data
        return 0.0
    s = sorted(values)                                         # percentiles need sorted input
    if len(s) == 1:                                            # single sample
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)                             # fractional rank
    f, c = math.floor(k), math.ceil(k)                         # bracketing indices
    if f == c:                                                 # exact hit
        return float(s[int(k)])
    return float(s[f] + (s[c] - s[f]) * (k - f))              # linear interpolation (numpy-style)


def summarize(records, wall_s):
    n = len(records)                                           # completed requests
    latencies = [r["latency_s"] for r in records]             # client-side end-to-end latency
    ttfts = [r["ttft_s"] for r in records]                    # server-reported time-to-first-token
    total_tokens = sum(r["tokens"] for r in records)          # all generated tokens
    return {
        "completed": n,                                        # how many finished
        "wall_s": round(wall_s, 4),                            # total wall time of the run
        "throughput_tok_s": round(total_tokens / wall_s, 2) if wall_s > 0 else 0.0,  # aggregate
        "req_per_s": round(n / wall_s, 3) if wall_s > 0 else 0.0,  # request rate achieved
        "latency_p50_s": round(percentile(latencies, 50), 4),  # median latency
        "latency_p90_s": round(percentile(latencies, 90), 4),  # tail
        "latency_p99_s": round(percentile(latencies, 99), 4),  # worst-case-ish
        "ttft_mean_s": round(sum(ttfts) / n, 4) if n else 0.0,  # mean time-to-first-token
        "total_tokens": total_tokens,                          # raw token count
    }
