import sys                                                       # args
import json                                                      # read results
import base64                                                     # inline images
from pathlib import Path                                          # paths

ROOT = Path(__file__).resolve().parent.parent                    # repo root
RESULTS = ROOT / "results"                                        # benchmark outputs
OUT = Path(__file__).resolve().parent / "index.html"            # generated dashboard

CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1000px;margin:30px auto;
 padding:0 16px;color:#1a1a1a;line-height:1.5}
h1{border-bottom:3px solid #2563eb;padding-bottom:8px}
h2{color:#1e3a8a;margin-top:32px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{border:1px solid #d1d5db;padding:7px 10px;text-align:right}
th{background:#eff6ff;text-align:center}
td:first-child,th:first-child{text-align:left}
tr:nth-child(even){background:#f8fafc}
img{max-width:100%;border:1px solid #e5e7eb;border-radius:8px;margin:8px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.muted{color:#64748b;font-size:13px}
.best{font-weight:700;color:#15803d}
"""


def _img_tag(name):
    p = RESULTS / name                                          # PNG produced by benchmark/plot.py
    if not p.exists():
        return f"<p class='muted'>[{name} not generated yet]</p>"
    b64 = base64.b64encode(p.read_bytes()).decode()            # inline so the file is self-contained
    return f"<img src='data:image/png;base64,{b64}' alt='{name}'>"


def _bench_table(rows):
    head = ("<tr><th>engine</th><th>concurrency</th><th>throughput (tok/s)</th>"
            "<th>p50 (s)</th><th>p99 (s)</th><th>TTFT (s)</th></tr>")
    best = max((r["throughput_tok_s"] for r in rows), default=0)  # highlight peak throughput
    body = ""
    for r in sorted(rows, key=lambda r: (r["engine"], r["concurrency"])):
        cls = " class='best'" if r["throughput_tok_s"] == best else ""
        body += (f"<tr><td>{r['engine']}</td><td>{r['concurrency']}</td>"
                 f"<td{cls}>{r['throughput_tok_s']}</td><td>{r['latency_p50_s']}</td>"
                 f"<td>{r['latency_p99_s']}</td><td>{r['ttft_mean_s']}</td></tr>")
    return f"<table>{head}{body}</table>"


def _quant_table(rows):
    head = ("<tr><th>precision</th><th>throughput (tok/s)</th><th>peak mem (MiB)</th>"
            "<th>perplexity</th><th>agreement vs fp16</th></tr>")
    body = ""
    for r in rows:
        body += (f"<tr><td>{r['quant']}</td><td>{r['throughput_tok_s']}</td>"
                 f"<td>{r['peak_mem_mib']}</td><td>{r['perplexity']}</td>"
                 f"<td>{r['token_agreement_vs_fp16']}</td></tr>")
    return f"<table>{head}{body}</table>"


def main():
    bench_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "latest.json"
    rows = json.loads(bench_path.read_text()) if bench_path.exists() else []
    quant_path = RESULTS / "quant.json"
    quant = json.loads(quant_path.read_text()) if quant_path.exists() else []

    note = "" if quant else ("<p class='muted'>Quantization table appears after running "
                             "<code>benchmark/quant_compare.py</code> on a GPU.</p>")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>LLM Inference Engine — Benchmark Dashboard</title><style>{CSS}</style></head><body>
<h1>LLM Inference Engine — Benchmark Dashboard</h1>
<p class="muted">Naive baseline vs from-scratch continuous-batching engine (and vLLM on GPU).
Generated from <code>results/</code>.</p>

<h2>Throughput &amp; latency vs concurrency</h2>
{_bench_table(rows) if rows else "<p class='muted'>Run benchmark/runner.py first.</p>"}
<div class="grid">{_img_tag('throughput.png')}{_img_tag('latency_p99.png')}
{_img_tag('latency_p50.png')}{_img_tag('ttft.png')}</div>

<h2>Quantization study (FP16 vs INT8 vs INT4)</h2>
{_quant_table(quant) if quant else note}
</body></html>"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(rows)} benchmark rows, {len(quant)} quant rows)")


if __name__ == "__main__":
    main()
