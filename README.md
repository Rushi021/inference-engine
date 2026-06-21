# Distributed LLM Inference & Serving Platform

A self-hosted LLM inference engine built three ways — a **naive baseline**, a **from-scratch
engine** with a hand-written KV cache and continuous-batching scheduler, and a **vLLM-backed**
production baseline — then benchmarked under load for throughput, latency (p50/p99), and GPU
utilization. Includes an FP16-vs-INT8 quantization study and a GCP/GKE autoscaling deployment.

> Status: **Phase 0 complete** (scaffold + local CPU model load). Build proceeds Phase 0 → 7.

## Why this project

Most "I used an LLM API" projects show no systems depth. This one demonstrates the internals that
production inference engines live or die on — KV caching, in-flight batching, GPU utilization,
quantization tradeoffs, and cloud autoscaling — all measured, not asserted.

## Repository layout

| Directory     | Purpose |
|---------------|---------|
| `engine/`     | Inference internals: model loader, KV cache, batching scheduler (Phases 0–3) |
| `server/`     | FastAPI serving layer exposing the engine variants (Phase 1+) |
| `benchmark/`  | Load-test client, sweep runner, and plotting (Phase 4) |
| `deploy/`     | Dockerfile, GCP VM scripts, GKE manifests (Phase 6) |
| `dashboard/`  | Results dashboard (Phase 7) |
| `tests/`      | pytest suite |
| `scripts/`    | One-off utilities (smoke tests, runbooks) |

## Quickstart (local, CPU, no cost)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Prove the model loads and generates on CPU
python scripts/smoke_generate.py "What is a KV cache?"

# Run the test suite
pytest -q
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct` — tiny, ungated, and CPU-friendly so iteration is
cheap. Override with `MODEL_ID=...`.

## Build phases

0. **Scaffold + local model load** ✅
1. Naive baseline FastAPI inference server
2. From-scratch KV cache
3. Continuous batching scheduler
4. Benchmark harness & load testing
5. vLLM baseline + quantization comparison (GPU)
6. Containerize & deploy on GCP / GKE (GPU)
7. Results dashboard & benchmark-driven README
