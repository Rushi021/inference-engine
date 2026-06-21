# Project: Distributed LLM Inference & Serving Platform — Claude Code Build Plan

This document is your complete build guide. Use it in two ways:
1. **Paste the "MASTER BRIEF" section into Claude Code first** to establish context for the whole project.
2. **Then feed phases one at a time** (Phase 0 → Phase 7). Do not paste all phases at once — Claude Code works best one phase per session, with you reviewing and testing between each.

---

## MASTER BRIEF (paste this first)

I'm building a portfolio project: a self-hosted LLM inference and serving engine with benchmarked optimizations, deployed on Google Cloud. The goal is to demonstrate production ML systems skills for big tech SDE/ML/AI interviews. I need to deeply understand every component because I'll defend this in technical interviews.

Key principles for how I want you to work:
- I am new to cloud/Kubernetes/GPU infra, so explain infra concepts as we go, in plain language, before running commands.
- Build incrementally. One phase at a time. After each phase, write tests and update the README before moving on.
- Comment code with concise, right-aligned inline comments on each meaningful line, plain descriptive language, no docstrings or block comments.
- Prioritize my understanding over speed. When you make an architectural decision, briefly explain why.
- I'm on a $1000 GCP credit budget. Never leave GPU instances running. Always remind me to shut things down. Default to local/CPU work; only use GPUs when actively benchmarking.

The architecture, in order of build:
1. A naive baseline inference server (sequential generation, no optimizations) — the control group.
2. A from-scratch optimized engine with my own KV cache and a continuous batching scheduler — so I understand these from first principles.
3. vLLM as a production baseline to benchmark against.
4. A benchmark harness measuring throughput (tokens/sec), latency (p50/p99), and GPU utilization under varying concurrency.
5. Quantization comparison (FP16 vs INT8) measuring the throughput-vs-quality trade-off.
6. Containerization + deployment on GCP (single GPU VM first, then GKE with autoscaling).
7. A results dashboard and a benchmark-driven README with graphs.

Tech stack: Python, PyTorch, Hugging Face Transformers, FastAPI for the serving API, vLLM for the baseline, Docker, Google Cloud (Compute Engine GPU VM, then GKE), Locust or k6 for load testing, and matplotlib/a simple frontend for the dashboard. Model: a small open model that fits a single mid-tier GPU — Qwen2.5-0.5B or Llama-3.2-1B — so iteration is cheap.

Do not write any code yet. Confirm you understand the plan and the constraints, then wait for me to send Phase 0.

---

## PHASE 0 — Project scaffold & local dev environment (local, no GPU, no cost)

Goal: clean repo structure, dependency management, and a "hello world" that proves the model loads and generates locally on CPU.

Paste to Claude Code:

> Phase 0. Set up the project scaffold locally — no cloud, no GPU yet, CPU only.
> 1. Create a clean repo structure: `engine/` (inference code), `server/` (FastAPI app), `benchmark/` (load tests + harness), `deploy/` (Docker + GCP/k8s configs, empty for now), `dashboard/`, `tests/`, plus README, .gitignore (Python/Jupyter/venv/.env/large model files), and a requirements file or pyproject.
> 2. Set up a virtual environment and pin core deps: torch (CPU build for now), transformers, fastapi, uvicorn, pydantic, pytest. Don't install vLLM yet (it needs GPU).
> 3. Write a minimal script that loads Qwen2.5-0.5B-Instruct (or Llama-3.2-1B) from Hugging Face and generates one completion from a prompt, printed to console. This is just to confirm the model loads and runs on my CPU.
> 4. Add a basic pytest that asserts the model loads and produces non-empty output.
> 5. Explain the repo structure choices and what each directory is for before I commit.
>
> After this works, show me the structure and the test passing, then stop.

What you learn: project hygiene, HF model loading, the generation API.

---

## PHASE 1 — Naive baseline inference server (local CPU, no cost)

Goal: a working FastAPI server that does the dumbest possible generation — one request at a time, sequential, no batching, no KV cache reuse. This is your control group; every optimization later is measured against this.

Paste to Claude Code:

> Phase 1. Build the naive baseline inference server in `server/`.
> 1. A FastAPI app with a `/generate` endpoint taking a prompt and max_tokens, returning the completion plus timing metadata (time to first token, total time, tokens generated, tokens/sec).
> 2. Generation should be deliberately naive: process one request fully before the next, standard HF `.generate()`, no custom batching. Add a comment explaining that this is the intentional baseline.
> 3. Add a `/health` endpoint.
> 4. Instrument timing carefully — I want per-request latency and throughput captured accurately, because these numbers become my baseline benchmarks.
> 5. Write tests hitting the endpoint with the test client.
> 6. Explain where the performance bottlenecks are in this naive version and which ones the next phases will fix — this framing matters for my interview story.
>
> Show me the server running and a sample request/response with timing, then stop.

What you learn: serving APIs, why naive serving wastes the GPU, the metrics that matter.

---

## PHASE 2 — From-scratch KV cache (local CPU, the core learning phase)

Goal: implement your own KV cache so you understand it cold. This is the single most-asked LLM-systems interview concept.

Paste to Claude Code:

> Phase 2. Implement a KV cache from scratch in `engine/`, separate from HF's built-in caching, so I understand the mechanism.
> 1. First, explain in plain language what the KV cache is, why autoregressive generation without it is O(n^2) in attention compute, and how caching makes each new token O(n). Use a concrete small example.
> 2. Implement a minimal decode loop that manages the key/value tensors manually across generation steps — appending each step's K and V, reusing cached ones rather than recomputing. Keep it readable over clever.
> 3. Build it so I can run generation WITH my KV cache vs WITHOUT (recomputing every step) and measure the speedup on CPU.
> 4. Add tests verifying cached and uncached generation produce identical token outputs (correctness) and that cached is faster.
> 5. Walk me through the memory cost of the KV cache — the formula for how much memory it uses as a function of layers, heads, head_dim, sequence length, batch size. I need to be able to recite this.
>
> Show me the correctness test passing and the with/without timing, then stop.

What you learn: the KV cache from first principles, its memory arithmetic, why it's the central constraint in LLM serving. Highest interview-signal phase.

---

## PHASE 3 — Continuous batching scheduler (local CPU, the second core learning phase)

Goal: implement a request scheduler that batches multiple in-flight requests and adds/removes them mid-generation. This is the hard part and the thing almost no student has built.

Paste to Claude Code:

> Phase 3. Implement continuous (in-flight) batching in `engine/`.
> 1. Explain the difference between static batching (wait for a full batch, all finish together) and continuous batching (requests join and leave the batch at different steps), and why continuous batching dramatically improves GPU utilization for variable-length generation. This is a key interview talking point — make sure I understand it.
> 2. Build a scheduler that maintains a set of active sequences, runs one decode step across the whole batch at a time, evicts sequences that hit EOS or max_tokens, and admits waiting requests into freed slots.
> 3. Integrate it with the KV cache from Phase 2 (each sequence has its own cache state in the batch).
> 4. Expose it through the FastAPI server as an alternative engine, so I can switch between naive (Phase 1) and batched (this phase).
> 5. Add tests: concurrent requests get correct individual outputs, and batched throughput beats naive under concurrency.
> 6. Explain the scheduling decisions you made and the tradeoffs (e.g., what happens under memory pressure, how you handle the batch being full).
>
> Show me concurrent requests returning correct results and a throughput comparison vs naive, then stop.

What you learn: the scheduling layer that production engines live or die on, GPU utilization reasoning, concurrency handling.

---

## PHASE 4 — Benchmark harness & load testing (local first, then your first GPU run)

Goal: a rigorous, repeatable benchmark that produces the graphs for your README. This phase has the first (brief, cheap) GPU usage.

Paste to Claude Code:

> Phase 4. Build a benchmark harness in `benchmark/`.
> 1. A load-testing setup (Locust or a custom async client) that fires N concurrent requests at a configurable rate and records: throughput (tokens/sec aggregate), per-request latency p50/p90/p99, time-to-first-token, and GPU utilization (via nvidia-smi sampling when on GPU).
> 2. A runner that benchmarks each engine variant — naive, my batched engine — under a sweep of concurrency levels (1, 4, 16, 64 concurrent requests) and saves results to structured files (JSON/CSV).
> 3. A plotting script that turns results into clean graphs: throughput vs concurrency, latency percentiles vs concurrency, one chart per metric comparing the engine variants.
> 4. Explain how to run this locally on CPU first (small numbers) to validate the harness works, before I spend any GPU money.
> 5. Give me a separate short checklist of exactly what to do when I move to a GPU instance — how to confirm GPU is being used, how to sample utilization, and the reminder to shut the instance down.
>
> Validate the harness on CPU with small numbers and show me a sample graph, then stop. I'll run the real GPU benchmarks after the deploy phase.

What you learn: benchmarking methodology, percentile latency, load testing, the discipline of measurement.

---

## PHASE 5 — vLLM baseline + quantization comparison (GPU phase — budget-aware)

Goal: bring in the production baseline and the quantization trade-off study. This needs GPU; do it in a focused session to limit cost.

Paste to Claude Code:

> Phase 5. Add vLLM as a production baseline and a quantization comparison. This phase needs a GPU, so first give me the cheapest way to do it and a plan to minimize runtime before we touch the cloud.
> 1. Explain what vLLM does that my engine doesn't (PagedAttention especially) and why it'll be faster — I need to explain this gap intelligently rather than pretend my engine wins.
> 2. Add a vLLM-backed engine variant behind the same server interface, so the benchmark harness can hit it identically to my engines.
> 3. Set up a quantization comparison: serve the model in FP16 vs INT8 (or available quantized format), and extend the harness to measure throughput, memory footprint, AND output quality (so I can show the quality-vs-speed tradeoff, not just speed). Suggest a simple, defensible quality metric for this.
> 4. Give me a tight, ordered runbook for the GPU session: what to launch, what to benchmark, in what order, so I get all three engines (naive, mine, vLLM) and both quant levels measured in the shortest possible GPU time. Then the shutdown reminder.
>
> Prepare everything that can be prepared on CPU/locally first. Give me the GPU runbook. Don't assume the GPU is running yet.

What you learn: vLLM/PagedAttention, quantization in practice, honest baseline comparison, quality-vs-throughput tradeoffs.

---

## PHASE 6 — Containerize & deploy on GCP (infra phase — explained step by step)

Goal: get it running on Google Cloud, first as a single GPU VM, then on GKE with autoscaling. Since you're new to this, the prompt forces step-by-step teaching.

Paste to Claude Code:

> Phase 6. Containerize and deploy to Google Cloud. I am new to Docker, GCP, and Kubernetes, so teach me each step before running it, and keep cost control front of mind throughout.
> 1. Write a Dockerfile for the serving app (GPU-capable base image). Explain each layer.
> 2. First deployment target: a single GCP Compute Engine GPU VM. Walk me through, in plain steps: creating the instance (which machine type and GPU is cheapest that works, and prefer a spot/preemptible instance), getting my container onto it, running it, and hitting the endpoint from my laptop. Explain every gcloud command before I run it.
> 3. Then GKE: explain what Kubernetes gives me here (autoscaling under load) in plain language, then provide the minimal manifests — a deployment, a service, and a Horizontal Pod Autoscaler — to run the server on GKE with a GPU node pool that scales with load. Explain each manifest field that matters.
> 4. Show me how to drive load at the GKE deployment with my benchmark harness and observe autoscaling happen — this is a great thing to capture for the README.
> 5. Give me an explicit teardown checklist: delete the GKE cluster, delete node pools, stop/delete VMs, confirm nothing is still billing. Make this impossible to forget.
>
> Start with the Dockerfile and the single-VM path. Don't move to GKE until I confirm the VM deployment worked. Explain before executing at every step.

What you learn: Docker, GCP Compute Engine, Kubernetes basics, autoscaling, cloud cost management. This is the infra skill block you're missing.

---

## PHASE 7 — Dashboard & benchmark-driven README (local, no cost)

Goal: the presentation layer that makes this land in 6 seconds with a recruiter and 20 minutes with an interviewer.

Paste to Claude Code:

> Phase 7. Build the results dashboard and the final README.
> 1. A simple dashboard (a clean static page or small Streamlit app in `dashboard/`) that displays the benchmark graphs and a summary table comparing all engine variants and quant levels across throughput, p99 latency, and memory.
> 2. Rewrite the README as the centerpiece of the project: a crisp problem statement, an architecture diagram (describe it in mermaid or ASCII), the engine variants explained, the headline benchmark results with graphs embedded, the key findings (where my engine wins/loses vs vLLM and why, the quantization tradeoff, the autoscaling demo), what I learned, and how to run it. Keep it professional and benchmark-driven, not hype.
> 3. Add a short "Key Engineering Decisions" section capturing the tradeoffs I made — this is what interviewers read.
> 4. Make sure setup instructions actually work from a clean clone (ideally a single docker compose for local).
>
> Show me the README and dashboard, then stop. After I review, help me write the GitHub repo description and a LinkedIn post about it.

What you learn: technical communication, the part most students skip and that disproportionately drives recruiter response.

---

## Budget discipline (read before any GPU phase)

- A100 ≈ $3-4/hr; $1000 ≈ ~250-300 GPU-hours. Plenty if disciplined.
- Phases 0-4 are CPU/local — effectively free. Do all of that first.
- Only Phases 5 and 6 need GPU. Batch your GPU work into focused sessions.
- Always use spot/preemptible instances.
- Set a GCP budget alert at $200 so you get warned early.
- After every GPU session, run the teardown checklist and confirm in the billing console that nothing is running.

## Interview narrative this project gives you

"I built an LLM inference engine three ways — a naive baseline, a from-scratch version with my own KV cache and continuous batching scheduler, and a vLLM-backed version — then benchmarked all three under load on GPU instances on GCP. I measured throughput and p99 latency across concurrency levels, studied the FP16-vs-INT8 quantization tradeoff, and deployed it on GKE with autoscaling. My engine got within X% of vLLM's throughput, and I can explain exactly where vLLM's PagedAttention pulls ahead and why."

That's a sentence that earns you 20 minutes of deep technical conversation in any ML/infra-adjacent interview, at any company.
