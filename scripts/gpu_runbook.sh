#!/usr/bin/env bash
# One-shot GPU benchmark: runs naive + batched + vLLM + quantization on ONE box (same GPU,
# fair comparison), then leaves results in results/latest.json + results/quant.json.
# Works on a bare CUDA VM (GCP Deep Learning "common-cu*" image): installs pip+venv itself.
set -e

echo "== deps (apt: pip, venv, git) =="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv git >/dev/null

echo "== python venv + pip installs =="
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip
pip install -q vllm bitsandbytes                                              # vLLM brings matched torch+CUDA
pip install -q fastapi "uvicorn[standard]" pydantic httpx accelerate matplotlib  # our serving + harness deps

echo "== clone =="
cd ~ && rm -rf llm-inference-engine
git clone -q https://github.com/Rushi021/inference-engine.git
cd llm-inference-engine
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "== 1/4 my engines (naive + batched) =="
python benchmark/runner.py --concurrency 1,4,16,64 --requests 64 --max-tokens 128
cp results/latest.json results/mine.json

echo "== 2/4 start vLLM (OpenAI server) =="
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --dtype half --enforce-eager \
    --max-model-len 2048 > vllm.log 2>&1 &                                    # background
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8000/health >/dev/null && echo "vLLM up" || { echo "vLLM FAILED:"; tail -30 vllm.log; }

echo "== 3/4 benchmark vLLM through the SAME harness =="
python benchmark/runner.py --url http://127.0.0.1:8000 --api openai \
    --model Qwen/Qwen2.5-0.5B-Instruct --engines vllm \
    --concurrency 1,4,16,64 --requests 64 --max-tokens 128
python - <<'PY'
import json                                                                   # merge mine + vllm rows
mine = json.load(open("results/mine.json"))
vllm = [r for r in json.load(open("results/latest.json")) if r["engine"] == "vllm"]
json.dump(mine + vllm, open("results/latest.json", "w"), indent=2)
print("merged", len(mine), "mine +", len(vllm), "vllm rows")
PY

echo "== 4/4 quantization study =="
python benchmark/quant_compare.py fp16,int8,int4

echo ""
echo "############ COPY EVERYTHING BELOW THIS LINE AND PASTE BACK ############"
echo "===== latest.json ====="
cat results/latest.json
echo "===== quant.json ====="
cat results/quant.json
echo "############ END ############"
