# GPU serving image for the inference engine (naive + batched engines behind FastAPI).
# Base: NVIDIA CUDA runtime — provides the CUDA/cuDNN libraries torch needs on the GPU VM/GKE node.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Predictable Python behavior; cache HF weights on a mounted volume, not the image layer.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# System Python + pip (the CUDA base image ships no Python).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements FIRST so the (slow) dependency layer is cached and only rebuilds when deps change.
# On Linux the default torch wheel is the CUDA build, so this image is GPU-ready.
COPY requirements-serve.txt .
RUN pip3 install --no-cache-dir -r requirements-serve.txt

# Copy only what the server needs (keeps the image small; see .dockerignore for exclusions).
COPY engine/ engine/
COPY server/ server/

EXPOSE 8000

# 0.0.0.0 so the container is reachable from outside; port 8000 matches the service/probe config.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
