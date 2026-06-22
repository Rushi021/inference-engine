from typing import Literal                                       # restrict engine choices
from fastapi import FastAPI                                      # serving framework
from pydantic import BaseModel, Field                            # request/response validation

from engine.naive import naive_generate                          # Phase 1 baseline engine
from engine.batched import batched_generate                      # Phase 3 continuous-batching engine
from engine.config import MODEL_ID, DEFAULT_MAX_NEW_TOKENS       # shared defaults

app = FastAPI(title="LLM Inference Engine", version="3.0")       # serves both engine variants


class GenerateRequest(BaseModel):
    prompt: str                                                  # user prompt
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=1024)  # bounded reply length
    engine: Literal["naive", "batched"] = "batched"              # which engine handles this request


class GenerateResponse(BaseModel):
    completion: str                                              # generated text
    engine: str                                                  # which engine produced this
    model: str                                                   # which model
    prompt_tokens: int                                           # input token count
    tokens_generated: int                                        # output token count
    time_to_first_token_s: float                                 # TTFT latency
    total_time_s: float                                          # end-to-end latency
    tokens_per_sec: float                                        # per-request throughput


_FIELDS = ("completion", "prompt_tokens", "tokens_generated",   # response fields both engines return
           "time_to_first_token_s", "total_time_s", "tokens_per_sec")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "engines": ["naive", "batched"]}  # liveness + what is available


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    fn = batched_generate if req.engine == "batched" else naive_generate  # pick the engine
    result = fn(req.prompt, req.max_tokens)                      # run it
    fields = {k: result[k] for k in _FIELDS}                     # keep only response fields
    return GenerateResponse(engine=req.engine, model=MODEL_ID, **fields)  # completion + timing metadata
