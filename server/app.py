from fastapi import FastAPI                                      # serving framework
from pydantic import BaseModel, Field                            # request/response validation

from engine.naive import naive_generate                          # the Phase 1 baseline engine
from engine.config import MODEL_ID, DEFAULT_MAX_NEW_TOKENS       # shared defaults

app = FastAPI(title="LLM Inference — Naive Baseline", version="1.0")  # the control-group server


class GenerateRequest(BaseModel):
    prompt: str                                                  # user prompt
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=1024)  # bounded reply length


class GenerateResponse(BaseModel):
    completion: str                                              # generated text
    engine: str                                                  # which engine produced this
    model: str                                                   # which model
    prompt_tokens: int                                           # input token count
    tokens_generated: int                                        # output token count
    time_to_first_token_s: float                                 # TTFT latency
    total_time_s: float                                          # end-to-end latency
    tokens_per_sec: float                                        # per-request throughput


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "engine": "naive"}  # liveness + what is loaded


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    result = naive_generate(req.prompt, req.max_tokens)          # sequential, no batching
    return GenerateResponse(engine="naive", model=MODEL_ID, **result)  # completion + timing metadata
