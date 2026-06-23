import asyncio                                                   # drive async adapter test
import math                                                      # isfinite
import httpx                                                     # ASGI client
from fastapi import FastAPI, Request                             # fake OpenAI server

from benchmark.quality import perplexity, token_agreement, EVAL_TEXTS
from benchmark.load_test import run_load
from engine.model import load_model


def test_token_agreement():
    assert token_agreement([1, 2, 3], [1, 2, 3]) == 1.0        # identical
    assert token_agreement([1, 2, 3], [9, 9, 9]) == 0.0        # disjoint
    assert token_agreement([1, 2, 3, 4], [1, 2, 9, 9]) == 0.5  # half match
    assert token_agreement([], [1]) == 0.0                     # empty safe


def test_perplexity_on_real_model_is_sane():
    model, tok, _ = load_model()                               # fp32 on CPU/MPS
    ppl = perplexity(model, tok, EVAL_TEXTS)                   # quality on coherent text
    assert math.isfinite(ppl)                                  # well-defined
    assert 1.0 < ppl < 1000.0                                  # plausible LM perplexity range


def _fake_openai_app():
    app = FastAPI()                                            # stands in for a vLLM server

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        return {"choices": [{"text": "ok ok"}],                # OpenAI-shaped response
                "usage": {"completion_tokens": body["max_tokens"]}}

    return app


def test_openai_adapter_path():
    async def go():
        transport = httpx.ASGITransport(app=_fake_openai_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=30) as client:
            return await run_load(client, engine="vllm", concurrency=2, num_requests=4,
                                  max_tokens=5, api="openai", model="test-model")
    s = asyncio.run(go())
    assert s["completed"] == 4                                 # all requests parsed
    assert s["total_tokens"] == 20                             # 4 requests x 5 tokens from usage block
    assert s["throughput_tok_s"] > 0                           # adapter produced metrics
