from engine.model import load_model, generate                   # the two things Phase 0 must prove work


def test_model_loads():
    model, tokenizer, device = load_model()                     # download + load once (cached)
    assert model is not None                                    # weights actually loaded
    assert tokenizer is not None                                # tokenizer actually loaded
    assert device in {"cpu", "mps", "cuda"}                     # resolved to a real device


def test_generation_non_empty():
    reply = generate("Say hello.", max_new_tokens=16)           # tiny generation to keep CPU fast
    assert isinstance(reply, str)                               # decoded to text
    assert len(reply.strip()) > 0                               # produced actual content
