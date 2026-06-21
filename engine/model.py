import torch                                                   # tensor + inference runtime
from functools import lru_cache                                 # cache the loaded model across calls
from transformers import AutoModelForCausalLM, AutoTokenizer    # HF model + tokenizer loaders

from engine.config import MODEL_ID, DEFAULT_DTYPE               # project-wide defaults


def pick_device():
    if torch.cuda.is_available():                              # real GPU (cloud phases)
        return "cuda"
    if torch.backends.mps.is_available():                      # Apple Silicon GPU
        return "mps"
    return "cpu"                                               # default for local Phases 0-4


@lru_cache(maxsize=2)                                          # cache on fully-resolved args
def _load_cached(model_id: str, device: str):
    dtype = torch.float32 if device == "cpu" else torch.float16  # fp16 only helps on GPU/MPS
    tokenizer = AutoTokenizer.from_pretrained(model_id)       # download/cache tokenizer
    model = AutoModelForCausalLM.from_pretrained(            # download/cache weights
        model_id, torch_dtype=dtype                           # match dtype to device
    )
    model.to(device)                                          # move weights onto chosen device
    model.eval()                                              # disable dropout, inference mode
    return model, tokenizer, device                          # caller needs all three


def load_model(model_id: str = MODEL_ID, device: str | None = None):
    device = device or pick_device()                          # resolve device before caching
    return _load_cached(model_id, device)                    # concrete args → reliable cache hits


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]          # single-turn chat format
    return tokenizer.apply_chat_template(                    # wrap with the model's chat template
        messages, tokenize=False, add_generation_prompt=True  # leave room for the assistant reply
    )


@torch.inference_mode()                                       # no autograd graph during generation
def generate(prompt: str, max_new_tokens: int = 64, model_id: str = MODEL_ID):
    model, tokenizer, device = load_model(model_id)           # reuse cached model
    text = build_prompt(tokenizer, prompt)                    # apply chat template
    inputs = tokenizer(text, return_tensors="pt").to(device)  # tokenize onto device
    output_ids = model.generate(                              # standard HF autoregressive decode
        **inputs,
        max_new_tokens=max_new_tokens,                        # cap reply length
        do_sample=False,                                      # greedy → deterministic for tests
    )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]    # drop the prompt, keep only new tokens
    return tokenizer.decode(new_ids, skip_special_tokens=True)  # decode reply to text
