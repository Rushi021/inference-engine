import torch                                                     # dtypes
from transformers import AutoModelForCausalLM, AutoTokenizer      # loaders

from engine.config import MODEL_ID                               # default model

# NOTE: int8/int4 require an NVIDIA GPU (bitsandbytes is CUDA-only). fp16 loads anywhere.
# This module is exercised on GPU in Phase 5; perplexity/quality (benchmark/quality.py) runs on CPU.


def load_quantized(model_id: str = MODEL_ID, quant: str = "fp16"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)          # tokenizer is quant-independent

    if quant == "fp16":                                         # half precision, the quality reference
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda")
    elif quant == "int8":                                       # 8-bit weights via bitsandbytes
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="cuda")
    elif quant == "int4":                                       # 4-bit (NF4) weights
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="cuda")
    else:
        raise ValueError(f"unknown quant level: {quant!r}")     # guard typos

    model.eval()                                                # inference mode
    return model, tokenizer
