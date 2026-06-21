import os                                                      # read env overrides

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"                 # tiny, ungated, CPU-friendly
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)         # allow override without code edits

DEFAULT_MAX_NEW_TOKENS = 64                                     # short by default to keep CPU fast
DEFAULT_DTYPE = "float32"                                       # CPU has no fast fp16; fp32 is correct here
