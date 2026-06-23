import math                                                     # exp for perplexity
import torch                                                     # forward pass

# A small, fixed evaluation set — coherent English. Perplexity on this is a defensible, model-agnostic
# quality proxy: lower perplexity = the model assigns higher probability to fluent text.
EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Machine learning models are trained on large datasets to make predictions.",
    "In the morning, she poured a cup of coffee and read the newspaper quietly.",
    "The capital of France is Paris, a city famous for its art and architecture.",
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure.",
]


@torch.inference_mode()
def perplexity(model, tokenizer, texts=None) -> float:
    texts = texts or EVAL_TEXTS                                 # default eval set
    device = next(model.parameters()).device                   # run on the model's device
    total_nll, total_tokens = 0.0, 0                           # accumulate negative log-likelihood
    for t in texts:
        ids = tokenizer(t, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:                                   # need at least one prediction target
            continue
        out = model(input_ids=ids, labels=ids)                # HF returns mean token cross-entropy
        n = ids.shape[1] - 1                                   # number of predicted tokens
        total_nll += out.loss.item() * n                      # undo the mean to sum NLL
        total_tokens += n
    return math.exp(total_nll / total_tokens)                 # perplexity = exp(mean NLL)


def token_agreement(reference_ids, other_ids) -> float:
    n = min(len(reference_ids), len(other_ids))               # compare the overlapping prefix
    if n == 0:
        return 0.0
    same = sum(1 for a, b in zip(reference_ids[:n], other_ids[:n]) if a == b)  # exact matches
    return same / n                                           # fraction of greedy tokens that match fp16
