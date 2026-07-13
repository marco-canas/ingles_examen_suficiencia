"""Text generation: wrap the model's token-level ``generate`` with a tokenizer.

Sampling controls (all optional, combine freely):
    temperature -- flatten (>1) or sharpen (<1) the distribution
    top_k       -- keep only the k most likely tokens
    top_p       -- keep the smallest set of tokens whose probability sums to p
"""

from __future__ import annotations

import torch

from .model import GPT


@torch.no_grad()
def generate_text(model: GPT, tokenizer, prompt: str = "\n", max_new_tokens: int = 300,
                  temperature: float = 0.8, top_k: int | None = None,
                  top_p: float | None = None, device: str = "cpu",
                  seed: int | None = None) -> str:
    """Generate a continuation of ``prompt`` and return the full decoded string."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt) if prompt else []
    if len(ids) == 0:                      # need at least one token to start
        ids = tokenizer.encode("\n") or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p)
    return tokenizer.decode(out[0].tolist())
