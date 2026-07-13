"""gptlab -- a tiny, CPU-friendly GPT built from scratch for teaching.

Companion package for the PyCon Colombia 2026 workshop
"LLMs in Depth: How an LLM Works Mathematically (and Its Implementation with PyTorch)".

The notebooks build each component up by hand; this package holds the clean,
tested "source of truth" versions that the training script and later notebooks
import so everything stays consistent with the shipped checkpoint.
"""

from __future__ import annotations

from . import data, embeddings, generate, modern, tokenizer, train, utils
from .config import GPTConfig, TrainConfig
from .generate import generate_text
from .model import (
    GPT,
    Block,
    CausalSelfAttention,
    LayerNorm,
    MLP,
    load_gpt,
    save_checkpoint,
)
from .train import estimate_loss, get_lr

__all__ = [
    "GPTConfig",
    "TrainConfig",
    "GPT",
    "Block",
    "CausalSelfAttention",
    "MLP",
    "LayerNorm",
    "save_checkpoint",
    "load_gpt",
    "generate_text",
    "estimate_loss",
    "get_lr",
    "data",
    "embeddings",
    "generate",
    "modern",
    "tokenizer",
    "train",
    "utils",
]

__version__ = "0.1.0"
