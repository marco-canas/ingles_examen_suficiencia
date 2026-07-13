"""Small utilities: reproducibility, device/thread setup, param counting, plots.

Plotting helpers import matplotlib lazily (inside the functions) so that simply
importing :mod:`gptlab` never forces a matplotlib import.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """This workshop targets CPU-only machines, so we always return ``cpu``.

    (Everything still works unchanged if you later move to a GPU.)
    """
    return torch.device("cpu")


def configure_cpu_threads(n_threads: int | None = None) -> int:
    """Set the number of intra-op threads PyTorch uses for CPU math.

    Defaults to the machine's logical core count. Returns the value set so the
    notebooks can print it.
    """
    if n_threads is None:
        n_threads = max(1, os.cpu_count() or 2)
    torch.set_num_threads(n_threads)
    return n_threads


def count_params(model: torch.nn.Module) -> int:
    """Total number of parameters (shared/tied tensors counted once)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def human_params(n: int) -> str:
    """Format a parameter count like ``0.95M`` or ``12.3K``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# --------------------------------------------------------------------------- #
# Plot helpers (matplotlib imported lazily)
# --------------------------------------------------------------------------- #
def plot_losses(history: dict, title: str = "Training curve"):
    """Plot train/val loss vs iteration from a ``train()`` history dict."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["iter"], history["train"], label="train", marker="o", ms=3)
    ax.plot(history["iter"], history["val"], label="val", marker="o", ms=3)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_attention(attn: torch.Tensor, tokens: list[str] | None = None, title: str = "Attention"):
    """Heat-map a single (T, T) attention matrix.

    ``attn`` may be (T, T) or a batched/head tensor; pass a single head/slice.
    """
    import matplotlib.pyplot as plt

    a = attn.detach().cpu().float()
    while a.dim() > 2:  # squeeze leading batch/head dims
        a = a[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(a.numpy(), cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xlabel("key position (attended to)")
    ax.set_ylabel("query position (attending from)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_embeddings_2d(coords: np.ndarray, labels: list[str], title: str = "Embeddings (2D)"):
    """Scatter 2D-projected embeddings with text labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.6)
    for (x, y), label in zip(coords, labels):
        ax.annotate(label, (x, y), fontsize=8, alpha=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax
