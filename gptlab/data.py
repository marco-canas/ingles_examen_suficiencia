"""Corpus loading, encoding, and batching for the language model.

The pipeline is deliberately tiny:

    download_shakespeare() -> load_text() -> prepare_splits(text, tokenizer)
                                                 -> get_batch(split_tensor, ...)
"""

from __future__ import annotations

import os
from urllib.request import urlretrieve

import numpy as np
import torch

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DEFAULT_PATH = os.path.join("data", "tinyshakespeare.txt")


def download_shakespeare(path: str = DEFAULT_PATH) -> str:
    """Download Tiny Shakespeare to ``path`` if it is not already present."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urlretrieve(SHAKESPEARE_URL, path)
    return path


def load_text(path: str = DEFAULT_PATH) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def to_tensor(ids) -> torch.Tensor:
    """Turn a sequence of token ids into a 1-D ``int64`` tensor (for indexing)."""
    return torch.from_numpy(np.asarray(ids, dtype=np.int64))


def encode_dataset(text: str, tokenizer, val_frac: float = 0.1):
    """Encode ``text`` and split into (train, val) as ``uint16`` numpy arrays.

    ``uint16`` is enough for any vocab < 65,536 and keeps the arrays small on
    disk / in RAM.
    """
    ids = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    n_val = int(len(ids) * val_frac)
    train_ids = ids[: len(ids) - n_val]
    val_ids = ids[len(ids) - n_val:]
    return train_ids, val_ids


def prepare_splits(text: str, tokenizer, val_frac: float = 0.1) -> dict[str, torch.Tensor]:
    """Convenience wrapper: encode, split, and return int64 tensors ready for training."""
    train_ids, val_ids = encode_dataset(text, tokenizer, val_frac)
    return {"train": to_tensor(train_ids), "val": to_tensor(val_ids)}


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"):
    """Sample a batch of contiguous ``block_size`` windows.

    Returns ``(x, y)`` where ``y`` is ``x`` shifted one token to the right --
    i.e. the next-token targets for a causal language model.
    """
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
