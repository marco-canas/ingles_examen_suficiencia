"""Configuration objects for the playground GPT.

Two dataclasses:

* ``GPTConfig``   -- the *architecture* (what the model IS).
* ``TrainConfig`` -- the *optimization* (how the model LEARNS).

Keeping them separate mirrors how real training scripts are organized and lets
us swap one without touching the other (e.g. train the same architecture with a
faster/slower profile).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Architecture hyper-parameters for the classic GPT-2-style model.

    The defaults are the "playground" configuration described in the workshop
    plan: ~0.95M parameters, small enough to train on a CPU in a few minutes,
    large enough to reproduce recognizable Shakespeare structure.

    Note: ``vocab_size`` is normally *overwritten* by ``build_tokenizer`` with
    the actual size of the trained tokenizer, so the embedding table always
    matches the data (see ``gptlab.tokenizer``).
    """

    vocab_size: int = 1024      # upper bound; set from the trained tokenizer
    block_size: int = 64        # context length (in tokens)
    n_layer: int = 4            # number of transformer blocks
    n_head: int = 4             # attention heads per block
    n_embd: int = 128           # embedding / residual-stream width
    dropout: float = 0.1        # dropout probability (0.0 disables it)
    bias: bool = True           # use bias terms in Linear/LayerNorm layers
    tokenizer: str = "bpe"      # "bpe" (subword) or "char" (character-level)

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    """Optimization hyper-parameters for the training loop."""

    max_iters: int = 2500
    batch_size: int = 16
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 20
    val_frac: float = 0.1
    seed: int = 1337
    profile: str = "playground"

    @staticmethod
    def for_profile(profile: str = "playground") -> "TrainConfig":
        """Return a preset training config.

        * ``"demo"``       -- ~1000 iters, ~2 min: quick live run.
        * ``"playground"`` -- default full run (~4-10 min on an i7 8th-gen CPU).
        """
        if profile == "demo":
            return TrainConfig(
                max_iters=1000,
                warmup_iters=50,
                eval_interval=200,
                eval_iters=20,
                profile="demo",
            )
        if profile == "playground":
            return TrainConfig(profile="playground")
        raise ValueError(f"unknown profile: {profile!r} (use 'demo' or 'playground')")
