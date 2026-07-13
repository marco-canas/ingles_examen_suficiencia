"""Training loop with the standard modern "tricks": AdamW, weight decay,
linear warmup + cosine learning-rate decay, and gradient clipping.
"""

from __future__ import annotations

import math

import torch

from .config import GPTConfig, TrainConfig
from .data import get_batch
from .model import GPT, save_checkpoint


def get_lr(it: int, tcfg: TrainConfig) -> float:
    """Linear warmup for the first ``warmup_iters`` steps, then cosine decay to ``min_lr``."""
    if it < tcfg.warmup_iters:
        return tcfg.lr * (it + 1) / tcfg.warmup_iters
    if it >= tcfg.max_iters:
        return tcfg.min_lr
    ratio = (it - tcfg.warmup_iters) / max(1, tcfg.max_iters - tcfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))   # 1 -> 0
    return tcfg.min_lr + coeff * (tcfg.lr - tcfg.min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, data: dict, tcfg: TrainConfig, cfg: GPTConfig,
                  device: str = "cpu") -> dict[str, float]:
    """Average loss over ``eval_iters`` batches for each split (less noisy than one batch)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(data[split], cfg.block_size, tcfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model: GPT, data: dict, tcfg: TrainConfig, cfg: GPTConfig, device: str = "cpu",
          ckpt_path: str | None = None, tokenizer_meta: dict | None = None, log_fn=print) -> dict:
    """Train ``model`` on ``data`` (dict with 'train'/'val' int64 tensors).

    Returns a history dict with lists ``iter``/``train``/``val`` for plotting.
    If ``ckpt_path`` is given, saves the final model there (with ``tokenizer_meta``).
    """
    model.to(device)
    optimizer = model.configure_optimizers(tcfg.weight_decay, tcfg.lr)
    history = {"iter": [], "train": [], "val": []}
    model.train()

    for it in range(tcfg.max_iters + 1):
        lr = get_lr(it, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters:
            losses = estimate_loss(model, data, tcfg, cfg, device)
            history["iter"].append(it)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            if log_fn:
                log_fn(f"iter {it:5d} | train {losses['train']:.4f} | "
                       f"val {losses['val']:.4f} | lr {lr:.2e}")

        if it == tcfg.max_iters:
            break

        x, y = get_batch(data["train"], cfg.block_size, tcfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

    if ckpt_path is not None:
        meta = {"tokenizer": tokenizer_meta or {}, "train_config": vars(tcfg)}
        save_checkpoint(model, ckpt_path, **meta)
        if log_fn:
            log_fn(f"saved checkpoint -> {ckpt_path}")
    return history
