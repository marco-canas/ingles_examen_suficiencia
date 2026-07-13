"""The classic GPT-2-style language model, built from small transparent pieces.

This is the *source of truth* for the model that notebook 05 assembles and that
the shipped checkpoint was trained with. It stays deliberately clean and
flag-free; the modern variants (RMSNorm, RoPE, SwiGLU, GQA) live in
:mod:`gptlab.modern` and are demonstrated in notebook 08.

Components (bottom-up):
    LayerNorm -> CausalSelfAttention -> MLP -> Block -> GPT
"""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (PyTorch's built-in always has one).

    Normalises each token vector to zero mean / unit variance, then applies a
    learned scale (gamma) and shift (beta):  y = gamma * (x - mu) / sigma + beta.
    """

    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Attention is computed explicitly (rather than via a fused kernel) so it can
    be read line-by-line and can return the attention weights for visualisation:

        att = softmax( (Q K^T) / sqrt(head_dim) + causal_mask )
        y   = att V
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # One matrix produces Q, K, V together (then we split).
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular causal mask: position t may attend to <= t.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))   # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = self.attn_dropout(att) @ v                            # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)          # reassemble heads
        y = self.resid_dropout(self.c_proj(y))
        if return_attn:
            return y, att
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """A transformer block, pre-norm style:

        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))

    Normalising *before* each sub-layer (pre-norm) keeps a clean residual
    highway and makes deep stacks trainable.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            a, att = self.attn(self.ln_1(x), return_attn=True)
            x = x + a
            x = x + self.mlp(self.ln_2(x))
            return x, att
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """A minimal GPT: token + position embeddings -> N blocks -> LM head.

    The LM head weight is *tied* to the token embedding (they are the same
    tensor), which saves parameters and typically improves quality.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # position embeddings
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=LayerNorm(cfg.n_embd, cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight       # weight tying

        self.apply(self._init_weights)
        # GPT-2 scaled init for the residual projections.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_attn: bool = False):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        attns = []
        for block in self.transformer.h:
            if return_attn:
                x, att = block(x, return_attn=True)
                attns.append(att)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_attn:
            return logits, loss, attns
        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Count parameters (tied wte/lm_head counted once)."""
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.transformer.wpe.weight.numel()
            total -= self.transformer.wte.weight.numel()  # tied with lm_head
        return total

    def configure_optimizers(self, weight_decay: float, lr: float,
                             betas: tuple[float, float] = (0.9, 0.99)) -> torch.optim.Optimizer:
        """AdamW with weight decay on matmul weights only (not biases/norms)."""
        decay, no_decay, seen = [], [], set()
        for p in self.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
        """Autoregressively extend ``idx`` by ``max_new_tokens`` tokens."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]           # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # last step only
            logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def _filter_logits(logits: torch.Tensor, top_k: int | None = None,
                   top_p: float | None = None) -> torch.Tensor:
    """Apply optional top-k and/or top-p (nucleus) filtering in-place-safe."""
    logits = logits.clone()
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k)[0][..., -1, None]
        logits[logits < thresh] = float("-inf")
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        # keep at least the top token: shift the mask right by one.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def save_checkpoint(model: GPT, path: str, **meta) -> None:
    """Save model weights + architecture config + arbitrary metadata."""
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), **meta}, path)


def load_gpt(path: str, map_location: str = "cpu"):
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns ``(model, config, checkpoint_dict)``.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt
