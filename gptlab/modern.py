"""Modern architecture upgrades used by today's LLMs (Llama / Mistral / etc.).

These are the drop-in replacements demonstrated in notebook 08. They are kept
*out* of the clean classic model (:mod:`gptlab.model`) so the main narrative
stays uncluttered, and reassembled here into ``GPTModern`` so the notebook can
prove they actually train.

Mapping to the classic components:
    LayerNorm            -> RMSNorm
    learned position emb -> RoPE (rotary), applied inside attention
    MLP (GELU)           -> SwiGLU (gated, 2/3 hidden rule)
    Multi-Head Attention -> Grouped-Query Attention (fewer K/V heads)
    (bias terms)         -> removed (bias-free Linear layers)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig
from .model import _filter_logits


# --------------------------------------------------------------------------- #
# 1. RMSNorm  --  y = x / sqrt(mean(x^2) + eps) * gamma
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-Mean-Square norm: like LayerNorm but no mean subtraction and no bias.

    Cheaper (one statistic, one parameter vector) with equivalent quality.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# --------------------------------------------------------------------------- #
# 2. RoPE  --  rotate (q, k) by a position-dependent angle (relative positions)
# --------------------------------------------------------------------------- #
def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    """Return (cos, sin) tables of shape (seq_len, head_dim) for RoPE."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (hd/2,)
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)              # (T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (T, hd)  -- duplicate for the two halves
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings. ``x``: (B, n_head, T, hd); ``cos``/``sin``: (1,1,T,hd)."""
    return x * cos + rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# 3. SwiGLU  --  gated feed-forward:  down( SiLU(gate(x)) * up(x) )
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """Gated MLP. Uses three matrices, so hidden is scaled by 2/3 (then rounded)
    to keep the parameter count comparable to a standard 4x GELU MLP."""

    def __init__(self, dim: int, hidden: int | None = None, multiple_of: int = 32,
                 bias: bool = False):
        super().__init__()
        if hidden is None:
            hidden = int(2 / 3 * 4 * dim)
            hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden
        self.w_gate = nn.Linear(dim, hidden, bias=bias)
        self.w_up = nn.Linear(dim, hidden, bias=bias)
        self.w_down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# --------------------------------------------------------------------------- #
# 4. Grouped-Query Attention (+ RoPE)
# --------------------------------------------------------------------------- #
class GroupedQueryAttention(nn.Module):
    """Attention with fewer K/V heads than Q heads.

    ``n_kv_head < n_head`` shrinks the K/V projections and (at inference) the
    KV-cache by a factor of ``n_head / n_kv_head``, with little quality loss.
    RoPE is applied to Q and K before the dot product.
    """

    def __init__(self, cfg: GPTConfig, n_kv_head: int | None = None, base: float = 10000.0):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = n_kv_head or cfg.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        self.hd = cfg.n_embd // cfg.n_head
        self.n_embd = cfg.n_embd

        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.hd, bias=False)
        self.kv_proj = nn.Linear(cfg.n_embd, 2 * self.n_kv_head * self.hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        cos, sin = build_rope_cache(cfg.block_size, self.hd, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)  # (B, nh, T, hd)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.hd)
        k = kv[:, :, 0].transpose(1, 2)   # (B, n_kv, T, hd)
        v = kv[:, :, 1].transpose(1, 2)

        cos = self.cos[:T].view(1, 1, T, self.hd)
        sin = self.sin[:T].view(1, 1, T, self.hd)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        rep = self.n_head // self.n_kv_head        # broadcast K/V heads to match Q heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        if return_attn:
            return y, att
        return y


# --------------------------------------------------------------------------- #
# Assembled modern model (used to prove the pieces integrate & train)
# --------------------------------------------------------------------------- #
class ModernBlock(nn.Module):
    def __init__(self, cfg: GPTConfig, n_kv_head: int | None = None):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = GroupedQueryAttention(cfg, n_kv_head=n_kv_head)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPTModern(nn.Module):
    """The classic GPT reassembled with RMSNorm + RoPE + SwiGLU + GQA, bias-free.

    Note there is no position-embedding table: RoPE injects position inside
    attention instead.
    """

    def __init__(self, cfg: GPTConfig, n_kv_head: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.n_kv_head = n_kv_head or max(1, cfg.n_head // 2)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([ModernBlock(cfg, self.n_kv_head) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight        # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def configure_optimizers(self, weight_decay: float, lr: float,
                             betas: tuple[float, float] = (0.9, 0.99)):
        decay, no_decay, seen = [], [], set()
        for p in self.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = _filter_logits(logits[:, -1, :] / max(temperature, 1e-8),
                                    top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
