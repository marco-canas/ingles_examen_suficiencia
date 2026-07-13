import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "08_modern_tricks_rope_rmsnorm_swiglu_gqa.ipynb")

cells = [
    md(r"""
# 08 · Modern Tricks — RMSNorm, RoPE, SwiGLU, GQA

Our model is a faithful **GPT-2-style** architecture. Since 2019 a handful of
upgrades have become the de-facto standard in models like **Llama, Mistral,
Qwen, DeepSeek, and Gemini**. They keep the same skeleton and swap four parts:

| Classic (our model) | Modern replacement | Why |
|---|---|---|
| LayerNorm | **RMSNorm** | cheaper, same quality |
| learned position embeddings | **RoPE** (rotary) | relative positions, extrapolates |
| GELU MLP | **SwiGLU** | gated, better use of parameters |
| Multi-Head Attention | **Grouped-Query Attention** | smaller KV-cache, faster inference |
| bias terms | **bias-free** | tiny simplification, no quality loss |

We demonstrate each in isolation (math → interpretation → runnable demo → *exact
drop-in mapping*), then assemble them into `GPTModern` and prove it trains. The
clean implementations live in `gptlab/modern.py`.
"""),
    code(BOOTSTRAP),
    code(r"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gptlab import GPTConfig

cfg = GPTConfig(vocab_size=1024, block_size=64, n_layer=4, n_head=4, n_embd=128)
"""),
    md(r"""
## 1 · RMSNorm

**Math.** LayerNorm subtracts the mean and divides by the standard deviation.
RMSNorm drops the mean-centring and the bias, normalizing only by the
root-mean-square:

$$ \mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma. $$

**Interpretation / why.** In practice the mean-subtraction contributes little,
so removing it costs no quality while using **one** statistic and **one**
parameter vector (just $\gamma$) instead of two — slightly faster and lighter.

**Drop-in mapping.** Replace `LayerNorm` with `RMSNorm` in `Block.ln_1`,
`Block.ln_2`, and `GPT.transformer.ln_f`.
"""),
    code(r"""
from gptlab.modern import RMSNorm
from gptlab.model import LayerNorm

x = torch.randn(4, cfg.n_embd)
rms, ln = RMSNorm(cfg.n_embd), LayerNorm(cfg.n_embd)
print("RMSNorm output RMS per row:", rms(x).pow(2).mean(-1).sqrt().round(decimals=3).tolist())
print("params  -> LayerNorm:", sum(p.numel() for p in ln.parameters()),
      "| RMSNorm:", sum(p.numel() for p in rms.parameters()), "(half)")
"""),
    md(r"""
## 2 · RoPE (Rotary Position Embeddings)

**Math.** Instead of *adding* a position vector, RoPE **rotates** the query and
key vectors by an angle proportional to their position. Pairing up dimensions,
position $m$ applies a rotation $R_m$ built from frequencies
$\theta_j = 10000^{-2j/d}$. The key identity is that the attention score between
positions $m$ and $n$ becomes a function of their **relative** offset:

$$ (R_m q)^\top (R_n k) = q^\top R_{n-m}\, k. $$

**Interpretation / why.** Relative positioning matches how language works
("the previous word" is relative, not absolute) and lets the model generalize to
sequence lengths it never saw in training. Below: place the *same* content vector
at every position and watch the attention dot-product depend only on the distance.

**Drop-in mapping.** Remove `wpe` (the learned position table) and call
`apply_rotary` on `q` and `k` inside `CausalSelfAttention`, right after
splitting into heads.
"""),
    code(r"""
from gptlab.modern import build_rope_cache, apply_rotary

T, hd = 48, 16
cos, sin = build_rope_cache(T, hd)
base = torch.randn(hd)
X = base.view(1, 1, 1, hd).expand(1, 1, T, hd).clone()      # identical content everywhere
Xr = apply_rotary(X, cos.view(1, 1, T, hd), sin.view(1, 1, T, hd))
rel_dot = (Xr[0, 0, 0] * Xr[0, 0]).sum(-1)                  # dot(pos 0, pos j) vs distance j

plt.figure(figsize=(6, 3.2))
plt.plot(range(T), rel_dot.detach())
plt.xlabel("relative distance (j - 0)"); plt.ylabel("q·k after RoPE")
plt.title("RoPE encodes RELATIVE position (same content at all positions)")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
"""),
    md(r"""
## 3 · SwiGLU

**Math.** The classic MLP is $W_2\,\mathrm{GELU}(W_1 x)$. SwiGLU adds a
multiplicative **gate**:

$$ \mathrm{SwiGLU}(x) = W_{\text{down}}\big(\, \mathrm{SiLU}(W_{\text{gate}}\,x) \odot (W_{\text{up}}\,x)\,\big), \qquad \mathrm{SiLU}(z) = z\,\sigma(z). $$

**Interpretation / why.** The gate lets the network *modulate* information
per-feature (route some through, suppress others), which empirically learns
better than a plain activation. Because it uses **three** matrices instead of
two, the hidden size is scaled by $\tfrac{2}{3}$ (then rounded) to keep the
parameter count comparable — verify below.

**Drop-in mapping.** Replace the `MLP` module inside `Block` with `SwiGLU`.
"""),
    code(r"""
from gptlab.modern import SwiGLU
from gptlab.model import MLP

mlp, swiglu = MLP(cfg), SwiGLU(cfg.n_embd)
p_mlp = sum(p.numel() for p in mlp.parameters())
p_swi = sum(p.numel() for p in swiglu.parameters())
print(f"GELU MLP    hidden=4*{cfg.n_embd}={4*cfg.n_embd:4d} -> {p_mlp:,} params")
print(f"SwiGLU      hidden(2/3 rule)={swiglu.hidden:4d}    -> {p_swi:,} params  (comparable)")
print("SwiGLU output shape:", tuple(swiglu(torch.randn(2, 5, cfg.n_embd)).shape))
"""),
    md(r"""
## 4 · Grouped-Query Attention (GQA)

**Math.** Multi-Head Attention uses $H$ query heads and $H$ key/value heads. GQA
keeps $H$ query heads but only $H_{kv} < H$ key/value heads, each shared by a
group of $H / H_{kv}$ query heads. (MQA is the extreme $H_{kv}=1$.)

**Interpretation / why.** At inference the **KV-cache** stores $K, V$ for every
past token; its size is proportional to $H_{kv}$. Reducing $H_{kv}$ shrinks the
cache and the memory bandwidth per step — the main bottleneck in fast decoding —
with little quality loss.

**Drop-in mapping.** Replace `CausalSelfAttention` with
`GroupedQueryAttention(cfg, n_kv_head=...)` (which also folds in RoPE).
"""),
    code(r"""
from gptlab.modern import GroupedQueryAttention

for n_kv in [4, 2, 1]:
    gqa = GroupedQueryAttention(cfg, n_kv_head=n_kv)
    y = gqa(torch.randn(2, 16, cfg.n_embd))
    kv_cache_ratio = n_kv / cfg.n_head
    tag = {4: "= MHA", 1: "= MQA"}.get(n_kv, "")
    print(f"n_kv_head={n_kv} {tag:5s} | out {tuple(y.shape)} | "
          f"KV-cache size vs MHA: {kv_cache_ratio:.0%}")
"""),
    md(r"""
## Putting it together: `GPTModern`

`gptlab.modern.GPTModern` is our classic `GPT` reassembled with **all four**
swaps (RMSNorm + RoPE + SwiGLU + GQA, bias-free). Same config, no learned
position table. We check it forwards correctly (initial loss ≈ $\ln V$) and
compare its size to the classic model.
"""),
    code(r"""
import math
from gptlab.model import GPT
from gptlab.modern import GPTModern
from gptlab.utils import count_params, human_params

classic = GPT(cfg)
modern = GPTModern(cfg, n_kv_head=2)

xb = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
_, loss = modern(xb, xb)
print(f"GPTModern initial loss = {loss.item():.3f}  vs  ln(V) = {math.log(cfg.vocab_size):.3f}")
print(f"classic params : {human_params(count_params(classic))}")
print(f"modern  params : {human_params(count_params(modern))}  (n_kv_head=2)")
"""),
    md(r"""
### Proof it trains

A short run on real Shakespeare tokens: if the loss drops, the modern components
are correctly wired and optimizing — no full retraining needed to make the point.
"""),
    code(r"""
from gptlab.data import get_batch, load_text, prepare_splits
from gptlab.tokenizer import load_bpe

tok = load_bpe(os.path.join(CKPT_DIR, "bpe_tokenizer.json"))
data = prepare_splits(load_text(DATA), tok, val_frac=0.1)

opt = modern.configure_optimizers(weight_decay=0.1, lr=1e-3)
modern.train()
losses = []
for step in range(60):
    x, y = get_batch(data["train"], cfg.block_size, batch_size=16)
    _, loss = modern(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(modern.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
print(f"GPTModern loss: {losses[0]:.3f} -> {losses[-1]:.3f}  ({'✅ learning' if losses[-1] < losses[0] else '⚠'})")

plt.figure(figsize=(6, 3.2))
plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss")
plt.title("GPTModern — 60 training steps"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
    md(r"""
## Summary

| Component | Classic | Modern | Effect |
|---|---|---|---|
| Norm | LayerNorm (2 params/dim) | RMSNorm (1 param/dim) | cheaper |
| Position | learned table `wpe` | RoPE (rotate q,k) | relative, extrapolates |
| MLP | GELU, 4× | SwiGLU, ⅔·4× | gated, better quality/param |
| Attention | MHA (H kv heads) | GQA (H_kv < H) | smaller KV-cache |

Same skeleton, better parts. Swapping them into `gptlab/model.py` — using the
drop-in mappings above — is an excellent exercise, and a great way to *feel* how
today's frontier models are built from the pieces you now understand end-to-end.

🎉 **You've built a language model from the mathematics up.** Congratulations, and
thank you for coming to the workshop!
"""),
]

build(OUT, cells)
