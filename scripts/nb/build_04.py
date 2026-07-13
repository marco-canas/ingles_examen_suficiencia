import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "04_normalization_and_transformer_block.ipynb")

cells = [
    md(r"""
# 04 · Normalization & the Transformer Block

Attention gathers information; a few supporting pieces make a *stack* of
attention trainable and expressive:

- **LayerNorm** — keeps activations well-scaled,
- **residual connections** — give gradients a clean highway,
- the **MLP** (feed-forward) — processes each position after mixing,
- **positional embeddings** — tell the model *where* each token is.

We assemble these into the **Transformer block**, the unit we will stack.
"""),
    code(BOOTSTRAP),
    md(r"""
## Layer Normalization (mathematics)

For a single token vector $x \in \mathbb{R}^{d}$, LayerNorm standardizes across
its features, then applies a learned scale $\gamma$ and shift $\beta$:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2,
$$
$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta.
$$

**Interpretation.** Each token is re-centred to mean 0 and unit variance *across
its own features* (independently of batch or sequence length), which stops
activations from exploding or shrinking as they pass through many layers.
$\gamma, \beta$ let the network undo the normalization if it needs to. Note this
is **per-token** — unlike BatchNorm it does not mix information across examples.
"""),
    code(r"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta

ln = LayerNorm(8)
x = torch.randn(3, 8) * 5 + 2                 # arbitrary scale/shift
y = ln(x)
print("input  per-row mean/std:", x.mean(-1).round(decimals=2).tolist(),
      x.std(-1, unbiased=False).round(decimals=2).tolist())
print("output per-row mean/std:", y.mean(-1).round(decimals=3).tolist(),
      y.std(-1, unbiased=False).round(decimals=2).tolist(), " (≈ 0 mean, ≈ 1 std)")
"""),
    md(r"""
## Residual connections (mathematics)

Instead of replacing its input, each sub-layer **adds** to it:

$$ y = x + f(x). $$

The Jacobian is $\dfrac{\partial y}{\partial x} = I + \dfrac{\partial f}{\partial x}$.
That identity term means gradients always have a direct path back to earlier
layers, even if $\partial f / \partial x$ is tiny — this is what lets us stack
many blocks without gradients vanishing. Think of a **residual stream**: a
running representation that each block reads from and writes a *correction* into.
"""),
    md(r"""
### Pre-norm vs post-norm

The original Transformer put the norm *after* the sub-layer (post-norm). Modern
GPTs use **pre-norm** — normalize the input to each sub-layer, keeping the
residual stream itself un-normalized:

$$
x \leftarrow x + \mathrm{Attention}(\mathrm{LN}(x)), \qquad
x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x)).
$$

Pre-norm keeps the residual highway clean and makes deep stacks train stably
without delicate learning-rate warmup gymnastics.
"""),
    md(r"""
## The MLP / feed-forward network (mathematics)

After attention mixes information *across* positions, a small MLP processes each
position *independently*. It expands the width by a factor of 4, applies a
non-linearity, and projects back:

$$ \mathrm{MLP}(x) = W_2\,\phi(W_1 x + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{4d \times d}. $$

Modern GPTs use **GELU** as $\phi$, a smooth version of ReLU:
$\mathrm{GELU}(x) = x\,\Phi(x)$, where $\Phi$ is the standard-normal CDF.

**Interpretation.** Attention decides *what to combine*; the MLP is where most of
the model's parameters live and where per-token "thinking"/feature-transformation
happens. The 4× hidden layer gives it room to compute non-linear combinations.
"""),
    code(r"""
import torch.nn.functional as F
import matplotlib.pyplot as plt

xs = torch.linspace(-4, 4, 200)
plt.figure(figsize=(6, 3.2))
plt.plot(xs, F.gelu(xs), label="GELU")
plt.plot(xs, F.relu(xs), "--", label="ReLU")
plt.axhline(0, color="k", lw=0.5); plt.axvline(0, color="k", lw=0.5)
plt.legend(); plt.title("GELU is a smooth ReLU"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
    md(r"""
## Positional embeddings (why order matters)

Attention is **permutation-equivariant**: shuffle the input tokens and the
outputs shuffle the same way — it has *no inherent sense of order*. But "dog
bites man" ≠ "man bites dog". We inject order by adding a learned **position
vector** to each token embedding:

$$ h_t = \underbrace{E_{x_t}}_{\text{token emb}} + \underbrace{P_t}_{\text{position emb}}. $$

GPT-2 learns the table $P \in \mathbb{R}^{T_{\max} \times d}$ directly (one row
per position). Let's demonstrate the permutation problem, then fix it.
"""),
    code(r"""
from gptlab.model import CausalSelfAttention
from gptlab import GPTConfig

cfg = GPTConfig(vocab_size=100, block_size=8, n_head=2, n_embd=16, dropout=0.0)
attn = CausalSelfAttention(cfg).eval()
wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)

seq = torch.tensor([[5, 9, 2, 7]])
perm = torch.tensor([[7, 2, 9, 5]])              # reversed

with torch.no_grad():
    # Without positions: attention output for the LAST token depends only on the
    # *set* of tokens it can see, not their order (for a single query over all).
    out_seq = attn(wte(seq))
    out_perm = attn(wte(perm))
print("Without positional info, token embeddings are order-agnostic inputs.")
print("Adding P_t breaks the symmetry so order carries information:")
pos = nn.Embedding(cfg.block_size, cfg.n_embd)
h1 = wte(seq) + pos(torch.arange(4))
h2 = wte(perm) + pos(torch.arange(4))
print("max |h(seq) - reverse(h(perm))| =",
      (h1 - h2.flip(1)).abs().max().item(), "(≠ 0 → order now matters)")
"""),
    md(r"""
## Assembling the block

Putting it together, one **pre-norm Transformer block** is:

$$
x \leftarrow x + \mathrm{Attn}(\mathrm{LN}_1(x)), \qquad
x \leftarrow x + \mathrm{MLP}(\mathrm{LN}_2(x)).
$$

This is exactly `gptlab.model.Block`. We stack $N$ of these in the next notebook.
"""),
    code(r"""
import inspect
from gptlab.model import Block, MLP

print(inspect.getsource(MLP))
print(inspect.getsource(Block))
"""),
    code(r"""
# Shape check: a block maps (B, T, n_embd) -> (B, T, n_embd) (shape-preserving).
block = Block(cfg)
h = torch.randn(2, 4, cfg.n_embd)
print("block input :", tuple(h.shape))
print("block output:", tuple(block(h).shape), " (identical — it writes into the residual stream)")
"""),
    md(r"""
We now have every ingredient: token + position embeddings, attention, norm,
residuals, and the MLP.

---
**Next → `05_assembling_the_gpt.ipynb`:** stack the blocks into a full GPT and
watch the initial loss land on our $\ln V$ baseline.
"""),
]

build(OUT, cells)
