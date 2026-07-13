import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "03_attention.ipynb")

cells = [
    md(r"""
# 03 · Attention — the core of modern LLMs

Embeddings give each token a vector, but those vectors are still **isolated**.
Language is contextual: to represent "it" you must look back at what "it" refers
to. **Self-attention** is the mechanism that lets every position gather
information from the others. It is *the* idea behind the Transformer.

We build it up: a single head from scratch → why the $\sqrt{d_k}$ scaling exists
→ causal masking → multiple heads → visualizing what attention does.
"""),
    code(BOOTSTRAP),
    md(r"""
## Queries, Keys, Values (mathematics)

Start from a sequence of token vectors stacked into $X \in \mathbb{R}^{T \times d}$
($T$ positions, width $d$). From $X$ we compute three linear projections:

$$ Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V. $$

The analogy is a **soft dictionary lookup**:
- a **query** $q_t$ asks *"what am I looking for?"*,
- a **key** $k_s$ advertises *"what do I offer?"*,
- a **value** $v_s$ is *"what I actually pass on if selected."*

**Scaled dot-product attention** is then

$$
\mathrm{Attention}(Q,K,V) = \underbrace{\mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right)}_{A \;\in\; \mathbb{R}^{T\times T}} V.
$$

Row $t$ of $A$ is a probability distribution over positions: **how much position
$t$ attends to each position $s$**. The output at $t$ is a weighted average of
the value vectors — position $t$'s new representation, assembled from the whole
sequence.
"""),
    md(r"""
### A single head, from scratch

No `nn.Module` yet — just the four lines of math on a small random example.
"""),
    code(r"""
import torch

def single_head_attention(x, Wq, Wk, Wv, causal=True, return_attn=True):
    q, k, v = x @ Wq, x @ Wk, x @ Wv          # each (T, d_k)
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / d_k ** 0.5     # (T, T)
    if causal:
        mask = torch.tril(torch.ones(scores.shape))
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)      # rows sum to 1
    out = attn @ v                            # (T, d_k)
    return (out, attn) if return_attn else out

torch.manual_seed(0)
T, d, d_k = 5, 8, 8
x = torch.randn(T, d)
Wq, Wk, Wv = (torch.randn(d, d_k) * 0.5 for _ in range(3))
out, attn = single_head_attention(x, Wq, Wk, Wv)
print("attention matrix (rows = query position, cols = key position):")
print(attn.round(decimals=2))
print("row sums:", attn.sum(dim=-1).round(decimals=3).tolist(), "  (each row is a distribution)")
"""),
    md(r"""
## Why divide by $\sqrt{d_k}$?

If the entries of $q$ and $k$ are independent with unit variance, their dot
product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has variance $\approx d_k$. Large
scores push softmax into a **saturated** regime — nearly one-hot — where
gradients vanish and only one position gets attended to. Dividing by $\sqrt{d_k}$
rescales the variance back to $\approx 1$, keeping the distribution soft and
trainable. Let's verify the variance claim empirically.
"""),
    code(r"""
d_k = 64
q = torch.randn(10000, d_k)
k = torch.randn(10000, d_k)
dots = (q * k).sum(dim=-1)
print(f"Var(q·k)            = {dots.var().item():6.2f}   (~ d_k = {d_k})")
print(f"Var(q·k / sqrt(d_k)) = {(dots / d_k ** 0.5).var().item():6.2f}   (~ 1, as intended)")
"""),
    md(r"""
## Causal masking (mathematics)

A language model may only use the **past**: predicting token $t$ must not peek at
tokens $> t$. We enforce this by masking the score matrix *before* the softmax,
setting forbidden entries to $-\infty$ so they receive zero probability:

$$
\tilde{A}_{ts} =
\begin{cases}
\dfrac{q_t \cdot k_s}{\sqrt{d_k}} & s \le t \\[4pt]
-\infty & s > t
\end{cases}
\qquad\Longrightarrow\qquad
\mathrm{softmax}(\tilde{A}_{t,\cdot})_s = 0 \text{ for } s > t.
$$

The result is a **lower-triangular** attention matrix: position $t$ can attend to
$\{0, \dots, t\}$ only. That single trick is what makes the model *autoregressive*.
"""),
    code(r"""
import matplotlib.pyplot as plt

mask = torch.tril(torch.ones(8, 8))
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(mask, cmap="Greys", vmin=0, vmax=1)
ax.set_title("Causal mask (1 = allowed, 0 = blocked)")
ax.set_xlabel("key position s"); ax.set_ylabel("query position t")
for t in range(8):
    for s in range(8):
        ax.text(s, t, int(mask[t, s].item()), ha="center", va="center",
                color="white" if mask[t, s] else "black", fontsize=8)
plt.tight_layout(); plt.show()
"""),
    md(r"""
## Multi-head attention

One head learns one kind of relationship. **Multi-head** attention runs $h$
attentions in parallel on slices of width $d_k = d / h$, then concatenates and
projects them:

$$
\mathrm{MHA}(X) = \big[\, \mathrm{head}_1 \,\|\, \cdots \,\|\, \mathrm{head}_h \,\big]\, W_O,
\qquad \mathrm{head}_i = \mathrm{Attention}(X W_Q^i, X W_K^i, X W_V^i).
$$

**Interpretation.** Different heads specialize — one may track the previous
token, another may link verbs to subjects, another may attend to punctuation.
Below is the exact module the model uses (`gptlab.model.CausalSelfAttention`):
it fuses the three projections into one matrix for speed and reshapes into heads.
"""),
    code(r"""
import inspect
from gptlab.model import CausalSelfAttention
print(inspect.getsource(CausalSelfAttention))
"""),
    md(r"""
## Visualizing attention on real tokens

Let's push a real snippet of Shakespeare through an attention layer and look at
the weight matrix. If a trained checkpoint exists we use its learned weights
(rich patterns); otherwise we use a freshly initialized layer, which still shows
the tell-tale **causal triangle** and softmax normalization.
"""),
    code(r"""
from gptlab import GPTConfig
from gptlab.tokenizer import load_bpe
from gptlab.utils import plot_attention

tok = load_bpe(os.path.join(CKPT_DIR, "bpe_tokenizer.json"))
snippet = "To be, or not to be, that is the question"
ids = torch.tensor([tok.encode(snippet)])
T = ids.shape[1]
token_strs = [tok.decode([i]) for i in ids[0].tolist()]

cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=max(64, T), n_head=4, n_embd=128)

ckpt_file = os.path.join(CKPT_DIR, "gpt_playground.pt")
if os.path.exists(ckpt_file):
    from gptlab.model import load_gpt
    model, cfg, _ = load_gpt(ckpt_file)
    x = model.transformer.drop(model.transformer.wte(ids) +
                               model.transformer.wpe(torch.arange(T)))
    _, att = model.transformer.h[0].attn(model.transformer.h[0].ln_1(x), return_attn=True)
    source = "trained (layer 0)"
else:
    attn_layer = CausalSelfAttention(cfg)
    x = torch.randn(1, T, cfg.n_embd)
    _, att = attn_layer(x, return_attn=True)
    source = "untrained"

# att: (1, n_head, T, T) -> show head 0
fig, ax = plot_attention(att[0, 0], tokens=token_strs,
                         title=f"Attention weights — {source}, head 0")
plt.show()
"""),
    md(r"""
**Reading the heat-map.** Every row sums to 1 and the upper triangle is exactly
zero — the causal mask at work. Bright cells show where a query position pulls
information from. With trained weights you will often see structured stripes
(e.g. attending to the previous token or to sentence delimiters); untrained,
the mass is spread smoothly across the allowed (past) positions.

Attention is the expensive, powerful heart of the model. Everything in the next
notebook exists to *support* it: normalization to keep it stable, a feed-forward
network to process what it gathers, and residual connections to preserve the
signal.

---
**Next → `04_normalization_and_transformer_block.ipynb`.**
"""),
]

build(OUT, cells)
