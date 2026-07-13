import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "07_generation_and_sampling.ipynb")

cells = [
    md(r"""
# 07 · Generation & Sampling

A trained GPT outputs a probability distribution over the next token. **Text
generation** turns that into a sequence by sampling one token at a time and
feeding it back in — *autoregression*:

$$ x_{t+1} \sim p_\theta(\cdot \mid x_{1:t}), \qquad \text{then append and repeat.} $$

*How* we pick from $p_\theta$ dramatically changes the output. This notebook uses
the shipped checkpoint to explore **temperature**, **top-k**, and **top-p**.
"""),
    code(BOOTSTRAP),
    code(r"""
from gptlab.model import load_gpt
from gptlab.tokenizer import load_tokenizer

model, cfg, ckpt = load_gpt(os.path.join(CKPT_DIR, "gpt_playground.pt"))
meta = ckpt.get("tokenizer", {"kind": "bpe", "path": "bpe_tokenizer.json"})
tok_file = os.path.basename(meta["path"].replace("\\", "/"))     # robust across OSes
tok = load_tokenizer(meta["kind"], os.path.join(CKPT_DIR, tok_file))
print("loaded checkpoint:", cfg)
print("tokenizer:", meta["kind"], "| vocab:", tok.vocab_size)
"""),
    md(r"""
## Temperature (mathematics)

Before the softmax we divide the logits by a **temperature** $\tau$:

$$ p_i = \mathrm{softmax}(z / \tau)_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}. $$

- $\tau \to 0$: the distribution collapses onto the argmax → **greedy**,
  repetitive, "safe".
- $\tau = 1$: the model's raw distribution.
- $\tau > 1$: flatter → more surprising, more mistakes.

Let's see temperature reshape a real next-token distribution.
"""),
    code(r"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ctx = torch.tensor([tok.encode("ROMEO:")])
with torch.no_grad():
    logits = model(ctx)[0][0, -1]              # next-token logits

fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
for ax, tau in zip(axes, [0.5, 1.0, 1.5]):
    probs = F.softmax(logits / tau, dim=-1)
    top = torch.topk(probs, 10)
    ax.bar([tok.decode([i]) for i in top.indices.tolist()], top.values.tolist())
    ax.set_title(f"τ = {tau}"); ax.tick_params(axis="x", rotation=90)
axes[0].set_ylabel("probability")
plt.tight_layout(); plt.show()
"""),
    md(r"""
## Top-k and top-p (nucleus) sampling

Sampling from the *full* distribution occasionally picks absurd low-probability
tokens. Two truncation strategies fix this:

- **Top-k**: keep only the $k$ highest-probability tokens, renormalize, sample.
- **Top-p (nucleus)**: keep the *smallest* set of tokens whose cumulative
  probability first exceeds $p$, renormalize, sample. The cutoff adapts to how
  confident the model is at each step.

Both are implemented in `gptlab.model._filter_logits`. Compare the styles:
"""),
    code(r"""
from gptlab.generate import generate_text

variants = [
    dict(temperature=0.01, label="greedy (τ≈0)"),
    dict(temperature=0.8, top_k=40, label="temp 0.8 + top-k 40"),
    dict(temperature=1.0, top_p=0.9, label="temp 1.0 + top-p 0.9"),
    dict(temperature=1.3, label="hot τ=1.3 (no truncation)"),
]
for v in variants:
    label = v.pop("label")
    print(f"\n===== {label} =====")
    print(generate_text(model, tok, prompt="JULIET:\n", max_new_tokens=120, seed=0, **v))
"""),
    md(r"""
## Your turn — interactive generation

Change `PROMPT` and the sampling knobs and re-run. Good defaults for this model
are `temperature≈0.8` with `top_k≈100`.
"""),
    code(r"""
PROMPT = "KING RICHARD:\n"
print(generate_text(model, tok, prompt=PROMPT, max_new_tokens=250,
                    temperature=0.8, top_k=100, seed=None))
"""),
    md(r"""
## Inference trick: the KV-cache (concept)

Naive generation recomputes attention over the *entire* prefix at every step —
$O(T^2)$ repeated work. But the keys and values for past tokens never change.
Real systems **cache** them: at each new step only the newest token's $Q, K, V$
are computed, the new $K, V$ are appended to the cache, and attention runs once
against the cache — turning per-step cost from $O(T^2)$ into $O(T)$.

We keep generation simple here (no cache) because our sequences are short, but
this single optimization is what makes large-model chat responsive. It is also
exactly why **Grouped-Query Attention** (notebook 08) matters: fewer K/V heads
means a proportionally smaller KV-cache.

---
**Next → `08_modern_tricks_rope_rmsnorm_swiglu_gqa.ipynb`:** how today's LLMs
upgrade this classic architecture.
"""),
]

build(OUT, cells)
