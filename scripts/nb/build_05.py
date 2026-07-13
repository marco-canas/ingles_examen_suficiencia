import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "05_assembling_the_gpt.ipynb")

cells = [
    md(r"""
# 05 · Assembling the GPT

Time to wire every component into one model. A GPT is a short, clear pipeline:

```
tokens ─▶ token embedding  ┐
                            ├─▶ + ─▶ [ Block × N ] ─▶ LayerNorm ─▶ LM head ─▶ logits
positions ─▶ pos embedding ─┘
```

This notebook builds the full `GPT`, checks its size, walks a batch through it,
and confirms the **initial loss lands on the $\ln V$ baseline** from notebook 00.
"""),
    code(BOOTSTRAP),
    md(r"""
## The full model (mathematics)

Given input token ids $x_{1:T}$:

$$
h^0_t = E_{x_t} + P_t \quad (\text{token + position embedding})
$$
$$
h^{\ell} = \mathrm{Block}_\ell(h^{\ell-1}), \quad \ell = 1 \dots N
$$
$$
z_t = \mathrm{LN}(h^N_t)\,W_{\text{head}}^\top \in \mathbb{R}^{V} \quad (\text{logits over the vocabulary})
$$

The training loss is the mean cross-entropy of the next-token predictions:

$$ \mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} \mathrm{CE}\big(z_t,\; x_{t+1}\big). $$

Here is the exact class (`gptlab.model.GPT`) that later notebooks and the shipped
checkpoint use:
"""),
    code(r"""
import inspect
from gptlab.model import GPT
print(inspect.getsource(GPT.__init__))
print(inspect.getsource(GPT.forward))
"""),
    md(r"""
## Weight tying

Notice `self.transformer.wte.weight = self.lm_head.weight`: the **input embedding
table and the output projection are the same matrix**. Intuitively, the vector
that *represents* a token and the vector that *scores* it should live in the same
space. Tying them saves $V \times d$ parameters (a big fraction of a small model)
and usually improves quality.
"""),
    code(r"""
from gptlab import GPTConfig
from gptlab.tokenizer import load_bpe
from gptlab.utils import count_params, human_params

tok = load_bpe(os.path.join(CKPT_DIR, "bpe_tokenizer.json"))
cfg = GPTConfig(vocab_size=tok.vocab_size)     # block=64, n_layer=4, n_head=4, n_embd=128
model = GPT(cfg)
print(cfg)
print("total params    :", human_params(count_params(model)))
print("non-embedding   :", human_params(model.get_num_params(non_embedding=True)))
print("tied? wte is lm_head:", model.transformer.wte.weight is model.lm_head.weight)
"""),
    md(r"""
## Where do the parameters live?

A quick breakdown shows the model is dominated by the **blocks** (attention +
MLP), with embeddings a smaller share — exactly why keeping the vocabulary small
(notebook 01) kept the whole model tiny.
"""),
    code(r"""
def block_breakdown(model, cfg):
    wte = model.transformer.wte.weight.numel()
    wpe = model.transformer.wpe.weight.numel()
    per_block = sum(p.numel() for p in model.transformer.h[0].parameters())
    ln_f = sum(p.numel() for p in model.transformer.ln_f.parameters())
    print(f"token embedding (tied w/ head): {wte:>9,}")
    print(f"position embedding            : {wpe:>9,}")
    print(f"per block                     : {per_block:>9,}  x {cfg.n_layer} = {per_block*cfg.n_layer:,}")
    print(f"final LayerNorm               : {ln_f:>9,}")

block_breakdown(model, cfg)
"""),
    md(r"""
## Sanity check — the initial loss

Before any training the model is random, so its best guess is roughly uniform
over the vocabulary. Cross-entropy should therefore be close to
$\ln V = \ln(1024) \approx 6.93$. If it is, the whole forward pass is wired
correctly.
"""),
    code(r"""
import math
import torch
from gptlab.data import load_text, prepare_splits
from gptlab.data import get_batch

data = prepare_splits(load_text(DATA), tok, val_frac=0.1)
x, y = get_batch(data["train"], cfg.block_size, batch_size=4)
logits, loss = model(x, y)
print("input  x:", tuple(x.shape))
print("logits  :", tuple(logits.shape), " (batch, time, vocab)")
print(f"initial loss = {loss.item():.3f}   vs   ln(V) = {math.log(cfg.vocab_size):.3f}  ✅")
"""),
    md(r"""
The model is assembled and correct — but it only produces noise so far. In the
next notebook we **train** it.

---
**Next → `06_training.ipynb`.**
"""),
]

build(OUT, cells)
