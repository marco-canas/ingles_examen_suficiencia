import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "00_intro_and_setup.ipynb")

cells = [
    md(r"""
# 00 · Introduction & Setup

### LLMs in Depth — How an LLM Works Mathematically (and Its Implementation with PyTorch)
**PyCon Colombia 2026 · Universidad EAFIT · Advanced Workshop**

Welcome! Across this series we build a small **GPT-style language model from
scratch** in PyTorch — small enough to train on a laptop **CPU**, yet complete
enough to be *the real thing*.

## Three lenses on every component
Each building block is examined from three angles:

1. **The mathematics** — the exact equations.
2. **The interpretation** — what the math *means* and how to picture it.
3. **The code** — a small PyTorch implementation you can run and modify.

## Roadmap
| # | Notebook | Component |
|---|----------|-----------|
| 00 | Intro & setup | orientation, the language-model objective |
| 01 | Data & tokenization | text → tokens, the **BPE** algorithm |
| 02 | Embeddings | tokens → vectors (**Skip-Gram**) |
| 03 | Attention | the core: **scaled dot-product attention** |
| 04 | Normalization & block | LayerNorm, residuals, MLP, positions |
| 05 | Assembling the GPT | putting it all together |
| 06 | Training | the learning loop and its tricks |
| 07 | Generation | sampling text from the model |
| 08 | Modern tricks | RMSNorm, RoPE, SwiGLU, GQA |

Everything runs on CPU — no GPU required.
"""),
    md(r"""
## Setup

The cell below puts the repo on the import path and loads `gptlab`, our small
companion package. The notebooks *build each component by hand*; `gptlab` holds
the clean, tested versions that later notebooks import so everything stays
consistent with the shipped checkpoint.
"""),
    code(BOOTSTRAP),
    md(r"""
## What *is* a language model?

A language model assigns a probability to a sequence of tokens
$x_1, x_2, \dots, x_T$. By the **chain rule of probability**, any joint
distribution factorizes into a product of **next-token** conditionals:

$$ p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}). $$

A GPT is exactly a parameterized function $p_\theta(x_t \mid x_{<t})$ that
predicts the next token from all previous ones. Training **maximizes the
likelihood** of real text — equivalently, it **minimizes the average negative
log-likelihood**, the *cross-entropy loss*:

$$ \mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}). $$

**Interpretation.** The model plays one game over and over: *given what you've
read so far, guess the next token.* Attention, embeddings, and normalization all
exist to make that guess better. A model that outputs a **uniform** distribution
over a vocabulary of size $V$ has loss $\ln V$ — our sanity-check baseline.
"""),
    md(r"""
## A 60-second PyTorch refresher

Two ideas power everything: **tensors** (n-dimensional arrays) and **autograd**
(automatic differentiation). PyTorch records the operations you apply to tensors
and can replay them backwards to compute gradients.
"""),
    code(r"""
import torch
import torch.nn.functional as F

# A scalar function of a tensor, and its gradient — autograd in four lines.
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()          # y = x1^2 + x2^2 + x3^2
y.backward()                # populates x.grad with dy/dx = 2x
print("x     =", x.detach().tolist())
print("dy/dx =", x.grad.tolist(), "  (should equal 2*x)")
"""),
    md(r"""
### The two functions we will use constantly

**Softmax** turns a vector of scores (*logits*) $z$ into a probability
distribution:

$$ \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}. $$

**Cross-entropy** measures how surprised the model is by the true token. If the
true class is $c$,

$$ \text{CE} = -\log \mathrm{softmax}(z)_c. $$
"""),
    code(r"""
logits = torch.tensor([2.0, 0.5, -1.0])
probs = F.softmax(logits, dim=-1)
print("probs:", [round(p, 3) for p in probs.tolist()], "| sum =", round(probs.sum().item(), 3))

# Cross-entropy when the true next token is index 0:
target = torch.tensor([0])
ce = F.cross_entropy(logits.unsqueeze(0), target)
print("cross-entropy (true = 0):", round(ce.item(), 3))

import math
V = 1024
print(f"baseline loss of a uniform model over V={V}:  ln(V) = {math.log(V):.3f}")
"""),
    md(r"""
That baseline — $\ln V$ — is worth remembering: when we build the full model in
notebook 05, its **initial** loss (before any training) should land very close
to it. If it does, our wiring is correct.

---
**Next → `01_data_and_tokenization_bpe.ipynb`:** turning raw text into the
integer tokens the model actually consumes.
"""),
]

build(OUT, cells)
