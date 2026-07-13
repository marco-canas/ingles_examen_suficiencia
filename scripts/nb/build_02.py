import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "02_embeddings_skipgram.ipynb")

cells = [
    md(r"""
# 02 · Embeddings (Skip-Gram)

A token id like `742` carries no meaning — id 742 is not "more" than id 741. We
need to turn each token into a **vector** whose geometry encodes meaning. That
vector is an **embedding**.

To build real intuition we train *static word embeddings* with the classic
**Skip-Gram with Negative Sampling (SGNS)** algorithm (Word2Vec, 2013). It is
simple, fast, and — crucially — the resulting vectors are **interpretable**: we
can look at nearest neighbours and even do analogies. At the end we connect this
to the *contextual* embeddings a GPT learns.
"""),
    code(BOOTSTRAP),
    md(r"""
## From one-hot to dense vectors

The naive encoding of a token is a **one-hot** vector: all zeros except a single
1 at the token's index. One-hot vectors are useless as *meaning*: every pair is
equidistant (orthogonal), so "king" is exactly as similar to "queen" as to
"banana".

An **embedding matrix** $E \in \mathbb{R}^{V \times d}$ maps each token to a
dense $d$-dimensional vector — multiplying a one-hot by $E$ just *selects a row*:

$$ \text{embedding}(i) = e_i = E_{i,:} \in \mathbb{R}^{d}. $$

The whole point of training is to arrange these rows so that **geometry reflects
meaning**: related tokens land near each other.

### The distributional hypothesis
> *"You shall know a word by the company it keeps."* — J.R. Firth

Words that appear in similar contexts should get similar vectors. Skip-Gram
turns this slogan into an objective.
"""),
    code(r"""
from gptlab.data import load_text
from gptlab.embeddings import simple_word_tokenize, Vocab, subsample

text = load_text(DATA)
words = simple_word_tokenize(text)
vocab = Vocab(words, min_count=5)
print(f"word tokens: {len(words):,} | vocabulary (count>=5): {len(vocab):,}")
print("most common:", vocab.itos[:12])

ids = vocab.encode(words)
ids = subsample(ids, vocab.counts, threshold=1e-3)   # drop very frequent words
print(f"tokens after subsampling: {len(ids):,}")
"""),
    md(r"""
## The Skip-Gram objective (mathematics)

For each **center** word $c$ we try to predict its **context** words $o$ (those
within a small window). The model keeps two vectors per word: an *input* vector
$v_w$ (the embedding we keep) and an *output* vector $u_w$ (used only in
training). The probability of a true context word is a sigmoid of a dot product,
and we contrast it against $K$ random **negative** words $n_k$:

$$
\mathcal{L} = -\log \sigma(v_c \cdot u_o)
              -\sum_{k=1}^{K} \log \sigma(-\,v_c \cdot u_{n_k}),
\qquad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

**Interpretation.** The first term pulls the center vector *towards* words it
actually co-occurs with; the second term pushes it *away* from random words.
Negatives are sampled from a smoothed unigram distribution
$P(w) \propto \text{count}(w)^{0.75}$, which slightly favours rarer words. After
training, dot product (cosine) between input vectors measures **semantic
similarity**.
"""),
    md(r"""
### The model in code

Two embedding tables, one dot-product loss. This is exactly
`gptlab.embeddings.SkipGramNS`; we spell it out here so nothing is hidden.
"""),
    code(r"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)    # kept: the word vectors
        self.out_emb = nn.Embedding(vocab_size, dim)   # scratch: context vectors
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, pos, neg):
        v = self.in_emb(center)                                   # (B, D)
        u_pos = self.out_emb(pos)                                 # (B, D)
        u_neg = self.out_emb(neg)                                 # (B, K, D)
        pos_score = (v * u_pos).sum(dim=1)                        # (B,)
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)   # (B, K)
        return -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(1)).mean()

print(SkipGramNS(len(vocab)))
"""),
    md(r"""
## Training

We build (center, context) pairs within a window of 2, then train for a few
epochs. On CPU this takes well under a minute.
"""),
    code(r"""
from gptlab.embeddings import build_skipgram_pairs, train_skipgram

centers, contexts = build_skipgram_pairs(ids, window=2)
print(f"training pairs: {len(centers):,}")

sg = SkipGramNS(len(vocab), dim=64)
history = train_skipgram(sg, centers, contexts, vocab,
                         epochs=5, batch_size=512, n_neg=5, lr=5e-3)
"""),
    code(r"""
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.5))
plt.plot(range(1, len(history) + 1), history, "o-")
plt.xlabel("epoch"); plt.ylabel("SGNS loss"); plt.title("Skip-Gram training")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
"""),
    md(r"""
## Interpretation — do the vectors *mean* anything?

Two tests. First, **nearest neighbours** by cosine similarity: words that keep
similar company should cluster. Second, the famous **analogies** via vector
arithmetic ($\text{king} - \text{man} + \text{woman} \approx \text{queen}$).

Our corpus is tiny and the vectors are low-dimensional, so results are
*suggestive rather than perfect* — but the structure is unmistakable.
"""),
    code(r"""
from gptlab.embeddings import nearest_neighbors

for w in ["king", "love", "death", "night", "sword", "heart"]:
    if w in vocab.stoi:
        nbrs = nearest_neighbors(sg.in_emb.weight, vocab, w, k=6)
        print(f"{w:>6}: " + ", ".join(f"{n}({s:.2f})" for n, s in nbrs))
"""),
    code(r"""
from gptlab.embeddings import analogy

for a, b, c in [("man", "woman", "king"), ("king", "queen", "lord")]:
    if all(w in vocab.stoi for w in (a, b, c)):
        res = analogy(sg.in_emb.weight, vocab, a, b, c, k=4)
        print(f"{a} : {b}  ::  {c} : ?  ->  " + ", ".join(f"{n}({s:.2f})" for n, s in res))
"""),
    md(r"""
## Visualizing the space (PCA to 2D)

64 dimensions are impossible to picture, so we project the most frequent words
down to 2D with PCA. Semantically related words tend to fall near one another.
"""),
    code(r"""
import numpy as np
from sklearn.decomposition import PCA
from gptlab.utils import plot_embeddings_2d

n_show = 120
vecs = sg.in_emb.weight.detach().numpy()[:n_show]
labels = vocab.itos[:n_show]
coords = PCA(n_components=2).fit_transform(vecs)
fig, ax = plot_embeddings_2d(coords, labels, title="Skip-Gram embeddings (PCA, top-120 words)")
plt.show()
"""),
    md(r"""
## Bridge: static vectors vs. the embeddings inside a GPT

Skip-Gram gives every word **one fixed vector**. But the word "bank" means
different things in "river bank" and "savings bank". A GPT solves this with two
differences:

1. Its token embedding table `nn.Embedding(vocab_size, n_embd)` is **learned
   jointly** with the whole model, by backprop from the language-modeling loss —
   not from a separate co-occurrence objective.
2. Those static vectors are only the *input*. The **attention** layers
   (notebook 03) then mix information across positions, producing a
   **contextual** representation that changes with the surrounding tokens.

So this notebook's SGNS vectors are the perfect mental model for *what an
embedding is*; the GPT simply learns them end-to-end and then makes them
context-dependent.

---
**Next → `03_attention.ipynb`:** the mechanism at the heart of every modern LLM.
"""),
]

build(OUT, cells)
