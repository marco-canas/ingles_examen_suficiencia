"""Word2Vec-style Skip-Gram with Negative Sampling (SGNS), from scratch.

This is a *standalone, pedagogical* module: it teaches what an embedding is by
training static word vectors on the corpus and letting us inspect neighbours and
analogies. It is intentionally separate from the GPT, whose embeddings are
learned jointly and are *contextual*. Notebook 02 draws that bridge explicitly.

Pipeline:

    tokens = simple_word_tokenize(text)
    vocab  = Vocab(tokens, min_count=5)
    ids    = vocab.encode(tokens)                # + optional subsampling
    centers, contexts = build_skipgram_pairs(ids, window=2)
    model  = SkipGramNS(len(vocab), dim=64)
    train_skipgram(model, centers, contexts, vocab, ...)
    nearest_neighbors(model.in_emb.weight, vocab, "king")
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def simple_word_tokenize(text: str) -> list[str]:
    """Lowercase and split into alphabetic word tokens (good enough for teaching)."""
    return re.findall(r"[a-z]+", text.lower())


class Vocab:
    """Maps words <-> integer ids, keeping words with count >= ``min_count``."""

    def __init__(self, tokens: list[str], min_count: int = 5):
        counts = Counter(tokens)
        self.itos = [w for w, c in counts.most_common() if c >= min_count]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.array([counts[w] for w in self.itos], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi[w] for w in tokens if w in self.stoi]


def subsample(ids: list[int], counts: np.ndarray, threshold: float = 1e-3,
              rng: np.random.Generator | None = None) -> list[int]:
    """Randomly drop very frequent words (word2vec subsampling).

    Keeps word w with probability ``sqrt(threshold / freq) + threshold / freq``.
    Reduces the dominance of tokens like "the"/"and" and speeds up training.
    """
    rng = rng or np.random.default_rng(0)
    freq = counts / counts.sum()
    keep_prob = (np.sqrt(threshold / freq) + threshold / freq).clip(max=1.0)
    draws = rng.random(len(ids))
    return [i for i, r in zip(ids, draws) if r < keep_prob[i]]


def build_skipgram_pairs(ids: list[int], window: int = 2):
    """Build (center, context) pairs within +/- ``window`` tokens."""
    centers, contexts = [], []
    n = len(ids)
    for pos in range(n):
        c = ids[pos]
        start = max(0, pos - window)
        end = min(n, pos + window + 1)
        for j in range(start, end):
            if j == pos:
                continue
            centers.append(c)
            contexts.append(ids[j])
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


class SkipGramNS(nn.Module):
    """Skip-Gram with Negative Sampling.

    Two embedding tables:
      * ``in_emb``  -- the "word" vectors we keep and visualise.
      * ``out_emb`` -- the "context" vectors used only during training.

    For a (center v, positive context u+) pair and K negatives u-_k the loss is
        -log sigma(v . u+) - sum_k log sigma(-v . u-_k)
    i.e. push v towards real contexts and away from random ones.
    """

    def __init__(self, vocab_size: int, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.in_emb = nn.Embedding(vocab_size, dim)
        self.out_emb = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        v = self.in_emb(center)                    # (B, D)
        u_pos = self.out_emb(pos)                  # (B, D)
        u_neg = self.out_emb(neg)                  # (B, K, D)
        pos_score = (v * u_pos).sum(dim=1)                       # (B,)
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # (B, K)
        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()


def _neg_sampling_cdf(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    p = counts ** power
    p = p / p.sum()
    return np.cumsum(p)


def train_skipgram(model: SkipGramNS, centers: np.ndarray, contexts: np.ndarray,
                   vocab: Vocab, epochs: int = 5, batch_size: int = 512, n_neg: int = 5,
                   lr: float = 5e-3, seed: int = 0, log_fn=print) -> list[float]:
    """Train SGNS with minibatches; returns per-log loss history."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    cdf = _neg_sampling_cdf(vocab.counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_pairs = len(centers)
    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        perm = rng.permutation(n_pairs)
        running = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]
            c = torch.from_numpy(centers[idx])
            p = torch.from_numpy(contexts[idx])
            neg = np.searchsorted(cdf, rng.random((len(idx), n_neg))).astype(np.int64)
            neg = torch.from_numpy(neg)
            loss = model(c, p, neg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        avg = running / max(1, n_batches)
        history.append(avg)
        if log_fn:
            log_fn(f"epoch {epoch + 1}/{epochs} | loss {avg:.4f}")
    return history


@torch.no_grad()
def _normalized(weight: torch.Tensor) -> torch.Tensor:
    return F.normalize(weight, dim=1)


def nearest_neighbors(weight: torch.Tensor, vocab: Vocab, word: str, k: int = 10):
    """Return the ``k`` nearest words to ``word`` by cosine similarity."""
    if word not in vocab.stoi:
        raise KeyError(f"{word!r} not in vocabulary")
    w = _normalized(weight)
    q = w[vocab.stoi[word]]
    sims = w @ q
    top = torch.topk(sims, k + 1).indices.tolist()
    return [(vocab.itos[i], float(sims[i])) for i in top if i != vocab.stoi[word]][:k]


def analogy(weight: torch.Tensor, vocab: Vocab, a: str, b: str, c: str, k: int = 5):
    """Solve 'a is to b as c is to ?' via vector arithmetic (b - a + c)."""
    for w in (a, b, c):
        if w not in vocab.stoi:
            raise KeyError(f"{w!r} not in vocabulary")
    w = _normalized(weight)
    target = w[vocab.stoi[b]] - w[vocab.stoi[a]] + w[vocab.stoi[c]]
    target = F.normalize(target, dim=0)
    sims = w @ target
    exclude = {vocab.stoi[a], vocab.stoi[b], vocab.stoi[c]}
    top = torch.topk(sims, k + len(exclude)).indices.tolist()
    return [(vocab.itos[i], float(sims[i])) for i in top if i not in exclude][:k]
