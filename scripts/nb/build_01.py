import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "01_data_and_tokenization_bpe.ipynb")

cells = [
    md(r"""
# 01 · Data & Tokenization (BPE)

A neural network consumes **numbers**, not text. *Tokenization* is the bridge:
it maps a string to a sequence of integer ids (and back). This notebook covers

- our corpus (**Tiny Shakespeare**),
- the simplest scheme (**character-level**),
- the **Byte-Pair Encoding (BPE)** algorithm — mathematics + a from-scratch toy,
- a **vocab-size study** that justifies our choice *from the data*, and
- building the train/validation token arrays the model will train on.
"""),
    code(BOOTSTRAP),
    md(r"""
## The corpus

Tiny Shakespeare is ~1 MB of the Bard's plays — small enough to train on in
minutes, large enough to learn real structure (character names, line breaks,
archaic diction). It is committed in `data/` for offline reproducibility.
"""),
    code(r"""
from gptlab.data import download_shakespeare, load_text

download_shakespeare(DATA)
text = load_text(DATA)
print(f"characters : {len(text):,}")
print(f"unique chars: {len(set(text))}")
print("-" * 60)
print(text[:250])
"""),
    md(r"""
## Lens 1 — character-level tokenization

The simplest possible vocabulary is *the set of characters that appear*. Each
character gets an integer id. There is **no out-of-vocabulary problem** and the
vocab is tiny (~65), but sequences are long (one token per character).
"""),
    code(r"""
from gptlab.tokenizer import CharTokenizer

char_tok = CharTokenizer(text)
print("char vocab size:", char_tok.vocab_size)
ids = char_tok.encode("To be, or not to be")
print("encoded:", ids)
print("decoded:", repr(char_tok.decode(ids)))
"""),
    md(r"""
## Lens 2 — subword tokenization with BPE

Character-level is simple but wasteful: the model must relearn common spellings
one character at a time. Word-level explodes the vocabulary and cannot spell new
words. **Byte-Pair Encoding (BPE)** is the pragmatic middle ground used by GPT-2,
GPT-4, Llama, and friends.

### The algorithm (mathematics)

Start with a base vocabulary of the 256 raw **bytes** (so *any* text is
representable — zero out-of-vocabulary). Then repeat, greedily:

1. Count the frequency of every adjacent pair of tokens in the corpus:
$$ \text{count}(a, b) = \#\{\, t : (x_t, x_{t+1}) = (a, b) \,\}. $$
2. Merge the **most frequent** pair $(a^\*, b^\*) = \arg\max_{(a,b)} \text{count}(a,b)$
   into a single **new** token, extending the vocabulary by one.
3. Rewrite the corpus with the merge applied, and repeat until the vocabulary
   reaches the target size (or no pair repeats often enough).

**Interpretation.** BPE discovers *statistically frequent chunks* — common
letter pairs, then syllables, then whole frequent words like `the` — and gives
each its own id. Frequent text becomes short; rare text falls back to smaller
pieces. It is lossless data compression repurposed as tokenization.
"""),
    md(r"""
### A from-scratch toy BPE (illustrative)

We are **not** going to build the production tokenizer by hand — a well-tested
library does that below. But ~20 lines make the merge loop concrete on a toy
string. Watch the vocabulary grow one frequent pair at a time.
"""),
    code(r"""
from collections import Counter

def get_stats(ids):
    counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

toy = "the cat sat on the mat, the cat ran to the hat"
ids = list(toy.encode("utf-8"))          # start from raw bytes
vocab = {i: bytes([i]) for i in range(256)}
next_id = 256
print(f"start: {len(ids)} byte-tokens")
for step in range(6):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    if stats[pair] < 2:
        break
    ids = merge(ids, pair, next_id)
    vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
    print(f"merge {step + 1}: {pair} (seen {stats[pair]}x) -> new token {next_id} = {vocab[next_id]!r}")
    next_id += 1
print(f"end:   {len(ids)} tokens  |  learned chunks: "
      f"{[vocab[i].decode('utf-8', 'replace') for i in range(256, next_id)]}")
"""),
    md(r"""
## Choosing the vocabulary size *from the data*

How big should the vocabulary be? Too small and we barely compress; too large
and many tokens are **rare** — their embeddings hardly get trained. We don't
guess: we **measure**. For each candidate size we train a BPE tokenizer and look at

- **compression** (characters per token — higher = more context per step), and
- **token-frequency health** (what fraction of tokens are seen very rarely).

Two facts make the size *self-limiting*:
- `min_frequency` means a merge is only created if the pair actually recurs, so
  a small corpus that cannot support the ceiling yields a **smaller** vocab.
- We always read the **actual** size back with `get_vocab_size()`.
"""),
    code(r"""
import numpy as np
from gptlab.tokenizer import train_bpe

candidates = [256, 512, 1024, 2048]
rows = []
for ceiling in candidates:
    tok = train_bpe(text, vocab_size=ceiling, min_frequency=2)
    ids = np.asarray(tok.encode(text))
    actual = tok.get_vocab_size()
    chars_per_tok = len(text) / len(ids)
    counts = np.bincount(ids, minlength=actual)
    pct_rare = float((counts < 100).mean()) * 100      # % of vocab seen <100 times
    rows.append((ceiling, actual, chars_per_tok, pct_rare))
    print(f"ceiling {ceiling:5d} | actual {actual:5d} | "
          f"chars/token {chars_per_tok:5.3f} | tokens seen <100x: {pct_rare:5.1f}%")
"""),
    code(r"""
import matplotlib.pyplot as plt

ceilings = [r[0] for r in rows]
comp = [r[2] for r in rows]
rare = [r[3] for r in rows]

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(ceilings, comp, "o-", color="tab:blue", label="chars/token")
ax1.set_xlabel("vocab size (ceiling)")
ax1.set_ylabel("compression (chars/token)", color="tab:blue")
ax1.set_xscale("log", base=2)
ax2 = ax1.twinx()
ax2.plot(ceilings, rare, "s--", color="tab:red", label="% rare tokens")
ax2.set_ylabel("% tokens seen < 100x", color="tab:red")
ax1.set_title("Vocab-size study on Tiny Shakespeare")
fig.tight_layout()
plt.show()
"""),
    md(r"""
**Reading the curves.** Compression rises quickly and then flattens — classic
diminishing returns. Meanwhile the share of *rarely-seen* tokens grows with the
vocabulary. We pick **1024**: near the compression knee, with the rare-token
fraction still modest — subword tokens that connect nicely to the embeddings
story in notebook 02, while keeping the model's embedding table small enough to
train on CPU.
"""),
    code(r"""
# Train and SAVE the tokenizer we will actually use (matches the shipped checkpoint).
tok = train_bpe(text, vocab_size=1024, min_frequency=2,
                save_path=os.path.join(CKPT_DIR, "bpe_tokenizer.json"))
print("actual vocab size:", tok.get_vocab_size())

sample = "First Citizen:\nBefore we proceed"
enc = tok.encode(sample)
print("round-trip OK:", tok.decode(enc) == sample)
print(f"'{sample[:20]}...' -> {len(enc)} tokens (vs {len(sample)} chars)")
print("chars/token over full corpus:", round(len(text) / len(tok.encode(text)), 3))
"""),
    md(r"""
### Aside: what does the real GPT-2 tokenizer look like?

For comparison, `tiktoken` gives us GPT-2's actual BPE (vocab **50,257**). It
compresses more, but such a large vocabulary would make our tiny model's
embedding table (`vocab_size × n_embd`) dominate the parameter count and slow
CPU training — which is exactly why we trained a small custom BPE.
"""),
    code(r"""
try:
    import tiktoken
    gpt2 = tiktoken.get_encoding("gpt2")
    s = text[:300]
    print("GPT-2 vocab size :", gpt2.n_vocab)
    print("GPT-2 tokens     :", len(gpt2.encode(s)), "for 300 chars")
    print("our BPE-1024     :", len(tok.encode(s)), "for 300 chars")
except Exception as e:
    print("tiktoken not available (optional):", e)
"""),
    md(r"""
## Building the training data

Finally we encode the whole corpus and split it into **train** and
**validation** streams of token ids (stored compactly as `uint16`). The model
will learn on `train` and we watch `val` to detect overfitting.
"""),
    code(r"""
from gptlab.data import prepare_splits

data = prepare_splits(text, tok, val_frac=0.1)
print("train tokens:", len(data["train"]), "| val tokens:", len(data["val"]))
print("first 20 train ids:", data["train"][:20].tolist())
print("decoded            :", repr(tok.decode(data["train"][:20].tolist())))
"""),
    md(r"""
We now have integer sequences the model can consume.

---
**Next → `02_embeddings_skipgram.ipynb`:** how a bare integer id becomes a
*meaningful vector* — and what "meaning" even means for a token.
"""),
]

build(OUT, cells)
