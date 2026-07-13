# LLMs in Depth — a tiny GPT from the mathematics up

> **PyCon Colombia 2026 · Universidad EAFIT, Medellín · Advanced Workshop (Spanish)**
> *How an LLM Works Mathematically (and Its Implementation with PyTorch)*

Build a small **GPT-style language model from scratch** in PyTorch — small enough
to train on a **laptop CPU in ~8 minutes**, complete enough to be *the real
thing*. Every component (embeddings, BPE tokenization, attention, normalization,
training/inference tricks) is studied through **three lenses**:

1. **The mathematics** — the exact equations.
2. **The interpretation** — what the math means and how to picture it.
3. **The code** — a small PyTorch implementation you can run and modify.

The result is a "playground GPT" trained on Tiny Shakespeare that produces
recognizable Shakespearean text — character names, dialogue, line breaks —
from about **0.93M parameters**.

```
KING RICHARD:
And, by that I live,
Why, here is not a gracious while.

PAULINA:
I am what instracted, or thou is the day:
I will not have you not on himself.
```

---

## What you'll build

A faithful GPT-2-style decoder-only Transformer:

- a trained **byte-level BPE** tokenizer (small vocab, ~1024),
- **Skip-Gram** word embeddings (Word2Vec) for intuition,
- **multi-head causal self-attention**, **pre-norm LayerNorm**, residual MLP blocks,
- learned positional embeddings, weight-tied LM head,
- a full **training loop** (AdamW, warmup + cosine LR, gradient clipping),
- **sampling** (temperature, top-k, top-p),
- and a tour of the **modern stack** (RMSNorm, RoPE, SwiGLU, GQA).

## Hardware requirements

Designed for a modest, **CPU-only** machine (the reference target is an Intel
i7 8th-gen, 12 GB RAM, no GPU):

| | |
|---|---|
| OS | Windows / macOS / Linux |
| Python | 3.10 – 3.13 (tested on 3.11) |
| RAM | ~2 GB free is plenty |
| Disk | < 1 GB (mostly PyTorch) |
| GPU | **not required** |
| Full training time | ~8–10 min (`playground`), ~3 min (`demo`) |

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-fork-url> pycon_repo
cd pycon_repo

# 2. Create a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install the package + notebook tooling (installs the CPU build of PyTorch)
pip install -e ".[notebooks]"
```

`pip install torch` pulls the **CPU** build by default (no CUDA) — exactly what
we want. If you prefer, `pip install -r requirements.txt` works too.

### Verify the install

```bash
python scripts/smoke_test.py       # runs the whole stack in a few seconds
```

## Running the notebooks

The notebooks live in [`notebooks/`](notebooks/) and are numbered `00 → 08`.

### Option A — VS Code (recommended, no server to manage)

1. Install the **Python** and **Jupyter** extensions (VS Code will recommend them
   automatically — see [.vscode/extensions.json](.vscode/extensions.json)).
2. Open the repo folder, open any `notebooks/*.ipynb`.
3. When prompted for a kernel, pick the **`.venv`** interpreter.
4. Run cells with `Shift+Enter`, or "Run All". VS Code manages the kernel for
   you — **no need to start a Jupyter server**.

### Option B — Jupyter Lab (browser)

Everything is fully compatible with a classic server:

```bash
jupyter lab      # then open notebooks/00_intro_and_setup.ipynb
```

Work through them in order — each notebook ends by pointing to the next.

| # | Notebook | Component |
|---|----------|-----------|
| 00 | [Intro & setup](notebooks/00_intro_and_setup.ipynb) | the language-model objective, PyTorch refresher |
| 01 | [Data & tokenization](notebooks/01_data_and_tokenization_bpe.ipynb) | Tiny Shakespeare, the **BPE** algorithm, vocab-size study |
| 02 | [Embeddings](notebooks/02_embeddings_skipgram.ipynb) | **Skip-Gram** word vectors, neighbours, analogies |
| 03 | [Attention](notebooks/03_attention.ipynb) | **scaled dot-product** & multi-head attention (the core) |
| 04 | [Normalization & block](notebooks/04_normalization_and_transformer_block.ipynb) | LayerNorm, residuals, MLP, positions |
| 05 | [Assembling the GPT](notebooks/05_assembling_the_gpt.ipynb) | wire it all together, sanity-check the loss |
| 06 | [Training](notebooks/06_training.ipynb) | the loop and its tricks (run it live) |
| 07 | [Generation](notebooks/07_generation_and_sampling.ipynb) | temperature, top-k, top-p sampling |
| 08 | [Modern tricks](notebooks/08_modern_tricks_rope_rmsnorm_swiglu_gqa.ipynb) | RMSNorm, RoPE, SwiGLU, GQA |

## The `gptlab` package

The notebooks build each piece by hand for teaching; [`gptlab/`](gptlab/) holds
the clean, tested "source of truth" that later notebooks and the shipped
checkpoint import, so everything stays consistent.

| Module | Contents |
|---|---|
| `config.py` | `GPTConfig`, `TrainConfig` (+ `demo`/`playground` profiles) |
| `tokenizer.py` | trained byte-level BPE + a character-level fallback |
| `data.py` | download / load / encode / batch Tiny Shakespeare |
| `embeddings.py` | `SkipGramNS`, training, nearest-neighbours, analogies |
| `model.py` | `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` |
| `train.py` | LR schedule, loss estimation, the training loop |
| `generate.py` | text generation with sampling controls |
| `modern.py` | `RMSNorm`, RoPE, `SwiGLU`, `GroupedQueryAttention`, `GPTModern` |
| `utils.py` | seeding, CPU threads, param counting, plots |

## Reproducing the checkpoint

The repo ships a pretrained checkpoint (`checkpoints/gpt_playground.pt`, ~3.8 MB)
and its tokenizer (`checkpoints/bpe_tokenizer.json`). Recreate them with:

```bash
python scripts/train_playground.py --profile playground   # ~8 min on CPU
# quicker:
python scripts/train_playground.py --profile demo         # ~3 min
# character-level variant:
python scripts/train_playground.py --tokenizer char
```

## Repository layout

```
pycon_repo/
├── notebooks/         # the workshop, 00 → 08
├── gptlab/            # the clean, importable implementation
├── scripts/           # train_playground.py, smoke_test.py, notebook builders
├── data/              # Tiny Shakespeare (committed)
├── checkpoints/       # pretrained weights + tokenizer (committed)
├── requirements.txt   # dependencies (also in pyproject.toml)
└── pyproject.toml     # `pip install -e .`
```

> Notebooks are generated from readable Python in `scripts/nb/`
> (`python scripts/build_all_notebooks.py`) so the teaching content is easy to
> version and regenerate.

## Credits & references

- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
  and [minGPT](https://github.com/karpathy/minGPT).
- Dataset: Tiny Shakespeare (char-rnn).
- Core papers: *Attention Is All You Need* (Vaswani et al., 2017),
  *Language Models are Unsupervised Multitask Learners* (GPT-2, Radford et al., 2019),
  *Efficient Estimation of Word Representations* (Word2Vec, Mikolov et al., 2013),
  RoPE (Su et al., 2021), RMSNorm (Zhang & Sennrich, 2019), SwiGLU (Shazeer, 2020),
  GQA (Ainslie et al., 2023).

## License

[MIT](LICENSE) — free to use, modify, and share.
