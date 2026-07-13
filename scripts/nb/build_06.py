import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nbkit import BOOTSTRAP, build, code, md  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "notebooks", "06_training.ipynb")

cells = [
    md(r"""
# 06 · Training

Now we teach the model. Training is a loop: sample a batch of text, predict the
next token everywhere, measure the loss, and nudge the weights downhill. This
notebook shows the data batching, the **optimization "tricks"** that make it work
well, and then runs a full training that you can watch converge.

> ⏱️ On a CPU the full run ("playground" profile) takes **~8–10 minutes**. Set
> `PROFILE = "demo"` for a quick ~3-minute pass. During the workshop this is a
> great moment to take questions while it trains.
"""),
    code(BOOTSTRAP),
    code(r"""
import os
# "playground" = full run (~8-10 min CPU); "demo" = quick ~3 min run.
PROFILE = os.environ.get("GPTLAB_PROFILE", "playground")
print("training profile:", PROFILE)
"""),
    md(r"""
## Batching: inputs and shifted targets

We cut the token stream into random windows of length `block_size`. The target
`y` is simply the input `x` shifted **one token to the right** — at every
position the model must predict the *next* token. One window of length $T$ thus
provides $T$ training signals at once.
"""),
    code(r"""
from gptlab import GPTConfig
from gptlab.data import get_batch, load_text, prepare_splits
from gptlab.tokenizer import load_bpe

tok = load_bpe(os.path.join(CKPT_DIR, "bpe_tokenizer.json"))
cfg = GPTConfig(vocab_size=tok.vocab_size)
data = prepare_splits(load_text(DATA), tok, val_frac=0.1)

x, y = get_batch(data["train"], cfg.block_size, batch_size=2)
print("x[0][:12]:", x[0, :12].tolist())
print("y[0][:12]:", y[0, :12].tolist(), " <- x shifted left by 1")
print("decoded x:", repr(tok.decode(x[0, :12].tolist())))
"""),
    md(r"""
## The optimization tricks (mathematics)

**AdamW.** Adam keeps running estimates of the gradient's mean $m_t$ and
variance $v_t$ and takes a per-parameter step
$\theta \leftarrow \theta - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$.
The *W* adds **decoupled weight decay** (an $L_2$ pull toward 0), applied only to
matmul weights — not to biases or LayerNorm gains.

**Learning-rate schedule: warmup + cosine decay.** Start tiny, ramp up linearly
for a few steps (avoids early divergence), then decay smoothly along a cosine to
a small floor:

$$
\eta_t =
\begin{cases}
\eta_{\max}\,\dfrac{t}{t_{\text{warm}}} & t < t_{\text{warm}} \\[8pt]
\eta_{\min} + \tfrac12\big(\eta_{\max}-\eta_{\min}\big)\Big(1+\cos\pi\,\dfrac{t-t_{\text{warm}}}{t_{\max}-t_{\text{warm}}}\Big) & \text{otherwise}
\end{cases}
$$

**Gradient clipping.** Rescale the gradient if its norm exceeds a threshold
($\lVert g\rVert \le c$), which prevents rare huge updates from destabilizing
training.
"""),
    code(r"""
import matplotlib.pyplot as plt
from gptlab import TrainConfig
from gptlab.train import get_lr

tcfg = TrainConfig.for_profile(PROFILE)
lrs = [get_lr(t, tcfg) for t in range(tcfg.max_iters + 1)]
plt.figure(figsize=(6, 3.2))
plt.plot(lrs)
plt.xlabel("iteration"); plt.ylabel("learning rate")
plt.title(f"Warmup + cosine schedule ({PROFILE})"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
    md(r"""
## The training loop

`gptlab.train.train` implements the loop: for each step, fetch a batch, set the
scheduled learning rate, forward → loss → backward → clip → optimizer step, and
periodically estimate train/val loss. Watch the numbers fall.
"""),
    code(r"""
import time
from gptlab.model import GPT
from gptlab.train import train
from gptlab.utils import set_seed, human_params, count_params

set_seed(1337)
model = GPT(cfg)
print("training", human_params(count_params(model)), "params for", tcfg.max_iters, "iters...\n")

t0 = time.time()
history = train(model, data, tcfg, cfg)
print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
"""),
    code(r"""
from gptlab.utils import plot_losses
fig, ax = plot_losses(history, title=f"Training curve ({PROFILE})")
plt.show()
"""),
    md(r"""
## Does it generate anything yet?

Even this short run produces text with real Shakespearean *shape* — capitalized
speaker names, line breaks, mostly-real words. (Coherent meaning is beyond a
~1M-parameter model on 1 MB of text — and that's the honest, expected ceiling.)
"""),
    code(r"""
from gptlab.generate import generate_text
print(generate_text(model, tok, prompt="\n", max_new_tokens=300,
                    temperature=0.8, top_k=100, seed=0))
"""),
    md(r"""
## The shipped checkpoint

The repository ships `checkpoints/gpt_playground.pt`, trained with the exact
`playground` profile above (~8 min, final val loss ≈ 3.75). Notebook 07 loads it
so you can explore generation without waiting for training. You can reproduce it
any time with:

```bash
python scripts/train_playground.py --profile playground
```

---
**Next → `07_generation_and_sampling.ipynb`.**
"""),
]

build(OUT, cells)
