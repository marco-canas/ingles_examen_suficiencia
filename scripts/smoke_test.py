"""Fast end-to-end smoke test: exercises the whole stack in a few seconds.

Not a training run -- just proves that tokenizer -> data -> model -> train
-> generate all wire together and that the modern variant runs too.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import torch

from gptlab import GPTConfig, TrainConfig, GPT
from gptlab.data import download_shakespeare, load_text, prepare_splits
from gptlab.generate import generate_text
from gptlab.modern import GPTModern
from gptlab.tokenizer import build_tokenizer
from gptlab.train import train
from gptlab.utils import count_params, human_params, set_seed


def main() -> None:
    set_seed(0)
    download_shakespeare()
    text = load_text()[:20000]                    # small slice keeps it fast

    cfg = GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=2, n_embd=64, tokenizer="bpe")
    tok = build_tokenizer(cfg, text)
    assert tok.decode(tok.encode(text[:100])) == text[:100], "tokenizer round-trip failed"
    print(f"[ok] tokenizer round-trip, vocab_size={cfg.vocab_size}")

    data = prepare_splits(text, tok)
    model = GPT(cfg)
    print(f"[ok] model built: {human_params(count_params(model))} params")

    tcfg = TrainConfig(max_iters=20, warmup_iters=5, eval_interval=10, eval_iters=5, batch_size=8)
    hist = train(model, data, tcfg, cfg, log_fn=None)
    assert hist["train"][-1] < hist["train"][0], "loss did not decrease"
    print(f"[ok] trained 20 iters: loss {hist['train'][0]:.3f} -> {hist['train'][-1]:.3f}")

    sample = generate_text(model, tok, prompt="The ", max_new_tokens=20, top_k=20, seed=0)
    assert isinstance(sample, str) and len(sample) > 0
    print(f"[ok] generation: {sample!r}")

    gm = GPTModern(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = gm(x, x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    print(f"[ok] modern GPT forward: loss {loss.item():.3f}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
