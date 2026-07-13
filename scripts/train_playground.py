"""Train the playground GPT and save a checkpoint + tokenizer.

Usage:
    python scripts/train_playground.py                     # bpe, playground profile
    python scripts/train_playground.py --profile demo       # quick ~2 min run
    python scripts/train_playground.py --tokenizer char     # character-level

Outputs (into --out, default ./checkpoints):
    gpt_playground.pt          model weights + config + tokenizer metadata
    bpe_tokenizer.json         (bpe)  trained tokenizer
    char_tokenizer.json        (char) character vocabulary
"""

from __future__ import annotations

import argparse
import os
import time

from gptlab import GPTConfig, TrainConfig, GPT
from gptlab.data import download_shakespeare, load_text, prepare_splits
from gptlab.generate import generate_text
from gptlab.tokenizer import build_tokenizer
from gptlab.train import train
from gptlab.utils import configure_cpu_threads, count_params, human_params, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the playground GPT.")
    parser.add_argument("--profile", default="playground", choices=["demo", "playground"])
    parser.add_argument("--tokenizer", default="bpe", choices=["bpe", "char"])
    parser.add_argument("--vocab-size", type=int, default=1024, help="BPE vocab ceiling")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--data", default=os.path.join("data", "tinyshakespeare.txt"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    threads = configure_cpu_threads()
    set_seed(1337)
    print(f"[setup] threads={threads} | tokenizer={args.tokenizer} | profile={args.profile}")

    download_shakespeare(args.data)
    text = load_text(args.data)
    print(f"[data] {len(text):,} chars, {len(set(text))} unique chars")

    cfg = GPTConfig(vocab_size=args.vocab_size, tokenizer=args.tokenizer)
    tok_name = "bpe_tokenizer.json" if args.tokenizer == "bpe" else "char_tokenizer.json"
    tok_path = os.path.join(args.out, tok_name)
    tok = build_tokenizer(cfg, text, save_path=tok_path)
    print(f"[tokenizer] actual vocab_size={cfg.vocab_size} -> {tok_path}")

    data = prepare_splits(text, tok, val_frac=0.1)
    print(f"[data] train={len(data['train']):,} tokens | val={len(data['val']):,} tokens")

    model = GPT(cfg)
    print(f"[model] {human_params(count_params(model))} params "
          f"(non-embedding {human_params(model.get_num_params(non_embedding=True))})")

    tcfg = TrainConfig.for_profile(args.profile)
    ckpt_path = os.path.join(args.out, "gpt_playground.pt")
    tokenizer_meta = {"kind": args.tokenizer, "path": os.path.basename(tok_path)}

    t0 = time.time()
    train(model, data, tcfg, cfg, ckpt_path=ckpt_path, tokenizer_meta=tokenizer_meta)
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min")

    print("\n[sample] --------------------------------------------------")
    print(generate_text(model, tok, prompt="\n", max_new_tokens=400,
                        temperature=0.8, top_k=100, seed=0))


if __name__ == "__main__":
    main()
