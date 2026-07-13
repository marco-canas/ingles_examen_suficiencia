"""Tokenizers: a trained byte-level BPE and a simple character-level fallback.

Both expose the same tiny interface so the rest of the codebase does not care
which one is in use:

    tok.encode(text) -> list[int]
    tok.decode(ids)  -> str
    tok.vocab_size   -> int   (also tok.get_vocab_size())

``build_tokenizer(cfg, text)`` builds the right one for ``cfg.tokenizer`` and,
crucially, sets ``cfg.vocab_size`` to the *actual* trained size so the model's
embedding table always matches the data.
"""

from __future__ import annotations

import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class BPETokenizer:
    """Thin wrapper around a HuggingFace byte-level BPE ``Tokenizer``.

    Byte-level BPE has two properties we rely on:
      * zero out-of-vocabulary: the 256 raw bytes are always in the base alphabet;
      * lossless round-trip: ``decode(encode(x)) == x``.
    """

    kind = "bpe"

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids) -> str:
        return self._tok.decode([int(i) for i in ids])

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def get_vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def save(self, path: str) -> None:
        self._tok.save(str(path))


def train_bpe(text: str, vocab_size: int = 1024, min_frequency: int = 2,
              save_path: str | None = None) -> BPETokenizer:
    """Train a byte-level BPE tokenizer on ``text``.

    ``vocab_size`` is an *upper bound*: with ``min_frequency`` a merge is only
    created when the byte-pair recurs at least that many times, so a small
    corpus that cannot support ``vocab_size`` well-attested merges simply yields
    a smaller vocabulary. Always read the true size back with ``get_vocab_size``.
    """
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # ByteLevel splits into bytes mapped to a printable unicode alphabet.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    tok = BPETokenizer(tokenizer)
    if save_path is not None:
        tok.save(save_path)
    return tok


def load_bpe(path: str) -> BPETokenizer:
    return BPETokenizer(Tokenizer.from_file(str(path)))


class CharTokenizer:
    """Character-level tokenizer: the vocabulary is the set of characters seen.

    Tiny vocab (~65 for Shakespeare), trains fast, and often looks a touch
    crisper on tiny data. Used for comparison in notebook 01 and available via
    ``cfg.tokenizer = "char"``.
    """

    kind = "char"

    def __init__(self, text: str):
        chars = sorted(set(text))
        self._vocab = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab_size(self) -> int:
        return len(self._vocab)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self._vocab}, f, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str) -> "CharTokenizer":
        obj = cls.__new__(cls)
        with open(path, encoding="utf-8") as f:
            chars = json.load(f)["chars"]
        obj._vocab = chars
        obj.stoi = {c: i for i, c in enumerate(chars)}
        obj.itos = {i: c for i, c in enumerate(chars)}
        return obj


def build_tokenizer(cfg, text: str, min_frequency: int = 2, save_path: str | None = None):
    """Build the tokenizer selected by ``cfg.tokenizer`` and sync ``cfg.vocab_size``.

    Returns the tokenizer; mutates ``cfg.vocab_size`` in place so the model
    matches the trained vocabulary exactly.
    """
    if cfg.tokenizer == "char":
        tok = CharTokenizer(text)
        if save_path is not None:
            tok.save(save_path)
    elif cfg.tokenizer == "bpe":
        tok = train_bpe(text, vocab_size=cfg.vocab_size, min_frequency=min_frequency,
                        save_path=save_path)
    else:
        raise ValueError(f"unknown tokenizer: {cfg.tokenizer!r} (use 'bpe' or 'char')")
    cfg.vocab_size = tok.get_vocab_size()
    return tok


def load_tokenizer(kind: str, path: str):
    """Reload a saved tokenizer of the given kind (used when loading a checkpoint)."""
    if kind == "bpe":
        return load_bpe(path)
    if kind == "char":
        return CharTokenizer.from_file(path)
    raise ValueError(f"unknown tokenizer kind: {kind!r}")
