"""Tiny helper to build .ipynb notebooks from Python (via nbformat).

Notebooks are generated (not hand-edited) so the teaching content lives in
readable Python source under scripts/nb/ and can be regenerated at any time:

    python scripts/build_all_notebooks.py
"""

from __future__ import annotations

import os

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(src: str):
    return nbf.v4.new_code_cell(src.strip("\n"))


def build(path: str, cells: list, title: str | None = None) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {path} ({len(cells)} cells)")


# Shared preamble used by most notebooks. `import gptlab` works out of the box
# after `pip install -e .`; the sys.path fallback also covers the case where
# only requirements were installed. Also defines REPO_ROOT / DATA / CKPT_DIR so
# file paths resolve no matter the working directory (notebooks/ or repo root).
BOOTSTRAP = """
import sys, os


def _find_repo_root():
    for base in (os.getcwd(), os.path.abspath(os.path.join(os.getcwd(), ".."))):
        if os.path.isdir(os.path.join(base, "gptlab")):
            return base
    return os.getcwd()


REPO_ROOT = _find_repo_root()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gptlab
from gptlab.utils import set_seed, configure_cpu_threads

set_seed(1337)
n_threads = configure_cpu_threads()
DATA = os.path.join(REPO_ROOT, "data", "tinyshakespeare.txt")
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")
print(f"gptlab {gptlab.__version__} ready | CPU threads: {n_threads}")
""".strip()
