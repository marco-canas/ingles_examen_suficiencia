"""Regenerate every notebook in notebooks/ from the builders in scripts/nb/.

    python scripts/build_all_notebooks.py
"""

import glob
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))

builders = sorted(glob.glob(os.path.join(HERE, "nb", "build_*.py")))
for path in builders:
    runpy.run_path(path, run_name="__main__")
print(f"\nbuilt {len(builders)} notebooks")
