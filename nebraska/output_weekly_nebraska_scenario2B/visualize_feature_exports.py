from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parents[2] / "visualization" / "plot_feature_exports.py"
    sys.argv = [
        str(script_path),
        "--folder",
        str(Path(__file__).resolve().parent),
        *sys.argv[1:],
    ]
    runpy.run_path(str(script_path), run_name="__main__")
