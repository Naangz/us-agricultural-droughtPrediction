"""Run every Nebraska BiLSTM scenario script in sequence.

The scenario scripts already call ``plt.savefig(...)`` for their figures.
This runner forces a non-interactive matplotlib backend in each child process
so ``plt.show()`` does not block training progress while the images are still
written to disk.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
import tensorflow as tf

# 1. Batasi thread intra/inter op (Coba set ke jumlah P-Core kamu, yaitu 6 atau 12)
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

# 2. Set environment variable untuk OpenMP (jika TF kamu pakai backend OMP)
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

BASE_DIR = Path(__file__).resolve().parent
SCENARIO_PATTERN = re.compile(r"^BiLSTM_Scenario(\d+)([A-Z]?)_.*\.py$")


def scenario_sort_key(path: Path) -> tuple[int, str, str]:
    match = SCENARIO_PATTERN.match(path.name)
    if match is None:
        return (999, path.name.lower(), path.name)
    number = int(match.group(1))
    suffix = match.group(2)
    return (number, suffix, path.name.lower())


def discover_scenario_scripts() -> list[Path]:
    scripts = [
        path
        for path in BASE_DIR.glob("BiLSTM_Scenario*.py")
        if path.name != "training_optimizations.py"
    ]
    return sorted(scripts, key=scenario_sort_key)


def run_script(script_path: Path) -> int:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    print("\n" + "=" * 100)
    print(f"Running: {script_path.name}")
    print("=" * 100)

    completed = subprocess.run(
        [sys.executable, "-u", str(script_path)],
        cwd=str(BASE_DIR),
        env=env,
        check=False,
    )
    return completed.returncode


def main() -> int:
    scripts = discover_scenario_scripts()
    if not scripts:
        print("No BiLSTM scenario scripts found.")
        return 1

    print(f"Found {len(scripts)} scenario scripts in {BASE_DIR}")

    for script_path in scripts:
        return_code = run_script(script_path)
        if return_code != 0:
            print(f"\nStopped because {script_path.name} exited with code {return_code}.")
            return return_code

    print("\nAll scenario scripts finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())