"""Run a quick Active Learning smoke test.

This script executes train.py with tiny AL settings and checks for expected files
in the output directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    params_path = os.path.join(repo_root, "params", "train_xlmroberta.json")
    with open(params_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_name = cfg.get("model_type", "model")
    out_base = cfg.get("output_folder", "models/xlmroberta")

    # Run train with AL
    cmd = [
        sys.executable,
        os.path.join(repo_root, "esnlir", "training", "train.py"),
        "--config-file", params_path,
        "--active-learning",
        "--al_strategy", "NegE",
        "--al_L", "1",
        "--al_K", "2",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=repo_root)

    # Check outputs
    strategy = "NegE"
    model_dir_name = f"{model_name.split('/')[-1]}_active_{strategy}"
    out_dir = os.path.join(out_base, model_dir_name)
    sel_file = os.path.join(out_dir, "selected_indices_iter_0.csv")
    metrics_file = os.path.join(out_dir, "metrics_iter_0.json")
    missing = [p for p in [sel_file, metrics_file] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Smoke test failed. Missing files: {missing}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
