from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="FacePercept-Bench entry point")
    p.add_argument("--quick", action="store_true", help="Run a lightweight wiring check")
    p.add_argument("--model", default="qwen2_5_vl_3b")
    p.add_argument("--embedding-backend", default="clip_vit_b32")
    p.add_argument("--n-per-class", type=int, default=250)
    p.add_argument("--skip-inference", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("FacePercept-Bench")
    print(f"model={args.model} embedding={args.embedding_backend} n_per_class={args.n_per_class}")
    if args.quick:
        Path("results").mkdir(exist_ok=True)
        payload = {
            "mode": "quick",
            "note": "Lightweight repository wiring check. Use the full local pipeline for scientific runs.",
        }
        Path("results/quick_check.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Quick check complete: results/quick_check.json")
        return

    print("This public entry point documents the validated benchmark interface.")
    print("For full VLM/encoder execution, use the research modules under src/facepercept and the scripts directory.")


if __name__ == "__main__":
    main()
