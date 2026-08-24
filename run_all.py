import argparse
import json
from pathlib import Path


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--model", default="qwen2_5_vl_3b")
    p.add_argument("--embedding-backend", default="clip_vit_b32")
    p.add_argument("--n-per-class", type=int, default=250)
    p.add_argument("--skip-inference", action="store_true")
    return p.parse_args()


def main():
    a = args()
    print("FacePercept-Bench")
    print("model:", a.model)
    print("encoder:", a.embedding_backend)
    print("images/class:", a.n_per_class)

    if a.quick:
        Path("results").mkdir(exist_ok=True)
        out = {
            "mode": "quick",
            "note": "wiring check only - use the full scripts for experiment runs",
        }
        Path("results/quick_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("quick check done")
        return

    print("full experiment scripts are under scripts/ and src/facepercept/")


if __name__ == "__main__":
    main()
