from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    df = pd.read_csv("results/final_real_centroid_results.csv")
    x = np.arange(len(df))
    e = df["SynthMinusReal"].to_numpy()
    lo = df["CI_Low"].to_numpy()
    hi = df["CI_High"].to_numpy()
    yerr = np.vstack([e - lo, hi - e])

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.errorbar(x, e, yerr=yerr, fmt="o", capsize=7, linewidth=1.8, markersize=8)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Encoder"])
    ax.set_ylabel("Median centroid-distance difference (Synthetic - Real)")
    ax.set_title("Synthetic Faces Are Closer to the Real-Face Reference Centroid")
    ax.text(0.02, 0.02,
            "Reference centroid estimated from real faces in the training split only.\n"
            "Negative values indicate greater representation-space centrality for synthetic faces.",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/final_real_centroid_robustness.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/final_real_centroid_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
