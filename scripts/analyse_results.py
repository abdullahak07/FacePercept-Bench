from pathlib import Path
import pandas as pd


def main():
    p = Path("results/final_real_centroid_results.csv")
    if not p.exists():
        raise SystemExit("Missing validated results CSV")
    df = pd.read_csv(p)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
