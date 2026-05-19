from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.rollout_analysis import load_rollout, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a QUBE simulator or hardware CSV log.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--upright-deg", type=float, default=10.0)
    args = parser.parse_args()

    df = load_rollout(args.csv_path)
    metrics = summarize(df, args.upright_deg)
    modes = Counter(mode for mode in df["mode"] if mode)

    print(f"log: {args.csv_path}")
    print(f"rows: {len(df)}")
    if modes:
        print(f"modes: {dict(modes)}")
    for key, value in metrics.items():
        if key == "source":
            continue
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
