from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.rollout_analysis import load_rollout, plot_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay two QUBE CSV rollouts with AIS-style summary metrics.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", type=str, default=None)
    parser.add_argument("--right-label", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/report_figures/rollout_comparison.png"))
    parser.add_argument("--title", type=str, default="QUBE rollout comparison")
    args = parser.parse_args()

    left = load_rollout(args.left, args.left_label)
    right = load_rollout(args.right, args.right_label)
    left_summary, right_summary = plot_comparison(left, right, args.output, args.title)

    print(f"Wrote comparison plot to {args.output}")
    for label, summary in (("Left", left_summary), ("Right", right_summary)):
        print(f"{label} summary ({summary['source']}):")
        for key, value in summary.items():
            if key == "source":
                continue
            print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
