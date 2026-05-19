from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.rollout_analysis import load_rollout, plot_rollout


DEFAULT_INPUTS = [
    ("classic_validation", Path("runs/classic_validation.csv"), "Classical validation with AIS upright balance"),
    ("sac_eval", Path("runs/sac_eval.csv"), "SAC simulation rollout"),
    ("sac_upright_balance", Path("runs/sac_upright_balance_eval.csv"), "SAC upright-balance rollout"),
    ("real_sac_rollout", Path("runs/real_sac_rollout.csv"), "SAC hardware rollout"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AIS-style analytic plots for QUBE rollout CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/report_figures"))
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("NAME", "CSV", "TITLE"),
        help="Additional plot input: output-name csv-path plot-title. Can be repeated.",
    )
    args = parser.parse_args()

    plot_specs = list(DEFAULT_INPUTS)
    if args.input:
        plot_specs.extend((name, Path(csv_path), title) for name, csv_path, title in args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    generated = 0
    for name, csv_path, title in plot_specs:
        if not csv_path.exists():
            print(f"Skipping missing input: {csv_path}")
            continue
        df = load_rollout(csv_path, label=name)
        output = args.output_dir / f"{name}_timeseries.png"
        summary = plot_rollout(df, output, title)
        summary["plot"] = str(output)
        metrics.append(summary)
        generated += 1
        print(f"Wrote {output}")

    if metrics:
        metrics_path = args.output_dir / "rollout_metrics.csv"
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        print(f"Wrote {metrics_path}")
    print(f"Generated {generated} plot(s).")


if __name__ == "__main__":
    main()
