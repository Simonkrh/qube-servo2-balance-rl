from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qube_sim.env import QubeServo2SwingUpEnv  # noqa: E402
from qube_sim.parameters import QubeServo2Parameters  # noqa: E402


def parse_degrees(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAC from paired +theta/-theta initial states."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "swingup_smooth" / "sac_2m_ultra_smooth_50k.zip",
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--theta-deg", default="2,5,10,15")
    parser.add_argument("--alpha-deg", default="-6,-3,0,3,6")
    parser.add_argument("--theta-dot", type=float, default=0.0)
    parser.add_argument("--alpha-dot", type=float, default=0.0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "runs" / "mirrored_theta_rollouts.csv",
        help="Trajectory CSV output.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "runs" / "mirrored_theta_summary.csv",
        help="Per-rollout summary CSV output.",
    )
    return parser.parse_args()


def make_env() -> QubeServo2SwingUpEnv:
    return QubeServo2SwingUpEnv(
        params=QubeServo2Parameters.reference_sim2real(),
        render_mode=None,
        reset_mode="uniform",
    )


def rollout(
    env: QubeServo2SwingUpEnv,
    model: SAC,
    *,
    theta_deg: float,
    alpha_deg: float,
    theta_dot: float,
    alpha_dot: float,
    seconds: float,
    deterministic: bool,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    initial_state = np.array(
        [
            np.deg2rad(theta_deg),
            np.deg2rad(alpha_deg),
            theta_dot,
            alpha_dot,
        ],
        dtype=np.float64,
    )
    obs, _ = env.reset(options={"state": initial_state})
    max_steps = int(seconds / env.params.dt)

    rows: list[dict[str, float]] = []
    total_reward = 0.0
    balanced_steps = 0
    terminated_at = seconds

    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        time_s = (step - 1) * env.params.dt
        total_reward += float(reward)
        balanced_steps += int(info["is_balanced"])
        rows.append(
            {
                "time": time_s,
                "initial_theta_deg": theta_deg,
                "initial_alpha_deg": alpha_deg,
                "theta_deg": float(info["theta_deg"]),
                "alpha_deg": float(info["alpha_deg"]),
                "theta_dot": float(info["theta_dot"]),
                "alpha_dot": float(info["alpha_dot"]),
                "voltage": float(info["voltage"]),
                "reward": float(reward),
                "is_balanced": float(info["is_balanced"]),
            }
        )
        if terminated or truncated:
            terminated_at = time_s
            break

    theta_values = np.array([row["theta_deg"] for row in rows], dtype=np.float64)
    alpha_values = np.array([row["alpha_deg"] for row in rows], dtype=np.float64)
    voltage_values = np.array([row["voltage"] for row in rows], dtype=np.float64)
    summary = {
        "initial_theta_deg": theta_deg,
        "initial_alpha_deg": alpha_deg,
        "steps": float(len(rows)),
        "duration_s": float(terminated_at),
        "total_reward": total_reward,
        "balanced_fraction": balanced_steps / max(len(rows), 1),
        "mean_abs_alpha_deg": float(np.mean(np.abs(alpha_values))) if len(rows) else 0.0,
        "max_abs_alpha_deg": float(np.max(np.abs(alpha_values))) if len(rows) else 0.0,
        "mean_theta_deg": float(np.mean(theta_values)) if len(rows) else 0.0,
        "max_abs_theta_deg": float(np.max(np.abs(theta_values))) if len(rows) else 0.0,
        "mean_voltage": float(np.mean(voltage_values)) if len(rows) else 0.0,
        "mean_abs_voltage": float(np.mean(np.abs(voltage_values))) if len(rows) else 0.0,
    }
    return summary, rows


def paired_delta_rows(summary_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    by_key = {
        (abs(row["initial_theta_deg"]), row["initial_alpha_deg"], np.sign(row["initial_theta_deg"])): row
        for row in summary_rows
    }
    deltas: list[dict[str, float]] = []
    for theta_abs in sorted({abs(row["initial_theta_deg"]) for row in summary_rows}):
        for alpha_deg in sorted({row["initial_alpha_deg"] for row in summary_rows}):
            negative = by_key.get((theta_abs, alpha_deg, -1.0))
            positive = by_key.get((theta_abs, alpha_deg, 1.0))
            if negative is None or positive is None:
                continue
            deltas.append(
                {
                    "theta_abs_deg": theta_abs,
                    "alpha_deg": alpha_deg,
                    "negative_minus_positive_reward": negative["total_reward"] - positive["total_reward"],
                    "negative_minus_positive_balanced_fraction": (
                        negative["balanced_fraction"] - positive["balanced_fraction"]
                    ),
                    "negative_minus_positive_mean_abs_alpha_deg": (
                        negative["mean_abs_alpha_deg"] - positive["mean_abs_alpha_deg"]
                    ),
                    "negative_minus_positive_mean_theta_deg": negative["mean_theta_deg"] - positive["mean_theta_deg"],
                    "negative_minus_positive_max_abs_theta_deg": (
                        negative["max_abs_theta_deg"] - positive["max_abs_theta_deg"]
                    ),
                    "negative_mean_theta_deg": negative["mean_theta_deg"],
                    "positive_mean_theta_deg": positive["mean_theta_deg"],
                    "negative_mean_voltage": negative["mean_voltage"],
                    "positive_mean_voltage": positive["mean_voltage"],
                    "negative_minus_positive_duration_s": negative["duration_s"] - positive["duration_s"],
                }
            )
    return deltas


def print_pair_report(deltas: list[dict[str, float]]) -> None:
    print("\nPair deltas: negative-theta rollout minus positive-theta rollout")
    print(
        "theta_abs  alpha  d_reward  d_balanced  d_abs_alpha  "
        "neg_mean_theta  pos_mean_theta  d_duration"
    )
    for row in deltas:
        print(
            f"{row['theta_abs_deg']:8.1f} "
            f"{row['alpha_deg']:6.1f} "
            f"{row['negative_minus_positive_reward']:9.1f} "
            f"{row['negative_minus_positive_balanced_fraction']:10.3f} "
            f"{row['negative_minus_positive_mean_abs_alpha_deg']:11.3f} "
            f"{row['negative_mean_theta_deg']:15.3f} "
            f"{row['positive_mean_theta_deg']:15.3f} "
            f"{row['negative_minus_positive_duration_s']:10.3f}"
        )

    if not deltas:
        return
    mean_reward_delta = float(np.mean([row["negative_minus_positive_reward"] for row in deltas]))
    mean_balance_delta = float(np.mean([row["negative_minus_positive_balanced_fraction"] for row in deltas]))
    mean_alpha_delta = float(np.mean([row["negative_minus_positive_mean_abs_alpha_deg"] for row in deltas]))
    mean_negative_theta = float(np.mean([row["negative_mean_theta_deg"] for row in deltas]))
    mean_positive_theta = float(np.mean([row["positive_mean_theta_deg"] for row in deltas]))
    mean_negative_voltage = float(np.mean([row["negative_mean_voltage"] for row in deltas]))
    mean_positive_voltage = float(np.mean([row["positive_mean_voltage"] for row in deltas]))
    print("\nAverages over all pairs")
    print(f"negative - positive reward: {mean_reward_delta:.1f}")
    print(f"negative - positive balanced fraction: {mean_balance_delta:.3f}")
    print(f"negative - positive mean |alpha| deg: {mean_alpha_delta:.3f}")
    print(f"mean theta after negative starts: {mean_negative_theta:.3f} deg")
    print(f"mean theta after positive starts: {mean_positive_theta:.3f} deg")
    print(f"mean voltage after negative starts: {mean_negative_voltage:.3f} V")
    print(f"mean voltage after positive starts: {mean_positive_voltage:.3f} V")


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    theta_values = parse_degrees(args.theta_deg)
    alpha_values = parse_degrees(args.alpha_deg)
    model = SAC.load(args.model)
    env = make_env()

    summary_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, float]] = []
    try:
        for theta_abs in theta_values:
            for alpha_deg in alpha_values:
                for sign in (-1.0, 1.0):
                    theta_deg = sign * abs(theta_abs)
                    summary, rows = rollout(
                        env,
                        model,
                        theta_deg=theta_deg,
                        alpha_deg=alpha_deg,
                        theta_dot=args.theta_dot,
                        alpha_dot=args.alpha_dot,
                        seconds=args.seconds,
                        deterministic=args.deterministic,
                    )
                    summary_rows.append(summary)
                    trajectory_rows.extend(rows)
    finally:
        env.close()

    deltas = paired_delta_rows(summary_rows)
    print_pair_report(deltas)
    write_csv(args.csv, trajectory_rows)
    write_csv(args.summary_csv, summary_rows)
    print(f"\nWrote trajectory CSV to {args.csv}")
    print(f"Wrote summary CSV to {args.summary_csv}")


if __name__ == "__main__":
    main()
