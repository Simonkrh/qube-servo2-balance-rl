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

from qube_sim.env import QubeServo2SwingUpEnv, make_reference_sim2real_env, make_reference_upright_balance_env
from qube_sim.parameters import QubeServo2Parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC policy in the QUBE simulator.")
    parser.add_argument("--model", type=Path, default=Path("models/sac_qube_reference_500k.zip"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--reference-profile", action="store_true")
    parser.add_argument(
        "--upright-balance-profile",
        action="store_true",
        help="Evaluate from near-upright starts with the balance-focused reward profile.",
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional trajectory CSV output. For multiple episodes, the episode index is included.",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> QubeServo2SwingUpEnv:
    if args.upright_balance_profile:
        env = make_reference_upright_balance_env(domain_randomization=False)
        env.render_mode = None if args.no_render else "human"
        return env

    if args.reference_profile:
        env = make_reference_sim2real_env()
        env.render_mode = None if args.no_render else "human"
        return env

    return QubeServo2SwingUpEnv(
        params=QubeServo2Parameters.reference_sim2real(),
        render_mode=None if args.no_render else "human",
        reset_mode="uniform",
    )


def main() -> None:
    args = parse_args()
    model = SAC.load(args.model)
    env = make_env(args)
    max_steps = int(args.seconds / env.params.dt)

    try:
        rows = []
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            balanced_steps = 0
            voltage_delta_sum = 0.0
            steps = 0

            for steps in range(1, max_steps + 1):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                total_reward += reward
                balanced_steps += int(info["is_balanced"])
                voltage_delta_sum += abs(float(info.get("voltage_delta", 0.0)))
                if args.csv is not None:
                    rows.append(
                        {
                            "episode": episode + 1,
                            "time": (steps - 1) * env.params.dt,
                            "theta": info["theta"],
                            "alpha": info["alpha"],
                            "theta_dot": info["theta_dot"],
                            "alpha_dot": info["alpha_dot"],
                            "voltage": info["voltage"],
                            "reward": reward,
                            "is_balanced": int(info["is_balanced"]),
                            "disturbance_active": int(info.get("disturbance_active", False)),
                        }
                    )
                if terminated or truncated:
                    break

            print(
                f"Episode {episode + 1}: "
                f"steps={steps}, reward={total_reward:.1f}, "
                f"balanced_fraction={balanced_steps / max(steps, 1):.3f}, "
                f"mean_abs_voltage_delta={voltage_delta_sum / max(steps, 1):.3f}"
            )

        if args.csv is not None and rows:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {len(rows)} trajectory samples to {args.csv}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
