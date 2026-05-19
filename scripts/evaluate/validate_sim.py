from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qube_sim.controllers import AisEnergySwingUpPD, EnergySwingUpPD
from qube_sim.env import QubeServo2SwingUpEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate QUBE simulator with classic swing-up/balance control.")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--csv", type=Path, default=Path("runs/classic_validation.csv"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--controller",
        choices=["ais", "servo"],
        default="ais",
        help="Classical controller to validate. 'ais' uses the imported stronger upright balance handoff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    env = QubeServo2SwingUpEnv(
        render_mode="human" if args.render else None,
        quantize_encoders=False,
        normalized_action=False,
    )
    controller = AisEnergySwingUpPD(env.params) if args.controller == "ais" else EnergySwingUpPD(env.params)
    obs, info = env.reset(options={"state": np.array([0.0, np.pi - 0.05, 0.0, 0.0])})
    del obs

    rows = []
    steps = int(args.seconds / env.params.dt)
    balanced_steps = 0
    for step in range(steps):
        voltage = controller(env.state)
        _, reward, terminated, truncated, info = env.step(np.array([voltage], dtype=np.float32))
        rows.append(
            {
                "time": step * env.params.dt,
                "theta": info["theta"],
                "alpha": info["alpha"],
                "theta_dot": info["theta_dot"],
                "alpha_dot": info["alpha_dot"],
                "voltage": info["voltage"],
                "reward": reward,
                "mode": getattr(controller, "last_mode", args.controller),
            }
        )
        balanced_steps += int(info["is_balanced"])
        if terminated or truncated:
            break

    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    env.close()
    print(f"Wrote {len(rows)} samples to {args.csv}")
    print(f"Balanced fraction: {balanced_steps / max(len(rows), 1):.3f}")


if __name__ == "__main__":
    main()
