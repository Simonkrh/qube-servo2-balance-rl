from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qube_sim.env import QubeServo2SwingUpEnv, make_reference_sim2real_env
from qube_sim.parameters import QubeServo2Parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QUBE policy recovery after external torque pulses.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--reference-profile", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--kick-interval-seconds", type=float, default=2.0)
    parser.add_argument("--kick-duration-seconds", type=float, default=0.06)
    parser.add_argument("--pendulum-torque", type=float, default=0.006)
    parser.add_argument("--arm-torque", type=float, default=0.0)
    parser.add_argument("--stable-seconds", type=float, default=0.25)
    parser.add_argument("--balance-deg", type=float, default=12.0)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> QubeServo2SwingUpEnv:
    if args.reference_profile:
        return make_reference_sim2real_env()
    return QubeServo2SwingUpEnv(
        params=QubeServo2Parameters.reference_sim2real(),
        reset_mode="uniform",
    )


def main() -> None:
    args = parse_args()
    model = SAC.load(args.model)
    env = make_env(args)
    max_steps = int(args.seconds / env.params.dt)
    warmup_steps = int(args.warmup_seconds / env.params.dt)
    kick_interval_steps = max(1, int(args.kick_interval_seconds / env.params.dt))
    kick_duration_steps = max(1, int(args.kick_duration_seconds / env.params.dt))
    stable_steps = max(1, int(args.stable_seconds / env.params.dt))
    balance_rad = np.deg2rad(args.balance_deg)

    episode_rewards: list[float] = []
    episode_balanced: list[float] = []
    episode_recovered: list[float] = []
    episode_recovery_times: list[float] = []

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(
                seed=args.seed + episode,
                options={"state": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)},
            )
            total_reward = 0.0
            balanced_steps = 0
            kicks = 0
            recovered_kicks = 0
            recovery_times: list[float] = []
            max_abs_alpha = 0.0
            active_kick_start: int | None = None
            stable_count = 0

            for step in range(1, max_steps + 1):
                if step >= warmup_steps and (step - warmup_steps) % kick_interval_steps == 0:
                    sign = -1.0 if kicks % 2 else 1.0
                    env.inject_disturbance(
                        arm_torque=sign * args.arm_torque,
                        pendulum_torque=sign * args.pendulum_torque,
                        duration_steps=kick_duration_steps,
                    )
                    kicks += 1
                    active_kick_start = step
                    stable_count = 0

                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                total_reward += reward
                balanced = abs(info["alpha"]) < balance_rad
                balanced_steps += int(balanced)
                max_abs_alpha = max(max_abs_alpha, abs(info["alpha"]))

                if active_kick_start is not None and step > active_kick_start + kick_duration_steps:
                    stable_count = stable_count + 1 if balanced else 0
                    if stable_count >= stable_steps:
                        recovered_kicks += 1
                        recovery_times.append((step - active_kick_start) * env.params.dt)
                        active_kick_start = None
                        stable_count = 0

                if terminated or truncated:
                    break

            balanced_fraction = balanced_steps / max(step, 1)
            recovered_fraction = recovered_kicks / max(kicks, 1)
            avg_recovery_time = float(np.mean(recovery_times)) if recovery_times else float("nan")
            episode_rewards.append(total_reward)
            episode_balanced.append(balanced_fraction)
            episode_recovered.append(recovered_fraction)
            if recovery_times:
                episode_recovery_times.extend(recovery_times)

            print(
                f"Episode {episode + 1}: steps={step}, reward={total_reward:.1f}, "
                f"balanced_fraction={balanced_fraction:.3f}, "
                f"max_abs_alpha_deg={np.rad2deg(max_abs_alpha):.1f}, "
                f"recovered_kicks={recovered_kicks}/{kicks}, "
                f"avg_recovery_s={avg_recovery_time:.2f}"
            )

        print(
            "Summary: "
            f"avg_reward={np.mean(episode_rewards):.1f}, "
            f"avg_balanced={np.mean(episode_balanced):.3f}, "
            f"avg_recovered={np.mean(episode_recovered):.3f}, "
            f"avg_recovery_s={np.mean(episode_recovery_times) if episode_recovery_times else float('nan'):.2f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
