from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from qube_sim.env import (
    QubeServo2SwingUpEnv,
    make_default_randomized_env,
    make_reference_upright_balance_env,
    make_reference_recovery_env,
    make_reference_sim2real_env,
    scaled_reference_recovery_disturbances,
)
from qube_sim.parameters import QubeServo2Parameters


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


class EtaCallback(BaseCallback):
    def __init__(self, additional_timesteps: int, print_interval: int = 10_000) -> None:
        super().__init__()
        self.additional_timesteps = additional_timesteps
        self.print_interval = print_interval
        self.start_wall_time = 0.0
        self.start_timesteps = 0
        self.next_print = 0

    def _on_training_start(self) -> None:
        self.start_wall_time = time.monotonic()
        self.start_timesteps = self.model.num_timesteps
        self.next_print = self.start_timesteps + self.print_interval

    def _on_step(self) -> bool:
        current = self.model.num_timesteps
        if current < self.next_print and current < self.start_timesteps + self.additional_timesteps:
            return True

        completed = max(0, current - self.start_timesteps)
        elapsed = max(1e-9, time.monotonic() - self.start_wall_time)
        fps = completed / elapsed
        remaining = max(0, self.additional_timesteps - completed)
        eta = remaining / fps if fps > 0 else 0.0
        percent = 100.0 * min(completed, self.additional_timesteps) / max(1, self.additional_timesteps)
        print(
            f"[progress] {percent:5.1f}% | "
            f"{completed:,}/{self.additional_timesteps:,} steps | "
            f"{fps:,.0f} fps | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
        )
        self.next_print = current + self.print_interval
        return True


class ResumeReplayWarmupCallback(BaseCallback):
    def __init__(self, warmup_steps: int) -> None:
        super().__init__()
        self.warmup_steps = max(0, warmup_steps)
        self.start_timesteps = 0
        self.original_gradient_steps: int | None = None
        self.restored = False

    def _on_training_start(self) -> None:
        self.start_timesteps = self.model.num_timesteps
        self.original_gradient_steps = self.model.gradient_steps
        if self.warmup_steps > 0:
            self.model.gradient_steps = 0
            print(
                "Loaded model without replay buffer; collecting "
                f"{self.warmup_steps:,} policy steps before critic updates."
            )

    def _on_step(self) -> bool:
        if self.restored or self.original_gradient_steps is None:
            return True

        if self.model.num_timesteps - self.start_timesteps >= self.warmup_steps:
            self.model.gradient_steps = self.original_gradient_steps
            self.restored = True
            if self.warmup_steps > 0:
                print("Replay warmup complete; SAC gradient updates resumed.")
        return True

    def _on_training_end(self) -> None:
        if self.original_gradient_steps is not None:
            self.model.gradient_steps = self.original_gradient_steps


def replay_buffer_path_for_model(model_path: Path) -> Path:
    return replay_buffer_paths_for_model(model_path)[0]


def replay_buffer_paths_for_model(model_path: Path) -> list[Path]:
    suffix = "".join(model_path.suffixes)
    if suffix == ".zip":
        stem_path = model_path.with_suffix("")
    else:
        stem_path = model_path

    paths = [stem_path.with_name(f"{stem_path.name}_replay_buffer.pkl")]
    checkpoint_match = re.match(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps$", stem_path.name)
    if checkpoint_match:
        prefix = checkpoint_match.group("prefix")
        steps = checkpoint_match.group("steps")
        paths.append(stem_path.with_name(f"{prefix}_replay_buffer_{steps}_steps.pkl"))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on the QUBE-Servo 2 swing-up simulator.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/sac_qube"))
    parser.add_argument("--model-out", type=Path, default=Path("models/sac_qube_servo2"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument(
        "--reference-profile",
        action="store_true",
        help="Use the sim-to-real profile adapted from jonurce/Inverted_Pendulum_RL.",
    )
    parser.add_argument(
        "--recovery-disturbances",
        action="store_true",
        help="Train with random external torque pulses for better push recovery.",
    )
    parser.add_argument(
        "--disturbance-scale",
        type=float,
        default=1.0,
        help="Scale recovery disturbance torque strengths when --recovery-disturbances is set.",
    )
    parser.add_argument(
        "--smooth-balance",
        action="store_true",
        help="Penalize rapid voltage changes near upright to reduce balancing chatter.",
    )
    parser.add_argument(
        "--upright-balance-profile",
        action="store_true",
        help="Train a direct SAC policy from upright with balance-focused reward and smooth voltage penalties.",
    )
    parser.add_argument(
        "--voltage-smoothness-weight",
        type=float,
        default=0.0,
        help="Global voltage-change penalty weight used with --smooth-balance.",
    )
    parser.add_argument(
        "--upright-voltage-smoothness-weight",
        type=float,
        default=0.2,
        help="Near-upright voltage-change penalty weight used with --smooth-balance.",
    )
    parser.add_argument(
        "--balance-voltage-weight",
        type=float,
        default=0.10,
        help="Voltage magnitude penalty used by --upright-balance-profile.",
    )
    parser.add_argument(
        "--balance-voltage-smoothness-weight",
        type=float,
        default=2.0,
        help="Voltage-change penalty used by --upright-balance-profile.",
    )
    parser.add_argument(
        "--balance-upright-voltage-smoothness-weight",
        type=float,
        default=4.0,
        help="Extra near-upright voltage-change penalty used by --upright-balance-profile.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a saved SAC .zip checkpoint. When set, --timesteps means additional steps.",
    )
    parser.add_argument(
        "--replay-buffer",
        type=Path,
        default=None,
        help="Optional replay buffer .pkl to load when resuming.",
    )
    parser.add_argument(
        "--resume-warmup-steps",
        type=int,
        default=25_000,
        help="When resuming without a replay buffer, collect this many policy steps before gradient updates.",
    )
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    def apply_smoothness(env: QubeServo2SwingUpEnv) -> QubeServo2SwingUpEnv:
        if args.smooth_balance:
            env.voltage_smoothness_weight = args.voltage_smoothness_weight
            env.upright_voltage_smoothness_weight = args.upright_voltage_smoothness_weight
        return env

    def make_env():
        if args.upright_balance_profile:
            env = make_reference_upright_balance_env(
                disturbance_scale=args.disturbance_scale if args.recovery_disturbances else 0.0,
                domain_randomization=not args.no_domain_randomization,
                voltage_penalty_weight=args.balance_voltage_weight,
                voltage_smoothness_weight=args.balance_voltage_smoothness_weight,
                upright_voltage_smoothness_weight=args.balance_upright_voltage_smoothness_weight,
            )
            return Monitor(env)

        if args.reference_profile:
            if args.no_domain_randomization:
                env = QubeServo2SwingUpEnv(
                    params=QubeServo2Parameters.reference_sim2real(),
                    reset_mode="uniform",
                    disturbance_config=(
                        scaled_reference_recovery_disturbances(args.disturbance_scale)
                        if args.recovery_disturbances
                        else None
                    ),
                )
            elif args.recovery_disturbances:
                env = make_reference_recovery_env(args.disturbance_scale)
            else:
                env = make_reference_sim2real_env()
            return Monitor(apply_smoothness(env))

        params = QubeServo2Parameters.from_json(args.calibration) if args.calibration else None
        if args.no_domain_randomization:
            env = QubeServo2SwingUpEnv(params=params)
        elif params:
            env = QubeServo2SwingUpEnv(
                params=params,
                domain_randomization={
                    "arm_damping": (0.7, 1.4),
                    "pendulum_damping": (0.5, 1.8),
                    "motor_voltage_scale": (0.85, 1.15),
                    "terminal_resistance": (0.95, 1.05),
                    "torque_constant": (0.95, 1.05),
                    "back_emf_constant": (0.95, 1.05),
                },
                action_noise_std=0.02,
                observation_noise_std=0.001,
            )
        else:
            env = make_default_randomized_env()
        return Monitor(apply_smoothness(env))

    if args.check_env:
        check_env(QubeServo2SwingUpEnv(), warn=True)

    env = DummyVecEnv([make_env])
    checkpoint = CheckpointCallback(
        save_freq=25_000,
        save_path=str(args.log_dir / "checkpoints"),
        name_prefix="sac_qube",
        save_replay_buffer=True,
    )
    callback_list: list[BaseCallback] = [checkpoint, EtaCallback(args.timesteps, args.progress_interval)]

    if args.resume_from:
        model = SAC.load(
            args.resume_from,
            env=env,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(args.log_dir),
        )
        reset_num_timesteps = False
        replay_buffer_candidates = [args.replay_buffer] if args.replay_buffer else replay_buffer_paths_for_model(args.resume_from)
        replay_buffer_path = next((path for path in replay_buffer_candidates if path.exists()), replay_buffer_candidates[0])
        if replay_buffer_path.exists():
            model.load_replay_buffer(replay_buffer_path)
            print(f"Loaded replay buffer from {replay_buffer_path}.")
        else:
            callback_list.append(ResumeReplayWarmupCallback(args.resume_warmup_steps))
            searched = ", ".join(str(path) for path in replay_buffer_candidates)
            print(f"No replay buffer found. Searched: {searched}.")
        print(f"Resuming from {args.resume_from}; training for {args.timesteps} additional steps.")
    else:
        reference_like = args.reference_profile or args.upright_balance_profile
        buffer_size = 400_000 if reference_like else 300_000
        batch_size = 1024 if reference_like else 256
        tau = 0.005 if reference_like else 0.02
        learning_starts = 2_000 if args.upright_balance_profile else 10_000
        gamma = 0.995 if args.upright_balance_profile else 0.99
        model = SAC(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(args.log_dir),
            learning_rate=3e-4,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs={"net_arch": [256, 256]},
        )
        reset_num_timesteps = True

    callbacks = CallbackList(callback_list)
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(args.model_out)
    model.save_replay_buffer(replay_buffer_path_for_model(args.model_out))
    env.close()
    print(f"Saved model to {args.model_out}.zip")
    print(f"Saved replay buffer to {replay_buffer_path_for_model(args.model_out)}")


if __name__ == "__main__":
    main()
