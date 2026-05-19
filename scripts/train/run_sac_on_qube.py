from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import serial
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[2]
HARDWARE_DIR = ROOT / "qube_hardware"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_DIR))

from QUBE import QUBE  # noqa: E402
from qube_sim.dynamics import wrap_pi  # noqa: E402
from qube_sim.parameters import QubeServo2Parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained SAC policy on the real QUBE."
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models" / "sac_qube_servo2.zip"
    )
    parser.add_argument(
        "--calibration", type=Path, default=ROOT / "runs" / "qube_calibration.json"
    )
    parser.add_argument(
        "--reference-profile",
        action="store_true",
        help="Use the 6 V sim-to-real parameter profile instead of a calibration JSON.",
    )
    parser.add_argument("--port", default="auto")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=300.0)
    parser.add_argument(
        "--max-voltage",
        type=float,
        default=None,
        help="Clip sent motor voltage. Defaults to the selected training/profile voltage limit.",
    )
    parser.add_argument(
        "--voltage-slew-rate",
        type=float,
        default=None,
        help="Optional sent-voltage slew limit in V/s to reduce balancing chatter.",
    )
    parser.add_argument("--warmup-seconds", type=float, default=0.1)
    parser.add_argument("--velocity-filter", type=float, default=0.35)
    parser.add_argument("--max-alpha-dot", type=float, default=50.0)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs" / "real_sac_rollout.csv"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Predict actions but send 0 V to the motor.",
    )
    return parser.parse_args()


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def safe_update(qube: QUBE, attempts: int = 3, settle: float = 0.03) -> bool:
    for attempt in range(attempts):
        try:
            qube.update()
            return True
        except serial.SerialException as error:
            if attempt == attempts - 1:
                print(f"Serial frame failed after {attempts} attempts: {error}")
                return False
            try:
                qube.master.reset_input_buffer()
                qube.master.reset_output_buffer()
            except serial.SerialException:
                return False
            time.sleep(settle)
    return False


def stop_qube(qube: QUBE) -> None:
    try:
        qube.setMotorVoltage(0)
        qube.setRGB(0, 999, 0)
        safe_update(qube, attempts=1)
    except Exception as error:
        print(f"Could not send final stop command: {error}")


def make_observation(
    theta_deg: float,
    pendulum_down_zero_deg: float,
    previous_theta: float,
    previous_alpha: float,
    previous_alpha_dot: float,
    rpm: float,
    dt: float,
    last_voltage: float,
    params: QubeServo2Parameters,
    velocity_filter: float,
    max_alpha_dot: float,
) -> tuple[np.ndarray, float, float, float, float]:
    theta = np.deg2rad(theta_deg)
    down_zero = np.deg2rad(wrap_deg(pendulum_down_zero_deg))
    alpha = wrap_pi(float(down_zero + np.pi))
    theta_dot = rpm * 2.0 * np.pi / 60.0

    if dt > 0:
        alpha_delta = wrap_pi(float(alpha - previous_alpha))
        measured_alpha_dot = alpha_delta / dt
        measured_alpha_dot = float(
            np.clip(measured_alpha_dot, -max_alpha_dot, max_alpha_dot)
        )
        alpha_dot = previous_alpha_dot + velocity_filter * (
            measured_alpha_dot - previous_alpha_dot
        )
    else:
        alpha_dot = previous_alpha_dot

    obs = np.array(
        [
            np.sin(theta),
            np.cos(theta),
            np.sin(alpha),
            np.cos(alpha),
            np.clip(theta_dot / params.max_arm_velocity, -1.0, 1.0),
            np.clip(alpha_dot / params.max_pendulum_velocity, -1.0, 1.0),
            np.clip(last_voltage / params.voltage_limit, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return obs, theta, alpha, theta_dot, alpha_dot


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.reference_profile:
        params = QubeServo2Parameters.reference_sim2real()
    else:
        params = (
            QubeServo2Parameters.from_json(args.calibration)
            if args.calibration.exists()
            else QubeServo2Parameters()
        )
    max_voltage = params.voltage_limit if args.max_voltage is None else args.max_voltage
    model = SAC.load(args.model)

    qube = QUBE(args.port, args.baudrate)
    period = 1.0 / args.rate
    fields = [
        "time",
        "dt",
        "loop_hz",
        "motor_angle",
        "pendulum_angle_down_zero",
        "alpha_upright_zero_deg",
        "theta_dot",
        "alpha_dot",
        "raw_action",
        "policy_voltage",
        "sent_voltage",
        "rpm",
        "current_raw",
    ]
    loop_dts = []

    try:
        qube.setRGB(999, 0, 0)
        qube.setMotorVoltage(0)
        qube.resetMotorEncoder()
        qube.resetPendulumEncoder()
        if not safe_update(qube):
            raise RuntimeError(
                "Could not establish reliable serial communication with the QUBE."
            )
        time.sleep(0.3)

        previous_time = time.monotonic()
        previous_theta = np.deg2rad(qube.getMotorAngle())
        previous_alpha = wrap_pi(np.deg2rad(wrap_deg(qube.getPendulumAngle())) + np.pi)
        previous_alpha_dot = 0.0
        last_voltage = 0.0

        start = previous_time
        next_tick = start
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            while True:
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= args.seconds:
                    break

                dt = now - previous_time
                if elapsed < args.warmup_seconds:
                    qube.setMotorVoltage(0)
                    if not safe_update(qube):
                        break
                    previous_time = now
                    previous_theta = np.deg2rad(qube.getMotorAngle())
                    previous_alpha = wrap_pi(
                        np.deg2rad(wrap_deg(qube.getPendulumAngle())) + np.pi
                    )
                    previous_alpha_dot = 0.0
                    next_tick += period
                    sleep_time = next_tick - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                obs, theta, alpha, theta_dot, alpha_dot = make_observation(
                    qube.getMotorAngle(),
                    qube.getPendulumAngle(),
                    previous_theta,
                    previous_alpha,
                    previous_alpha_dot,
                    qube.getMotorRPM(),
                    dt,
                    last_voltage,
                    params,
                    args.velocity_filter,
                    args.max_alpha_dot,
                )
                action, _ = model.predict(obs, deterministic=True)
                raw_action = float(np.asarray(action).reshape(-1)[0])
                policy_voltage = raw_action * params.voltage_limit
                target_voltage = float(
                    np.clip(policy_voltage, -max_voltage, max_voltage)
                )
                if args.voltage_slew_rate is not None and dt > 0:
                    max_delta = max(0.0, args.voltage_slew_rate) * dt
                    target_voltage = last_voltage + float(
                        np.clip(target_voltage - last_voltage, -max_delta, max_delta)
                    )
                sent_voltage = 0.0 if args.dry_run else target_voltage

                qube.setMotorVoltage(sent_voltage)
                if not safe_update(qube):
                    break
                last_voltage = sent_voltage

                writer.writerow(
                    {
                        "time": elapsed,
                        "dt": dt,
                        "loop_hz": 1.0 / dt if dt > 0 else 0.0,
                        "motor_angle": qube.getMotorAngle(),
                        "pendulum_angle_down_zero": qube.getPendulumAngle(),
                        "alpha_upright_zero_deg": np.rad2deg(alpha),
                        "theta_dot": theta_dot,
                        "alpha_dot": alpha_dot,
                        "raw_action": raw_action,
                        "policy_voltage": policy_voltage,
                        "sent_voltage": sent_voltage,
                        "rpm": qube.getMotorRPM(),
                        "current_raw": qube.getMotorCurrent(),
                    }
                )

                previous_time = now
                loop_dts.append(dt)
                previous_theta = theta
                previous_alpha = alpha
                previous_alpha_dot = alpha_dot

                next_tick += period
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stop_qube(qube)
        if loop_dts:
            actual_rate = 1.0 / float(np.mean(loop_dts))
            jitter_ms = 1000.0 * float(np.std(loop_dts))
            max_dt_ms = 1000.0 * float(np.max(loop_dts))
            print(
                f"Loop timing: avg {actual_rate:.1f} Hz | "
                f"jitter std {jitter_ms:.3f} ms | max dt {max_dt_ms:.3f} ms"
            )
        print(f"Wrote rollout log to {args.out}")


if __name__ == "__main__":
    main()
