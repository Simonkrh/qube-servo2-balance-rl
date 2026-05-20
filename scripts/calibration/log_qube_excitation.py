from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parents[2]
HARDWARE_DIR = ROOT / "qube_hardware"
if str(HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_DIR))

from QUBE import MAX_MOTOR_SPEED_COMMAND, QUBE  # noqa: E402


MOTOR_NEUTRAL_COMMAND = 999


def motor_command_from_voltage(voltage: float) -> int:
    raw = int((voltage / 24.0) * MOTOR_NEUTRAL_COMMAND)
    return max(-MAX_MOTOR_SPEED_COMMAND, min(MAX_MOTOR_SPEED_COMMAND, raw))


def effective_voltage_from_command(command: int) -> float:
    return 24.0 * command / MOTOR_NEUTRAL_COMMAND


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log QUBE-Servo 2 excitation data for simulator calibration."
    )
    parser.add_argument("--port", default="auto")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "qube_excitation.csv")
    parser.add_argument("--mode", choices=["sine", "step"], default="sine")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--amplitude", type=float, default=2.0)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--step-hold", type=float, default=2.0)
    parser.add_argument(
        "--max-angle-deg",
        type=float,
        default=90.0,
        help=(
            "Stop if the motor angle moves farther than this from its starting angle. "
            "Use 0 to disable this software limit."
        ),
    )
    parser.add_argument(
        "--step-values",
        default="-2,-1,0,1,2,0",
        help="Comma-separated requested voltages used in step mode.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not reset encoders before logging.",
    )
    return parser.parse_args()


def voltage_at(t: float, args: argparse.Namespace, step_values: list[float]) -> float:
    if args.mode == "sine":
        return args.amplitude * math.sin(2.0 * math.pi * args.frequency * t)

    index = int(t // args.step_hold) % len(step_values)
    return step_values[index]


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


def main() -> None:
    args = parse_args()
    step_values = [float(item.strip()) for item in args.step_values.split(",") if item.strip()]
    if not step_values:
        raise ValueError("--step-values must contain at least one voltage")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    qube = QUBE(args.port, args.baudrate)
    period = 1.0 / args.rate
    fields = [
        "time",
        "motor_angle",
        "pendulum_angle",
        "rpm",
        "requested_voltage",
        "motor_command",
        "effective_voltage",
        "current_raw",
    ]

    try:
        qube.setRGB(999, 999, 0)
        qube.setMotorVoltage(0)
        if not args.no_reset:
            qube.resetMotorEncoder()
            qube.resetPendulumEncoder()
        if not safe_update(qube):
            raise RuntimeError("Could not establish reliable serial communication with the QUBE.")
        time.sleep(0.5)
        if not safe_update(qube):
            raise RuntimeError("Could not read QUBE state after encoder reset.")

        start = time.monotonic()
        next_tick = start
        start_angle = qube.getMotorAngle()
        print(f"Starting motor angle reference: {start_angle:.1f} deg")
        bad_frames = 0
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            while True:
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= args.seconds:
                    break

                requested_voltage = voltage_at(elapsed, args, step_values)
                command = motor_command_from_voltage(requested_voltage)
                effective_voltage = effective_voltage_from_command(command)

                qube.setMotorVoltage(requested_voltage)
                if not safe_update(qube):
                    bad_frames += 1
                    if bad_frames >= 5:
                        print("Too many consecutive serial failures; stopping the run.")
                        break
                    next_tick += period
                    time.sleep(0.1)
                    continue
                bad_frames = 0
                angle_delta = qube.getMotorAngle() - start_angle
                if args.max_angle_deg > 0 and abs(angle_delta) >= args.max_angle_deg:
                    print(
                        "Motor angle safety limit reached "
                        f"({angle_delta:.1f} deg from start); stopping the run."
                    )
                    break
                writer.writerow(
                    {
                        "time": elapsed,
                        "motor_angle": qube.getMotorAngle(),
                        "pendulum_angle": qube.getPendulumAngle(),
                        "rpm": qube.getMotorRPM(),
                        "requested_voltage": requested_voltage,
                        "motor_command": command,
                        "effective_voltage": effective_voltage,
                        "current_raw": qube.getMotorCurrent(),
                    }
                )

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
        print(f"Wrote log to {args.out}")


if __name__ == "__main__":
    main()
