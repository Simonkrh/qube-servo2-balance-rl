from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qube_sim.dynamics import rk4_step
from qube_sim.parameters import QubeServo2Parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a first-pass QUBE simulator calibration from logs.")
    parser.add_argument("--log", type=Path, default=ROOT / "runs" / "qube_excitation.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "qube_calibration.json")
    return parser.parse_args()


def load_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    if len(rows) < 20:
        raise ValueError(f"{path} does not contain enough samples")

    time = np.array([row["time"] for row in rows], dtype=np.float64)
    voltage = np.array([row["effective_voltage"] for row in rows], dtype=np.float64)
    omega = np.array([row["rpm"] for row in rows], dtype=np.float64) * 2.0 * np.pi / 60.0
    return time, voltage, omega


def simulate_omega(
    time: np.ndarray,
    voltage: np.ndarray,
    params: QubeServo2Parameters,
) -> np.ndarray:
    state = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float64)
    omega = []
    for index in range(len(time)):
        omega.append(state[2])
        if index == len(time) - 1:
            break

        dt = time[index + 1] - time[index]
        substeps = max(1, int(np.ceil(dt / 0.0125)))
        for _ in range(substeps):
            state, _ = rk4_step(state, voltage[index], params, dt / substeps)
    return np.array(omega)


def fit(time: np.ndarray, voltage: np.ndarray, omega: np.ndarray) -> tuple[QubeServo2Parameters, float]:
    best: tuple[float, QubeServo2Parameters] | None = None
    base = QubeServo2Parameters(voltage_limit=9.60960960960961)

    damping_values = np.linspace(0.0010, 0.0035, 11)
    scale_values = np.linspace(1.0, 2.5, 13)
    coulomb_values = [0.0, 0.00005, 0.0001, 0.0002]
    mask = time > 0.2

    for arm_damping in damping_values:
        for motor_voltage_scale in scale_values:
            for coulomb in coulomb_values:
                params = replace(
                    base,
                    arm_damping=float(arm_damping),
                    motor_voltage_scale=float(motor_voltage_scale),
                    coulomb_friction_arm=float(coulomb),
                )
                simulated = simulate_omega(time, voltage, params)
                rmse = float(np.sqrt(np.mean((simulated[mask] - omega[mask]) ** 2)))
                if best is None or rmse < best[0]:
                    best = (rmse, params)

    assert best is not None
    return best[1], best[0]


def main() -> None:
    args = parse_args()
    time, voltage, omega = load_log(args.log)
    params, rmse = fit(time, voltage, omega)
    params.to_json(args.out)
    print(f"Wrote calibration to {args.out}")
    print(f"Omega RMSE: {rmse:.3f} rad/s")
    print(f"arm_damping: {params.arm_damping:.6g}")
    print(f"coulomb_friction_arm: {params.coulomb_friction_arm:.6g}")
    print(f"motor_voltage_scale: {params.motor_voltage_scale:.3f}")
    print(f"voltage_limit: {params.voltage_limit:.3f}")


if __name__ == "__main__":
    main()
