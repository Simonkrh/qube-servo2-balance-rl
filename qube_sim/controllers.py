from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qube_sim.dynamics import wrap_pi
from qube_sim.parameters import QubeServo2Parameters


@dataclass
class EnergySwingUpPD:
    """Energy swing-up plus PD balance controller for simulator validation."""

    params: QubeServo2Parameters
    swing_gain: float = 36.0
    swing_acceleration_limit: float = 6.0
    swing_direction: float = -1.0
    balance_angle: float = np.deg2rad(20.0)
    handoff_angle: float = np.deg2rad(65.0)
    kp_theta: float = -2.0
    kd_theta: float = -2.0
    kp_alpha: float = -30.0
    kd_alpha: float = -2.5

    def __call__(self, state: np.ndarray) -> float:
        theta, alpha, theta_dot, alpha_dot = state
        alpha = wrap_pi(float(alpha))

        if abs(alpha) <= self.balance_angle:
            voltage = (
                self.kp_theta * theta
                + self.kp_alpha * alpha
                + self.kd_theta * theta_dot
                + self.kd_alpha * alpha_dot
            )
            return float(np.clip(voltage, -self.params.voltage_limit, self.params.voltage_limit))

        beta = wrap_pi(alpha - np.pi)
        mp = self.params.pendulum_mass
        lp = self.params.pendulum_com_length
        jp = self.params.pendulum_inertia_pivot
        desired_energy = 2.0 * mp * self.params.gravity * lp
        energy = 0.5 * jp * alpha_dot**2 + mp * self.params.gravity * lp * (1.0 - np.cos(beta))

        switching = np.sign(alpha_dot * np.cos(beta))
        if switching == 0:
            switching = np.sign(beta) or 1.0
        acceleration = self.swing_gain * (energy - desired_energy) * switching
        acceleration = np.clip(acceleration, -self.swing_acceleration_limit, self.swing_acceleration_limit)

        handoff_scale = np.clip(abs(alpha) / self.handoff_angle, 0.15, 1.0)
        voltage_per_acceleration = (
            self.params.terminal_resistance
            * self.params.arm_mass
            * self.params.arm_length
            / self.params.torque_constant
        )
        voltage = self.swing_direction * voltage_per_acceleration * acceleration * handoff_scale
        return float(np.clip(voltage, -self.params.voltage_limit, self.params.voltage_limit))


@dataclass
class AisUprightBalancePD:
    """Upright PD balance law adapted from AIS4002_RLpendulum.

    The AIS controller was tuned for the practical report's upright capture
    phase. It is kept separate from the original Servo2 controller so existing
    workflows can still select the previous baseline when needed.
    """

    params: QubeServo2Parameters
    theta_gain: float = 1.93
    alpha_gain: float = 45.40
    theta_dot_gain: float = 1.40
    alpha_dot_gain: float = 3.08
    voltage_limit: float | None = None

    def __call__(self, state: np.ndarray) -> float:
        theta, alpha, theta_dot, alpha_dot = state
        feedback_state = np.array(
            [theta, wrap_pi(float(alpha)), theta_dot, alpha_dot],
            dtype=np.float64,
        )
        gains = np.array(
            [
                self.theta_gain,
                self.alpha_gain,
                self.theta_dot_gain,
                self.alpha_dot_gain,
            ],
            dtype=np.float64,
        )
        limit = self.params.voltage_limit if self.voltage_limit is None else self.voltage_limit
        voltage = -float(np.dot(gains, feedback_state))
        return float(np.clip(voltage, -limit, limit))


@dataclass
class AisEnergySwingUpPD:
    """AIS-style swing-up with the stronger upright balance handoff.

    This mirrors the simple AIS energy-pumping structure while using the
    Servo2 simulator's parameter object and angle convention: alpha = 0 is
    upright, alpha = +/-pi is hanging down.
    """

    params: QubeServo2Parameters
    swing_gain: float = 3.0
    upright_threshold: float = np.deg2rad(30.0)
    theta_gain: float = 1.93
    alpha_gain: float = 45.40
    theta_dot_gain: float = 1.40
    alpha_dot_gain: float = 3.08
    arm_centering_gain: float = 0.2
    arm_centering_rate_gain: float = 0.05

    def __post_init__(self) -> None:
        self.balance = AisUprightBalancePD(
            params=self.params,
            theta_gain=self.theta_gain,
            alpha_gain=self.alpha_gain,
            theta_dot_gain=self.theta_dot_gain,
            alpha_dot_gain=self.alpha_dot_gain,
        )
        self.last_mode = "swingup"

    def __call__(self, state: np.ndarray) -> float:
        theta, alpha, theta_dot, alpha_dot = state
        alpha_error = wrap_pi(float(alpha))

        if abs(alpha_error) < self.upright_threshold:
            self.last_mode = "ais_balance"
            return self.balance(state)

        self.last_mode = "ais_swingup"
        pump_direction = np.sign(alpha_dot * np.cos(alpha_error) + 1e-6)
        voltage = self.swing_gain * self.params.voltage_limit * pump_direction
        voltage -= self.arm_centering_gain * theta + self.arm_centering_rate_gain * theta_dot
        return float(np.clip(voltage, -self.params.voltage_limit, self.params.voltage_limit))
