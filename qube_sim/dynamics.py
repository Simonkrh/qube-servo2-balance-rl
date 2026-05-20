from __future__ import annotations

import numpy as np

from qube_sim.parameters import QubeServo2Parameters


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def smooth_sign(value: float, width: float) -> float:
    if width <= 0:
        return float(np.sign(value))
    return float(np.tanh(value / width))


def motor_torque(voltage: float, theta_dot: float, params: QubeServo2Parameters) -> tuple[float, float]:
    voltage = float(np.clip(voltage, -params.voltage_limit, params.voltage_limit))
    if voltage > params.motor_voltage_deadband:
        effective_voltage = voltage - params.motor_voltage_deadband
        directional_scale = params.positive_motor_voltage_scale
    elif voltage < -params.motor_voltage_deadband:
        effective_voltage = voltage + params.motor_voltage_deadband
        directional_scale = params.negative_motor_voltage_scale
    else:
        effective_voltage = 0.0
        directional_scale = 1.0

    motor_voltage = params.motor_voltage_scale * directional_scale * effective_voltage
    current = (motor_voltage - params.back_emf_constant * theta_dot) / params.terminal_resistance
    current = float(np.clip(current, -params.peak_current, params.peak_current))
    return params.torque_constant * current, current


def derivatives(
    state: np.ndarray,
    voltage: float,
    params: QubeServo2Parameters,
    arm_disturbance_torque: float = 0.0,
    pendulum_disturbance_torque: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Continuous Furuta pendulum dynamics.

    State is [theta, alpha, theta_dot, alpha_dot], where alpha = 0 is upright
    and alpha = +/-pi is hanging down. The pendulum hinge axis is aligned with
    the rotary arm, matching the QUBE rotary pendulum geometry.
    """

    theta, alpha, theta_dot, alpha_dot = state

    mp = params.pendulum_mass
    lr = params.arm_length
    lp = params.pendulum_com_length
    jp = params.pendulum_inertia_com
    jr = params.shaft_inertia
    g = params.gravity

    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)

    m11 = jr + mp * lr**2 + mp * lp**2 * sin_a**2
    m12 = mp * lr * lp * cos_a
    m22 = jp + mp * lp**2

    tau, current = motor_torque(voltage, theta_dot, params)

    arm_friction = params.arm_damping * theta_dot
    pend_friction = params.pendulum_damping * alpha_dot
    arm_friction += params.coulomb_friction_arm * smooth_sign(theta_dot, params.stiction_velocity)
    pend_friction += params.coulomb_friction_pendulum * smooth_sign(alpha_dot, params.stiction_velocity)

    rhs1 = (
        tau
        + arm_disturbance_torque
        - arm_friction
        - params.arm_stiffness * theta
        - 2.0 * mp * lp**2 * sin_a * cos_a * theta_dot * alpha_dot
        + mp * lr * lp * sin_a * alpha_dot**2
    )
    rhs2 = (
        -pend_friction
        + pendulum_disturbance_torque
        + mp * lp**2 * sin_a * cos_a * theta_dot**2
        + mp * g * lp * sin_a
    )

    mass_matrix = np.array([[m11, m12], [m12, m22]], dtype=np.float64)
    theta_ddot, alpha_ddot = np.linalg.solve(mass_matrix, np.array([rhs1, rhs2]))
    return np.array([theta_dot, alpha_dot, theta_ddot, alpha_ddot], dtype=np.float64), current


def rk4_step(
    state: np.ndarray,
    voltage: float,
    params: QubeServo2Parameters,
    dt: float,
    arm_disturbance_torque: float = 0.0,
    pendulum_disturbance_torque: float = 0.0,
) -> tuple[np.ndarray, float]:
    k1, current = derivatives(state, voltage, params, arm_disturbance_torque, pendulum_disturbance_torque)
    k2, _ = derivatives(
        state + 0.5 * dt * k1,
        voltage,
        params,
        arm_disturbance_torque,
        pendulum_disturbance_torque,
    )
    k3, _ = derivatives(
        state + 0.5 * dt * k2,
        voltage,
        params,
        arm_disturbance_torque,
        pendulum_disturbance_torque,
    )
    k4, _ = derivatives(state + dt * k3, voltage, params, arm_disturbance_torque, pendulum_disturbance_torque)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[1] = wrap_pi(float(next_state[1]))
    return next_state, current
