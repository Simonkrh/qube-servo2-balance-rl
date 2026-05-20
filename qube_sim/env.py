from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised before deps install
    raise ImportError(
        "QubeServo2SwingUpEnv requires gymnasium. Install dependencies with "
        "`python3 -m pip install -r requirements.txt`."
    ) from exc

from qube_sim.dynamics import rk4_step, wrap_pi
from qube_sim.parameters import QubeServo2Parameters

DisturbanceConfig = dict[str, Any]


REFERENCE_RECOVERY_DISTURBANCES: DisturbanceConfig = {
    "probability_per_second": 0.6,
    "warmup_steps": 150,
    "min_interval_steps": 80,
    "duration_steps": (2, 8),
    "arm_torque": (-0.012, 0.012),
    "pendulum_torque": (-0.006, 0.006),
}


def scaled_reference_recovery_disturbances(
    disturbance_scale: float = 1.0,
) -> DisturbanceConfig:
    disturbance_config = dict(REFERENCE_RECOVERY_DISTURBANCES)
    disturbance_config["arm_torque"] = tuple(
        disturbance_scale * value
        for value in REFERENCE_RECOVERY_DISTURBANCES["arm_torque"]
    )
    disturbance_config["pendulum_torque"] = tuple(
        disturbance_scale * value
        for value in REFERENCE_RECOVERY_DISTURBANCES["pendulum_torque"]
    )
    return disturbance_config


class QubeServo2SwingUpEnv(gym.Env):
    """Gymnasium environment for QUBE-Servo 2 swing-up and balance training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        params: QubeServo2Parameters | None = None,
        render_mode: str | None = None,
        domain_randomization: dict[str, tuple[float, float]] | None = None,
        quantize_encoders: bool = True,
        normalized_action: bool = True,
        reset_mode: str = "down",
        action_noise_std: float = 0.0,
        observation_noise_std: float = 0.0,
        disturbance_config: DisturbanceConfig | None = None,
        voltage_smoothness_weight: float = 0.0,
        upright_voltage_smoothness_weight: float = 0.0,
        voltage_penalty_weight: float = 0.02,
        reward_profile: str = "swingup",
        left_recovery_probability: float = 0.5,
        recovery_theta_abs_range: tuple[float, float] = (
            np.deg2rad(5.0),
            np.deg2rad(25.0),
        ),
        recovery_alpha_range: tuple[float, float] = (-np.deg2rad(8.0), np.deg2rad(8.0)),
        recovery_theta_dot_range: tuple[float, float] = (-2.0, 2.0),
        recovery_alpha_dot_range: tuple[float, float] = (-5.0, 5.0),
        recovery_reset_probability: float = 1.0,
        arm_center_deadband: float = np.deg2rad(3.0),
        arm_centering_weight: float = 0.0,
        upright_arm_centering_weight: float = 0.0,
    ) -> None:
        self.base_params = params or QubeServo2Parameters()
        self.params = self.base_params
        self.render_mode = render_mode
        self.domain_randomization = domain_randomization
        self.quantize_encoders = quantize_encoders
        self.normalized_action = normalized_action
        self.reset_mode = reset_mode
        self.action_noise_std = action_noise_std
        self.observation_noise_std = observation_noise_std
        self.disturbance_config = disturbance_config
        self.voltage_smoothness_weight = voltage_smoothness_weight
        self.upright_voltage_smoothness_weight = upright_voltage_smoothness_weight
        self.voltage_penalty_weight = voltage_penalty_weight
        self.reward_profile = reward_profile
        self.left_recovery_probability = left_recovery_probability
        self.recovery_theta_abs_range = recovery_theta_abs_range
        self.recovery_alpha_range = recovery_alpha_range
        self.recovery_theta_dot_range = recovery_theta_dot_range
        self.recovery_alpha_dot_range = recovery_alpha_dot_range
        self.recovery_reset_probability = recovery_reset_probability
        self.arm_center_deadband = arm_center_deadband
        self.arm_centering_weight = arm_centering_weight
        self.upright_arm_centering_weight = upright_arm_centering_weight

        action_bound = 1.0 if normalized_action else self.base_params.voltage_limit
        self.action_space = spaces.Box(
            low=np.array([-action_bound], dtype=np.float32),
            high=np.array([action_bound], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.state = np.zeros(4, dtype=np.float64)
        self.last_current = 0.0
        self.last_voltage = 0.0
        self.last_voltage_delta = 0.0
        self.elapsed_steps = 0
        self._action_delay: deque[float] = deque()
        self._disturbance_steps_remaining = 0
        self._disturbance_arm_torque = 0.0
        self._disturbance_pendulum_torque = 0.0
        self._last_disturbance_step = -1_000_000
        self.disturbance_events = 0
        self.screen = None
        self.clock = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        self.params = self.base_params.randomized(
            self.np_random, self.domain_randomization
        )

        if "state" in options:
            self.state = np.array(options["state"], dtype=np.float64)
        else:
            theta = self.np_random.normal(0.0, 0.02)
            use_recovery_reset = self.reset_mode == "upright_recovery" or (
                self.reset_mode == "mixed_reference_recovery"
                and self.np_random.random() < self.recovery_reset_probability
            )
            if use_recovery_reset:
                theta_abs = self.np_random.uniform(*self.recovery_theta_abs_range)
                theta_sign = (
                    -1.0
                    if self.np_random.random() < self.left_recovery_probability
                    else 1.0
                )
                theta = theta_sign * theta_abs
                alpha = self.np_random.uniform(*self.recovery_alpha_range)
                theta_dot = self.np_random.uniform(*self.recovery_theta_dot_range)
                alpha_dot = self.np_random.uniform(*self.recovery_alpha_dot_range)
            elif self.reset_mode in {"uniform", "mixed_reference_recovery"}:
                theta = self.np_random.uniform(-np.pi / 6.0, np.pi / 6.0)
                alpha = self.np_random.uniform(-np.pi, np.pi)
            elif self.reset_mode == "upright":
                alpha = self.np_random.normal(0.0, self.params.reset_down_std)
            else:
                alpha = wrap_pi(
                    np.pi + self.np_random.normal(0.0, self.params.reset_down_std)
                )
            if not use_recovery_reset:
                theta_dot = self.np_random.normal(0.0, self.params.reset_velocity_std)
                alpha_dot = self.np_random.normal(0.0, self.params.reset_velocity_std)
            self.state = np.array(
                [theta, alpha, theta_dot, alpha_dot], dtype=np.float64
            )

        self.elapsed_steps = 0
        self.last_current = 0.0
        self.last_voltage = 0.0
        self.last_voltage_delta = 0.0
        self._action_delay = deque([0.0] * self.params.action_delay_steps)
        self._disturbance_steps_remaining = 0
        self._disturbance_arm_torque = 0.0
        self._disturbance_pendulum_torque = 0.0
        self._last_disturbance_step = -1_000_000
        self.disturbance_events = 0
        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        commanded_voltage = (
            raw_action * self.params.voltage_limit
            if self.normalized_action
            else raw_action
        )
        commanded_voltage = float(
            np.clip(
                commanded_voltage, -self.params.voltage_limit, self.params.voltage_limit
            )
        )
        if self.action_noise_std > 0:
            commanded_voltage += float(
                self.np_random.normal(0.0, self.action_noise_std)
            )
        commanded_voltage = float(
            np.clip(
                commanded_voltage, -self.params.voltage_limit, self.params.voltage_limit
            )
        )

        if self.params.action_delay_steps > 0:
            self._action_delay.append(commanded_voltage)
            voltage = self._action_delay.popleft()
        else:
            voltage = commanded_voltage

        previous_voltage = self.last_voltage
        self._maybe_start_random_disturbance()
        arm_torque, pendulum_torque = self._current_disturbance()

        sub_dt = self.params.dt / self.params.integration_substeps
        current = self.last_current
        for _ in range(self.params.integration_substeps):
            self.state, current = rk4_step(
                self.state,
                voltage,
                self.params,
                sub_dt,
                arm_disturbance_torque=arm_torque,
                pendulum_disturbance_torque=pendulum_torque,
            )

        self.state[2] = float(
            np.clip(
                self.state[2],
                -3.0 * self.params.max_arm_velocity,
                3.0 * self.params.max_arm_velocity,
            )
        )
        self.state[3] = float(
            np.clip(
                self.state[3],
                -3.0 * self.params.max_pendulum_velocity,
                3.0 * self.params.max_pendulum_velocity,
            )
        )
        self.last_current = current
        self.last_voltage = voltage
        self.last_voltage_delta = voltage - previous_voltage
        self.elapsed_steps += 1
        if self._disturbance_steps_remaining > 0:
            self._disturbance_steps_remaining -= 1

        reward = self._reward(voltage, previous_voltage)
        terminated = bool(
            abs(self.state[0]) > self.params.max_arm_angle
            or abs(self.state[2]) > 2.5 * self.params.max_arm_velocity
            or abs(self.state[3]) > 2.5 * self.params.max_pendulum_velocity
        )
        truncated = self.elapsed_steps >= self.params.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def inject_disturbance(
        self,
        *,
        arm_torque: float = 0.0,
        pendulum_torque: float = 0.0,
        duration_steps: int = 1,
    ) -> None:
        """Apply an external torque pulse for the next environment steps."""

        self._disturbance_arm_torque = float(arm_torque)
        self._disturbance_pendulum_torque = float(pendulum_torque)
        self._disturbance_steps_remaining = max(0, int(duration_steps))
        self._last_disturbance_step = self.elapsed_steps
        self.disturbance_events += int(self._disturbance_steps_remaining > 0)

    def _sample_range(self, value: Any) -> float:
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(
                    f"Disturbance range must have two values, got {value!r}."
                )
            low, high = float(value[0]), float(value[1])
            return float(self.np_random.uniform(low, high))
        return float(value)

    def _maybe_start_random_disturbance(self) -> None:
        if not self.disturbance_config or self._disturbance_steps_remaining > 0:
            return

        warmup_steps = int(self.disturbance_config.get("warmup_steps", 0))
        min_interval_steps = int(self.disturbance_config.get("min_interval_steps", 0))
        if self.elapsed_steps < warmup_steps:
            return
        if self.elapsed_steps - self._last_disturbance_step < min_interval_steps:
            return

        probability_per_second = float(
            self.disturbance_config.get("probability_per_second", 0.0)
        )
        step_probability = np.clip(probability_per_second * self.params.dt, 0.0, 1.0)
        if self.np_random.random() >= step_probability:
            return

        duration_steps = round(
            self._sample_range(self.disturbance_config.get("duration_steps", 1))
        )
        self.inject_disturbance(
            arm_torque=self._sample_range(
                self.disturbance_config.get("arm_torque", 0.0)
            ),
            pendulum_torque=self._sample_range(
                self.disturbance_config.get("pendulum_torque", 0.0)
            ),
            duration_steps=max(1, int(duration_steps)),
        )

    def _current_disturbance(self) -> tuple[float, float]:
        if self._disturbance_steps_remaining <= 0:
            return 0.0, 0.0
        return self._disturbance_arm_torque, self._disturbance_pendulum_torque

    def _reward(self, voltage: float, previous_voltage: float) -> float:
        theta, alpha, theta_dot, alpha_dot = self.state
        if self.reward_profile == "upright_balance":
            return self._upright_balance_reward(voltage, previous_voltage)

        upright = np.cos(alpha)
        upright_closeness = np.exp(-10.0 * alpha**2)
        down_closeness = np.exp(-1.0 * wrap_pi(alpha - np.pi) ** 2)
        pendulum_stability = np.exp(-1.0 * alpha_dot**2)
        arm_stability = np.exp(-1.0 * theta_dot**2)

        velocity_penalty = -0.3 * np.tanh((theta_dot**2 + alpha_dot**2) / 10.0)
        position_penalty = -0.1 * np.tanh(theta**2 / 2.0)

        limit_distance = np.clip(
            0.8 - 0.2 * (self.params.max_arm_angle - abs(theta)), 0.0, 1.0
        )
        limit_penalty = -15.0 * limit_distance**3

        energy_like = (
            self.params.pendulum_mass
            * self.params.gravity
            * self.params.pendulum_com_length
            * (1.0 + upright)
            + 0.5 * self.params.pendulum_inertia_pivot * alpha_dot**2
        )

        reward = 20.0
        reward += 2.0 * upright
        reward += velocity_penalty
        reward += position_penalty
        reward += 10.0 * upright_closeness * pendulum_stability
        reward -= 10.0 * down_closeness * pendulum_stability
        reward += 5.0 * upright_closeness * arm_stability
        reward += limit_penalty
        reward += 2.0 - 0.15 * abs(energy_like)
        reward -= (
            self.voltage_penalty_weight
            * (voltage / max(self.params.voltage_limit, 1e-9)) ** 2
        )
        reward -= self._arm_centering_penalty(theta, upright_closeness)
        voltage_delta = (voltage - previous_voltage) / max(
            self.params.voltage_limit, 1e-9
        )
        reward -= self.voltage_smoothness_weight * voltage_delta**2
        reward -= (
            self.upright_voltage_smoothness_weight
            * upright_closeness
            * voltage_delta**2
        )
        return float(reward)

    def _upright_balance_reward(self, voltage: float, previous_voltage: float) -> float:
        """Balance-focused reward shaped after the smoother AIS upright policy.

        This profile is meant for policies trained from an upright initial
        condition. It strongly rewards small alpha error, quiet velocities,
        centered arm position, and smooth voltage changes.
        """

        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_error = wrap_pi(float(alpha))
        voltage_scale = max(self.params.voltage_limit, 1e-9)
        voltage_delta = (voltage - previous_voltage) / voltage_scale

        alpha_closeness = np.exp(-45.0 * alpha_error**2)
        theta_closeness = np.exp(-1.5 * theta**2)
        velocity_quiet = np.exp(-0.08 * theta_dot**2 - 0.04 * alpha_dot**2)
        inside_tight_balance = float(abs(alpha_error) < np.deg2rad(8.0))
        inside_balance = float(abs(alpha_error) < np.deg2rad(15.0))

        reward = 2.0
        reward += 8.0 * alpha_closeness
        reward += 5.0 * alpha_closeness * velocity_quiet
        reward += 2.0 * theta_closeness
        reward += 3.0 * inside_tight_balance
        reward += 1.0 * inside_balance
        reward -= 0.8 * theta**2
        reward -= 0.04 * theta_dot**2
        reward -= 0.025 * alpha_dot**2
        reward -= self.voltage_penalty_weight * (voltage / voltage_scale) ** 2
        reward -= self.voltage_smoothness_weight * voltage_delta**2
        reward -= (
            self.upright_voltage_smoothness_weight * alpha_closeness * voltage_delta**2
        )
        reward -= self._arm_centering_penalty(theta, alpha_closeness)
        return float(reward)

    def _arm_centering_penalty(self, theta: float, upright_closeness: float) -> float:
        center_error = max(0.0, abs(float(theta)) - self.arm_center_deadband)
        center_penalty = center_error**2
        return (
            self.arm_centering_weight * center_penalty
            + self.upright_arm_centering_weight * upright_closeness * center_penalty
        )

    def _get_obs(self) -> np.ndarray:
        theta, alpha, theta_dot, alpha_dot = self.state

        if self.quantize_encoders:
            res = self.params.encoder_resolution_rad
            theta = round(theta / res) * res
            alpha = round(alpha / res) * res

        if self.observation_noise_std > 0:
            theta += float(self.np_random.normal(0.0, self.observation_noise_std))
            alpha += float(self.np_random.normal(0.0, self.observation_noise_std))

        obs = np.array(
            [
                np.sin(theta),
                np.cos(theta),
                np.sin(alpha),
                np.cos(alpha),
                np.clip(theta_dot / self.params.max_arm_velocity, -1.0, 1.0),
                np.clip(alpha_dot / self.params.max_pendulum_velocity, -1.0, 1.0),
                self.last_voltage / self.params.voltage_limit,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict[str, Any]:
        theta, alpha, theta_dot, alpha_dot = self.state
        return {
            "theta": theta,
            "alpha": alpha,
            "theta_deg": np.rad2deg(theta),
            "alpha_deg": np.rad2deg(alpha),
            "theta_dot": theta_dot,
            "alpha_dot": alpha_dot,
            "voltage": self.last_voltage,
            "voltage_delta": self.last_voltage_delta,
            "current": self.last_current,
            "is_balanced": abs(alpha) < np.deg2rad(12.0),
            "disturbance_active": self._disturbance_steps_remaining > 0,
            "disturbance_events": self.disturbance_events,
            "disturbance_arm_torque": self._disturbance_arm_torque
            if self._disturbance_steps_remaining > 0
            else 0.0,
            "disturbance_pendulum_torque": (
                self._disturbance_pendulum_torque
                if self._disturbance_steps_remaining > 0
                else 0.0
            ),
        }

    def render(self):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install pygame to use render_mode='human'.") from exc

        width, height = 640, 480
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((width, height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((width, height))
        surface.fill((250, 250, 248))
        center = np.array([width // 2, height // 2 + 40], dtype=float)

        theta, alpha, *_ = self.state
        arm_len = 150
        pend_len = 170
        arm_end = center + arm_len * np.array([np.cos(theta), np.sin(theta)])
        pend_tip = arm_end + pend_len * np.array([np.sin(alpha), -np.cos(alpha)])

        pygame.draw.circle(surface, (35, 43, 58), center.astype(int), 8)
        pygame.draw.line(surface, (37, 99, 235), center, arm_end, 8)
        pygame.draw.circle(surface, (37, 99, 235), arm_end.astype(int), 7)
        pygame.draw.line(surface, (230, 90, 60), arm_end, pend_tip, 6)
        pygame.draw.circle(surface, (230, 90, 60), pend_tip.astype(int), 9)
        font = pygame.font.Font(None, 24)
        surface.blit(font.render("rotary arm", True, (37, 99, 235)), (18, 18))
        surface.blit(font.render("pendulum", True, (230, 90, 60)), (18, 44))

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None


def make_default_randomized_env() -> QubeServo2SwingUpEnv:
    ranges = {
        "arm_damping": (0.5, 1.8),
        "pendulum_damping": (0.5, 1.8),
        "motor_voltage_scale": (0.8, 1.2),
        "terminal_resistance": (0.95, 1.05),
        "torque_constant": (0.95, 1.05),
        "back_emf_constant": (0.95, 1.05),
        "pendulum_mass": (0.97, 1.03),
        "pendulum_length": (0.99, 1.01),
    }
    params = replace(QubeServo2Parameters(), action_delay_steps=1)
    return QubeServo2SwingUpEnv(
        params=params,
        domain_randomization=ranges,
        normalized_action=True,
        reset_mode="down",
        action_noise_std=0.02,
        observation_noise_std=0.001,
    )


def make_reference_sim2real_env() -> QubeServo2SwingUpEnv:
    ranges = {
        "arm_damping": (0.7, 1.3),
        "pendulum_damping": (0.5, 1.8),
        "arm_stiffness": (0.7, 1.3),
        "terminal_resistance": (0.9, 1.1),
        "torque_constant": (0.9, 1.1),
        "back_emf_constant": (0.9, 1.1),
        "shaft_inertia_override": (0.8, 1.2),
        "pendulum_mass": (0.9, 1.1),
        "pendulum_length": (0.95, 1.05),
        "dt": (0.8, 1.2),
    }
    return QubeServo2SwingUpEnv(
        params=replace(QubeServo2Parameters.reference_sim2real(), action_delay_steps=1),
        domain_randomization=ranges,
        normalized_action=True,
        reset_mode="uniform",
        action_noise_std=0.03,
        observation_noise_std=0.0015,
    )


def make_reference_recovery_env(disturbance_scale: float = 1.0) -> QubeServo2SwingUpEnv:
    env = make_reference_sim2real_env()
    env.disturbance_config = scaled_reference_recovery_disturbances(disturbance_scale)
    return env


def make_reference_upright_balance_env(
    disturbance_scale: float = 0.0,
    domain_randomization: bool = True,
    voltage_penalty_weight: float = 0.10,
    voltage_smoothness_weight: float = 2.0,
    upright_voltage_smoothness_weight: float = 4.0,
) -> QubeServo2SwingUpEnv:
    ranges = {
        "arm_damping": (0.8, 1.2),
        "pendulum_damping": (0.7, 1.5),
        "arm_stiffness": (0.8, 1.2),
        "terminal_resistance": (0.95, 1.05),
        "torque_constant": (0.95, 1.05),
        "back_emf_constant": (0.95, 1.05),
        "pendulum_mass": (0.95, 1.05),
        "pendulum_length": (0.98, 1.02),
    }
    return QubeServo2SwingUpEnv(
        params=replace(QubeServo2Parameters.reference_sim2real(), action_delay_steps=1),
        domain_randomization=ranges if domain_randomization else None,
        normalized_action=True,
        reset_mode="upright",
        action_noise_std=0.015,
        observation_noise_std=0.001,
        disturbance_config=(
            scaled_reference_recovery_disturbances(disturbance_scale)
            if disturbance_scale > 0.0
            else None
        ),
        reward_profile="upright_balance",
        voltage_penalty_weight=voltage_penalty_weight,
        voltage_smoothness_weight=voltage_smoothness_weight,
        upright_voltage_smoothness_weight=upright_voltage_smoothness_weight,
    )


def make_left_recovery_balance_env(
    disturbance_scale: float = 1.0,
    domain_randomization: bool = True,
    left_recovery_probability: float = 0.7,
    recovery_reset_probability: float = 0.5,
) -> QubeServo2SwingUpEnv:
    ranges = {
        "arm_damping": (0.8, 1.25),
        "pendulum_damping": (0.7, 1.5),
        "arm_stiffness": (0.8, 1.2),
        "terminal_resistance": (0.95, 1.05),
        "torque_constant": (0.95, 1.05),
        "back_emf_constant": (0.95, 1.05),
        "positive_motor_voltage_scale": (0.85, 1.15),
        "negative_motor_voltage_scale": (0.75, 1.15),
        "motor_voltage_deadband": (0.0, 4.0),
        "pendulum_mass": (0.95, 1.05),
        "pendulum_length": (0.98, 1.02),
    }
    return QubeServo2SwingUpEnv(
        params=replace(
            QubeServo2Parameters.reference_sim2real(),
            action_delay_steps=1,
            motor_voltage_deadband=0.05,
        ),
        domain_randomization=ranges if domain_randomization else None,
        normalized_action=True,
        reset_mode="mixed_reference_recovery",
        action_noise_std=0.02,
        observation_noise_std=0.0015,
        disturbance_config=(
            scaled_reference_recovery_disturbances(disturbance_scale)
            if disturbance_scale > 0.0
            else None
        ),
        reward_profile="upright_balance",
        left_recovery_probability=left_recovery_probability,
        recovery_reset_probability=recovery_reset_probability,
        voltage_smoothness_weight=0.08,
        upright_voltage_smoothness_weight=0.20,
        arm_center_deadband=np.deg2rad(3.0),
        arm_centering_weight=10.0,
        upright_arm_centering_weight=80.0,
    )


def make_left_recovery_balance_env(
    disturbance_scale: float = 1.0,
    domain_randomization: bool = True,
    left_recovery_probability: float = 0.7,
    recovery_reset_probability: float = 0.5,
) -> QubeServo2SwingUpEnv:
    ranges = {
        "arm_damping": (0.8, 1.25),
        "pendulum_damping": (0.7, 1.5),
        "arm_stiffness": (0.8, 1.2),
        "terminal_resistance": (0.95, 1.05),
        "torque_constant": (0.95, 1.05),
        "back_emf_constant": (0.95, 1.05),
        "positive_motor_voltage_scale": (0.85, 1.15),
        "negative_motor_voltage_scale": (0.75, 1.15),
        "motor_voltage_deadband": (0.0, 4.0),
        "pendulum_mass": (0.95, 1.05),
        "pendulum_length": (0.98, 1.02),
    }
    return QubeServo2SwingUpEnv(
        params=replace(
            QubeServo2Parameters.reference_sim2real(),
            action_delay_steps=1,
            motor_voltage_deadband=0.05,
        ),
        domain_randomization=ranges if domain_randomization else None,
        normalized_action=True,
        reset_mode="mixed_reference_recovery",
        action_noise_std=0.02,
        observation_noise_std=0.0015,
        disturbance_config=(
            scaled_reference_recovery_disturbances(disturbance_scale)
            if disturbance_scale > 0.0
            else None
        ),
        reward_profile="upright_balance",
        left_recovery_probability=left_recovery_probability,
        recovery_reset_probability=recovery_reset_probability,
        voltage_smoothness_weight=0.08,
        upright_voltage_smoothness_weight=0.20,
        arm_center_deadband=np.deg2rad(3.0),
        arm_centering_weight=10.0,
        upright_arm_centering_weight=80.0,
    )
