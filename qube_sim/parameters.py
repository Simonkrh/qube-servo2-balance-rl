from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QubeServo2Parameters:
    """Physical and numerical parameters for the QUBE-Servo 2 pendulum.

    Most hardware constants come from the QUBE-Servo 2 user manual. The viscous
    damping defaults are the Quanser courseware values, but those are
    unit-specific and should be identified from your own logged data before
    trusting sim-to-real transfer.
    """

    gravity: float = 9.81

    # DC motor and amplifier.
    nominal_voltage: float = 18.0
    terminal_resistance: float = 8.4
    torque_constant: float = 0.042
    back_emf_constant: float = 0.042
    rotor_inertia: float = 4.0e-6
    rotor_inductance: float = 1.16e-3
    peak_current: float = 2.0
    continuous_current: float = 0.5
    recommended_voltage_limit: float = 10.0
    maximum_voltage_limit: float = 15.0
    motor_voltage_scale: float = 1.0

    # Shaft attachment hub.
    hub_mass: float = 0.0106
    hub_radius: float = 0.0111
    hub_inertia: float = 0.6e-6

    # Rotary arm and pendulum module.
    arm_mass: float = 0.095
    arm_length: float = 0.085
    pendulum_mass: float = 0.024
    pendulum_length: float = 0.129
    shaft_inertia_override: float | None = None

    # Experimentally identified in Quanser courseware; tune for your unit.
    arm_damping: float = 0.0015
    pendulum_damping: float = 0.0005
    arm_stiffness: float = 0.0

    # Numerics and sensor/actuator modelling.
    dt: float = 0.01
    integration_substeps: int = 4
    encoder_counts_per_rev: int = 2048
    velocity_filter_alpha: float = 1.0
    action_delay_steps: int = 0
    voltage_limit: float = 10.0
    coulomb_friction_arm: float = 0.0
    coulomb_friction_pendulum: float = 0.0
    stiction_velocity: float = 0.02

    # Episode shaping and safety limits.
    max_episode_steps: int = 1000
    max_arm_angle: float = 2.0 * np.pi
    max_arm_velocity: float = 40.0
    max_pendulum_velocity: float = 50.0
    reset_down_std: float = 0.08
    reset_velocity_std: float = 0.04

    @property
    def pendulum_com_length(self) -> float:
        return 0.5 * self.pendulum_length

    @property
    def pendulum_inertia_com(self) -> float:
        return (1.0 / 12.0) * self.pendulum_mass * self.pendulum_length**2

    @property
    def pendulum_inertia_pivot(self) -> float:
        return self.pendulum_inertia_com + self.pendulum_mass * self.pendulum_com_length**2

    @property
    def arm_inertia(self) -> float:
        # Published QUBE-Servo 2 pendulum papers commonly use this value
        # (about 5.7e-5 kg m^2) for the rotary arm inertia.
        return (1.0 / 12.0) * self.arm_mass * self.arm_length**2

    @property
    def shaft_inertia(self) -> float:
        if self.shaft_inertia_override is not None:
            return self.shaft_inertia_override
        return self.arm_inertia + self.hub_inertia + self.rotor_inertia

    @property
    def encoder_resolution_rad(self) -> float:
        return 2.0 * np.pi / self.encoder_counts_per_rev

    def randomized(
        self,
        rng: np.random.Generator,
        ranges: dict[str, tuple[float, float]] | None,
    ) -> "QubeServo2Parameters":
        """Return a copy with uniform multiplicative domain randomization."""

        if not ranges:
            return self

        values: dict[str, Any] = {}
        for name, (low, high) in ranges.items():
            base = getattr(self, name)
            if base is None:
                continue
            values[name] = base * rng.uniform(low, high)
        return replace(self, **values)

    @classmethod
    def reference_sim2real(cls) -> "QubeServo2Parameters":
        """Parameters adapted from jonurce/Inverted_Pendulum_RL's successful setup.

        This profile intentionally differs from the manual-only defaults: it
        includes a cable/arm centering spring, lower damping values, a 6 V action
        limit, and a ±150 degree arm travel limit.
        """

        return cls(
            terminal_resistance=8.94,
            torque_constant=0.0431,
            back_emf_constant=0.0431,
            arm_mass=0.053,
            arm_length=0.086,
            pendulum_mass=0.024,
            pendulum_length=0.128,
            shaft_inertia_override=0.0000572 + 0.00006,
            arm_damping=0.0004,
            pendulum_damping=0.000003,
            arm_stiffness=0.002,
            voltage_limit=6.0,
            recommended_voltage_limit=6.0,
            max_arm_angle=5.0 * np.pi / 6.0,
            max_episode_steps=2000,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "QubeServo2Parameters":
        with Path(path).open() as f:
            data = json.load(f)
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in data.items() if key in allowed}
        return cls(**values)

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as f:
            json.dump(self.__dict__, f, indent=2, sort_keys=True)
            f.write("\n")
