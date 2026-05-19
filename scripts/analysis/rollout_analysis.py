from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wrap_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def wrap_rad(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _series_or_default(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df:
        return df[name].astype(float)
    return pd.Series(np.full(len(df), default), index=df.index, dtype=float)


def load_rollout(path: Path, label: str | None = None) -> pd.DataFrame:
    """Load simulator or hardware CSVs into shared report columns.

    Supported inputs include:
    - Servo2 simulator CSVs from validate_sim/evaluate_sac.
    - Servo2 hardware CSVs from run_sac_on_qube.
    - Hardware CSVs with theta_deg/alpha_deg fields.
    """

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} has no rows")
    if "episode" in df:
        df = df[df["episode"] == df["episode"].min()].copy()

    out = pd.DataFrame()
    if {"theta", "alpha", "voltage"}.issubset(df.columns):
        out["time_s"] = _series_or_default(df, "time")
        out["theta_deg"] = np.rad2deg(df["theta"].astype(float).to_numpy())
        out["alpha_deg"] = wrap_deg(np.rad2deg(wrap_rad(df["alpha"].astype(float).to_numpy())))
        out["theta_dot_rad"] = _series_or_default(df, "theta_dot")
        out["alpha_dot_rad"] = _series_or_default(df, "alpha_dot")
        out["voltage_applied"] = df["voltage"].astype(float)
    elif {"time", "motor_angle", "alpha_upright_zero_deg", "sent_voltage"}.issubset(df.columns):
        out["time_s"] = df["time"].astype(float)
        out["theta_deg"] = df["motor_angle"].astype(float)
        out["alpha_deg"] = wrap_deg(df["alpha_upright_zero_deg"].astype(float).to_numpy())
        out["theta_dot_rad"] = _series_or_default(df, "theta_dot")
        out["alpha_dot_rad"] = _series_or_default(df, "alpha_dot")
        out["voltage_applied"] = df["sent_voltage"].astype(float)
    elif {"time_s", "theta_deg", "alpha_deg", "voltage_applied"}.issubset(df.columns):
        out["time_s"] = df["time_s"].astype(float)
        out["theta_deg"] = df["theta_deg"].astype(float)
        out["alpha_deg"] = wrap_deg(df["alpha_deg"].astype(float).to_numpy())
        out["theta_dot_rad"] = _series_or_default(df, "theta_dot_rad")
        out["alpha_dot_rad"] = _series_or_default(df, "alpha_dot_rad")
        out["voltage_applied"] = df["voltage_applied"].astype(float)
    else:
        raise ValueError(
            f"{path} does not match a known QUBE CSV schema. "
            f"Columns: {', '.join(df.columns)}"
        )

    out["source"] = label or path.stem
    out["mode"] = df["mode"].astype(str) if "mode" in df else ""
    if len(out):
        out["time_s"] = out["time_s"] - float(out["time_s"].iloc[0])
    return out


def longest_upright_seconds(df: pd.DataFrame, threshold_deg: float = 10.0) -> float:
    if len(df) < 2:
        return 0.0
    times = df["time_s"].to_numpy(dtype=float)
    mask = df["alpha_deg"].abs().to_numpy(dtype=float) < threshold_deg
    best = 0.0
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            best = max(best, times[index - 1] - times[start])
            start = None
    if start is not None:
        best = max(best, times[-1] - times[start])
    return float(best)


def summarize(df: pd.DataFrame, threshold_deg: float = 10.0) -> dict[str, float | str]:
    voltage = df["voltage_applied"].to_numpy(dtype=float)
    return {
        "source": str(df["source"].iloc[0]) if len(df) else "",
        "duration_s": float(df["time_s"].iloc[-1]) if len(df) else 0.0,
        "upright_ratio": float((df["alpha_deg"].abs() < threshold_deg).mean()) if len(df) else 0.0,
        "longest_upright_s": longest_upright_seconds(df, threshold_deg),
        "closest_alpha_deg": float(df["alpha_deg"].abs().min()) if len(df) else 0.0,
        "max_abs_theta_deg": float(df["theta_deg"].abs().max()) if len(df) else 0.0,
        "mean_abs_voltage_v": float(np.mean(np.abs(voltage))) if len(df) else 0.0,
        "rms_voltage_v": float(np.sqrt(np.mean(voltage**2))) if len(df) else 0.0,
        "final_theta_deg": float(df["theta_deg"].iloc[-1]) if len(df) else 0.0,
        "final_alpha_deg": float(df["alpha_deg"].iloc[-1]) if len(df) else 0.0,
    }


def add_summary_box(axis: plt.Axes, summary: dict[str, float | str], x: float = 0.01) -> None:
    text = (
        f"{summary['source']}\n"
        f"upright<10deg: {summary['upright_ratio']:.3f}\n"
        f"longest: {summary['longest_upright_s']:.2f} s\n"
        f"max |theta|: {summary['max_abs_theta_deg']:.1f} deg\n"
        f"RMS voltage: {summary['rms_voltage_v']:.2f} V"
    )
    axis.text(
        x,
        0.97,
        text,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.45", "alpha": 0.88},
    )


def shade_modes(axis: plt.Axes, df: pd.DataFrame, modes: set[str], color: str = "#2ca02c") -> None:
    if "mode" not in df or not df["mode"].astype(bool).any():
        return
    active = df["mode"].isin(modes).to_numpy()
    times = df["time_s"].to_numpy(dtype=float)
    start: float | None = None
    label_used = False
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = times[index]
        elif not is_active and start is not None:
            axis.axvspan(start, times[index - 1], color=color, alpha=0.10, label=None if label_used else "balance mode")
            label_used = True
            start = None
    if start is not None:
        axis.axvspan(start, times[-1], color=color, alpha=0.10, label=None if label_used else "balance mode")


def plot_rollout(df: pd.DataFrame, output: Path, title: str, threshold_deg: float = 10.0) -> dict[str, float | str]:
    summary = summarize(df, threshold_deg)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    balance_modes = {"ais_balance", "balance", "example_balance", "rl_balance"}
    for axis in axes:
        shade_modes(axis, df, balance_modes)
        axis.grid(True, alpha=0.25)

    axes[0].plot(df["time_s"], df["alpha_deg"], linewidth=1.3)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axhline(threshold_deg, color="black", linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].axhline(-threshold_deg, color="black", linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].set_ylabel("alpha [deg]")
    add_summary_box(axes[0], summary)

    axes[1].plot(df["time_s"], df["theta_deg"], linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axes[1].set_ylabel("theta [deg]")

    axes[2].plot(df["time_s"], df["voltage_applied"], linewidth=1.1)
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axes[2].set_ylabel("voltage [V]")

    axes[3].plot(df["time_s"], df["alpha_dot_rad"], linewidth=1.1)
    axes[3].axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axes[3].set_ylabel("alpha_dot [rad/s]")
    axes[3].set_xlabel("time [s]")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def plot_comparison(left: pd.DataFrame, right: pd.DataFrame, output: Path, title: str) -> tuple[dict[str, float | str], dict[str, float | str]]:
    left_summary = summarize(left)
    right_summary = summarize(right)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis in axes:
        axis.grid(True, alpha=0.25)

    axes[0].plot(left["time_s"], left["alpha_deg"], label=left_summary["source"], linewidth=1.6)
    axes[0].plot(right["time_s"], right["alpha_deg"], label=right_summary["source"], linewidth=1.2, alpha=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].axhline(10.0, color="black", linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].axhline(-10.0, color="black", linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].set_ylabel("alpha [deg]")
    axes[0].legend(loc="upper right")
    add_summary_box(axes[0], left_summary, x=0.01)
    add_summary_box(axes[0], right_summary, x=0.29)

    axes[1].plot(left["time_s"], left["theta_deg"], linewidth=1.6)
    axes[1].plot(right["time_s"], right["theta_deg"], linewidth=1.2, alpha=0.9)
    axes[1].set_ylabel("theta [deg]")

    axes[2].plot(left["time_s"], left["voltage_applied"], linewidth=1.6)
    axes[2].plot(right["time_s"], right["voltage_applied"], linewidth=1.2, alpha=0.9)
    axes[2].set_ylabel("voltage [V]")
    axes[2].set_xlabel("time [s]")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return left_summary, right_summary
