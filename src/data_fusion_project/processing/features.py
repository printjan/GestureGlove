# src/data_fusion_project/processing/features.py
"""
Feature assembly for the gesture dataset.

Two kinds of features are produced from a processed window:

1. Time-series channels -> a ``(T, C)`` matrix consumed by the convolutional part of a CNN.
   Channels are selected via :class:`FeatureConfig` and include raw acc/gyro per IMU,
   inter-IMU differences (``IMU2 - IMU1``), magnitudes, and roll/pitch from the
   orientation stage.

2. Scalar (per-window) features -> a ``(F,)`` vector for a dense branch. These cannot live
   in the time-series tensor because they summarize a whole window:
   - cross-correlation between corresponding wrist/finger axes
   - per-channel statistics (mean / std / min / max / rms)

The channel/feature *order* is deterministic and exposed via name lists so downstream code
can interpret the arrays unambiguously.
"""

from __future__ import annotations

import numpy as np

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.processing.config import FeatureConfig

logger = get_logger(__name__)

_ACC_AXES = ("X", "Y", "Z")
_GYR_AXES = ("X", "Y", "Z")


# ======================================================================================================================
# Time-series channel assembly
# ======================================================================================================================
def build_channels(processed: dict, orientation: dict, config: FeatureConfig, fs: float = 100.0, orientation_degrees: bool = True) -> tuple[np.ndarray, list[str]]:
    """
    Assembles the selected time-series channels into a (T, C) matrix.
    :param: processed (dict): per-IMU calibrated/filtered blocks, e.g. ``{"IMU1": {"acc": (T,3), "gyr": (T,3)}}``.
    :param: orientation (dict): per-IMU orientation angles, e.g. ``{"IMU1": {"roll": (T,), "pitch": (T,)}}``.
    :param: config (FeatureConfig): channel-selection options.
    :param: fs (float): global sampling rate in Hz.
    :param: orientation_degrees (bool): whether orientation angles are in degrees.
    :return: result (tuple): (channels, names) where channels is (T, C) float and names has length C.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []

    for imu in config.imus:
        if imu not in processed:
            continue
        acc = processed[imu]["acc"]
        gyr = processed[imu]["gyr"]

        if config.include_acc:
            for j, ax in enumerate(_ACC_AXES):
                columns.append(acc[:, j])
                names.append(f"{imu}_acc{ax}")
        if config.include_gyro:
            for j, ax in enumerate(_GYR_AXES):
                columns.append(gyr[:, j])
                names.append(f"{imu}_gyr{ax}")
        
        # Accelerometer Magnitudes
        if config.include_accelerometer_magnitude or config.include_acc_magnitude:
            acc_mag = np.linalg.norm(acc, axis=1)
            from data_fusion_project.processing.filters import apply_filter
            from data_fusion_project.processing.config import FilterType
            # Lowpass filter the squared/rectified magnitude to create a smooth physical envelope
            acc_mag = apply_filter(acc_mag[:, np.newaxis], FilterType.LOWPASS, cutoff_hz=8.0, fs=fs, order=2)[:, 0]
            columns.append(acc_mag)
            name = f"{imu}_accelerometer_magnitude" if config.include_accelerometer_magnitude else f"{imu}_acc_mag"
            names.append(name)
            
        # Gyroscope Magnitudes
        if config.include_gyroscope_magnitude or config.include_gyro_magnitude:
            gyr_mag = np.linalg.norm(gyr, axis=1)
            from data_fusion_project.processing.filters import apply_filter
            from data_fusion_project.processing.config import FilterType
            gyr_mag = apply_filter(gyr_mag[:, np.newaxis], FilterType.LOWPASS, cutoff_hz=8.0, fs=fs, order=2)[:, 0]
            columns.append(gyr_mag)
            name = f"{imu}_gyroscope_magnitude" if config.include_gyroscope_magnitude else f"{imu}_gyr_mag"
            names.append(name)

        # Linear Jerk
        if config.include_linear_jerk:
            dt = 1.0 / fs
            jerk = np.diff(acc, axis=0) / dt
            jerk = np.vstack([jerk[0:1], jerk])
            from data_fusion_project.processing.filters import apply_filter
            from data_fusion_project.processing.config import FilterType
            jerk = apply_filter(jerk, FilterType.LOWPASS, cutoff_hz=8.0, fs=fs, order=2)
            for j, ax in enumerate(_ACC_AXES):
                columns.append(jerk[:, j])
                names.append(f"{imu}_linear_jerk{ax}")

        # Angular Acceleration
        if config.include_angular_acceleration:
            dt = 1.0 / fs
            alpha = np.diff(gyr, axis=0) / dt
            alpha = np.vstack([alpha[0:1], alpha])
            for j, ax in enumerate(_GYR_AXES):
                columns.append(alpha[:, j])
                names.append(f"{imu}_angular_acceleration{ax}")

        # Short-Term Integrated Relative Yaw
        if config.include_relative_yaw:
            dt = 1.0 / fs
            gyr_z = gyr[:, 2]
            from data_fusion_project.processing.filters import apply_filter
            from data_fusion_project.processing.config import FilterType
            # Apply a high-pass filter at 0.5 Hz to remove DC offsets prior to integration and prevent linear drift
            gyr_z_hp = apply_filter(gyr_z[:, np.newaxis], FilterType.HIGHPASS, cutoff_hz=0.5, fs=fs, order=2)[:, 0]
            rel_yaw = np.cumsum(gyr_z_hp) * dt
            if not orientation_degrees:
                rel_yaw = rel_yaw * (np.pi / 180.0)
            columns.append(rel_yaw)
            names.append(f"{imu}_relative_yaw")

        # Gravity-Free Linear Acceleration
        if config.include_gravity_free_linear_acceleration:
            if imu in orientation:
                roll_val = orientation[imu]["roll"]
                pitch_val = orientation[imu]["pitch"]
                if orientation_degrees:
                    roll_rad = roll_val * (np.pi / 180.0)
                    pitch_rad = pitch_val * (np.pi / 180.0)
                else:
                    roll_rad = roll_val
                    pitch_rad = pitch_val
                g_x = -np.sin(pitch_rad)
                g_y = np.cos(pitch_rad) * np.sin(roll_rad)
                g_z = np.cos(pitch_rad) * np.cos(roll_rad)
                lin_acc = acc - np.column_stack([g_x, g_y, g_z])
                for j, ax in enumerate(_ACC_AXES):
                    columns.append(lin_acc[:, j])
                    names.append(f"{imu}_gravity_free_linear_acceleration{ax}")
            else:
                logger.warning("Gravity-free linear acceleration requested for %s but no orientation estimate is available; skipping.", imu)

    # Inter-IMU differences / relative features (finger relative to wrist): IMU2 - IMU1.
    if "IMU1" in processed and "IMU2" in processed:
        if config.include_relative_acceleration or config.include_diff_acc:
            diff = processed["IMU2"]["acc"] - processed["IMU1"]["acc"]
            name_prefix = "relative_acceleration" if config.include_relative_acceleration else "diff_acc"
            for j, ax in enumerate(_ACC_AXES):
                columns.append(diff[:, j])
                names.append(f"{name_prefix}{ax}")
        if config.include_relative_rotation or config.include_diff_gyro:
            diff = processed["IMU2"]["gyr"] - processed["IMU1"]["gyr"]
            name_prefix = "relative_rotation" if config.include_relative_rotation else "diff_gyr"
            for j, ax in enumerate(_GYR_AXES):
                columns.append(diff[:, j])
                names.append(f"{name_prefix}{ax}")
    elif config.include_diff_acc or config.include_diff_gyro or config.include_relative_acceleration or config.include_relative_rotation:
        logger.debug("Inter-IMU difference requested but both IMUs are not available; skipping.")

    # Orientation channels (roll/pitch per IMU). Iterate the computed orientations directly so
    # every estimated angle is included, even for IMUs not among the raw-channel IMUs.
    if config.include_orientation:
        for imu, angles in orientation.items():
            columns.append(angles["roll"])
            names.append(f"{imu}_roll")
            columns.append(angles["pitch"])
            names.append(f"{imu}_pitch")

    if not columns:
        raise ValueError("Feature configuration produced zero time-series channels.")

    channels = np.column_stack(columns).astype(np.float32)
    return channels, names


# ======================================================================================================================
# Scalar (per-window) features
# ======================================================================================================================
def _zero_lag_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Pearson correlation coefficient between two signals.
    :param: a (np.ndarray): first signal, shape (T,).
    :param: b (np.ndarray): second signal, shape (T,).
    :return: corr (float): zero-lag normalized correlation in [-1, 1] (0 if undefined).
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _max_cross_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """
    Computes the peak of the normalized cross-correlation and its lag.
    :param: a (np.ndarray): first signal, shape (T,).
    :param: b (np.ndarray): second signal, shape (T,).
    :return: result (tuple): (peak_correlation in [-1, 1], lag in samples; positive => b lags a).
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-12:
        return 0.0, 0
    corr = np.correlate(a, b, mode="full") / denom
    idx = int(np.argmax(np.abs(corr)))
    lag = idx - (len(a) - 1)
    return float(corr[idx]), lag


def cross_correlation_features(processed: dict) -> tuple[np.ndarray, list[str]]:
    """
    Computes inter-IMU cross-correlation features between corresponding wrist/finger axes.
    :param: processed (dict): per-IMU calibrated/filtered blocks (needs both "IMU1" and "IMU2").
    :return: result (tuple): (values (F,), names) — zero-lag corr, peak corr and lag per axis.
    """
    values: list[float] = []
    names: list[str] = []

    if "IMU1" not in processed or "IMU2" not in processed:
        return np.zeros(0, dtype=np.float32), names

    axis_specs = [("acc", j, ax) for j, ax in enumerate(_ACC_AXES)] + \
                 [("gyr", j, ax) for j, ax in enumerate(_GYR_AXES)]

    for sensor, j, ax in axis_specs:
        a = processed["IMU1"][sensor][:, j]
        b = processed["IMU2"][sensor][:, j]
        zero_lag = _zero_lag_correlation(a, b)
        peak, lag = _max_cross_correlation(a, b)
        values.extend([zero_lag, peak, float(lag)])
        names.extend([f"xcorr_{sensor}{ax}_zero", f"xcorr_{sensor}{ax}_peak", f"xcorr_{sensor}{ax}_lag"])

    return np.asarray(values, dtype=np.float32), names


def statistic_features(channels: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Computes per-channel summary statistics over a window.
    :param: channels (np.ndarray): time-series channel matrix, shape (T, C).
    :param: channel_names (list): names of the C channels.
    :return: result (tuple): (values (5*C,), names) — mean/std/min/max/rms per channel.
    """
    values: list[float] = []
    names: list[str] = []
    for c, name in enumerate(channel_names):
        col = channels[:, c]
        values.extend([
            float(col.mean()),
            float(col.std()),
            float(col.min()),
            float(col.max()),
            float(np.sqrt(np.mean(col * col))),
        ])
        names.extend([f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max", f"{name}_rms"])
    return np.asarray(values, dtype=np.float32), names


def build_scalar_features(processed: dict, channels: np.ndarray, channel_names: list[str],
                          config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """
    Assembles all enabled scalar features into a single (F,) vector.
    :param: processed (dict): per-IMU calibrated/filtered blocks.
    :param: channels (np.ndarray): assembled time-series channels, shape (T, C).
    :param: channel_names (list): names of the channels.
    :param: config (FeatureConfig): feature-selection options.
    :return: result (tuple): (values (F,), names); both empty when no scalar features are enabled.
    """
    parts: list[np.ndarray] = []
    names: list[str] = []

    if config.cross_correlation:
        vals, nm = cross_correlation_features(processed)
        parts.append(vals)
        names.extend(nm)

    if config.statistics:
        vals, nm = statistic_features(channels, channel_names)
        parts.append(vals)
        names.extend(nm)

    if not parts:
        return np.zeros(0, dtype=np.float32), names

    return np.concatenate(parts).astype(np.float32), names
