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
def build_channels(processed: dict, orientation: dict, config: FeatureConfig) -> tuple[np.ndarray, list[str]]:
    """
    Assembles the selected time-series channels into a (T, C) matrix.
    :param: processed (dict): per-IMU calibrated/filtered blocks, e.g. ``{"IMU1": {"acc": (T,3), "gyr": (T,3)}}``.
    :param: orientation (dict): per-IMU orientation angles, e.g. ``{"IMU1": {"roll": (T,), "pitch": (T,)}}``.
    :param: config (FeatureConfig): channel-selection options.
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
        if config.include_acc_magnitude:
            columns.append(np.linalg.norm(acc, axis=1))
            names.append(f"{imu}_acc_mag")
        if config.include_gyro_magnitude:
            columns.append(np.linalg.norm(gyr, axis=1))
            names.append(f"{imu}_gyr_mag")

    # Inter-IMU differences (finger relative to wrist): IMU2 - IMU1.
    if (config.include_diff_acc or config.include_diff_gyro) and "IMU1" in processed and "IMU2" in processed:
        if config.include_diff_acc:
            diff = processed["IMU2"]["acc"] - processed["IMU1"]["acc"]
            for j, ax in enumerate(_ACC_AXES):
                columns.append(diff[:, j])
                names.append(f"diff_acc{ax}")
        if config.include_diff_gyro:
            diff = processed["IMU2"]["gyr"] - processed["IMU1"]["gyr"]
            for j, ax in enumerate(_GYR_AXES):
                columns.append(diff[:, j])
                names.append(f"diff_gyr{ax}")
    elif config.include_diff_acc or config.include_diff_gyro:
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
