# src/data_fusion_project/processing/calibration.py
"""
Calibration estimation and application for the dual-IMU dataset.

Each recording session stores a ``calibration.csv`` (~5 s of stillness). During this
window the true angular velocity is zero and the only acceleration is gravity, which lets
us estimate, per IMU:

- gyroscope zero bias        (mean gyro over the still window)
- accelerometer bias vector  (mean acc minus the expected gravity vector)
- gravity magnitude ``g``     (norm of the mean acc, ~1.0 in g-units)
- gravity direction           (unit vector of the mean acc)

The estimated profile is then used to normalize raw windows as specified in the README:
``gyro_corrected = gyro_raw - gyro_bias`` and ``acc_norm = acc_raw / g``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.processing.config import CalibrationConfig

logger = get_logger(__name__)

# Sensor axis layout per IMU within a recording.
_ACC_AXES = ("accX", "accY", "accZ")
_GYR_AXES = ("gyrX", "gyrY", "gyrZ")


# ======================================================================================================================
# Calibration Profile
# ======================================================================================================================
@dataclass
class ImuCalibration:
    """
    Calibration constants for a single IMU.

    :param: gyro_bias (np.ndarray): gyroscope zero bias per axis (dps), shape (3,).
    :param: acc_bias (np.ndarray): accelerometer bias per axis (g), shape (3,).
    :param: gravity (float): estimated gravity magnitude in g-units (~1.0).
    :param: gravity_direction (np.ndarray): unit vector of measured gravity, shape (3,).
    """
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity: float = 1.0
    gravity_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


@dataclass
class CalibrationProfile:
    """
    Calibration constants for all IMUs of a session.

    :param: per_imu (dict): mapping of IMU name (e.g. "IMU1") to its :class:`ImuCalibration`.
    :param: source (str | None): path of the calibration file the profile was estimated from.
    """
    per_imu: dict[str, ImuCalibration] = field(default_factory=dict)
    source: str | None = None

    def get(self, imu: str) -> ImuCalibration:
        """
        Returns the calibration for an IMU, falling back to an identity calibration.
        :param: imu (str): IMU name, e.g. "IMU1".
        :return: calibration (ImuCalibration): stored or identity calibration.
        """
        return self.per_imu.get(imu, ImuCalibration())


def identity_profile(imus=("IMU1", "IMU2")) -> CalibrationProfile:
    """
    Builds a neutral profile that leaves signals unchanged (used when calibration is missing).
    :param: imus (tuple): IMU names to include.
    :return: profile (CalibrationProfile): identity calibration profile.
    """
    return CalibrationProfile(per_imu={imu: ImuCalibration() for imu in imus})


# ======================================================================================================================
# Estimation
# ======================================================================================================================
def _detect_imus(columns) -> list[str]:
    """
    Detects IMU prefixes (e.g. "IMU1", "IMU2") present in a dataframe's columns.
    :param: columns (Iterable[str]): column names.
    :return: imus (list): sorted list of detected IMU prefixes.
    """
    prefixes = set()
    for col in columns:
        if "_" in col:
            prefix, _, _ = col.partition("_")
            if prefix.upper().startswith("IMU"):
                prefixes.add(prefix)
    return sorted(prefixes)


def estimate_imu_calibration(cal_df: pd.DataFrame, imu: str, config: CalibrationConfig) -> ImuCalibration:
    """
    Estimates calibration constants for one IMU from a stillness recording.
    :param: cal_df (pd.DataFrame): calibration samples containing ``<imu>_acc*`` / ``<imu>_gyr*`` columns.
    :param: imu (str): IMU prefix, e.g. "IMU1".
    :param: config (CalibrationConfig): calibration options (anchor sample count, etc.).
    :return: calibration (ImuCalibration): estimated constants for the IMU.
    """
    df = cal_df.iloc[config.anchor_samples:] if config.anchor_samples > 0 else cal_df

    acc = df[[f"{imu}_{ax}" for ax in _ACC_AXES]].to_numpy(dtype=float)
    gyr = df[[f"{imu}_{ax}" for ax in _GYR_AXES]].to_numpy(dtype=float)

    gyro_bias = gyr.mean(axis=0)

    acc_mean = acc.mean(axis=0)
    gravity = float(np.linalg.norm(acc_mean))
    if gravity < 1e-6:
        logger.warning("[%s] Near-zero gravity magnitude in calibration; defaulting to 1.0 g.", imu)
        gravity = 1.0
        gravity_dir = np.array([0.0, 0.0, 1.0])
    else:
        gravity_dir = acc_mean / gravity

    # Accelerometer bias = deviation of the measured stillness vector from ideal gravity.
    acc_bias = acc_mean - gravity * gravity_dir  # ~0 by construction; kept for the remove_acc_bias path

    logger.debug("[%s] gyro_bias=%s, gravity=%.4f g, dir=%s", imu, np.round(gyro_bias, 3), gravity, np.round(gravity_dir, 3))
    return ImuCalibration(gyro_bias=gyro_bias, acc_bias=acc_bias, gravity=gravity, gravity_direction=gravity_dir)


def estimate_calibration(cal_source, config: CalibrationConfig, imus=None) -> CalibrationProfile:
    """
    Estimates a full calibration profile from a calibration CSV file or dataframe.
    :param: cal_source (str | Path | pd.DataFrame): path to ``calibration.csv`` or a loaded dataframe.
    :param: config (CalibrationConfig): calibration options.
    :param: imus (list | None): IMU prefixes to estimate; auto-detected from columns when None.
    :return: profile (CalibrationProfile): estimated calibration profile.
    """
    if isinstance(cal_source, pd.DataFrame):
        cal_df = cal_source
        source = None
    else:
        source = str(cal_source)
        cal_df = pd.read_csv(cal_source)

    if imus is None:
        imus = _detect_imus(cal_df.columns)

    per_imu = {imu: estimate_imu_calibration(cal_df, imu, config) for imu in imus}
    return CalibrationProfile(per_imu=per_imu, source=source)


# ======================================================================================================================
# Application
# ======================================================================================================================
def apply_calibration(acc: np.ndarray, gyr: np.ndarray, calib: ImuCalibration, config: CalibrationConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies a calibration to a window's accelerometer and gyroscope blocks.
    :param: acc (np.ndarray): accelerometer block, shape (T, 3), in g.
    :param: gyr (np.ndarray): gyroscope block, shape (T, 3), in dps.
    :param: calib (ImuCalibration): calibration constants for this IMU.
    :param: config (CalibrationConfig): calibration options controlling which corrections apply.
    :return: corrected (tuple): (acc_corrected, gyro_corrected) blocks of shape (T, 3).
    """
    acc_out = np.asarray(acc, dtype=float).copy()
    gyr_out = np.asarray(gyr, dtype=float).copy()

    if not config.enabled:
        return acc_out, gyr_out

    if config.remove_gyro_bias:
        gyr_out = gyr_out - calib.gyro_bias

    if config.remove_acc_bias:
        acc_out = acc_out - calib.acc_bias

    if config.normalize_acc_to_g and calib.gravity > 1e-6:
        acc_out = acc_out / calib.gravity

    return acc_out, gyr_out
