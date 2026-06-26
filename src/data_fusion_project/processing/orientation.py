# src/data_fusion_project/processing/orientation.py
"""
Roll / pitch estimation from IMU data, with selectable sensor-fusion filters.

Pipeline (per the project spec): roll and pitch are first derived from the *pre-filtered*
signals (gyro bias already removed by the calibration stage), then refined by one of the
fusion filters below.

Angle conventions (right-handed, aerospace style), with the accelerometer in g:
    roll  (phi)   = atan2( acc_y, acc_z )                       rotation about the X axis
    pitch (theta) = atan2(-acc_x, sqrt(acc_y^2 + acc_z^2))      rotation about the Y axis

Gyro rates are taken as roll-rate = gyr_x and pitch-rate = gyr_y. Yaw is intentionally
omitted: it is unobservable from accelerometer + gyroscope alone (no magnetometer).

Implemented fusion methods:
- ACCEL         : accelerometer-only angles (no drift, but noisy / sensitive to linear acc)
- GYRO          : integrated gyroscope only (smooth, but drifts over time)
- COMPLEMENTARY : classic complementary filter blending both paths
- KALMAN        : 2-state Kalman filter (angle + gyro bias), the well-known IMU formulation
"""

from __future__ import annotations

import numpy as np

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.processing.config import OrientationConfig, OrientationMethod

logger = get_logger(__name__)

_RAD2DEG = 180.0 / np.pi
_DEG2RAD = np.pi / 180.0


# ======================================================================================================================
# Angle primitives
# ======================================================================================================================
def accel_angles(acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes roll/pitch from the accelerometer alone (in radians).
    :param: acc (np.ndarray): accelerometer block, shape (T, 3) ordered X, Y, Z.
    :return: angles (tuple): (roll, pitch) arrays of shape (T,) in radians.
    """
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
    return roll, pitch


def _integrate_gyro(gyro_rate_rad: np.ndarray, dt: float, initial: float) -> np.ndarray:
    """
    Integrates a single-axis angular rate to an angle series (radians).
    :param: gyro_rate_rad (np.ndarray): angular rate per step (rad/s), shape (T,).
    :param: dt (float): sample period in seconds.
    :param: initial (float): initial angle (rad), typically the first accelerometer angle.
    :return: angle (np.ndarray): integrated angle, shape (T,) in radians.
    """
    angle = np.empty_like(gyro_rate_rad)
    angle[0] = initial
    for i in range(1, len(gyro_rate_rad)):
        angle[i] = angle[i - 1] + gyro_rate_rad[i] * dt
    return angle


# ======================================================================================================================
# Fusion filters
# ======================================================================================================================
def complementary_filter(acc_angle: np.ndarray, gyro_rate_rad: np.ndarray, dt: float, alpha: float) -> np.ndarray:
    """
    Fuses an accelerometer angle and a gyro rate with a complementary filter (radians).

    Recursion: ``angle[t] = alpha * (angle[t-1] + rate[t] * dt) + (1 - alpha) * acc_angle[t]``.
    A high ``alpha`` trusts the (smooth) gyro path short-term while the accelerometer
    anchors the long-term value to prevent drift.

    :param: acc_angle (np.ndarray): accelerometer-derived angle, shape (T,) in radians.
    :param: gyro_rate_rad (np.ndarray): gyro angular rate, shape (T,) in rad/s.
    :param: dt (float): sample period in seconds.
    :param: alpha (float): gyro weight in [0, 1].
    :return: angle (np.ndarray): fused angle, shape (T,) in radians.
    """
    angle = np.empty_like(acc_angle)
    angle[0] = acc_angle[0]
    for i in range(1, len(acc_angle)):
        predicted = angle[i - 1] + gyro_rate_rad[i] * dt
        angle[i] = alpha * predicted + (1.0 - alpha) * acc_angle[i]
    return angle


def kalman_filter(acc_angle: np.ndarray, gyro_rate_rad: np.ndarray, dt: float,
                  q_angle: float, q_bias: float, r_measure: float) -> np.ndarray:
    """
    Fuses an accelerometer angle and a gyro rate with a 2-state Kalman filter (radians).

    State ``x = [angle, gyro_bias]``. The gyro rate drives the prediction while the
    accelerometer angle is the measurement. This is the standard IMU Kalman formulation
    (Lauszus), which simultaneously tracks and removes a slowly-varying gyro bias.

    :param: acc_angle (np.ndarray): accelerometer-derived angle (measurement), shape (T,) in radians.
    :param: gyro_rate_rad (np.ndarray): gyro angular rate, shape (T,) in rad/s.
    :param: dt (float): sample period in seconds.
    :param: q_angle (float): process noise variance of the angle state.
    :param: q_bias (float): process noise variance of the gyro-bias state.
    :param: r_measure (float): measurement noise variance of the accelerometer angle.
    :return: angle (np.ndarray): fused angle, shape (T,) in radians.
    """
    n = len(acc_angle)
    out = np.empty(n, dtype=float)

    angle = float(acc_angle[0])
    bias = 0.0
    # Error covariance matrix P (2x2).
    p00, p01, p10, p11 = 0.0, 0.0, 0.0, 0.0
    out[0] = angle

    for i in range(1, n):
        rate = gyro_rate_rad[i] - bias
        angle += dt * rate

        # Predict error covariance.
        p00 += dt * (dt * p11 - p01 - p10 + q_angle)
        p01 -= dt * p11
        p10 -= dt * p11
        p11 += q_bias * dt

        # Innovation and Kalman gain.
        s = p00 + r_measure
        k0 = p00 / s
        k1 = p10 / s

        y = float(acc_angle[i]) - angle
        angle += k0 * y
        bias += k1 * y

        # Update error covariance.
        p00 -= k0 * p00
        p01 -= k0 * p01
        p10 -= k1 * p00
        p11 -= k1 * p01

        out[i] = angle

    return out


# ======================================================================================================================
# High-level orientation estimation
# ======================================================================================================================
def estimate_orientation(acc: np.ndarray, gyr: np.ndarray, fs: float, config: OrientationConfig) -> dict[str, np.ndarray]:
    """
    Estimates roll/pitch for a window using the configured fusion method.
    :param: acc (np.ndarray): accelerometer block, shape (T, 3), in g (pre-filtered/calibrated).
    :param: gyr (np.ndarray): gyroscope block, shape (T, 3), in dps (bias removed).
    :param: fs (float): sampling frequency in Hz.
    :param: config (OrientationConfig): orientation method and parameters.
    :return: angles (dict): {"roll": (T,), "pitch": (T,)} in degrees or radians per config.
    """
    dt = 1.0 / fs
    roll_acc, pitch_acc = accel_angles(acc)

    # Gyro rates: roll about X, pitch about Y; convert dps -> rad/s for integration.
    roll_rate = gyr[:, 0] * _DEG2RAD
    pitch_rate = gyr[:, 1] * _DEG2RAD

    method = config.method
    if method == OrientationMethod.ACCEL:
        roll, pitch = roll_acc, pitch_acc
    elif method == OrientationMethod.GYRO:
        roll = _integrate_gyro(roll_rate, dt, roll_acc[0])
        pitch = _integrate_gyro(pitch_rate, dt, pitch_acc[0])
    elif method == OrientationMethod.COMPLEMENTARY:
        roll = complementary_filter(roll_acc, roll_rate, dt, config.alpha)
        pitch = complementary_filter(pitch_acc, pitch_rate, dt, config.alpha)
    elif method == OrientationMethod.KALMAN:
        roll = kalman_filter(roll_acc, roll_rate, dt, config.kalman_q_angle, config.kalman_q_bias, config.kalman_r_measure)
        pitch = kalman_filter(pitch_acc, pitch_rate, dt, config.kalman_q_angle, config.kalman_q_bias, config.kalman_r_measure)
    else:
        raise ValueError(f"Unsupported orientation method: {method}")

    if config.degrees:
        roll = roll * _RAD2DEG
        pitch = pitch * _RAD2DEG

    return {"roll": roll, "pitch": pitch}
