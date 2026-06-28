# src/data_fusion_project/processing/config.py
"""
Configuration objects for the data processing / feature extraction pipeline.

The pipeline is split into four configurable stages, each represented by its own
dataclass and bundled into :class:`PipelineConfig`:

1. Calibration   -> :class:`CalibrationConfig`   (bias removal, gravity normalization)
2. Filtering     -> :class:`FilterConfig`        (low-/high-pass, gravity removal)
3. Orientation   -> :class:`OrientationConfig`   (roll/pitch + fusion filter)
4. Features      -> :class:`FeatureConfig`       (channel selection, scalar features)

All options have sensible defaults, so ``PipelineConfig()`` already yields a usable
configuration. Every stage can be toggled and re-parameterized independently, which
makes feature experiments cheap to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Sequence


# ======================================================================================================================
# Enumerations
# ======================================================================================================================
class FilterType(str, Enum):
    """Type of frequency-domain filter applied to a signal group."""
    NONE = "none"
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"


class OrientationMethod(str, Enum):
    """Algorithm used to estimate roll/pitch angles from the IMU signals."""
    NONE = "none"                    # do not compute orientation
    ACCEL = "accel"                  # roll/pitch from accelerometer only (noisy, no drift)
    GYRO = "gyro"                    # integrated gyroscope only (smooth, but drifts)
    COMPLEMENTARY = "complementary"  # complementary filter (acc + gyro)
    KALMAN = "kalman"                # 2-state Kalman filter (acc + gyro, estimates gyro bias)


# ======================================================================================================================
# Stage Configurations
# ======================================================================================================================
@dataclass
class CalibrationConfig:
    """
    Controls how the per-session ``calibration.csv`` is used to correct raw signals.

    :param: enabled (bool): master switch for calibration.
    :param: remove_gyro_bias (bool): subtract the gyroscope zero bias (``gyro - bias``).
    :param: normalize_acc_to_g (bool): divide accelerometer by the measured gravity magnitude.
    :param: remove_acc_bias (bool): subtract the full accelerometer bias vector. This also removes
            gravity, so it should usually stay False unless gravity removal is handled here.
    :param: anchor_samples (int): number of leading calibration samples ignored to avoid settling transients.
    """
    enabled: bool = True
    remove_gyro_bias: bool = True
    normalize_acc_to_g: bool = True
    remove_acc_bias: bool = False
    anchor_samples: int = 0


@dataclass
class FilterConfig:
    """
    Controls the digital filters applied to the (already calibrated) signals.

    Filtering happens before orientation estimation and feature assembly, so that both
    operate on de-noised data. Filters are zero-phase (``scipy.signal.sosfiltfilt``).

    :param: enabled (bool): master switch for filtering.
    :param: acc_filter (FilterType): filter applied to accelerometer channels.
    :param: acc_cutoff_hz (float | tuple): cutoff(s) for the accelerometer filter (Hz).
            A scalar for low/high-pass, a ``(low, high)`` tuple for band-pass.
    :param: gyro_filter (FilterType): filter applied to gyroscope channels.
    :param: gyro_cutoff_hz (float | tuple): cutoff(s) for the gyroscope filter (Hz).
    :param: order (int): Butterworth filter order.
    :param: remove_gravity (bool): additionally split the accelerometer into a gravity
            component (low-pass) and a linear-acceleration component (raw - gravity).
    :param: gravity_cutoff_hz (float): cutoff used to estimate the gravity component (Hz).
    :param: replace_acc_with_linear (bool): if gravity removal is on, replace the accelerometer
            channels with the gravity-free linear acceleration instead of keeping the raw values.
    """
    enabled: bool = True
    acc_filter: FilterType = FilterType.LOWPASS
    acc_cutoff_hz: float | tuple[float, float] = 8.0
    gyro_filter: FilterType = FilterType.LOWPASS
    gyro_cutoff_hz: float | tuple[float, float] = 12.0
    order: int = 2
    remove_gravity: bool = False
    gravity_cutoff_hz: float = 0.5
    replace_acc_with_linear: bool = False


@dataclass
class OrientationConfig:
    """
    Controls roll/pitch estimation and the sensor-fusion filter used to refine it.

    Roll/pitch are computed from the *pre-filtered* signals (gyro bias already removed),
    as required by the project spec, and then smoothed by the chosen fusion filter.

    :param: enabled (bool): master switch for orientation channels.
    :param: method (OrientationMethod): fusion algorithm (accel/gyro/complementary/kalman).
    :param: imus (tuple): IMUs to compute orientation for, e.g. ``("IMU1",)`` or ``("IMU1", "IMU2")``.
    :param: alpha (float): complementary-filter weight on the gyro path (0..1, typical 0.95-0.99).
    :param: degrees (bool): output angles in degrees (True) or radians (False).
    :param: kalman_q_angle (float): Kalman process noise for the angle state.
    :param: kalman_q_bias (float): Kalman process noise for the gyro-bias state.
    :param: kalman_r_measure (float): Kalman measurement noise of the accelerometer angle.
    """
    enabled: bool = True
    method: OrientationMethod = OrientationMethod.COMPLEMENTARY
    imus: Sequence[str] = ("IMU1",)
    alpha: float = 0.98
    degrees: bool = True
    kalman_q_angle: float = 0.001
    kalman_q_bias: float = 0.003
    kalman_r_measure: float = 0.03


@dataclass
class FeatureConfig:
    """
    Selects which channels end up in the time-series tensor and which scalar features
    are computed per window.

    Time-series channels form the ``(T, C)`` matrix consumed by a 1D-CNN. Scalar features
    (cross-correlation, statistics) are returned separately as a ``(F,)`` vector so they can
    feed a dense branch alongside the convolutional one.

    :param: imus (tuple): IMUs whose raw channels are included.
    :param: include_acc (bool): include accelerometer channels.
    :param: include_gyro (bool): include gyroscope channels.
    :param: include_acc_magnitude (bool): include the accelerometer magnitude channel per IMU.
    :param: include_gyro_magnitude (bool): include the gyroscope magnitude channel per IMU.
    :param: include_diff_acc (bool): include the inter-IMU acc difference (``IMU2_acc - IMU1_acc``).
    :param: include_diff_gyro (bool): include the inter-IMU gyro difference (``IMU2_gyr - IMU1_gyr``).
    :param: include_orientation (bool): include roll/pitch channels from the orientation stage.
    :param: cross_correlation (bool): compute inter-IMU cross-correlation scalar features.
    :param: statistics (bool): compute per-channel statistic scalar features (mean/std/min/max/rms).
    """
    imus: Sequence[str] = ("IMU1", "IMU2")
    include_acc: bool = True
    include_gyro: bool = True
    include_acc_magnitude: bool = False
    include_gyro_magnitude: bool = False
    include_diff_acc: bool = False
    include_diff_gyro: bool = False
    include_orientation: bool = True
    cross_correlation: bool = False
    statistics: bool = False
    
    # Real-Time Pre-Computed Features
    include_linear_jerk: bool = False
    include_angular_acceleration: bool = False
    include_relative_acceleration: bool = False
    include_relative_rotation: bool = False
    include_relative_yaw: bool = False
    include_accelerometer_magnitude: bool = False
    include_gyroscope_magnitude: bool = False
    include_gravity_free_linear_acceleration: bool = False


# ======================================================================================================================
# Top-level Pipeline Configuration
# ======================================================================================================================
@dataclass
class PipelineConfig:
    """
    Bundles all stage configurations plus global acquisition parameters.

    :param: sample_rate_hz (float): sampling rate of the recordings (100 Hz for this dataset).
    :param: window_size (int): fixed number of time steps per sample window (150 = 1.5 s).
    :param: pad_mode (str): how to coerce windows to ``window_size`` ("edge" repeats the last/first
            row, "zero" pads with zeros). Over-long windows are always truncated.
    :param: calibration (CalibrationConfig): calibration stage configuration.
    :param: filters (FilterConfig): filtering stage configuration.
    :param: orientation (OrientationConfig): orientation stage configuration.
    :param: features (FeatureConfig): feature-selection configuration.
    """
    sample_rate_hz: float = 100.0
    window_size: int = 150
    pad_mode: str = "edge"
    jitter_range: int = 0
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> dict:
        """
        Serializes the configuration to a plain (JSON/YAML-friendly) dictionary.
        :return: config (dict): nested dictionary with enum values converted to strings.
        """
        def _convert(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            return value

        return _convert(asdict(self))
