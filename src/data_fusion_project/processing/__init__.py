# src/data_fusion_project/processing/__init__.py
"""
Data processing & feature extraction pipeline for IMU gesture recognition.

This package is the interface between the recorded CSV dataset and a CNN: it reads the
``data/`` tree, calibrates and filters the signals, derives roll/pitch via selectable
sensor-fusion filters, assembles configurable features, and returns CNN-ready NumPy arrays.

Quick start
-----------
>>> from data_fusion_project.processing import load_dataset, PipelineConfig
>>> ds = load_dataset()                      # uses default configuration
>>> ds.X.shape                               # (N, T, C) -> Conv1D input
>>> X_tr, y_tr = ds.X[tr], ds.y[tr]

Configuration is fully declarative via :class:`PipelineConfig` and its stage configs
(:class:`CalibrationConfig`, :class:`FilterConfig`, :class:`OrientationConfig`,
:class:`FeatureConfig`), making feature experiments cheap to set up.
"""

from data_fusion_project.processing.config import (
    PipelineConfig,
    CalibrationConfig,
    FilterConfig,
    OrientationConfig,
    FeatureConfig,
    FilterType,
    OrientationMethod,
)
from data_fusion_project.processing.dataset import (
    GestureDataset,
    load_dataset,
    process_window,
)
from data_fusion_project.processing.calibration import (
    CalibrationProfile,
    ImuCalibration,
    estimate_calibration,
    identity_profile,
)
from data_fusion_project.processing.orientation import (
    estimate_orientation,
    accel_angles,
    complementary_filter,
    kalman_filter,
)
from data_fusion_project.processing import filters
from data_fusion_project.processing import features
from data_fusion_project.processing.splits import leave_sessions_out, stratified_split

__all__ = [
    # configuration
    "PipelineConfig",
    "CalibrationConfig",
    "FilterConfig",
    "OrientationConfig",
    "FeatureConfig",
    "FilterType",
    "OrientationMethod",
    # dataset interface
    "GestureDataset",
    "load_dataset",
    "process_window",
    # calibration
    "CalibrationProfile",
    "ImuCalibration",
    "estimate_calibration",
    "identity_profile",
    # orientation
    "estimate_orientation",
    "accel_angles",
    "complementary_filter",
    "kalman_filter",
    # filters & features modules
    "filters",
    "features",
    # splitting
    "leave_sessions_out",
    "stratified_split",
]
