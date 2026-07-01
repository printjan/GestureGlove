# src/data_fusion_project/training/model_training_pipeline/augmentation.py
"""
Spatial data augmentation utilities for IMU time-series.

Provides random 3D rotation augmentation using Rodrigues' rotation formula.
Augmentations are applied per-sample (each sample in a batch receives a unique
random rotation) to maximize the diversity of sensor orientations the model
sees during training.

This is critical for real-time generalization: the physical mounting angle
of the IMU straps varies between donning sessions, so the model must learn
gesture recognition that is invariant to small rotational offsets.
"""

from __future__ import annotations

import numpy as np

from data_fusion_project.core.logger_setup import get_logger

logger = get_logger(__name__)


def random_rotation_matrix(max_angle_deg: float = 180.0) -> np.ndarray:
    """
    Generates a random 3D rotation matrix within a maximum angle.

    Uses uniform axis sampling on the unit sphere and Rodrigues' rotation
    formula via quaternion representation.

    :param max_angle_deg: Maximum rotation angle in degrees.
    :return: 3x3 rotation matrix.
    """
    if max_angle_deg <= 0:
        return np.eye(3)

    # Random axis (uniformly distributed on the unit sphere)
    theta = np.random.uniform(0, np.pi * 2)
    phi = np.arccos(np.random.uniform(-1, 1))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    axis = np.array([x, y, z])

    # Random angle within ±max_angle_deg
    angle = np.random.uniform(-np.radians(max_angle_deg), np.radians(max_angle_deg))

    # Quaternion → Rotation matrix (Rodrigues)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
    ])


def apply_rotation_augmentation(
    X_batch: np.ndarray,
    channel_names: list[str],
    max_angle_deg: float = 15.0,
) -> np.ndarray:
    """
    Applies random 3D rotation augmentation to IMU coordinate triads.

    Identifies accelerometer and gyroscope XYZ groups per IMU prefix
    (e.g., ``IMU1_accX/Y/Z``, ``IMU2_gyrX/Y/Z``) and applies independent
    random rotations to each group for each sample in the batch.

    :param X_batch: Batch tensor of shape (N, T, C).
    :param channel_names: List of C channel names for index matching.
    :param max_angle_deg: Maximum rotation angle in degrees per axis.
    :return: Augmented copy of X_batch.
    """
    if max_angle_deg <= 0:
        return X_batch

    X_aug = X_batch.copy()
    N, T, C = X_aug.shape

    # Identify unique IMU prefixes (e.g., "IMU1", "IMU2")
    imus = set()
    for name in channel_names:
        parts = name.split("_")
        if len(parts) > 1:
            imus.add(parts[0])

    for imu in imus:
        # Find accelerometer XYZ triad indices
        acc_idx = [
            i for i, name in enumerate(channel_names)
            if name.startswith(f"{imu}_acc") and any(name.endswith(ax) for ax in ("X", "Y", "Z"))
        ]
        # Find gyroscope XYZ triad indices
        gyr_idx = [
            i for i, name in enumerate(channel_names)
            if name.startswith(f"{imu}_gyr") and any(name.endswith(ax) for ax in ("X", "Y", "Z"))
        ]

        for i in range(N):
            if len(acc_idx) == 3:
                R_acc = random_rotation_matrix(max_angle_deg)
                X_aug[i][:, acc_idx] = X_aug[i][:, acc_idx] @ R_acc.T
            if len(gyr_idx) == 3:
                R_gyr = random_rotation_matrix(max_angle_deg)
                X_aug[i][:, gyr_idx] = X_aug[i][:, gyr_idx] @ R_gyr.T

    return X_aug
