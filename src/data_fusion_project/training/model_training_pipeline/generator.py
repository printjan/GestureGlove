# src/data_fusion_project/training/model_training_pipeline/generator.py
"""
Dynamic temporal jittering batch generator for Keras training.

The existing data processing pipeline (``dataset.py``) applies jitter
**statically** — a single random offset is drawn once during
``load_dataset()`` and frozen for the entire training run. This means
every epoch sees the identical 150-sample crop.

This ``TimeSeriesJitterSequence`` implements **dynamic per-epoch jittering**:
the raw uncropped recording windows (shape ``(N, T_raw, C)`` where
``T_raw ≈ 174``) are pre-loaded and pre-scaled, and on every
``__getitem__`` call a fresh random offset is drawn independently for
each sample. Over 70 epochs of training the network sees ~70 unique
temporal translations of the same gesture, vastly improving robustness
to the causal filter group delay (20–40 ms) encountered during real-time
sliding-window inference.

When ``jitter_range = 0`` the generator degrades to a standard batched
Sequence that returns the exact centered window, preserving backward
compatibility with the static dataset loader.
"""

from __future__ import annotations

import random

import numpy as np
import keras

from data_fusion_project.training.model_training_pipeline.augmentation import (
    apply_rotation_augmentation,
)


class TimeSeriesJitterSequence(keras.utils.Sequence):
    """
    Keras Sequence that produces dynamically jittered + augmented batches.

    On each ``__getitem__`` call:
    1. Draws a per-sample random temporal offset within ``±jitter_range``.
    2. Slices a ``window_size``-sample window from the pre-scaled raw data.
    3. Applies optional 3D rotation augmentation.
    4. Shapes the output tensors according to the target model architecture
       (single tensor for early fusion / transformer, multi-input list for
       late fusion).

    :param X_raw: Pre-filtered & pre-scaled raw features, shape (N, T_raw, C).
    :param y: One-hot target labels, shape (N, num_classes).
    :param center_indices: Center start index per sample from .txt companion
        files, shape (N,). Each value marks where the perfectly centered
        150-sample window begins in the raw recording.
    :param window_size: Number of time steps to slice per sample (default: 150).
    :param jitter_range: Maximum temporal shift in samples (±). 0 = no jitter.
    :param batch_size: Batch size for training.
    :param augment_rotation: Maximum random 3D rotation angle in degrees.
        0.0 = no augmentation.
    :param channel_names: List of channel names for rotation group detection.
    :param model_type: Architecture identifier for output shaping.
        One of: ``"early_fusion_cnn"``, ``"late_fusion_cnn"``,
        ``"temporal_transformer"``.
    :param wrist_idx: Column indices for the wrist branch (late fusion only).
    :param finger_idx: Column indices for the finger branch (late fusion only).
    :param scalar_features: Pre-scaled scalar features, shape (N, F).
        Passed as a third input for the late fusion MLP branch.
    :param shuffle: Whether to shuffle sample order on epoch end.
    """

    def __init__(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        center_indices: np.ndarray,
        window_size: int = 150,
        jitter_range: int = 0,
        batch_size: int = 32,
        augment_rotation: float = 0.0,
        channel_names: list[str] | None = None,
        model_type: str = "early_fusion_cnn",
        wrist_idx: list[int] | None = None,
        finger_idx: list[int] | None = None,
        scalar_features: np.ndarray | None = None,
        shuffle: bool = True,
    ):
        self.X_raw = X_raw
        self.y = y
        self.center_indices = center_indices
        self.window_size = window_size
        self.jitter_range = jitter_range
        self.batch_size = batch_size
        self.augment_rotation = augment_rotation
        self.channel_names = channel_names or []
        self.model_type = model_type
        self.wrist_idx = wrist_idx or []
        self.finger_idx = finger_idx or []
        self.scalar_features = scalar_features
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X_raw))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X_raw) / self.batch_size))

    def on_epoch_end(self):
        """Re-shuffle sample order and draw new jitter offsets next epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X_batch = []

        # 1. Slice temporal window with per-sample random translation shift
        for idx in batch_idx:
            center = self.center_indices[idx]
            if self.jitter_range > 0:
                offset = random.randint(-self.jitter_range, self.jitter_range)
            else:
                offset = 0
            start = max(0, min(center + offset, self.X_raw.shape[1] - self.window_size))
            X_batch.append(self.X_raw[idx, start : start + self.window_size, :])

        X_batch = np.stack(X_batch)

        # 2. Apply on-the-fly random 3D rotation augmentation (per-sample unique)
        if self.augment_rotation > 0.0 and self.channel_names:
            X_batch = apply_rotation_augmentation(
                X_batch, self.channel_names, self.augment_rotation
            )

        # 3. Shape output based on model architecture
        y_batch = self.y[batch_idx]

        if self.model_type == "late_fusion_cnn":
            inputs = [X_batch[:, :, self.wrist_idx], X_batch[:, :, self.finger_idx]]
            if self.scalar_features is not None:
                inputs.append(self.scalar_features[batch_idx])
            return inputs, y_batch
        else:
            # early_fusion_cnn and temporal_transformer both use single tensor
            return X_batch, y_batch
