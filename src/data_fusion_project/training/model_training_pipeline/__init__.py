# src/data_fusion_project/training/model_training_pipeline/__init__.py
"""
Unified Model Training Pipeline.

Provides the architecture-agnostic training loop, temporal jitter generator,
and rotation augmentation utilities that support all three candidate
architectures (Early Fusion CNN, Late Fusion CNN, Temporal Transformer).
"""

from data_fusion_project.training.model_training_pipeline.pipeline import train_model
from data_fusion_project.training.model_training_pipeline.generator import TimeSeriesJitterSequence
from data_fusion_project.training.model_training_pipeline.augmentation import (
    apply_rotation_augmentation,
    random_rotation_matrix,
)

__all__ = [
    "train_model",
    "TimeSeriesJitterSequence",
    "apply_rotation_augmentation",
    "random_rotation_matrix",
]
