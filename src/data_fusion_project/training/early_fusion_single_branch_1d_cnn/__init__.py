# src/data_fusion_project/training/early_fusion_single_branch_1d_cnn/__init__.py
"""
Early Fusion Single-Branch Conv1D CNN model package.

Exports the model builder for the early fusion architecture, which concatenates
all sensor channels into a single input tensor and processes them through a
single-branch Conv1D pipeline.
"""

from data_fusion_project.training.early_fusion_single_branch_1d_cnn.model import (
    build_early_fusion_cnn,
)

__all__ = ["build_early_fusion_cnn"]
