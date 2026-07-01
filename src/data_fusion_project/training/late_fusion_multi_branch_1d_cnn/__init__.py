# src/data_fusion_project/training/late_fusion_multi_branch_1d_cnn/__init__.py
"""
Late Fusion Multi-Branch Conv1D CNN model package.

Exports the model builder for the late fusion architecture, which routes
wrist (IMU1) and finger (IMU2/diff) channels through independent Conv1D
branches before fusing them for classification.
"""

from data_fusion_project.training.late_fusion_multi_branch_1d_cnn.model import (
    build_late_fusion_cnn,
)

__all__ = ["build_late_fusion_cnn"]
