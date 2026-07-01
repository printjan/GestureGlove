# src/data_fusion_project/training/self_attention_temporal_transformer/__init__.py
"""
Self-Attention Temporal Transformer model package.

Exports the model builder for the lightweight temporal transformer architecture,
which applies multi-head self-attention along the time dimension to capture
long-range temporal dependencies in gesture sequences.
"""

from data_fusion_project.training.self_attention_temporal_transformer.model import (
    build_temporal_transformer,
)

__all__ = ["build_temporal_transformer"]
