# src/data_fusion_project/core/errors.py
"""
Exception definitions for the data_fusion_project package.
"""

from __future__ import annotations


class DataFusionError(Exception):
    """
    Base exception for data_fusion_project.
    """


class TomlSchemaError(DataFusionError, ValueError):
    """
    Raised when a TOML file cannot be parsed or does not match the expected schema.
    """


class JsonSchemaError(DataFusionError, ValueError):
    """
    Raised when a JSON file cannot be parsed or does not match the expected schema.
    """
