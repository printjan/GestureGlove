# src/data_fusion_project/core/json_writer.py
"""
JSON writer helper.
This module provides a helper to write dict/list structures to JSON files on disk.
"""

# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from data_fusion_project.core.logger_setup import get_logger

# ======================================================================================================================
# constants
# ======================================================================================================================
logger = get_logger(__name__)


def write_json(path: Path, data: Any, indent: int = 4, ensure_ascii: bool = False) -> None:
    """
    Write data to a JSON file. Automatically creates parent directories.

    :param: path (Path): Path to the target JSON file.
    :param: data (Any): Data to serialize (must be JSON-serializable).
    :param: indent (int): JSON formatting indent level. Default 4.
    :param: ensure_ascii (bool): Whether to escape non-ASCII characters. Default False.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        logger.info("Successfully wrote JSON to: %s", str(path))
    except Exception as e:
        logger.exception("Failed to write JSON to: %s", str(path))
        raise IOError(f"Failed to write JSON file: {path}. Error: {e}") from e
