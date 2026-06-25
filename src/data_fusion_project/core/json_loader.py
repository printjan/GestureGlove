# src/data_fusion_project/core/json_loader.py
"""
JSON loader with caching.
This module provides a cached JSON loader that invalidates entries based on file
modification time (mtime).
"""

# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

from data_fusion_project.core.logger_setup import get_logger

# ======================================================================================================================
# constants
# ======================================================================================================================
logger = get_logger(__name__)


def _json_cache_key(path: Path) -> Tuple[str, float]:
    """
    Build a cache key from resolved path and mtime.

    :param: path (Path): JSON file path.
    :return: key (tuple[str, float]): Cache key (resolved_path, mtime).
    :raises: FileNotFoundError: If the file does not exist.
    """
    resolved_path = str(path.resolve())
    mtime = path.stat().st_mtime
    return resolved_path, mtime


@lru_cache(maxsize=64)
def _load_json_cached(resolved_path: str, mtime: float) -> Any:
    """
    Load and parse JSON content (cached). Cache is invalidated automatically when
    mtime changes because mtime is part of the cache key.

    :param: resolved_path (str): Resolved absolute path to JSON file.
    :param: mtime (float): File modification time (cache invalidation token).
    :return: data (dict | list): Parsed JSON data.
    """
    path = Path(resolved_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded JSON file (cached): %s", resolved_path)
        return data
    except Exception as e:  # pragma: no cover
        logger.exception("Failed to parse JSON file: %s", resolved_path)
        raise ValueError(f"Failed to parse JSON file: {path}. Error: {e}") from e


def load_json(path: Path) -> Any:
    """
    Load JSON from disk using a cache keyed by (resolved_path, mtime).

    :param: path (Path): JSON file path.
    :return: data (dict | list): Parsed JSON content.
    :raises: FileNotFoundError: If the JSON file does not exist.
    :raises: ValueError: If JSON parsing fails.
    """
    if not path.exists():
        logger.error("JSON file not found: %s", str(path))
        raise FileNotFoundError(f"JSON file not found: {path}")

    resolved_path, mtime = _json_cache_key(path)
    logger.info("Loading JSON file: %s (mtime=%s)", resolved_path, mtime)
    return _load_json_cached(resolved_path, mtime)


def clear_json_caches() -> None:
    """
    Clear internal JSON caches.
    """
    _load_json_cached.cache_clear()
    logger.debug("JSON loader caches cleared.")
