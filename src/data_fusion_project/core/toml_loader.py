# src/data_fusion_project/core/toml_loader.py
"""
TOML loader with caching.
This module provides a cached TOML loader that invalidates entries based on file
modification time (mtime).

Minimal usage example (no execution):
    from pathlib import Path
    from data_fusion_project.core.toml_loader import load_toml
    data = load_toml(Path(".../devices.toml"))
"""



# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from data_fusion_project.core.logger_setup import get_logger
import data_fusion_project.core.errors as errors



# ======================================================================================================================
# constants
# ======================================================================================================================
logger = get_logger(__name__)



def _toml_cache_key(path: Path) -> Tuple[str, float]:
    """
    Build a cache key from resolved path and mtime.

    :param: path (Path): TOML file path.
    :return: key (tuple[str, float]): Cache key (resolved_path, mtime).
    :raises: FileNotFoundError: If the file does not exist.
    """

    resolved_path = str(path.resolve())
    mtime = path.stat().st_mtime
    return resolved_path, mtime



@lru_cache(maxsize=64)
def _load_toml_cached(resolved_path: str, mtime: float) -> Dict[str, Any]:
    """
    Load and parse TOML content (cached). Cache is invalidated automatically when
    mtime changes because mtime is part of the cache key.

    :param: resolved_path (str): Resolved absolute path to TOML file.
    :param: mtime (float): File modification time (cache invalidation token).
    :return: data (dict): Parsed TOML data.
    :raises: TomlSchemaError: If TOML parsing fails.
    """

    path = Path(resolved_path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
        logger.info("Loaded TOML file (cached): %s", resolved_path)
        return data
    except Exception as e:  # pragma: no cover
        logger.exception("Failed to parse TOML file: %s", resolved_path)
        raise errors.TomlSchemaError(f"Failed to parse TOML file: {path}. Error: {e}") from e



def load_toml(path: Path) -> Dict[str, Any]:
    """
    Load TOML from disk using a cache keyed by (resolved_path, mtime).

    :param: path (Path): TOML file path.
    :return: data (dict): Parsed TOML content.
    :raises: FileNotFoundError: If the TOML file does not exist.
    :raises: TomlSchemaError: If TOML parsing fails.
    """

    if not path.exists():
        logger.error("TOML file not found: %s", str(path))
        raise FileNotFoundError(f"TOML file not found: {path}")

    resolved_path, mtime = _toml_cache_key(path)
    logger.info("Loading TOML file: %s (mtime=%s)", resolved_path, mtime)
    return _load_toml_cached(resolved_path, mtime)



def clear_toml_caches() -> None:
    """
    Clear internal TOML caches.
    """

    _load_toml_cached.cache_clear()
    logger.debug("TOML loader caches cleared.")