# src/data_fusion_project/core/toml_writer.py
"""
TOML writer helper.
This module provides a pure-Python TOML serializer to write dictionaries/lists
to TOML files without requiring external packages like tomli_w.
"""

# ======================================================================================================================
# imports
# ======================================================================================================================
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from data_fusion_project.core.logger_setup import get_logger

# ======================================================================================================================
# constants
# ======================================================================================================================
logger = get_logger(__name__)


def serialize_value(val: Any) -> str:
    """
    Recursively serialize Python values to TOML representation.
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, (int, float)):
        return str(val)
    elif isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        return f'"{escaped}"'
    elif isinstance(val, list):
        if not val:
            return "[]"
        items = [serialize_value(x) for x in val]
        return "[" + ", ".join(items) + "]"
    elif isinstance(val, dict):
        # Format dictionary as inline table
        items = [f"{k} = {serialize_value(v)}" for k, v in val.items()]
        return "{ " + ", ".join(items) + " }"
    elif val is None:
        return '""'
    else:
        raise TypeError(f"Unsupported type for TOML serialization: {type(val)}")


def dumps_toml(data: Dict[str, Any]) -> str:
    """
    Serialize a dictionary to TOML format.
    """
    lines = []
    for k, v in data.items():
        if isinstance(v, list) and len(v) > 0 and all(isinstance(x, dict) for x in v):
            # Print list of dictionaries as standard array of tables
            for item in v:
                lines.append("")
                lines.append(f"[[{k}]]")
                for sub_k, sub_v in item.items():
                    lines.append(f"{sub_k} = {serialize_value(sub_v)}")
        elif isinstance(v, dict):
            lines.append("")
            lines.append(f"[{k}]")
            for sub_k, sub_v in v.items():
                lines.append(f"{sub_k} = {serialize_value(sub_v)}")
        else:
            lines.append(f"{k} = {serialize_value(v)}")
    return "\n".join(lines).strip() + "\n"


def write_toml(path: Path, data: Dict[str, Any]) -> None:
    """
    Write data to a TOML file. Automatically creates parent directories.

    :param: path (Path): Path to the target TOML file.
    :param: data (dict): Dictionary to serialize.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = dumps_toml(data)
        with path.open("w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Successfully wrote TOML to: %s", str(path))
    except Exception as e:
        logger.exception("Failed to write TOML to: %s", str(path))
        raise IOError(f"Failed to write TOML file: {path}. Error: {e}") from e
