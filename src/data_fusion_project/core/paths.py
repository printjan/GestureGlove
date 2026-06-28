# src/data_fusion_project/core/paths.py
"""
Path resolution logic and workspace directories for the data_fusion_project.
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

from data_fusion_project.core.logger_setup import get_logger
logger = get_logger(__name__)


def _is_project_root(candidate: Path) -> bool:
    """
    Check whether a candidate path looks like the project root.
    """
    return (
        (candidate / "config" / "devices.yml").is_file()
        or (candidate / ".git").is_dir()
    ) and (candidate / "src" / "data_fusion_project").is_dir()


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    Resolve the data_fusion_project root directory.
    Resolution policy:
    1) DATA_FUSION_PROJECT_ROOT environment variable.
    2) Walk upwards from current working directory and look for a project marker.
    3) Walk upwards from this file.
    """
    env_root = (os.getenv("DATA_FUSION_PROJECT_ROOT") or "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not _is_project_root(root):
            raise FileNotFoundError(
                f"DATA_FUSION_PROJECT_ROOT is set but does not point to a valid project root: {root}"
            )
        logger.debug("Resolved project root via DATA_FUSION_PROJECT_ROOT: %s", root)
        return root

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if _is_project_root(candidate):
            logger.debug("Resolved project root by walking up from CWD: %s", candidate)
            return candidate

    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if _is_project_root(candidate):
            logger.debug("Resolved project root by walking up from module path: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "Could not resolve project root. "
        "Set DATA_FUSION_PROJECT_ROOT to the git project directory."
    )


# --- Core Directory Configurations ---
BASE_DIRECTORY = get_project_root()

CONFIG_DIR = BASE_DIRECTORY / "config"
SCRIPTS_DIR = BASE_DIRECTORY / "scripts"
DOCUMENTATION_DIR = BASE_DIRECTORY / "documentation"
FIRMWARE_DIR = BASE_DIRECTORY / "firmware"
DATA_DIR = BASE_DIRECTORY / "data"

DEVICES_CONFIG_FILE = CONFIG_DIR / "devices.yml"
LOGS_DIR = BASE_DIRECTORY / "logs"
MODELS_DIR = BASE_DIRECTORY / "models"


# --- Gestures and Hierarchical Data Paths ---

GESTURES: list[str] = [
    "none",
    "swipe_left",
    "swipe_right",
    "circle_cw",
    "circle_ccw",
    "fist",
    "jerk_down",
    "jerk_up"
]


def get_gesture_dir(gesture_name: str) -> Path:
    """
    Get the directory for a specific gesture.
    """
    if gesture_name not in GESTURES:
        logger.warning("Gesture '%s' is not in the official GESTURES list: %s", gesture_name, GESTURES)
    return DATA_DIR / gesture_name


def get_session_dir(gesture_name: str, session_name: str) -> Path:
    """
    Get the directory for a specific recording session of a gesture.
    """
    return get_gesture_dir(gesture_name) / session_name


def get_calibration_file(gesture_name: str, session_name: str, index: int = 0) -> Path:
    """
    Get the path to the calibration file of a session.
    """
    return get_session_dir(gesture_name, session_name) / f"calibration_{index}.csv"


def get_session_metadata_file(gesture_name: str, session_name: str) -> Path:
    """
    Get the path to the recording session configuration JSON metadata file.
    """
    return get_session_dir(gesture_name, session_name) / "recording_session.json"


def get_next_recording_file(gesture_name: str, session_name: str) -> Path:
    """
    Find the next available recording file in the session directory.
    E.g. if 00001.csv and 00002.csv exist, returns path/to/00003.csv.
    """
    session_dir = get_session_dir(gesture_name, session_name)
    session_dir.mkdir(parents=True, exist_ok=True)

    existing_indices = []
    # Find all 5-digit CSV files
    for p in session_dir.glob("*.csv"):
        if p.stem.isdigit() and len(p.stem) == 5:
            existing_indices.append(int(p.stem))

    next_index = max(existing_indices, default=0) + 1
    return session_dir / f"{next_index:05d}.csv"
