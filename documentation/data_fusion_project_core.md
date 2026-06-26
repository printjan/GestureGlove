# Data Fusion Project Core Module `data_fusion_project`

This document describes the installable Python package `data_fusion_project`, which provides centralized utility functions for path resolution, interactive CLI helpers, structured logging, and configuration loading.

---

## Startup - Installation

### From the repo root:

You can install the module in editable mode so that any updates to the source code are reflected immediately in your scripts:

```bash
python -m pip install -e .
```

#### Creating an isolated Virtual Environment (Optional but recommended):

- **macOS/Linux:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip
  python -m pip install -e .
  ```

- **Windows PowerShell:**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -U pip
  python -m pip install -e .
  ```

---

## Terminology and Project Layout

### Project Structure:

```
data_fusion_project/
├── pyproject.toml
├── config/
│   ├── devices.yml             # active board serial mapping config
│   └── devices.example.yml     # template config
├── data/                       # gestural sensor data
│   └── <gesture_name>/
│       └── <recording_session>/
│           ├── calibration.csv # 3-5 second static recording to measure drift
│           ├── 00001.csv       # first sequential gesture recording
│           └── 00002.csv
├── scripts/                       # runnable entry points / CLIs
│   ├── record_data.py             # recording controller (entry point)
│   ├── build_dataset.py           # builds the CNN-ready processing dataset
│   ├── visualize_processing.py    # processing pipeline diagnostic plots
│   └── check_samples.py           # dataset sanity check
└── src/
    └── data_fusion_project/
        ├── __init__.py
        ├── core/                  # project-agnostic infrastructure
        │   ├── __init__.py
        │   ├── paths.py           # path resolution constants and utilities
        │   ├── logger_setup.py    # standardized aligned logging setup
        │   ├── cli_ui.py          # rich CLI formatting, spinners and prompt helpers
        │   ├── errors.py          # core package exceptions
        │   ├── json_loader.py
        │   ├── json_writer.py
        │   ├── toml_loader.py
        │   └── toml_writer.py
        ├── recording/             # hardware interfaces & stream synchronization
        │   ├── __init__.py
        │   ├── input_data.py      # serial receiver interface
        │   ├── device_resolution.py  # hardware port mapping
        │   └── sync.py            # timestamp window synchronization
        └── processing/            # offline calibration / feature pipeline (CNN-ready arrays)
```

### How `data_fusion_project` Finds Paths

The package determines the workspace root dynamically using `data_fusion_project.core.paths.get_project_root()` in accordance with the following rules:

1. `DATA_FUSION_PROJECT_ROOT` env var if set.
2. Walking up from the current working directory (a valid root must contain `config/devices.yml` or `.git` and `src/data_fusion_project/`).
3. Walking up from the installed module file path (ideal for editable installs).

**If resolution fails**, you can set the environment variable:

- **macOS/Linux:**
  ```bash
  export DATA_FUSION_PROJECT_ROOT="/path/to/DataFusionProject"
  ```

- **Windows PowerShell:**
  ```powershell
  $env:DATA_FUSION_PROJECT_ROOT = "C:\path\to\DataFusionProject"
  ```

**Verify Resolution:**
```bash
python3 -c "import data_fusion_project.core.paths as p; print(p.get_project_root())"
```

---

## Core Utilities (`data_fusion_project.core`)

### `data_fusion_project.core.paths`

This module centralizes all filesystem locations and helper directories.

**Important Constants:**
- `BASE_DIRECTORY`: Resolved repository root directory.
- `CONFIG_DIR`: Location of configurations (`config/`).
- `SCRIPTS_DIR`: Location of runnable scripts / entry points (`scripts/`).
- `DATA_DIR`: Location of the datasets folder (`data/`).
- `LOGS_DIR`: Location of application logs (`logs/`).
- `DEVICES_CONFIG_FILE`: File path to `config/devices.yml`.
- `GESTURES`: Official list of supported gestures:
  `["none", "swipe_left", "swipe_right", "circle_cw", "circle_ccw", "fist", "jerk_down", "jerk_up"]`

**Helper Functions:**
- `get_gesture_dir(gesture_name: str) -> Path`: Returns the directory path for a gesture (e.g. `data/swipe_left`).
- `get_session_dir(gesture_name: str, session_name: str) -> Path`: Returns the recording session path (e.g. `data/swipe_left/session_1`).
- `get_calibration_file(gesture_name: str, session_name: str) -> Path`: Returns path to the calibration CSV.
- `get_next_recording_file(gesture_name: str, session_name: str) -> Path`: Automatically increments the file index to return the next recording file in a session (e.g. `data/swipe_left/session_1/00003.csv`).

---

### `data_fusion_project.core.logger_setup`

A structured logger that pads file/function names and logs to stdout.

- `get_logger(name: str | None = None, level: str | int | None = None, *, use_colors: bool | None = None) -> logging.Logger`
  - Typical use: `logger = get_logger(__name__)`
  - **Aligned Format:** `[filename] [funcName] [LEVEL]: message`
  - **Automatic Line Wrapping:** Messages are wrapped to fit terminal width. Continuation lines are indented with `↳`.
  - Honors `LOG_LEVEL` environment variable (defaults to `INFO`).
  - Color styling mapping: `DEBUG` (blue), `INFO` (green), `WARNING` (yellow), `ERROR` (red), `CRITICAL` (magenta).

**Prompting helpers on the returned logger:**
- `logger.read_text(prompt)`: Prompts user with the logger header prefix.
- `logger.read_text_raw(prompt)`: Prompts user without the logger header prefix.

**UI Namespace (`logger.ui`):**
- `logger.ui.success(text)` - Prints message with green `✓` prefix.
- `logger.ui.error(text)` - Prints message with red `✗` prefix.
- `logger.ui.warning(text)` - Prints message with yellow `⚠` prefix.
- `logger.ui.info(text)` - Prints message with blue `ℹ` prefix.
- `logger.ui.hr(title=None)` - Prints a horizontal rule.
- `logger.ui.banner(title, subtitle=None)` - Prints a formatted heading banner.
- `logger.ui.box(lines, title=None)` - Prints lines inside a bordered box.
- `logger.ui.kv(items)` - Prints aligned key-value pairs.

**Example:**
```python
from data_fusion_project.core.logger_setup import get_logger

logger = get_logger(__name__, level="INFO")
logger.info("Starting processing...")
logger.ui.success("Sensors synchronized successfully!")
```

---

### `data_fusion_project.core.cli_ui`

Standard terminal helpers. Can be used standalone or via `logger.ui`.

- `get_terminal_width()`: Gets width of terminal.
- `is_interactive()`: Checks if stdout/stdin is a TTY.
- `ansi_enabled()`: Checks if color output is enabled (respects `NO_COLOR`).
- `confirm(prompt: str, default: bool | None = None) -> bool`: Yes/No confirmation prompt.
- `ask_choice(prompt: str, choices: list[str]) -> str`: Prompts the user to select from a list.
- `Spinner(message: str)`: Context manager displaying an animated progress spinner.
  ```python
  from data_fusion_project.core.cli_ui import Spinner
  with Spinner("Connecting to serial ports..."):
      # resolve ports...
      pass
  ```

---

### `data_fusion_project.core.errors`

Standard exception hierarchy:

- `DataFusionError`: Base exception.
- `TomlSchemaError`: Raised when parsing a TOML schema fails.
- `JsonSchemaError`: Raised when parsing a JSON schema fails.

---

### `data_fusion_project.core.toml_loader` & `toml_writer`

Reads and writes TOML configurations with file modification time based caching:

- `load_toml(path: Path) -> dict`: Automatically invalidates the cache when the file is updated.
- `write_toml(path: Path, data: dict) -> None`: Pure-python TOML writer (no extra dependencies).
