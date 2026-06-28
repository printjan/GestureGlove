# src/data_fusion_project/processing/dataset.py
"""
Dataset interface: turns the recorded CSV tree into CNN-ready NumPy arrays.

Directory layout consumed (produced by ``scripts/record_data.py``)::

    data/
    └── <gesture>/
        └── <session>/
            ├── recording_session.json           # properties of the session
            ├── calibration_<number>.csv         # ~5 s stillness, used for calibration
            ├── energy_distribution_<number>.csv  # motion energy distribution
            ├── 00001.csv                        # one gesture window (150 rows = 1.5 s @ 100 Hz)
            └── ...

For every window the pipeline runs: load -> calibrate -> filter -> orientation ->
feature assembly. The results are stacked into a :class:`GestureDataset`:

- ``X``        : float32 time-series tensor of shape ``(N, T, C)`` (Conv1D-ready)
- ``y``        : int label vector of shape ``(N,)``
- ``groups``   : session id per sample of shape ``(N,)`` (for leave-session-out splits)
- ``features`` : optional float32 scalar-feature matrix of shape ``(N, F)``

plus name lists for classes, channels and features.

Typical use::

    from data_fusion_project.processing import load_dataset, PipelineConfig
    ds = load_dataset()                     # defaults
    X, y = ds.X, ds.y                       # feed Conv1D(input_shape=X.shape[1:])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import random

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.core.paths import DATA_DIR, GESTURES
from data_fusion_project.processing.config import PipelineConfig
from data_fusion_project.processing import calibration as calib
from data_fusion_project.processing import filters as filt
from data_fusion_project.processing import features as feat
from data_fusion_project.processing.orientation import estimate_orientation
from data_fusion_project.processing.config import FilterType, OrientationMethod

logger = get_logger(__name__)

_ACC_AXES = ("accX", "accY", "accZ")
_GYR_AXES = ("gyrX", "gyrY", "gyrZ")
_RESERVED_CSV = {"calibration.csv", "energy_distribution.csv"}


# ======================================================================================================================
# Container
# ======================================================================================================================
@dataclass
class GestureDataset:
    """
    In-memory, framework-agnostic dataset container.

    :param: X (np.ndarray): time-series tensor, shape (N, T, C), float32.
    :param: y (np.ndarray): integer labels, shape (N,).
    :param: groups (np.ndarray): session id per sample, shape (N,) (object/str).
    :param: class_names (list): label index -> gesture name.
    :param: channel_names (list): channel index -> name (length C).
    :param: features (np.ndarray | None): scalar features, shape (N, F), or None.
    :param: feature_names (list): feature index -> name (length F).
    :param: sample_paths (list): source CSV path per sample (length N).
    :param: config (PipelineConfig | None): configuration used to build the dataset.
    """
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    class_names: list[str]
    channel_names: list[str]
    features: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    sample_paths: list[str] = field(default_factory=list)
    config: PipelineConfig | None = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_classes(self) -> int:
        """Number of distinct gesture classes."""
        return len(self.class_names)

    @property
    def input_shape(self) -> tuple[int, int]:
        """Per-sample time-series shape ``(T, C)`` for the CNN input layer."""
        return self.X.shape[1], self.X.shape[2]

    def class_distribution(self) -> dict[str, int]:
        """
        Counts samples per class.
        :return: counts (dict): mapping of gesture name to sample count.
        """
        counts = {name: 0 for name in self.class_names}
        for label in self.y:
            counts[self.class_names[int(label)]] += 1
        return counts

    def summary(self) -> str:
        """
        Builds a human-readable summary of the dataset.
        :return: text (str): multi-line summary string.
        """
        lines = [
            "─── GestureDataset ────────────────────────────────────",
            f"  samples   : {len(self)}",
            f"  X shape   : {self.X.shape}  (N, T, C)",
            f"  channels  : {len(self.channel_names)} -> {self.channel_names}",
        ]
        if self.features is not None and self.features.size:
            lines.append(f"  features  : {self.features.shape}  ({len(self.feature_names)} scalar features)")
        lines.append(f"  classes   : {self.n_classes}")
        for name, count in self.class_distribution().items():
            lines.append(f"    {name:<14} {count}")
        lines.append(f"  sessions  : {len(set(self.groups.tolist()))}")
        lines.append("────────────────────────────────────────────────────────")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """
        Saves the dataset arrays and metadata to a compressed ``.npz`` file.
        :param: path (str | Path): destination path (``.npz`` appended if missing).
        :return: path (Path): the written file path.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            y=self.y,
            groups=self.groups,
            class_names=np.array(self.class_names, dtype=object),
            channel_names=np.array(self.channel_names, dtype=object),
            features=self.features if self.features is not None else np.zeros((len(self), 0), dtype=np.float32),
            feature_names=np.array(self.feature_names, dtype=object),
            sample_paths=np.array(self.sample_paths, dtype=object),
        )
        logger.info("Saved dataset (%d samples) to %s", len(self), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GestureDataset":
        """
        Loads a dataset previously written by :meth:`save`.
        :param: path (str | Path): path to the ``.npz`` file.
        :return: dataset (GestureDataset): reconstructed dataset (without config).
        """
        data = np.load(path, allow_pickle=True)
        features = data["features"]
        if features.size == 0:
            features = None
        return cls(
            X=data["X"],
            y=data["y"],
            groups=data["groups"],
            class_names=list(data["class_names"]),
            channel_names=list(data["channel_names"]),
            features=features,
            feature_names=list(data["feature_names"]),
            sample_paths=list(data["sample_paths"]),
        )


# ======================================================================================================================
# Window loading & coercion
# ======================================================================================================================
def _coerce_length(window: np.ndarray, target: int, pad_mode: str) -> np.ndarray:
    """
    Forces a window to exactly ``target`` rows by truncating or padding.
    :param: window (np.ndarray): window array, shape (T0, C).
    :param: target (int): desired number of rows.
    :param: pad_mode (str): "edge" repeats the last row, "zero" pads with zeros.
    :return: coerced (np.ndarray): window of shape (target, C).
    """
    n = window.shape[0]
    if n == target:
        return window
    if n > target:
        return window[:target]

    pad_rows = target - n
    if pad_mode == "zero":
        pad = np.zeros((pad_rows, window.shape[1]), dtype=window.dtype)
    else:  # "edge"
        pad = np.repeat(window[-1:], pad_rows, axis=0)
    return np.vstack([window, pad])


def _split_imu_blocks(df: pd.DataFrame, imus) -> dict:
    """
    Extracts per-IMU accelerometer/gyroscope blocks from a window dataframe.
    :param: df (pd.DataFrame): window with ``<imu>_acc*`` / ``<imu>_gyr*`` columns.
    :param: imus (Iterable[str]): IMU prefixes to extract.
    :return: blocks (dict): mapping ``imu -> {"acc": (T,3), "gyr": (T,3)}``.
    """
    blocks = {}
    for imu in imus:
        acc_cols = [f"{imu}_{ax}" for ax in _ACC_AXES]
        gyr_cols = [f"{imu}_{ax}" for ax in _GYR_AXES]
        if not all(c in df.columns for c in acc_cols + gyr_cols):
            continue
        blocks[imu] = {
            "acc": df[acc_cols].to_numpy(dtype=float),
            "gyr": df[gyr_cols].to_numpy(dtype=float),
        }
    return blocks


# ======================================================================================================================
# Per-window processing
# ======================================================================================================================
def process_window(df: pd.DataFrame, profile: calib.CalibrationProfile, config: PipelineConfig):
    """
    Runs the full pipeline on a single window dataframe.
    :param: df (pd.DataFrame): raw window (already coerced to the configured length is not required).
    :param: profile (CalibrationProfile): calibration profile for the window's session.
    :param: config (PipelineConfig): pipeline configuration.
    :return: result (tuple): (channels (T, C), channel_names, scalar_features (F,), feature_names).
    """
    fs = config.sample_rate_hz
    imus = list(dict.fromkeys(list(config.features.imus) + list(config.orientation.imus)))
    blocks = _split_imu_blocks(df, imus)

    processed: dict[str, dict] = {}
    orientation: dict[str, dict] = {}

    for imu, block in blocks.items():
        acc, gyr = block["acc"], block["gyr"]

        # 1. Calibration (gyro bias, acc normalization).
        acc, gyr = calib.apply_calibration(acc, gyr, profile.get(imu), config.calibration)

        # 2. Filtering (low-/high-/band-pass + optional gravity removal).
        if config.filters.enabled:
            acc = filt.apply_filter(acc, config.filters.acc_filter, config.filters.acc_cutoff_hz, fs, config.filters.order)
            gyr = filt.apply_filter(gyr, config.filters.gyro_filter, config.filters.gyro_cutoff_hz, fs, config.filters.order)
            if config.filters.remove_gravity:
                linear, _gravity = filt.remove_gravity(acc, fs, config.filters.gravity_cutoff_hz, config.filters.order)
                if config.filters.replace_acc_with_linear:
                    acc = linear

        processed[imu] = {"acc": acc, "gyr": gyr}

        # 3. Orientation (roll/pitch) from the pre-filtered signals.
        if config.orientation.enabled and config.orientation.method != OrientationMethod.NONE and imu in config.orientation.imus:
            orientation[imu] = estimate_orientation(acc, gyr, fs, config.orientation)

    # 4. Feature assembly.
    channels, channel_names = feat.build_channels(
        processed,
        orientation,
        config.features,
        fs=fs,
        orientation_degrees=config.orientation.degrees
    )
    scalar, feature_names = feat.build_scalar_features(processed, channels, channel_names, config.features)
    return channels, channel_names, scalar, feature_names


# ======================================================================================================================
# Dataset loading
# ======================================================================================================================
def _iter_sessions(data_dir: Path, gestures):
    """
    Yields ``(gesture, session_dir)`` pairs for all sessions present on disk.
    :param: data_dir (Path): root data directory.
    :param: gestures (Iterable[str]): gesture names to scan.
    :return: generator: tuples of (gesture_name, session_directory).
    """
    for gesture in gestures:
        gesture_dir = data_dir / gesture
        if not gesture_dir.is_dir():
            continue
        for session_dir in sorted(p for p in gesture_dir.iterdir() if p.is_dir()):
            yield gesture, session_dir


def _session_sample_files(session_dir: Path) -> list[Path]:
    """
    Lists the gesture-sample CSVs of a session (excluding calibration/aux files).
    :param: session_dir (Path): session directory.
    :return: files (list): sorted list of sample CSV paths.
    """
    return sorted(
        p for p in session_dir.glob("*.csv")
        if p.name not in _RESERVED_CSV and p.stem.isdigit()
    )


def load_dataset(config: PipelineConfig | None = None, data_dir: str | Path | None = None,
                 gestures: list[str] | None = None, group_by: str = "session") -> GestureDataset:
    """
    Loads, processes and stacks all recorded gesture samples into a :class:`GestureDataset`.
    :param: config (PipelineConfig | None): pipeline configuration; defaults to ``PipelineConfig()``.
    :param: data_dir (str | Path | None): data root; defaults to the project ``data/`` directory.
    :param: gestures (list | None): gesture names/labels; defaults to the project ``GESTURES`` list.
    :param: group_by (str): grouping for the ``groups`` array — "session" (session name) or
            "gesture_session" (gesture + session, never merges sessions across gestures).
    :return: dataset (GestureDataset): the assembled dataset.
    :raises: FileNotFoundError: if the data directory does not exist.
    :raises: RuntimeError: if no valid samples were found.
    """
    config = config or PipelineConfig()
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    gestures = gestures if gestures is not None else list(GESTURES)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups: list[str] = []
    feat_list: list[np.ndarray] = []
    sample_paths: list[str] = []
    channel_names: list[str] | None = None
    feature_names: list[str] = []

    n_sessions = 0
    n_skipped = 0

    for gesture, session_dir in _iter_sessions(data_dir, gestures):
        label = gestures.index(gesture)
        sample_files = _session_sample_files(session_dir)
        if not sample_files:
            continue
        n_sessions += 1

        # Load session metadata recording_session.json (strict implementation, no fallback)
        metadata_file = session_dir / "recording_session.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Missing required session metadata file: {metadata_file}")

        import json
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        profiles_by_sample = {}
        recalibrations = metadata.get("recalibrations", [])
        if not recalibrations:
            raise ValueError(f"No recalibrations found in session metadata: {metadata_file}")

        for entry in recalibrations:
            cal_filename = entry["file"]
            sample_idx = entry["sample_index"]
            cal_path = session_dir / cal_filename
            if not cal_path.exists():
                raise FileNotFoundError(f"Calibration file '{cal_filename}' listed in metadata does not exist in {session_dir}")
            try:
                profiles_by_sample[sample_idx] = calib.estimate_calibration(cal_path, config.calibration)
            except Exception as exc:
                logger.error("Failed to estimate calibration from %s: %s", cal_path, exc)
                raise exc

        group = session_dir.name if group_by == "session" else f"{gesture}/{session_dir.name}"

        for csv_path in sample_files:
            # Resolve the closest calibration strictly before this sample index
            sample_stem = csv_path.stem
            sample_idx = int(sample_stem) if sample_stem.isdigit() else 0
            matching_scs = [sc for sc in profiles_by_sample.keys() if sc < sample_idx]
            active_sc = max(matching_scs) if matching_scs else min(profiles_by_sample.keys())
            profile = profiles_by_sample[active_sc]

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.error("Failed to read %s: %s", csv_path, exc)
                n_skipped += 1
                continue

            # Load start index companion file. If it doesn't exist, check if the CSV is pre-cropped.
            txt_path = csv_path.with_suffix('.txt')
            if not txt_path.exists():
                raw_len = len(df)
                if raw_len == config.window_size:
                    start_idx = 0
                else:
                    raise FileNotFoundError(
                        f"Missing required start index companion file: {txt_path} "
                        f"(CSV length is {raw_len}, which does not match target window_size {config.window_size})."
                    )
            else:
                try:
                    with open(txt_path, "r", encoding="utf-8") as txt_f:
                        start_idx = int(txt_f.read().strip())
                except Exception as exc:
                    logger.error("Failed to read/parse start index from %s: %s", txt_path, exc)
                    raise exc

            # Crop the raw dataframe to exactly config.window_size starting at start_idx with optional jitter
            raw_len = len(df)
            if hasattr(config, 'jitter_range') and config.jitter_range > 0:
                shift = random.randint(-config.jitter_range, config.jitter_range)
                start_idx = max(0, min(start_idx + shift, raw_len - config.window_size))

            df = df.iloc[start_idx : start_idx + config.window_size]

            if len(df) != config.window_size:
                raise ValueError(
                    f"Sliced window from {csv_path.name} starting at index {start_idx} has length {len(df)} "
                    f"instead of expected target {config.window_size} (raw file length was {raw_len})."
                )

            try:
                channels, ch_names, scalar, ft_names = process_window(df, profile, config)
            except Exception as exc:
                logger.error("Failed to process %s: %s", csv_path, exc)
                n_skipped += 1
                continue

            if channel_names is None:
                channel_names = ch_names
                feature_names = ft_names

            X_list.append(channels)
            y_list.append(label)
            groups.append(group)
            feat_list.append(scalar)
            sample_paths.append(str(csv_path))

    if not X_list:
        raise RuntimeError(f"No valid gesture samples found under {data_dir}.")

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    groups_arr = np.asarray(groups, dtype=object)

    features = None
    if feature_names:
        features = np.stack(feat_list).astype(np.float32)

    logger.info("Loaded %d samples from %d sessions (%d skipped). X shape %s.",
                len(X_list), n_sessions, n_skipped, X.shape)

    return GestureDataset(
        X=X,
        y=y,
        groups=groups_arr,
        class_names=list(gestures),
        channel_names=channel_names or [],
        features=features,
        feature_names=feature_names,
        sample_paths=sample_paths,
        config=config,
    )
