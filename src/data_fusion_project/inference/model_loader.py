# src/data_fusion_project/inference/model_loader.py
"""
Architecture-agnostic model loader for real-time inference.

This module encapsulates all architecture-specific dispatch logic required to
load any trained gesture classification model (Early Fusion CNN, Late Fusion
CNN, or Temporal Transformer) and prepare it for real-time inference.

The central entry point is ``load_inference_model(model_dir)``, which:

1.  Reads ``model_metadata.json`` to discover the ``model_type`` and training
    configuration (channels, scalers, architecture hyperparameters).
2.  Dispatches to the correct model builder (``build_early_fusion_cnn``,
    ``build_late_fusion_cnn``, or ``build_temporal_transformer``).
3.  Loads the saved weights from ``model.weights.h5``.
4.  Loads the architecture-appropriate scaler(s) from joblib artifacts.
5.  Constructs a ``transform_fn`` closure that maps raw ``(channels, channel_names)``
    tuples from the ``AsynchronousDataGrabber`` into model-ready tensors.
6.  Constructs a ``predict_fn`` closure that wraps ``model.predict()`` with the
    correct input format (single array for single-branch; named dict for late
    fusion multi-branch).

Returns an ``InferenceBundle`` dataclass containing the model, closures,
class names, pipeline configuration, and metadata.

Architecture dispatch table:

    +--------------------------+---------------------------+------------------------------+-----------------------+
    | model_type               | Builder                   | Scaler(s)                    | predict_fn input      |
    +--------------------------+---------------------------+------------------------------+-----------------------+
    | early_fusion_cnn         | build_early_fusion_cnn    | scaler_x.joblib              | np.ndarray            |
    | late_fusion_cnn          | build_late_fusion_cnn     | scaler_x_wrist + _finger     | dict (named inputs)   |
    | temporal_transformer     | build_temporal_transformer| scaler_x.joblib              | np.ndarray            |
    +--------------------------+---------------------------+------------------------------+-----------------------+

Design decisions:
- ``TimeSeriesScaler`` is imported from ``model_training_pipeline.pipeline`` so that
  joblib deserialization finds the exact same class path used during training.
- ``PipelineConfig`` reconstruction from metadata JSON is centralized here so both
  the inference script and any future programmatic consumers share the same logic.
- All architecture-specific branching is fully contained within this module. The
  inference script and ``AsynchronousDataGrabber`` remain architecture-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import joblib

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.processing import PipelineConfig
from data_fusion_project.processing.config import (
    CalibrationConfig,
    FilterConfig,
    FilterType,
    OrientationConfig,
    OrientationMethod,
    FeatureConfig,
)

# Import TimeSeriesScaler so that joblib can resolve the class during deserialization.
# This MUST match the class path used by the training pipeline when serializing.
from data_fusion_project.training.model_training_pipeline.pipeline import TimeSeriesScaler  # noqa: F401

logger = get_logger("ModelLoader")


@dataclass
class InferenceBundle:
    """
    Encapsulates all artifacts required for architecture-agnostic inference.

    Attributes
    ----------
    model : keras.Model
        The compiled Keras model with loaded weights.
    model_type : str
        Architecture identifier (``early_fusion_cnn``, ``late_fusion_cnn``,
        or ``temporal_transformer``).
    class_names : list[str]
        Ordered class label names matching the softmax output indices.
    pipeline_config : PipelineConfig
        Reconstructed signal processing configuration from training metadata.
    transform_fn : Callable
        Closure that maps ``(channels: np.ndarray, channel_names: list[str])``
        from the ``AsynchronousDataGrabber`` to model-ready input tensor(s).
    predict_fn : Callable
        Closure that wraps ``model.predict()`` with the correct input format
        for the architecture. Returns raw softmax probability array of shape
        ``(1, num_classes)``.
    metadata : dict
        Full raw ``model_metadata.json`` contents for audit/logging.
    model_dir : Path
        Resolved path to the training session directory.
    """

    model: Any
    model_type: str
    class_names: list[str]
    pipeline_config: PipelineConfig
    transform_fn: Callable[[np.ndarray, list[str]], Any]
    predict_fn: Callable[[Any], np.ndarray]
    metadata: dict
    model_dir: Path


def load_pipeline_config(metadata: dict) -> PipelineConfig:
    """
    Reconstructs a ``PipelineConfig`` instance from a ``model_metadata.json``
    dictionary.

    The metadata stores the pipeline config as a nested dict under the key
    ``"pipeline_config"``. This function deserializes enum types (``FilterType``,
    ``OrientationMethod``) and reconstructs the four sub-config dataclasses
    (``CalibrationConfig``, ``FilterConfig``, ``OrientationConfig``,
    ``FeatureConfig``).

    Parameters
    ----------
    metadata : dict
        Parsed contents of ``model_metadata.json``.

    Returns
    -------
    PipelineConfig
        Fully reconstructed pipeline configuration.
    """
    p_cfg_dict = metadata["pipeline_config"]

    cal_cfg = CalibrationConfig(**p_cfg_dict["calibration"])

    fil_cfg_dict = p_cfg_dict["filters"].copy()
    if "acc_filter" in fil_cfg_dict:
        fil_cfg_dict["acc_filter"] = FilterType(fil_cfg_dict["acc_filter"])
    if "gyro_filter" in fil_cfg_dict:
        fil_cfg_dict["gyro_filter"] = FilterType(fil_cfg_dict["gyro_filter"])
    fil_cfg = FilterConfig(**fil_cfg_dict)

    ori_cfg_dict = p_cfg_dict["orientation"].copy()
    if "method" in ori_cfg_dict:
        ori_cfg_dict["method"] = OrientationMethod(ori_cfg_dict["method"])
    ori_cfg = OrientationConfig(**ori_cfg_dict)

    feat_cfg = FeatureConfig(**p_cfg_dict["features"])

    return PipelineConfig(
        sample_rate_hz=p_cfg_dict.get("sample_rate_hz", 100.0),
        window_size=p_cfg_dict.get("window_size", 150),
        pad_mode=p_cfg_dict.get("pad_mode", "edge"),
        jitter_range=p_cfg_dict.get("jitter_range", 0),
        calibration=cal_cfg,
        filters=fil_cfg,
        orientation=ori_cfg,
        features=feat_cfg,
    )


def _resolve_session_dir(model_dir: Path) -> Path:
    """
    Resolves the latest training session subdirectory within a model folder.

    If ``model_dir`` itself contains ``model_metadata.json``, it is returned
    directly (caller pointed at a specific session). Otherwise, the function
    scans for ``training_session_*`` directories and returns the one with the
    highest sequential index.

    Parameters
    ----------
    model_dir : Path
        Either a model identifier directory (e.g. ``models/early_fusion_…``)
        or a specific session directory.

    Returns
    -------
    Path
        Resolved session directory.

    Raises
    ------
    FileNotFoundError
        If no training sessions or metadata files are found.
    """
    if (model_dir / "model_metadata.json").exists() or (model_dir / "metadata.json").exists():
        return model_dir

    def _session_index(p: Path) -> int:
        parts = p.name.split("_")
        try:
            return int(parts[2])
        except (IndexError, ValueError):
            return 0

    sessions = sorted(
        [p for p in model_dir.glob("training_session_*") if p.is_dir()],
        key=_session_index,
    )
    if sessions:
        return sessions[-1]

    raise FileNotFoundError(
        f"No training sessions found in '{model_dir}'. "
        f"Expected either a model_metadata.json or training_session_* subdirectories."
    )


def load_inference_model(model_dir: str | Path) -> InferenceBundle:
    """
    Loads a trained gesture classification model and prepares it for inference.

    This is the primary entry point for the inference pipeline. It reads the
    ``model_metadata.json`` to discover the architecture, loads the model
    builder and weights, deserializes the scaler(s), and constructs
    architecture-specific ``transform_fn`` and ``predict_fn`` closures.

    Parameters
    ----------
    model_dir : str or Path
        Path to the model identifier directory (e.g.
        ``models/early_fusion_single_branch_1d_cnn``) or a specific training
        session directory (e.g. ``models/…/training_session_2_20260701_223131``).
        The latest session is auto-resolved if a top-level model directory is
        provided.

    Returns
    -------
    InferenceBundle
        Dataclass containing the model, closures, class names, and metadata.

    Raises
    ------
    FileNotFoundError
        If model directory, metadata, or weight files are missing.
    ValueError
        If an unknown ``model_type`` is encountered in the metadata.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    # Resolve to the latest training session
    session_dir = _resolve_session_dir(model_dir)
    logger.info("Resolved model session: %s", session_dir)

    # ──────────────────────────────────────────────────────────────────────
    # 1. Load Metadata
    # ──────────────────────────────────────────────────────────────────────
    metadata_path = session_dir / "model_metadata.json"
    if not metadata_path.exists():
        metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata file not found in {session_dir}. "
            f"Expected model_metadata.json or metadata.json."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_type = metadata["model_type"]
    classes = metadata["classes"]
    channels = metadata["channels"]
    pipeline_config = load_pipeline_config(metadata)
    train_params = metadata.get("training_parameters", {})

    logger.info(
        "Loaded metadata: model_type=%s, classes=%d, channels=%d",
        model_type, len(classes), len(channels),
    )

    # ──────────────────────────────────────────────────────────────────────
    # 2. Build Model Architecture (Dynamic Input Binding)
    # ──────────────────────────────────────────────────────────────────────
    conv_filters = train_params.get("conv_filters", [32, 64])
    dense_units = train_params.get("dense_units", 16)
    window_size = pipeline_config.window_size

    if model_type == "early_fusion_cnn":
        from data_fusion_project.training.early_fusion_single_branch_1d_cnn import (
            build_early_fusion_cnn,
        )
        input_shape = (window_size, len(channels))
        model = build_early_fusion_cnn(
            input_shape=input_shape,
            num_classes=len(classes),
            conv_filters=conv_filters,
            dense_units=dense_units,
        )

    elif model_type == "late_fusion_cnn":
        from data_fusion_project.training.late_fusion_multi_branch_1d_cnn import (
            build_late_fusion_cnn,
        )
        wrist_channels = metadata.get("wrist_channels", [])
        finger_channels = metadata.get("finger_channels", [])
        feature_names = metadata.get("feature_names", [])

        wrist_shape = (window_size, len(wrist_channels)) if wrist_channels else None
        finger_shape = (window_size, len(finger_channels)) if finger_channels else None
        feat_shape = len(feature_names) if feature_names else None

        model = build_late_fusion_cnn(
            input_shape_wrist=wrist_shape,
            input_shape_finger=finger_shape,
            num_classes=len(classes),
            input_shape_feat=feat_shape,
            conv_filters=conv_filters,
            dense_units=dense_units,
        )

    elif model_type == "temporal_transformer":
        from data_fusion_project.training.self_attention_temporal_transformer import (
            build_temporal_transformer,
        )
        # Import custom layers so Keras can resolve them during weight loading
        from data_fusion_project.training.self_attention_temporal_transformer.model import (  # noqa: F401
            TransformerEncoderBlock,
            LearnablePositionalEncoding,
        )
        input_shape = (window_size, len(channels))
        model = build_temporal_transformer(
            input_shape=input_shape,
            num_classes=len(classes),
            d_model=train_params.get("d_model", 64),
            num_heads=train_params.get("num_heads", 4),
            num_blocks=train_params.get("num_blocks", 2),
            ff_dim=train_params.get("ff_dim", 128),
            dense_units=dense_units,
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}' in metadata. "
            f"Supported: early_fusion_cnn, late_fusion_cnn, temporal_transformer"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 3. Load Weights
    # ──────────────────────────────────────────────────────────────────────
    weights_path = session_dir / "model.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    model.load_weights(weights_path)
    logger.info("Loaded model weights from %s", weights_path)

    # ──────────────────────────────────────────────────────────────────────
    # 4. Load Scalers & Build Transform/Predict Closures
    # ──────────────────────────────────────────────────────────────────────
    if model_type == "late_fusion_cnn":
        transform_fn, predict_fn = _build_late_fusion_closures(
            session_dir, metadata, model,
        )
    else:
        transform_fn, predict_fn = _build_single_branch_closures(
            session_dir, metadata, model,
        )

    return InferenceBundle(
        model=model,
        model_type=model_type,
        class_names=classes,
        pipeline_config=pipeline_config,
        transform_fn=transform_fn,
        predict_fn=predict_fn,
        metadata=metadata,
        model_dir=session_dir,
    )


def _build_single_branch_closures(
    session_dir: Path,
    metadata: dict,
    model: Any,
) -> tuple[Callable, Callable]:
    """
    Builds transform_fn and predict_fn for single-branch architectures
    (Early Fusion CNN and Temporal Transformer).

    Single-branch models use one ``scaler_x.joblib`` and accept a single
    ``np.ndarray`` of shape ``(1, T, C)`` as input to ``model.predict()``.

    The ``transform_fn`` closure:
    1. Receives ``(channels, channel_names)`` from the ``AsynchronousDataGrabber``.
    2. Selects only the channels listed in ``metadata["channels"]``.
    3. Adds a batch dimension: ``(T, C) → (1, T, C)``.
    4. Applies the ``TimeSeriesScaler`` transformation.

    Parameters
    ----------
    session_dir : Path
        Path to the training session directory containing scaler artifacts.
    metadata : dict
        Parsed model metadata.
    model : keras.Model
        The loaded Keras model.

    Returns
    -------
    tuple[Callable, Callable]
        ``(transform_fn, predict_fn)`` closures.
    """
    scaler_path = session_dir / "scaler_x.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler_x = joblib.load(scaler_path)
    logger.info("Loaded single-branch scaler: %s", scaler_path)

    expected_channels = metadata["channels"]

    def transform_fn(channels: np.ndarray, channel_names: list[str]) -> np.ndarray:
        """Selects, batches, and scales channels for single-branch inference."""
        idx = [channel_names.index(ch) for ch in expected_channels]
        X = channels[:, idx][np.newaxis, :, :]  # (1, T, C)
        return scaler_x.transform(X)

    def predict_fn(frame: np.ndarray) -> np.ndarray:
        """Runs a forward pass on a single-branch model."""
        return model.predict(frame, verbose=0)

    return transform_fn, predict_fn


def _build_late_fusion_closures(
    session_dir: Path,
    metadata: dict,
    model: Any,
) -> tuple[Callable, Callable]:
    """
    Builds transform_fn and predict_fn for the Late Fusion Multi-Branch CNN.

    Late fusion models use separate scalers for wrist and finger branches, plus
    an optional scalar feature scaler. ``model.predict()`` receives a dict with
    named inputs matching the Keras Input layer names.

    The ``transform_fn`` closure:
    1. Receives ``(channels, channel_names)`` from the ``AsynchronousDataGrabber``.
    2. Splits channels into wrist and finger subsets using the channel lists from
       ``metadata["wrist_channels"]`` and ``metadata["finger_channels"]``.
    3. Adds batch dimensions: ``(T, C_w) → (1, T, C_w)``, ``(T, C_f) → (1, T, C_f)``.
    4. Applies the respective ``TimeSeriesScaler`` transformations.
    5. Returns a tuple ``(X_wrist_scaled, X_finger_scaled)`` (and optionally
       ``X_feat_scaled`` if the MLP branch is active).

    Parameters
    ----------
    session_dir : Path
        Path to the training session directory containing scaler artifacts.
    metadata : dict
        Parsed model metadata.
    model : keras.Model
        The loaded Keras model.

    Returns
    -------
    tuple[Callable, Callable]
        ``(transform_fn, predict_fn)`` closures.
    """
    wrist_channels = metadata.get("wrist_channels", [])
    finger_channels = metadata.get("finger_channels", [])
    feature_names = metadata.get("feature_names", [])

    # Load scalers
    scaler_wrist = None
    if wrist_channels:
        wrist_path = session_dir / "scaler_x_wrist.joblib"
        if not wrist_path.exists():
            raise FileNotFoundError(f"Wrist scaler not found: {wrist_path}")
        scaler_wrist = joblib.load(wrist_path)
        logger.info("Loaded wrist scaler: %s", wrist_path)

    scaler_finger = None
    if finger_channels:
        finger_path = session_dir / "scaler_x_finger.joblib"
        if not finger_path.exists():
            raise FileNotFoundError(f"Finger scaler not found: {finger_path}")
        scaler_finger = joblib.load(finger_path)
        logger.info("Loaded finger scaler: %s", finger_path)

    scaler_feat = None
    if feature_names:
        feat_path = session_dir / "scaler_feat.joblib"
        if feat_path.exists():
            scaler_feat = joblib.load(feat_path)
            logger.info("Loaded feature scaler: %s", feat_path)

    def transform_fn(channels: np.ndarray, channel_names: list[str]) -> tuple:
        """Splits, batches, and scales channels for late fusion inference."""
        result = []

        if wrist_channels and scaler_wrist is not None:
            wrist_idx = [channel_names.index(ch) for ch in wrist_channels]
            X_wrist = channels[:, wrist_idx][np.newaxis, :, :]
            result.append(scaler_wrist.transform(X_wrist))

        if finger_channels and scaler_finger is not None:
            finger_idx = [channel_names.index(ch) for ch in finger_channels]
            X_finger = channels[:, finger_idx][np.newaxis, :, :]
            result.append(scaler_finger.transform(X_finger))

        # Note: Scalar features (cross-correlation, statistics) are not available
        # from the real-time data grabber's process_window output. The MLP branch
        # will not receive input during live inference unless the feature extraction
        # is added to the real-time preprocessing pipeline. The model builder handles
        # this gracefully by making the MLP branch optional.

        return tuple(result)

    # Build the named input mapping for model.predict()
    input_names = [layer.name for layer in model.inputs]

    def predict_fn(frame: tuple) -> np.ndarray:
        """Runs a forward pass on the late fusion multi-branch model."""
        input_dict = {}
        for i, name in enumerate(input_names):
            if i < len(frame):
                input_dict[name] = frame[i]
        return model.predict(input_dict, verbose=0)

    return transform_fn, predict_fn
