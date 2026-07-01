# src/data_fusion_project/training/model_training_pipeline/pipeline.py
"""
Unified model training, evaluation, and artifact saving pipeline.

This module provides the architecture-agnostic ``train_model()`` function
that trains any of the three candidate gesture classifiers:

- Early Fusion Single-Branch Conv1D CNN
- Late Fusion Multi-Branch Conv1D CNN
- Self-Attention Temporal Transformer

All architecture-specific branching is isolated to model builder dispatch
and input routing. The pipeline handles:

1. Dynamic feature slicing (Optuna toggle integration)
2. Channel index routing (wrist / finger for late fusion)
3. Data splitting (stratified, chronological, balanced leave-session-out)
4. Architecture-dependent scaler fitting and serialization
5. Generator construction (dynamic jittering + rotation augmentation)
6. Model building, compilation, and training
7. Evaluation (classification report, confusion matrix)
8. Artifact saving (model files, scalers, metadata, plots)
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless training
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.core.paths import (
    get_model_run_dir,
    get_model_file,
    get_model_metadata_file,
    get_model_confusion_matrix_file,
    get_model_learning_curves_file,
)
from data_fusion_project.processing import (
    GestureDataset,
    leave_sessions_out_three_way,
    stratified_split_three_way,
    chronological_split_three_way,
)

# Model builders
from data_fusion_project.training.early_fusion_single_branch_1d_cnn.model import (
    build_early_fusion_cnn,
)
from data_fusion_project.training.late_fusion_multi_branch_1d_cnn.model import (
    build_late_fusion_cnn,
    parse_channel_indices,
)
from data_fusion_project.training.self_attention_temporal_transformer.model import (
    build_temporal_transformer,
)

# Augmentation
from data_fusion_project.training.model_training_pipeline.augmentation import (
    apply_rotation_augmentation,
)

logger = get_logger(__name__)


# ======================================================================================================================
# Feature Lists (from data quality audit — see documentation/model_training.md §2)
# ======================================================================================================================
ALL_37_FEATURES = [
    "IMU1_linear_jerkX", "IMU1_linear_jerkZ", "IMU2_linear_jerkZ",
    "IMU1_angular_accelerationY", "IMU1_angular_accelerationZ", "IMU2_angular_accelerationY",
    "IMU1_accX", "IMU1_accZ", "IMU1_gyrX", "IMU1_pitch",
    "IMU2_accX", "IMU2_accY", "IMU2_accZ", "IMU2_gyrX",
    "diff_accX", "diff_accZ", "IMU1_gyr_mag",
    "IMU1_accY", "IMU1_gyrY", "IMU1_gyrZ", "IMU1_acc_mag", "IMU1_roll",
    "IMU1_relative_yaw", "IMU1_linear_jerkY", "IMU1_angular_accelerationX",
    "IMU2_gyrY", "IMU2_gyrZ", "IMU2_gyr_mag", "IMU2_acc_mag", "IMU2_relative_yaw",
    "IMU2_linear_jerkX", "IMU2_linear_jerkY", "IMU2_angular_accelerationX",
    "IMU2_angular_accelerationZ", "diff_accY", "diff_gyrX", "diff_gyrY", "diff_gyrZ",
]


# ======================================================================================================================
# Feature Matching & Slicing
# ======================================================================================================================
def matches_feature(feature_name: str, channel_name: str) -> bool:
    """
    Fuzzy-matches a feature toggle name to a dataset channel name.

    Handles naming convention differences between the feature audit lists
    and the dataset column headers (e.g., ``gyr_mag`` vs ``gyroscope_magnitude``).
    """
    norm_feat = feature_name.lower().replace("_", "")
    norm_chan = channel_name.lower().replace("_", "")
    if norm_feat == norm_chan:
        return True

    replacements = {
        "gyrmag": "gyroscopemagnitude",
        "accmag": "accelerometermagnitude",
        "diffacc": "relativeacceleration",
        "diffgyr": "relativerotation",
    }
    for k, v in replacements.items():
        if norm_feat.replace(k, v) == norm_chan or norm_chan.replace(k, v) == norm_feat:
            return True

    return False


def slice_dataset_channels(
    ds: GestureDataset, feature_toggles: dict[str, bool]
) -> GestureDataset:
    """
    Slices dataset channels based on feature toggle flags.

    Only channels whose corresponding feature toggle is ``True`` are retained.
    The returned dataset has updated ``X`` and ``channel_names``.
    """
    active_features = [feat for feat, val in feature_toggles.items() if val]
    selected_indices = []
    new_channel_names = []

    for feat in active_features:
        for idx, chan in enumerate(ds.channel_names):
            if matches_feature(feat, chan):
                selected_indices.append(idx)
                new_channel_names.append(chan)
                break

    sliced_X = ds.X[:, :, selected_indices]

    return GestureDataset(
        X=sliced_X,
        y=ds.y,
        groups=ds.groups,
        class_names=ds.class_names,
        channel_names=new_channel_names,
        features=ds.features,
        feature_names=ds.feature_names,
        sample_paths=ds.sample_paths,
        config=ds.config,
    )


# ======================================================================================================================
# Time Series Scaler
# ======================================================================================================================
class TimeSeriesScaler:
    """
    StandardScaler wrapper for 3D time-series tensors of shape (N, T, C).

    Computes fit statistics across the combined (N × T) dimension for each
    channel independently, ensuring per-channel zero-mean unit-variance
    normalization.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> TimeSeriesScaler:
        if X is None or X.ndim != 3:
            raise ValueError("X must be a 3D array of shape (N, T, C).")
        N, T, C = X.shape
        self.scaler.fit(X.reshape(-1, C))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X is None:
            return None
        N, T, C = X.shape
        return self.scaler.transform(X.reshape(-1, C)).reshape(N, T, C)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ======================================================================================================================
# Plotting
# ======================================================================================================================
def plot_training_history(history: keras.callbacks.History, save_path: Path):
    """Plots training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history.get("accuracy", []), label="Train Acc", color="dodgerblue", linewidth=2)
    ax1.plot(history.history.get("val_accuracy", []), label="Val Acc", color="orange", linewidth=2)
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(history.history.get("loss", []), label="Train Loss", color="dodgerblue", linewidth=2)
    ax2.plot(history.history.get("val_loss", []), label="Val Loss", color="orange", linewidth=2)
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved training curves to %s", save_path)


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], save_path: Path
):
    """Plots and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    active_labels = [
        i for i in range(len(class_names))
        if np.sum(cm[i, :]) > 0 or np.sum(cm[:, i]) > 0
    ]
    if not active_labels:
        active_labels = list(range(min(2, len(class_names))))

    cm_active = cm[np.ix_(active_labels, active_labels)]
    active_names = [class_names[i] for i in active_labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_active, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_active.shape[1]),
        yticks=np.arange(cm_active.shape[0]),
        xticklabels=active_names,
        yticklabels=active_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_active.max() / 2.0
    for i in range(cm_active.shape[0]):
        for j in range(cm_active.shape[1]):
            ax.text(
                j, i, format(cm_active[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_active[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix to %s", save_path)


# ======================================================================================================================
# Model Identifier Mapping
# ======================================================================================================================
MODEL_TYPE_TO_FOLDER = {
    "early_fusion_cnn": "early_fusion_single_branch_1d_cnn",
    "late_fusion_cnn": "late_fusion_multi_branch_1d_cnn",
    "temporal_transformer": "slef_attention_temporal_transformer",
}


# ======================================================================================================================
# Unified Training Pipeline
# ======================================================================================================================
def train_model(
    ds: GestureDataset,
    model_type: str = "early_fusion_cnn",
    split_type: str = "leave-session-out",
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    epochs: int = 70,
    batch_size: int = 32,
    model_name: str | None = None,
    timestamp: str | None = None,
    seed: int = 42,
    augment_rotation: float = 0.0,
    feature_toggles: dict[str, bool] | None = None,
    # CNN-specific
    conv_filters: list[int] | None = None,
    dense_units: int = 16,
    # Transformer-specific
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    ff_dim: int = 128,
) -> tuple[keras.Model, dict, dict]:
    """
    Unified training, evaluation, and artifact saving pipeline.

    Supports all three candidate architectures via ``model_type``:
    - ``"early_fusion_cnn"``
    - ``"late_fusion_cnn"``
    - ``"temporal_transformer"``

    :param ds: Loaded GestureDataset (from ``load_dataset()``).
    :param model_type: Architecture identifier string.
    :param split_type: Split strategy: ``"stratified"``, ``"chronological"``,
        or ``"leave-session-out"``.
    :param test_fraction: Fraction of data reserved for testing.
    :param val_fraction: Fraction of data reserved for validation.
    :param epochs: Maximum training epochs (EarlyStopping may halt earlier).
    :param batch_size: Training batch size.
    :param model_name: Model identifier for the output directory. If None,
        derived from ``model_type``.
    :param timestamp: Timestamp string for the training session folder.
        If None, no artifacts are saved.
    :param seed: Random seed for reproducible splits.
    :param augment_rotation: Max 3D rotation angle in degrees (0 = disabled).
    :param feature_toggles: Dict mapping feature names to bool flags for
        dynamic channel slicing. None = use all loaded channels.
    :param conv_filters: Conv1D filter list for CNN architectures.
    :param dense_units: Classification head dense units (default: 16).
    :param d_model: Transformer projection dimensionality.
    :param num_heads: Transformer attention heads.
    :param num_blocks: Stacked transformer encoder blocks.
    :param ff_dim: Transformer feed-forward expansion dimensionality.
    :return: Tuple of (trained_model, history_dict, evaluation_report).
    """
    if conv_filters is None:
        conv_filters = [32, 64]

    # ──────────────────────────────────────────────────────────────────────
    # 1. Feature Slicing
    # ──────────────────────────────────────────────────────────────────────
    if feature_toggles is not None:
        ds = slice_dataset_channels(ds, feature_toggles)
        logger.info("Sliced dataset to active feature toggles. New X shape: %s", ds.X.shape)
    else:
        feature_toggles = {
            feat: any(matches_feature(feat, chan) for chan in ds.channel_names)
            for feat in ALL_37_FEATURES
        }

    n_classes = ds.n_classes
    logger.info(
        "Training pipeline: model_type=%s, split=%s, epochs=%d, batch=%d",
        model_type, split_type, epochs, batch_size,
    )

    # ──────────────────────────────────────────────────────────────────────
    # 2. Channel Index Routing (Late Fusion Only)
    # ──────────────────────────────────────────────────────────────────────
    wrist_idx, finger_idx = [], []
    if model_type == "late_fusion_cnn":
        wrist_idx, finger_idx = parse_channel_indices(ds.channel_names)
        logger.info("Late fusion routing: %d wrist, %d finger channels", len(wrist_idx), len(finger_idx))

    # ──────────────────────────────────────────────────────────────────────
    # 3. Data Splitting
    # ──────────────────────────────────────────────────────────────────────
    if split_type in ("leave_session_out", "leave-session-out"):
        # Detect balanced manual test/val sessions (V4 dataset)
        has_manual = (
            any("test_data" in str(g) for g in ds.groups)
            and any("validation_data" in str(g) for g in ds.groups)
        )
        if has_manual:
            logger.info("Detected balanced manual test_data/validation_data sessions.")
            test_mask = np.array(["test_data" in str(g) for g in ds.groups])
            val_mask = np.array(["validation_data" in str(g) for g in ds.groups])
            test_idx = np.where(test_mask)[0]
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(~test_mask & ~val_mask)[0]
        else:
            train_idx, val_idx, test_idx = leave_sessions_out_three_way(
                ds.groups, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
            )
    elif split_type == "chronological":
        train_idx, val_idx, test_idx = chronological_split_three_way(
            ds.y, test_fraction=test_fraction, val_fraction=val_fraction
        )
    else:
        train_idx, val_idx, test_idx = stratified_split_three_way(
            ds.y, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
        )

    logger.info(
        "Split: %d train, %d val, %d test samples.",
        len(train_idx), len(val_idx), len(test_idx),
    )

    # ──────────────────────────────────────────────────────────────────────
    # 4. Scaling (Architecture-Dependent Routing)
    # ──────────────────────────────────────────────────────────────────────
    y_train = ds.y[train_idx]
    y_val = ds.y[val_idx]
    y_test = ds.y[test_idx]
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_val_cat = to_categorical(y_val, num_classes=n_classes)
    y_test_cat = to_categorical(y_test, num_classes=n_classes)

    scaler_wrist = None
    scaler_finger = None
    scaler_x = None
    scaler_feat = None

    if model_type == "late_fusion_cnn":
        # Independent scalers for wrist and finger branches
        X_wrist = ds.X[:, :, wrist_idx] if wrist_idx else None
        X_finger = ds.X[:, :, finger_idx] if finger_idx else None

        train_inputs, val_inputs, test_inputs = [], [], []

        if X_wrist is not None:
            X_wrist_train = X_wrist[train_idx]
            if augment_rotation > 0.0:
                wrist_ch = [ds.channel_names[i] for i in wrist_idx]
                X_wrist_train = apply_rotation_augmentation(X_wrist_train, wrist_ch, augment_rotation)
            scaler_wrist = TimeSeriesScaler()
            train_inputs.append(scaler_wrist.fit_transform(X_wrist_train))
            val_inputs.append(scaler_wrist.transform(X_wrist[val_idx]))
            test_inputs.append(scaler_wrist.transform(X_wrist[test_idx]))

        if X_finger is not None:
            X_finger_train = X_finger[train_idx]
            if augment_rotation > 0.0:
                finger_ch = [ds.channel_names[i] for i in finger_idx]
                X_finger_train = apply_rotation_augmentation(X_finger_train, finger_ch, augment_rotation)
            scaler_finger = TimeSeriesScaler()
            train_inputs.append(scaler_finger.fit_transform(X_finger_train))
            val_inputs.append(scaler_finger.transform(X_finger[val_idx]))
            test_inputs.append(scaler_finger.transform(X_finger[test_idx]))

        # Always compute scalar features for late fusion MLP branch
        if ds.features is not None and ds.features.size > 0:
            scaler_feat = StandardScaler()
            train_inputs.append(scaler_feat.fit_transform(ds.features[train_idx]))
            val_inputs.append(scaler_feat.transform(ds.features[val_idx]))
            test_inputs.append(scaler_feat.transform(ds.features[test_idx]))

    else:
        # Single scaler for early fusion / transformer (all channels concatenated)
        X_train = ds.X[train_idx]
        if augment_rotation > 0.0:
            X_train = apply_rotation_augmentation(X_train, ds.channel_names, augment_rotation)
        scaler_x = TimeSeriesScaler()
        X_train_scaled = scaler_x.fit_transform(X_train)
        X_val_scaled = scaler_x.transform(ds.X[val_idx])
        X_test_scaled = scaler_x.transform(ds.X[test_idx])

        train_inputs = X_train_scaled
        val_inputs = X_val_scaled
        test_inputs = X_test_scaled

    # ──────────────────────────────────────────────────────────────────────
    # 5. Model Building
    # ──────────────────────────────────────────────────────────────────────
    if model_type == "early_fusion_cnn":
        model = build_early_fusion_cnn(
            input_shape=train_inputs.shape[1:],
            num_classes=n_classes,
            conv_filters=conv_filters,
            dense_units=dense_units,
        )
    elif model_type == "late_fusion_cnn":
        shape_wrist = train_inputs[0].shape[1:] if train_inputs else None
        shape_finger = train_inputs[1].shape[1:] if len(train_inputs) > 1 else None
        shape_feat = ds.features.shape[1] if (ds.features is not None and ds.features.size > 0) else None
        model = build_late_fusion_cnn(
            input_shape_wrist=shape_wrist,
            input_shape_finger=shape_finger,
            num_classes=n_classes,
            input_shape_feat=shape_feat,
            conv_filters=conv_filters,
            dense_units=dense_units,
        )
    elif model_type == "temporal_transformer":
        model = build_temporal_transformer(
            input_shape=train_inputs.shape[1:],
            num_classes=n_classes,
            d_model=d_model,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
            dense_units=dense_units,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         f"Choose from: early_fusion_cnn, late_fusion_cnn, temporal_transformer")

    # ──────────────────────────────────────────────────────────────────────
    # 6. Compilation
    # ──────────────────────────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ──────────────────────────────────────────────────────────────────────
    # 7. Callbacks
    # ──────────────────────────────────────────────────────────────────────
    callbacks_list = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-6, verbose=1),
    ]

    # ──────────────────────────────────────────────────────────────────────
    # 8. Training
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Starting model.fit for %s...", model_type)
    start_time = time.time()
    history = model.fit(
        train_inputs,
        y_train_cat,
        validation_data=(val_inputs, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1,
    )
    training_duration_s = time.time() - start_time
    logger.info("Training complete in %.2f seconds.", training_duration_s)

    # ──────────────────────────────────────────────────────────────────────
    # 9. Evaluation
    # ──────────────────────────────────────────────────────────────────────
    preds = model.predict(test_inputs)
    y_pred = np.argmax(preds, axis=1)

    report = classification_report(
        y_test, y_pred,
        labels=range(n_classes),
        target_names=ds.class_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(np.mean(y_pred == y_test))
    logger.info("Test Accuracy: %.4f", accuracy)

    # ──────────────────────────────────────────────────────────────────────
    # 10. Artifact Saving
    # ──────────────────────────────────────────────────────────────────────
    if model_name and timestamp:
        run_dir = get_model_run_dir(model_name, timestamp)

        # Model weights (always saved, backend-safe)
        weights_file = run_dir / "model.weights.h5"
        model.save_weights(weights_file)
        logger.info("Saved model weights to %s", weights_file)

        # Full Keras model (PyTorch backend workaround for macOS segfault)
        model_file = get_model_file(model_name, timestamp)
        if os.environ.get("KERAS_BACKEND") == "torch":
            model_file.touch()
            logger.warning(
                "Skipping full Keras model serialization on PyTorch backend. "
                "Weights preserved in model.weights.h5."
            )
        else:
            try:
                model.save(model_file)
                logger.info("Saved Keras model to %s", model_file)
            except Exception as e:
                model_file.touch()
                logger.warning("Failed to save full Keras model: %s. Weights preserved.", e)

        # Scalers
        if scaler_x:
            joblib.dump(scaler_x, run_dir / "scaler_x.joblib")
        if scaler_wrist:
            joblib.dump(scaler_wrist, run_dir / "scaler_x_wrist.joblib")
        if scaler_finger:
            joblib.dump(scaler_finger, run_dir / "scaler_x_finger.joblib")
        if scaler_feat:
            joblib.dump(scaler_feat, run_dir / "scaler_feat.joblib")
        logger.info("Saved scalers to %s", run_dir)

        # Plots
        curves_file = get_model_learning_curves_file(model_name, timestamp)
        cm_file = get_model_confusion_matrix_file(model_name, timestamp)
        plot_training_history(history, curves_file)
        plot_confusion_matrix_heatmap(y_test, y_pred, ds.class_names, cm_file)

        # Metadata (README.md-compliant schema)
        best_epoch = int(np.argmin(history.history["val_loss"]))
        sessions_used = sorted(list(set(ds.groups.tolist())))

        # Layer-by-layer model structure
        def format_shape(shape):
            if shape is None:
                return None
            if isinstance(shape, list):
                return [format_shape(s) for s in shape]
            if isinstance(shape, tuple):
                return [int(dim) if dim is not None else None for dim in shape]
            return str(shape)

        model_layers_info = []
        for layer in model.layers:
            try:
                out_shape = layer.output_shape
            except AttributeError:
                try:
                    out_shape = layer.input_shape
                except AttributeError:
                    out_shape = None
            model_layers_info.append({
                "layer_name": layer.name,
                "class_name": layer.__class__.__name__,
                "output_shape": format_shape(out_shape),
                "parameter_count": int(layer.count_params()),
            })

        model_structure = {
            "total_parameters": int(model.count_params()),
            "layers": model_layers_info,
        }

        total_samples = len(train_idx) + len(val_idx) + len(test_idx)

        # Build training_parameters dict (architecture-specific fields)
        training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "split_type": split_type,
            "test_fraction": test_fraction,
            "val_fraction": val_fraction,
            "seed": seed,
            "augment_rotation": augment_rotation,
            "jitter_range": ds.config.jitter_range if ds.config else 0,
            "sessions_used": [str(s) for s in sessions_used],
            "conv_filters": conv_filters,
            "dense_units": dense_units,
        }
        # Add transformer-specific params when applicable
        if model_type == "temporal_transformer":
            training_params.update({
                "d_model": d_model,
                "num_heads": num_heads,
                "num_blocks": num_blocks,
                "ff_dim": ff_dim,
                "attention_dropout": 0.1,
            })

        metadata = {
            "timestamp": timestamp,
            "model_name": model_name,
            "model_type": model_type,
            "training_duration_s": training_duration_s,
            "epochs_trained": len(history.epoch),
            "early_stopped": bool(len(history.epoch) < epochs),
            "classes": ds.class_names,
            "channels": ds.channel_names,
            "wrist_channels": (
                [ds.channel_names[i] for i in wrist_idx] if wrist_idx else []
            ),
            "finger_channels": (
                [ds.channel_names[i] for i in finger_idx] if finger_idx else []
            ),
            "feature_names": ds.feature_names,
            "feature_toggles": feature_toggles,
            "features_selection": {
                "default_selected_features": [f for f, v in feature_toggles.items() if v],
                "default_deselected_features": [f for f, v in feature_toggles.items() if not v],
            },
            "model_structure": model_structure,
            "machine_info": {
                "hostname": platform.node(),
                "os": f"{platform.system()}-{platform.release()}",
                "cpu": platform.processor(),
                "gpu": "MPS" if platform.system() == "Darwin" else "CUDA",
                "ram_gb": 64.0,
            },
            "training_parameters": training_params,
            "split_info": {
                "strategy": split_type,
                "total_samples": total_samples,
                "train_size_abs": len(train_idx),
                "val_size_abs": len(val_idx),
                "test_size_abs": len(test_idx),
                "train_fraction_real": float(len(train_idx)) / max(1, total_samples),
                "val_fraction_real": float(len(val_idx)) / max(1, total_samples),
                "test_fraction_real": float(len(test_idx)) / max(1, total_samples),
                "train_sessions": sorted(list(set(ds.groups[train_idx].tolist()))),
                "val_sessions": sorted(list(set(ds.groups[val_idx].tolist()))),
                "test_sessions": sorted(list(set(ds.groups[test_idx].tolist()))),
            },
            "performance": {
                "best_epoch": best_epoch + 1,
                "train_accuracy": float(history.history["accuracy"][best_epoch]),
                "train_loss": float(history.history["loss"][best_epoch]),
                "val_accuracy": float(history.history["val_accuracy"][best_epoch]),
                "val_loss": float(history.history["val_loss"][best_epoch]),
                "val_f1_score": float(report["macro avg"]["f1-score"]),
            },
            "evaluation": {
                "accuracy": accuracy,
                "macro_avg": report["macro avg"],
                "weighted_avg": report["weighted avg"],
                "per_class_metrics": {
                    label: report[label] for label in ds.class_names if label in report
                },
            },
            "pipeline_config": ds.config.to_dict() if ds.config else None,
        }

        metadata_file = get_model_metadata_file(model_name, timestamp)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata to %s", metadata_file)

    return model, history.history, report
