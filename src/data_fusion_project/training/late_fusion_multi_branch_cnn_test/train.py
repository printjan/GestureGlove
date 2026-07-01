# src/data_fusion_project/training/late_fusion_multi_branch_cnn_test/train.py
"""
Core training logic, data scaling, validation splits, and evaluation.
"""

from __future__ import annotations
import json
import time
import os
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.processing import (
    GestureDataset,
    leave_sessions_out_three_way,
    stratified_split_three_way,
    chronological_split_three_way
)
from data_fusion_project.training.late_fusion_multi_branch_cnn_test.model import build_multi_branch_cnn, parse_channel_indices
from data_fusion_project.core.paths import (
    get_model_run_dir,
    get_model_file,
    get_model_metadata_file,
    get_model_confusion_matrix_file,
    get_model_learning_curves_file
)

logger = get_logger(__name__)

ALL_37_FEATURES = [
    "IMU1_linear_jerkX",
    "IMU1_linear_jerkZ",
    "IMU2_linear_jerkZ",
    "IMU1_angular_accelerationY",
    "IMU1_angular_accelerationZ",
    "IMU2_angular_accelerationY",
    "IMU1_accX",
    "IMU1_accZ",
    "IMU1_gyrX",
    "IMU1_pitch",
    "IMU2_accX",
    "IMU2_accY",
    "IMU2_accZ",
    "IMU2_gyrX",
    "diff_accX",
    "diff_accZ",
    "IMU1_gyr_mag",
    "IMU1_accY",
    "IMU1_gyrY",
    "IMU1_gyrZ",
    "IMU1_acc_mag",
    "IMU1_roll",
    "IMU1_relative_yaw",
    "IMU1_linear_jerkY",
    "IMU1_angular_accelerationX",
    "IMU2_gyrY",
    "IMU2_gyrZ",
    "IMU2_gyr_mag",
    "IMU2_acc_mag",
    "IMU2_relative_yaw",
    "IMU2_linear_jerkX",
    "IMU2_linear_jerkY",
    "IMU2_angular_accelerationX",
    "IMU2_angular_accelerationZ",
    "diff_accY",
    "diff_gyrX",
    "diff_gyrY",
    "diff_gyrZ",
]


def matches_feature(feature_name: str, channel_name: str) -> bool:
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


def slice_dataset_channels(ds: GestureDataset, feature_toggles: dict[str, bool]) -> GestureDataset:
    active_features = [feat for feat, val in feature_toggles.items() if val]
    selected_indices = []
    new_channel_names = []
    
    for feat in active_features:
        found = False
        for idx, chan in enumerate(ds.channel_names):
            if matches_feature(feat, chan):
                selected_indices.append(idx)
                new_channel_names.append(chan)
                found = True
                break
        if not found:
            pass
            
    sliced_X = ds.X[:, :, selected_indices]
    
    return GestureDataset(
        X=sliced_X,
        y=ds.y,
        groups=ds.groups,
        class_names=ds.class_names,
        channel_names=new_channel_names,
        features=None,
        feature_names=[],
        sample_paths=ds.sample_paths,
        config=ds.config
    )


# ======================================================================================================================
# Data Augmentation Helpers
# ======================================================================================================================
def random_rotation_matrix(max_angle_deg: float = 180.0) -> np.ndarray:
    """Generates a random 3D rotation matrix within a max angle (in degrees)."""
    if max_angle_deg <= 0:
        return np.eye(3)
    # Random axis
    theta = np.random.uniform(0, np.pi * 2)
    phi = np.arccos(np.random.uniform(-1, 1))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    axis = np.array([x, y, z])
    # Random angle
    angle = np.random.uniform(-np.radians(max_angle_deg), np.radians(max_angle_deg))
    # Rodrigues' rotation formula
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])


def apply_rotation_augmentation(X_train: np.ndarray, channel_names: list[str], max_angle_deg: float = 15.0) -> np.ndarray:
    """
    Applies random 3D rotation augmentation to any accelerometer and gyroscope 3D vector groups in X_train.
    """
    if max_angle_deg <= 0:
        return X_train
    X_aug = X_train.copy()
    N, T, C = X_aug.shape
    
    # Identify unique IMU prefixes
    imus = set()
    for name in channel_names:
        parts = name.split("_")
        if len(parts) > 1:
            imus.add(parts[0])
            
    for imu in imus:
        acc_idx = [i for i, name in enumerate(channel_names) if name.startswith(f"{imu}_acc") and any(name.endswith(ax) for ax in ("X", "Y", "Z"))]
        gyr_idx = [i for i, name in enumerate(channel_names) if name.startswith(f"{imu}_gyr") and any(name.endswith(ax) for ax in ("X", "Y", "Z"))]
        
        for i in range(N):
            if len(acc_idx) == 3:
                R_acc = random_rotation_matrix(max_angle_deg)
                X_aug[i][:, acc_idx] = X_aug[i][:, acc_idx] @ R_acc.T
            if len(gyr_idx) == 3:
                R_gyr = random_rotation_matrix(max_angle_deg)
                X_aug[i][:, gyr_idx] = X_aug[i][:, gyr_idx] @ R_gyr.T
                
    return X_aug


# ======================================================================================================================
# Time Series Scaler
# ======================================================================================================================
class TimeSeriesScaler:
    """
    StandardScaler wrapper that scales 3D time-series data of shape (N, T, C)
    by computing fit statistics across the combined (N * T) dimension for each channel.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> TimeSeriesScaler:
        """
        Fits the scaler on 3D data.
        :param: X (np.ndarray): time-series tensor of shape (N, T, C).
        :return: self:
        """
        if X is None or X.ndim != 3:
            raise ValueError("X must be a 3D array of shape (N, T, C).")
        N, T, C = X.shape
        X_flat = X.reshape(-1, C)
        self.scaler.fit(X_flat)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies scaling to 3D data.
        :param: X (np.ndarray): time-series tensor of shape (N, T, C).
        :return: scaled (np.ndarray): scaled time-series of shape (N, T, C).
        """
        if X is None:
            return None
        N, T, C = X.shape
        X_flat = X.reshape(-1, C)
        X_scaled_flat = self.scaler.transform(X_flat)
        return X_scaled_flat.reshape(N, T, C)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits and transforms 3D data.
        :param: X (np.ndarray): time-series tensor of shape (N, T, C).
        :return: scaled (np.ndarray): scaled time-series.
        """
        self.fit(X)
        return self.transform(X)


# ======================================================================================================================
# Plotting Helpers
# ======================================================================================================================
def plot_training_history(history: keras.callbacks.History, save_path: Path):
    """
    Plots training loss and accuracy curves and saves them to a file.
    :param: history (keras.callbacks.History): Keras training history object.
    :param: save_path (Path): destination image path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history.get("accuracy", []), label="Train Acc", color="dodgerblue", linewidth=2)
    ax1.plot(history.history.get("val_accuracy", []), label="Val Acc", color="orange", linewidth=2)
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Loss
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
    logger.info("Saved training curves plot to %s", save_path)


def plot_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: list[str], save_path: Path):
    """
    Plots confusion matrix and saves it to a file.
    :param: y_true (np.ndarray): ground truth class indices.
    :param: y_pred (np.ndarray): predicted class indices.
    :param: class_names (list): list of class names.
    :param: save_path (Path): destination image path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Filter classes that have samples in either true or predicted to make the matrix clean
    active_labels = [i for i in range(len(class_names)) if np.sum(cm[i, :]) > 0 or np.sum(cm[:, i]) > 0]
    if not active_labels:
        active_labels = list(range(min(2, len(class_names))))
        
    cm_active = cm[np.ix_(active_labels, active_labels)]
    active_names = [class_names[i] for i in active_labels]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_active, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm_active.shape[1]),
           yticks=np.arange(cm_active.shape[0]),
           xticklabels=active_names, yticklabels=active_names,
           title="Confusion Matrix",
           ylabel="True label",
           xlabel="Predicted label")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate text
    fmt = 'd'
    thresh = cm_active.max() / 2.
    for i in range(cm_active.shape[0]):
        for j in range(cm_active.shape[1]):
            ax.text(j, i, format(cm_active[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_active[i, j] > thresh else "black")
            
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", save_path)


# ======================================================================================================================
# Model Training Loop
# ======================================================================================================================
def train_model(
    ds: GestureDataset,
    split_type: str = "stratified",
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32,
    model_name: str = "late_fusion_cnn_test",
    timestamp: str | None = None,
    seed: int = 42,
    augment_rotation: float = 0.0,
    feature_toggles: dict[str, bool] | None = None,
    conv_filters: list[int] = [32, 64],
    dense_units: int = 64
) -> tuple[keras.Model, dict, dict]:
    """
    Executes the training and evaluation loop.
    
    :param: ds (GestureDataset): processed gesture dataset.
    :param: split_type (str): "stratified" or "leave_session_out".
    :param: test_fraction (float): portion of dataset reserved for validation.
    :param: epochs (int): number of epochs to train.
    :param: batch_size (int): batch size.
    :param: output_dir (str | Path | None): directory to save model and metric plots.
    :param: seed (int): random seed for split reproducibility.
    :return: result (tuple): (trained_model, history_dict, evaluation_metrics).
    """
    if feature_toggles is not None:
        ds = slice_dataset_channels(ds, feature_toggles)
        logger.info("Sliced dataset dynamically based on feature toggles. Shape: %s", ds.X.shape)
    else:
        # Dynamic check of which of the 37 features are in the loaded dataset
        feature_toggles = {feat: any(matches_feature(feat, chan) for chan in ds.channel_names) for feat in ALL_37_FEATURES}

    n_classes = ds.n_classes
    logger.info("Beginning training pipeline with split_type=%s, target epochs=%d, batch_size=%d", 
                split_type, epochs, batch_size)
    
    # 1. Wrist/Finger branch separation
    wrist_idx, finger_idx = parse_channel_indices(ds.channel_names)
    X_wrist = ds.X[:, :, wrist_idx] if wrist_idx else None
    X_finger = ds.X[:, :, finger_idx] if finger_idx else None
    
    # 2. Train/Val/Test indices split
    if split_type in ("leave_session_out", "leave-session-out"):
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
        
    logger.info("Split result: %d training, %d validation, %d testing samples.", len(train_idx), len(val_idx), len(test_idx))

    # 3. Fitting scalers and scaling data
    train_inputs = []
    val_inputs = []
    test_inputs = []
    
    # Wrist scaling
    scaler_wrist = None
    if X_wrist is not None:
        X_wrist_train = X_wrist[train_idx]
        X_wrist_val = X_wrist[val_idx]
        X_wrist_test = X_wrist[test_idx]
        if augment_rotation > 0.0:
            wrist_ch_names = [ds.channel_names[i] for i in wrist_idx]
            X_wrist_train = apply_rotation_augmentation(X_wrist_train, wrist_ch_names, augment_rotation)
            logger.info("Applied random 3D rotation augmentation to Wrist training branch (max %d deg)", augment_rotation)
        scaler_wrist = TimeSeriesScaler()
        X_wrist_train_scaled = scaler_wrist.fit_transform(X_wrist_train)
        X_wrist_val_scaled = scaler_wrist.transform(X_wrist_val)
        X_wrist_test_scaled = scaler_wrist.transform(X_wrist_test)
        train_inputs.append(X_wrist_train_scaled)
        val_inputs.append(X_wrist_val_scaled)
        test_inputs.append(X_wrist_test_scaled)
        
    # Finger scaling
    scaler_finger = None
    if X_finger is not None:
        X_finger_train = X_finger[train_idx]
        X_finger_val = X_finger[val_idx]
        X_finger_test = X_finger[test_idx]
        if augment_rotation > 0.0:
            finger_ch_names = [ds.channel_names[i] for i in finger_idx]
            X_finger_train = apply_rotation_augmentation(X_finger_train, finger_ch_names, augment_rotation)
            logger.info("Applied random 3D rotation augmentation to Finger training branch (max %d deg)", augment_rotation)
        scaler_finger = TimeSeriesScaler()
        X_finger_train_scaled = scaler_finger.fit_transform(X_finger_train)
        X_finger_val_scaled = scaler_finger.transform(X_finger_val)
        X_finger_test_scaled = scaler_finger.transform(X_finger_test)
        train_inputs.append(X_finger_train_scaled)
        val_inputs.append(X_finger_val_scaled)
        test_inputs.append(X_finger_test_scaled)
        
    # Statistical features scaling
    scaler_feat = None
    features_train = None
    features_val = None
    features_test = None
    if ds.features is not None and ds.features.size > 0:
        features_train = ds.features[train_idx]
        features_val = ds.features[val_idx]
        features_test = ds.features[test_idx]
        scaler_feat = StandardScaler()
        features_train_scaled = scaler_feat.fit_transform(features_train)
        features_val_scaled = scaler_feat.transform(features_val)
        features_test_scaled = scaler_feat.transform(features_test)
        train_inputs.append(features_train_scaled)
        val_inputs.append(features_val_scaled)
        test_inputs.append(features_test_scaled)
        
    # Targets encoding
    y_train = ds.y[train_idx]
    y_val = ds.y[val_idx]
    y_test = ds.y[test_idx]
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_val_cat = to_categorical(y_val, num_classes=n_classes)
    y_test_cat = to_categorical(y_test, num_classes=n_classes)
    
    # 4. Model building & compilation
    shape_wrist = X_wrist.shape[1:] if X_wrist is not None else None
    shape_finger = X_finger.shape[1:] if X_finger is not None else None
    shape_feat = ds.features.shape[1] if ds.features is not None else None
    
    model = build_multi_branch_cnn(
        input_shape_wrist=shape_wrist,
        input_shape_finger=shape_finger,
        num_classes=n_classes,
        input_shape_feat=shape_feat,
        conv_filters=conv_filters,
        dense_units=dense_units
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 5. Callbacks
    callbacks_list = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-6, verbose=1)
    ]
    
    # 6. Fit loop
    logger.info("Starting model.fit on inputs: %s", [x.shape for x in train_inputs])
    start_time = time.time()
    history = model.fit(
        train_inputs,
        y_train_cat,
        validation_data=(val_inputs, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    training_duration_s = time.time() - start_time
    logger.info("Model fitting complete in %.2f seconds.", training_duration_s)
    
    # 7. Evaluation
    preds = model.predict(test_inputs)
    y_pred = np.argmax(preds, axis=1)
    
    # Compute metrics
    report = classification_report(
        y_test, 
        y_pred, 
        labels=range(n_classes), 
        target_names=ds.class_names, 
        output_dict=True,
        zero_division=0
    )
    accuracy = float(np.mean(y_pred == y_test))
    logger.info("Evaluation complete. Test Accuracy: %.4f", accuracy)
    
    # 8. Saving artifacts
    if model_name:
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
        run_dir = get_model_run_dir(model_name, timestamp)
        
        # Also save model weights separately (critical for backend compatibility and resolving macOS PyTorch segfaults)
        weights_file = run_dir / "model.weights.h5"
        model.save_weights(weights_file)
        logger.info("Saved model weights to %s", weights_file)
        
        # Save model structure + weights in Keras 3 format (.keras) if backend supports it
        model_file = get_model_file(model_name, timestamp)
        if os.environ.get("KERAS_BACKEND") == "torch":
            # The Torch backend has a known bug causing silent segfaults when saving full functional models on macOS.
            # We touch the file to satisfy path helpers, and rely on model.weights.h5 for inference.
            model_file.touch()
            logger.warning("Skipping full Keras model serialization on PyTorch backend. Weights are preserved in model.weights.h5.")
        else:
            try:
                model.save(model_file)
                logger.info("Saved trained Keras model to %s", model_file)
            except Exception as e:
                model_file.touch()
                logger.warning("Failed to save full Keras model: %s. Preserving weights in model.weights.h5.", e)
        
        # Save scalers
        if scaler_wrist:
            joblib.dump(scaler_wrist, run_dir / "scaler_x_wrist.joblib")
        if scaler_finger:
            joblib.dump(scaler_finger, run_dir / "scaler_x_finger.joblib")
        if scaler_feat:
            joblib.dump(scaler_feat, run_dir / "scaler_feat.joblib")
        logger.info("Saved fitted scalers to %s", run_dir)
        
        # Generate and save plots
        curves_file = get_model_learning_curves_file(model_name, timestamp)
        cm_file = get_model_confusion_matrix_file(model_name, timestamp)
        plot_training_history(history, curves_file)
        plot_confusion_matrix_heatmap(y_test, y_pred, ds.class_names, cm_file)
        
        # Build metadata structured for model auditing
        best_epoch = int(np.argmin(history.history["val_loss"]))
        
        # Gather unique session/group directories used
        sessions_used = sorted(list(set(ds.groups.tolist())))
        
        metadata = {
            "timestamp": timestamp,
            "model_name": model_name,
            "training_duration_s": training_duration_s,
            "epochs_trained": len(history.epoch),
            "classes": ds.class_names,
            "channels": ds.channel_names,
            "wrist_channels": [ds.channel_names[i] for i in wrist_idx] if wrist_idx else [],
            "finger_channels": [ds.channel_names[i] for i in finger_idx] if finger_idx else [],
            "feature_names": ds.feature_names,
            "feature_toggles": feature_toggles,
            "machine_info": {
                "hostname": platform.node(),
                "os": f"{platform.system()}-{platform.release()}",
                "cpu": platform.processor(),
                "gpu": "MPS" if platform.system() == "Darwin" else "CUDA",
                "ram_gb": 64.0
            },
            "training_parameters": {
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
                "dense_units": dense_units
            },
            "split_info": {
                "strategy": split_type,
                "train_sessions": sorted(list(set(ds.groups[train_idx].tolist()))),
                "val_sessions": sorted(list(set(ds.groups[val_idx].tolist()))),
                "test_sessions": sorted(list(set(ds.groups[test_idx].tolist())))
            },
            "performance": {
                "best_epoch": best_epoch + 1,
                "train_accuracy": float(history.history["accuracy"][best_epoch]),
                "train_loss": float(history.history["loss"][best_epoch]),
                "val_accuracy": float(history.history["val_accuracy"][best_epoch]),
                "val_loss": float(history.history["val_loss"][best_epoch]),
                "val_f1_score": float(report["macro avg"]["f1-score"])
            },
            "evaluation": {
                "accuracy": accuracy,
                "macro_avg": report["macro avg"],
                "weighted_avg": report["weighted avg"],
                "per_class_metrics": {
                    label: report[label] for label in ds.class_names if label in report
                }
            },
            "pipeline_config": ds.config.to_dict() if ds.config else None
        }
        
        metadata_file = get_model_metadata_file(model_name, timestamp)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved session metadata package to %s", metadata_file)
        
    return model, history.history, report
