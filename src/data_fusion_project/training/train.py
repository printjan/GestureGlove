# src/data_fusion_project/training/train.py
"""
Core training logic, data scaling, validation splits, and evaluation.
"""

from __future__ import annotations
import json
import time
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
from data_fusion_project.processing import GestureDataset, leave_sessions_out, stratified_split, chronological_split
from data_fusion_project.training.model import build_multi_branch_cnn, parse_channel_indices

logger = get_logger(__name__)


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
    epochs: int = 50,
    batch_size: int = 32,
    output_dir: str | Path | None = None,
    seed: int = 42
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
    n_classes = ds.n_classes
    logger.info("Beginning training pipeline with split_type=%s, target epochs=%d, batch_size=%d", 
                split_type, epochs, batch_size)
    
    # 1. Wrist/Finger branch separation
    wrist_idx, finger_idx = parse_channel_indices(ds.channel_names)
    X_wrist = ds.X[:, :, wrist_idx] if wrist_idx else None
    X_finger = ds.X[:, :, finger_idx] if finger_idx else None
    
    # 2. Train/Test indices split
    if split_type == "leave_session_out":
        train_idx, test_idx = leave_sessions_out(ds.groups, test_fraction=test_fraction, seed=seed)
    elif split_type == "chronological":
        train_idx, test_idx = chronological_split(ds.y, test_fraction=test_fraction)
    else:
        train_idx, test_idx = stratified_split(ds.y, test_fraction=test_fraction, seed=seed)
        
    logger.info("Split result: %d training samples, %d testing samples.", len(train_idx), len(test_idx))

    # 3. Fitting scalers and scaling data
    train_inputs = []
    test_inputs = []
    
    # Wrist scaling
    scaler_wrist = None
    if X_wrist is not None:
        X_wrist_train = X_wrist[train_idx]
        X_wrist_test = X_wrist[test_idx]
        scaler_wrist = TimeSeriesScaler()
        X_wrist_train_scaled = scaler_wrist.fit_transform(X_wrist_train)
        X_wrist_test_scaled = scaler_wrist.transform(X_wrist_test)
        train_inputs.append(X_wrist_train_scaled)
        test_inputs.append(X_wrist_test_scaled)
        
    # Finger scaling
    scaler_finger = None
    if X_finger is not None:
        X_finger_train = X_finger[train_idx]
        X_finger_test = X_finger[test_idx]
        scaler_finger = TimeSeriesScaler()
        X_finger_train_scaled = scaler_finger.fit_transform(X_finger_train)
        X_finger_test_scaled = scaler_finger.transform(X_finger_test)
        train_inputs.append(X_finger_train_scaled)
        test_inputs.append(X_finger_test_scaled)
        
    # Statistical features scaling
    scaler_feat = None
    features_train = None
    features_test = None
    if ds.features is not None and ds.features.size > 0:
        features_train = ds.features[train_idx]
        features_test = ds.features[test_idx]
        scaler_feat = StandardScaler()
        features_train_scaled = scaler_feat.fit_transform(features_train)
        features_test_scaled = scaler_feat.transform(features_test)
        train_inputs.append(features_train_scaled)
        test_inputs.append(features_test_scaled)
        
    # Targets encoding
    y_train = ds.y[train_idx]
    y_test = ds.y[test_idx]
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_test_cat = to_categorical(y_test, num_classes=n_classes)
    
    # 4. Model building & compilation
    shape_wrist = X_wrist.shape[1:] if X_wrist is not None else None
    shape_finger = X_finger.shape[1:] if X_finger is not None else None
    shape_feat = ds.features.shape[1] if ds.features is not None else None
    
    model = build_multi_branch_cnn(
        input_shape_wrist=shape_wrist,
        input_shape_finger=shape_finger,
        num_classes=n_classes,
        input_shape_feat=shape_feat
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
        validation_data=(test_inputs, y_test_cat),
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
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        weights_file = out_path / "model.weights.h5"
        model.save_weights(weights_file)
        logger.info("Saved trained Keras model weights to %s", weights_file)
        
        # Save scalers
        if scaler_wrist:
            joblib.dump(scaler_wrist, out_path / "scaler_x_wrist.joblib")
        if scaler_finger:
            joblib.dump(scaler_finger, out_path / "scaler_x_finger.joblib")
        if scaler_feat:
            joblib.dump(scaler_feat, out_path / "scaler_feat.joblib")
        logger.info("Saved fitted scalers to %s", out_path)
        
        # Generate and save plots
        plot_training_history(history, out_path / "learning_curves.png")
        plot_confusion_matrix_heatmap(y_test, y_pred, ds.class_names, out_path / "confusion_matrix.png")
        
        # Write metadata.json
        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "training_duration_s": training_duration_s,
            "epochs_trained": len(history.epoch),
            "split_type": split_type,
            "test_fraction": test_fraction,
            "seed": seed,
            "classes": ds.class_names,
            "channels": ds.channel_names,
            "wrist_channels": [ds.channel_names[i] for i in wrist_idx] if wrist_idx else [],
            "finger_channels": [ds.channel_names[i] for i in finger_idx] if finger_idx else [],
            "feature_names": ds.feature_names,
            "final_metrics": {
                "train_loss": float(history.history["loss"][-1]),
                "train_acc": float(history.history["accuracy"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "val_acc": float(history.history["val_accuracy"][-1]),
                "test_accuracy": accuracy
            },
            "pipeline_config": ds.config.to_dict() if ds.config else None
        }
        
        with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved session metadata package to %s", out_path / "metadata.json")
        
    return model, history.history, report
