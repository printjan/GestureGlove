# src/data_fusion_project/training/model.py
"""
Multi-Branch Conv1D CNN model definition using Keras.

Implements the Late Fusion Multi-Branch Conv1D CNN architecture:
- Wrist Conv1D Branch: filters temporal patterns on the wrist worn IMU.
- Finger Conv1D Branch: filters temporal patterns on the finger worn IMU and/or deltas.
- Dense MLP Branch: processes statistical/handcrafted features (optional).
- Concatenate & FC Classifier: fuses outputs and classifies into gesture labels.
"""

from __future__ import annotations
import keras
from keras import layers, Model, regularizers
import numpy as np

from data_fusion_project.core.logger_setup import get_logger

logger = get_logger(__name__)


def parse_channel_indices(channel_names: list[str]) -> tuple[list[int], list[int]]:
    """
    Parses channel names to separate wrist and finger IMU channel indices.
    
    Rules:
    - Wrist channels: contains 'IMU1' or 'wrist' (case-insensitive).
    - Finger channels: contains 'IMU2', 'finger', or 'diff' (case-insensitive).
    
    :param: channel_names (list): list of C channel names in GestureDataset.
    :return: indices (tuple): (wrist_indices, finger_indices).
    """
    wrist_idx = []
    finger_idx = []
    for idx, name in enumerate(channel_names):
        name_lower = name.lower()
        if "imu1" in name_lower or "wrist" in name_lower:
            wrist_idx.append(idx)
        elif "imu2" in name_lower or "finger" in name_lower or "diff" in name_lower:
            finger_idx.append(idx)
        else:
            # Fallback default (assume wrist)
            logger.warning("Unrecognized channel name '%s'. Grouping under wrist branch by default.", name)
            wrist_idx.append(idx)
    return wrist_idx, finger_idx


def _build_conv1d_branch(input_tensor: layers.Input, filters: list[int], branch_name: str) -> layers.Layer:
    """
    Constructs a Conv1D filtering branch dynamically.
    """
    x = input_tensor
    for i, f in enumerate(filters):
        kernel_size = 5 if i == 0 else 3
        x = layers.Conv1D(
            filters=f,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{branch_name}_conv{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"{branch_name}_bn{i+1}")(x)
        x = layers.ReLU(name=f"{branch_name}_relu{i+1}")(x)
        
        if i < len(filters) - 1:
            x = layers.MaxPooling1D(pool_size=2, name=f"{branch_name}_pool{i+1}")(x)
        else:
            x = layers.GlobalAveragePooling1D(name=f"{branch_name}_gap")(x)
    return x


def build_multi_branch_cnn(
    input_shape_wrist: tuple[int, int] | None,
    input_shape_finger: tuple[int, int] | None,
    num_classes: int,
    input_shape_feat: int | None = None,
    conv_filters: list[int] = [32, 64],
    dense_units: int = 64
) -> Model:
    """
    Builds the Late Fusion Multi-Branch Conv1D CNN model.
    
    :param: input_shape_wrist (tuple | None): (T, C_wrist) wrist time-series input shape.
    :param: input_shape_finger (tuple | None): (T, C_finger) finger time-series input shape.
    :param: num_classes (int): number of gesture classes.
    :param: input_shape_feat (int | None): F scalar features shape (optional).
    :param: conv_filters (list): number of filters in each Conv1D layer.
    :param: dense_units (int): number of units in classification dense layer.
    :return: model (Model): compiled or uncompiled Keras Functional Model.
    """
    inputs = []
    branch_outputs = []

    # 1. Wrist Conv1D Branch
    if input_shape_wrist is not None and input_shape_wrist[1] > 0:
        wrist_input = layers.Input(shape=input_shape_wrist, name="wrist_input")
        inputs.append(wrist_input)
        
        x1 = _build_conv1d_branch(wrist_input, conv_filters, "wrist")
        branch_outputs.append(x1)
        logger.info("Configured Wrist Conv1D branch with input shape %s and filters %s", input_shape_wrist, conv_filters)

    # 2. Finger Conv1D Branch
    if input_shape_finger is not None and input_shape_finger[1] > 0:
        finger_input = layers.Input(shape=input_shape_finger, name="finger_input")
        inputs.append(finger_input)
        
        x2 = _build_conv1d_branch(finger_input, conv_filters, "finger")
        branch_outputs.append(x2)
        logger.info("Configured Finger Conv1D branch with input shape %s and filters %s", input_shape_finger, conv_filters)

    # 3. Dense Feature MLP Branch (Handcrafted/statistical features)
    if input_shape_feat is not None and input_shape_feat > 0:
        feat_input = layers.Input(shape=(input_shape_feat,), name="feat_input")
        inputs.append(feat_input)
        
        x3 = layers.Dense(32, activation="relu", name="feat_dense")(feat_input)
        x3 = layers.Dropout(0.5, name="feat_dropout")(x3)
        
        branch_outputs.append(x3)
        logger.info("Configured Dense Feature MLP branch with input shape (%d,)", input_shape_feat)

    if not branch_outputs:
        raise ValueError("Cannot build model: all branches are empty. Check features configuration.")

    # 4. Fusion and Classification Layers
    if len(branch_outputs) > 1:
        fused = layers.Concatenate(name="concat_branch")(branch_outputs)
    else:
        fused = branch_outputs[0]

    y = layers.Dense(dense_units, activation="relu", name="classifier_dense")(fused)
    y = layers.Dropout(0.5, name="classifier_dropout")(y)
    output = layers.Dense(num_classes, activation="softmax", name="softmax_output")(y)

    model = Model(inputs=inputs, outputs=output, name="late_fusion_cnn")
    
    logger.info("Late Fusion Multi-Branch CNN built successfully. Input count: %d. Output classes: %d. Dense units: %d.", 
                len(inputs), num_classes, dense_units)
    return model
