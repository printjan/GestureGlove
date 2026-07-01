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


def build_multi_branch_cnn(
    input_shape_wrist: tuple[int, int] | None,
    input_shape_finger: tuple[int, int] | None,
    num_classes: int,
    input_shape_feat: int | None = None
) -> Model:
    """
    Builds the Late Fusion Multi-Branch Conv1D CNN model.
    
    :param: input_shape_wrist (tuple | None): (T, C_wrist) wrist time-series input shape.
    :param: input_shape_finger (tuple | None): (T, C_finger) finger time-series input shape.
    :param: num_classes (int): number of gesture classes.
    :param: input_shape_feat (int | None): F scalar features shape (optional).
    :return: model (Model): compiled or uncompiled Keras Functional Model.
    """
    inputs = []
    branch_outputs = []

    # 1. Wrist Conv1D Branch
    if input_shape_wrist is not None and input_shape_wrist[1] > 0:
        wrist_input = layers.Input(shape=input_shape_wrist, name="wrist_input")
        inputs.append(wrist_input)
        
        x1 = layers.Conv1D(filters=32, kernel_size=5, padding="same", kernel_regularizer=regularizers.l2(1e-4), name="wrist_conv1")(wrist_input)
        x1 = layers.BatchNormalization(name="wrist_bn1")(x1)
        x1 = layers.ReLU(name="wrist_relu1")(x1)
        x1 = layers.MaxPooling1D(pool_size=2, name="wrist_pool1")(x1)
        
        x1 = layers.Conv1D(filters=64, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(1e-4), name="wrist_conv2")(x1)
        x1 = layers.BatchNormalization(name="wrist_bn2")(x1)
        x1 = layers.ReLU(name="wrist_relu2")(x1)
        x1 = layers.GlobalAveragePooling1D(name="wrist_gap")(x1)
        
        branch_outputs.append(x1)
        logger.info("Configured Wrist Conv1D branch with input shape %s", input_shape_wrist)

    # 2. Finger Conv1D Branch
    if input_shape_finger is not None and input_shape_finger[1] > 0:
        finger_input = layers.Input(shape=input_shape_finger, name="finger_input")
        inputs.append(finger_input)
        
        x2 = layers.Conv1D(filters=32, kernel_size=5, padding="same", kernel_regularizer=regularizers.l2(1e-4), name="finger_conv1")(finger_input)
        x2 = layers.BatchNormalization(name="finger_bn1")(x2)
        x2 = layers.ReLU(name="finger_relu1")(x2)
        x2 = layers.MaxPooling1D(pool_size=2, name="finger_pool1")(x2)
        
        x2 = layers.Conv1D(filters=64, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(1e-4), name="finger_conv2")(x2)
        x2 = layers.BatchNormalization(name="finger_bn2")(x2)
        x2 = layers.ReLU(name="finger_relu2")(x2)
        x2 = layers.GlobalAveragePooling1D(name="finger_gap")(x2)
        
        branch_outputs.append(x2)
        logger.info("Configured Finger Conv1D branch with input shape %s", input_shape_finger)

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

    y = layers.Dense(64, activation="relu", name="classifier_dense")(fused)
    y = layers.Dropout(0.5, name="classifier_dropout")(y)
    output = layers.Dense(num_classes, activation="softmax", name="softmax_output")(y)

    model = Model(inputs=inputs, outputs=output, name="late_fusion_cnn")
    
    logger.info("Late Fusion Multi-Branch CNN built successfully. Input count: %d. Output classes: %d.", 
                len(inputs), num_classes)
    return model
