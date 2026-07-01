# src/data_fusion_project/training/late_fusion_multi_branch_1d_cnn/model.py
"""
Late Fusion Multi-Branch Conv1D CNN model definition.

Implements the production late fusion architecture with independent Conv1D
encoders for spatially distinct sensor groups:

- **Wrist Branch:** Processes IMU1 channels (arm-level, low-frequency translation).
- **Finger Branch:** Processes IMU2 and inter-IMU differential channels
  (fine-grained, high-frequency relative rotation).
- **MLP Branch (optional):** Processes scalar statistical features
  (cross-correlation, window statistics).

The branches are concatenated late before a shared classification head,
preventing spatial feature dilution while preserving independent kernel
specialization per sensor location.

Channel routing (wrist_idx, finger_idx) is determined dynamically in the
training pipeline by pattern-matching channel names, not in this builder.
"""

from __future__ import annotations

import keras
from keras import layers, Model, regularizers

from data_fusion_project.core.logger_setup import get_logger

logger = get_logger(__name__)


def _build_conv1d_branch(
    input_tensor: layers.Input,
    filters: list[int],
    branch_name: str,
    l2_reg: float = 1e-4,
) -> layers.Layer:
    """
    Constructs a Conv1D filtering branch dynamically.

    Layer graph per branch:
        For each filter f_i in filters:
            Conv1D(f_i, k=5 if first else 3) → BN → ReLU
            MaxPool(2) between consecutive blocks
            GlobalAveragePooling1D after last block

    :param input_tensor: Keras Input layer for this branch.
    :param filters: List of Conv1D filter counts.
    :param branch_name: Name prefix for layer naming (e.g., "wrist", "finger").
    :param l2_reg: L2 kernel regularization factor.
    :return: Output tensor after GAP.
    """
    x = input_tensor
    for i, f in enumerate(filters):
        kernel_size = 5 if i == 0 else 3
        x = layers.Conv1D(
            filters=f,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{branch_name}_conv{i + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"{branch_name}_bn{i + 1}")(x)
        x = layers.ReLU(name=f"{branch_name}_relu{i + 1}")(x)

        if i < len(filters) - 1:
            x = layers.MaxPooling1D(pool_size=2, name=f"{branch_name}_pool{i + 1}")(x)
        else:
            x = layers.GlobalAveragePooling1D(name=f"{branch_name}_gap")(x)
    return x


def parse_channel_indices(channel_names: list[str]) -> tuple[list[int], list[int]]:
    """
    Parses channel names to separate wrist and finger IMU channel indices.

    Rules:
    - Wrist channels: contains 'IMU1' or 'wrist' (case-insensitive).
    - Finger channels: contains 'IMU2', 'finger', or 'diff' (case-insensitive).
    - Unrecognized channels default to wrist branch with a warning.

    :param channel_names: List of C channel names from GestureDataset.
    :return: Tuple of (wrist_indices, finger_indices).
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
            logger.warning(
                "Unrecognized channel name '%s'. Grouping under wrist branch by default.", name
            )
            wrist_idx.append(idx)
    return wrist_idx, finger_idx


def build_late_fusion_cnn(
    input_shape_wrist: tuple[int, int] | None,
    input_shape_finger: tuple[int, int] | None,
    num_classes: int = 8,
    input_shape_feat: int | None = None,
    conv_filters: list[int] | None = None,
    dense_units: int = 16,
    dropout_rate: float = 0.5,
    l2_reg: float = 1e-4,
) -> Model:
    """
    Builds the Late Fusion Multi-Branch Conv1D CNN model.

    At least one of ``input_shape_wrist`` or ``input_shape_finger`` must be provided.
    The MLP branch is activated when ``input_shape_feat`` is not None and > 0.

    :param input_shape_wrist: (T, C_wrist) wrist time-series input shape, or None.
    :param input_shape_finger: (T, C_finger) finger time-series input shape, or None.
    :param num_classes: Number of gesture classes (default: 8).
    :param input_shape_feat: Scalar feature count F for the MLP branch, or None.
    :param conv_filters: List of Conv1D filter counts per layer. Default: [32, 64].
    :param dense_units: Units in the bottleneck classification dense layer (default: 16).
    :param dropout_rate: Dropout rate before the softmax output.
    :param l2_reg: L2 kernel regularization factor.
    :return: Uncompiled Keras Functional Model.
    """
    if conv_filters is None:
        conv_filters = [32, 64]

    inputs = []
    branch_outputs = []

    # 1. Wrist Conv1D Branch
    if input_shape_wrist is not None and input_shape_wrist[1] > 0:
        wrist_input = layers.Input(shape=input_shape_wrist, name="wrist_input")
        inputs.append(wrist_input)
        x1 = _build_conv1d_branch(wrist_input, conv_filters, "wrist", l2_reg)
        branch_outputs.append(x1)
        logger.info(
            "Configured Wrist Conv1D branch: input %s, filters %s",
            input_shape_wrist, conv_filters,
        )

    # 2. Finger Conv1D Branch
    if input_shape_finger is not None and input_shape_finger[1] > 0:
        finger_input = layers.Input(shape=input_shape_finger, name="finger_input")
        inputs.append(finger_input)
        x2 = _build_conv1d_branch(finger_input, conv_filters, "finger", l2_reg)
        branch_outputs.append(x2)
        logger.info(
            "Configured Finger Conv1D branch: input %s, filters %s",
            input_shape_finger, conv_filters,
        )

    # 3. Dense Feature MLP Branch (statistical/handcrafted features)
    if input_shape_feat is not None and input_shape_feat > 0:
        feat_input = layers.Input(shape=(input_shape_feat,), name="feat_input")
        inputs.append(feat_input)
        x3 = layers.Dense(32, activation="relu", name="feat_dense")(feat_input)
        x3 = layers.Dropout(dropout_rate, name="feat_dropout")(x3)
        branch_outputs.append(x3)
        logger.info("Configured Dense Feature MLP branch: input (%d,)", input_shape_feat)

    if not branch_outputs:
        raise ValueError(
            "Cannot build model: all branches are empty. "
            "Check that at least one of input_shape_wrist or input_shape_finger is provided."
        )

    # 4. Fusion and Classification Head
    if len(branch_outputs) > 1:
        fused = layers.Concatenate(name="concat_branch")(branch_outputs)
    else:
        fused = branch_outputs[0]

    y = layers.Dense(dense_units, activation="relu", name="classifier_dense")(fused)
    y = layers.Dropout(dropout_rate, name="classifier_dropout")(y)
    output = layers.Dense(num_classes, activation="softmax", name="softmax_output")(y)

    model = Model(inputs=inputs, outputs=output, name="late_fusion_cnn")

    logger.info(
        "Late Fusion Multi-Branch CNN built. Branches: %d, dense_units: %d, "
        "classes: %d, total params: %d.",
        len(inputs), dense_units, num_classes, model.count_params(),
    )
    return model
