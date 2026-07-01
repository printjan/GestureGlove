# src/data_fusion_project/training/early_fusion_single_branch_1d_cnn/model.py
"""
Early Fusion Single-Branch Conv1D CNN model definition.

Implements the single-branch architecture from the early fusion specification:
all sensor channels (wrist + finger) are concatenated into a single input tensor
of shape (T, C) and processed through a single Conv1D pipeline.

Architecture (Config 1 - Standard):
    Input(150, C) → Conv1D(32, k=5) → BN → ReLU → MaxPool(2)
                   → Conv1D(64, k=3) → BN → ReLU → GAP
                   → Dense(16) → Dropout(0.5) → Dense(8, softmax)

Architecture (Config 2 - Compact):
    Input(150, C) → Conv1D(16, k=5) → BN → ReLU → GAP
                   → Dense(16) → Dropout(0.5) → Dense(8, softmax)

The input shape is fully dynamic and determined at runtime from the loaded
dataset dimensions (Dynamic Input Binding Strategy).
"""

from __future__ import annotations

import keras
from keras import layers, Model, regularizers

from data_fusion_project.core.logger_setup import get_logger

logger = get_logger(__name__)


def build_early_fusion_cnn(
    input_shape: tuple[int, int],
    num_classes: int = 8,
    conv_filters: list[int] | None = None,
    dense_units: int = 16,
    dropout_rate: float = 0.5,
    l2_reg: float = 1e-4,
) -> Model:
    """
    Builds the Early Fusion Single-Branch Conv1D CNN model.

    The Conv1D stack is constructed dynamically from ``conv_filters``:
    - Each filter gets a Conv1D → BatchNorm → ReLU block.
    - MaxPooling1D(2) is inserted between consecutive conv blocks (not after the last).
    - The final conv block terminates with GlobalAveragePooling1D.

    :param input_shape: (T, C) time-series input shape. C is determined dynamically
        from the loaded dataset and is never hardcoded.
    :param num_classes: Number of gesture classes (default: 8, including 'none').
    :param conv_filters: List of filter counts per Conv1D layer. Default: [32, 64].
        Pass [16] for the compact single-layer configuration.
    :param dense_units: Number of units in the bottleneck classification dense layer.
        Default: 16 (capacity-constrained to prevent session-specific memorization).
    :param dropout_rate: Dropout rate before the softmax output.
    :param l2_reg: L2 kernel regularization factor.
    :return: Uncompiled Keras Functional Model.
    """
    if conv_filters is None:
        conv_filters = [32, 64]

    inp = layers.Input(shape=input_shape, name="input")
    x = inp

    for i, f in enumerate(conv_filters):
        kernel_size = 5 if i == 0 else 3
        x = layers.Conv1D(
            filters=f,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"conv{i + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn{i + 1}")(x)
        x = layers.ReLU(name=f"relu{i + 1}")(x)

        if i < len(conv_filters) - 1:
            x = layers.MaxPooling1D(pool_size=2, name=f"pool{i + 1}")(x)
        else:
            x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(dense_units, activation="relu", name="classifier_dense")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    output = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)

    model = Model(inputs=inp, outputs=output, name="early_fusion_cnn")

    logger.info(
        "Early Fusion Single-Branch CNN built. Input shape %s, filters %s, "
        "dense_units %d, classes %d, total params %d.",
        input_shape, conv_filters, dense_units, num_classes, model.count_params(),
    )
    return model
