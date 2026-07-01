# src/data_fusion_project/training/self_attention_temporal_transformer/model.py
"""
Lightweight Temporal Transformer (Self-Attention) model definition.

Implements a compact self-attention architecture for gesture time-series
classification. Instead of using convolutional spatial filters, this model
applies multi-head self-attention along the temporal dimension to capture
long-range dependencies across the full 150-sample (1.5 s) gesture window.

Architecture:
    Input(150, C) → Dense(d_model, linear)   [Projection]
                   → Add(learnable positional encoding)
                   × num_blocks:
                       → MultiHeadAttention(num_heads, key_dim)
                       → Residual + LayerNorm
                       → Dense(ff_dim, ReLU) → Dense(d_model)
                       → Residual + LayerNorm
                   → GlobalAveragePooling1D
                   → Dense(16) → Dropout(0.5) → Dense(8, softmax)

Critical design constraints (from specification):
- Low-pass smoothing of magnitude features is MANDATORY before feeding to this
  model. Attention mechanisms compute softmax weights based on dot-product
  comparisons, and high-frequency noise spikes cause attention weight saturation.
  This is a preprocessing concern handled by PipelineConfig, not by this builder.
- Positional encodings are learnable (not sinusoidal), as the sequence length
  is fixed at 150 and the dataset is too small for sinusoidal generalization.
"""

from __future__ import annotations

import keras
from keras import layers, Model

from data_fusion_project.core.logger_setup import get_logger

logger = get_logger(__name__)


class TransformerEncoderBlock(layers.Layer):
    """
    Single Transformer Encoder block: Multi-Head Self-Attention + Feed-Forward.

    Implements Pre-LN (Layer Normalization before attention/FFN) for more
    stable gradient flow in small-model regimes.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        attention_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.attention_dropout = attention_dropout
        self.ff_dropout = ff_dropout

        key_dim = max(1, d_model // num_heads)

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=attention_dropout,
        )
        self.drop1 = layers.Dropout(attention_dropout)

        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(ff_dropout),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(ff_dropout)

    def call(self, x, training=None):
        # Pre-LN Self-Attention
        x_norm = self.ln1(x)
        attn_out = self.mha(x_norm, x_norm, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = x + attn_out  # Residual connection

        # Pre-LN Feed-Forward
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm, training=training)
        ffn_out = self.drop2(ffn_out, training=training)
        x = x + ffn_out  # Residual connection

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "attention_dropout": self.attention_dropout,
            "ff_dropout": self.ff_dropout,
        })
        return config


class LearnablePositionalEncoding(layers.Layer):
    """
    Learnable positional encoding for fixed-length sequences.

    Adds a trainable weight matrix of shape (T, d_model) to the projected
    input tensor. Unlike sinusoidal encodings, learnable encodings can
    adapt to the specific temporal patterns of the gesture dataset.
    """

    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(self.sequence_length, self.d_model),
            initializer="uniform",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config


def build_temporal_transformer(
    input_shape: tuple[int, int],
    num_classes: int = 8,
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    ff_dim: int = 128,
    dense_units: int = 16,
    attention_dropout: float = 0.1,
    classifier_dropout: float = 0.5,
) -> Model:
    """
    Builds the Lightweight Temporal Transformer model.

    :param input_shape: (T, C) time-series input shape. T is typically 150
        and C is determined dynamically from the loaded dataset.
    :param num_classes: Number of gesture classes (default: 8).
    :param d_model: Dimensionality of the linear projection (default: 64).
    :param num_heads: Number of attention heads (default: 4, yielding key_dim=16).
    :param num_blocks: Number of stacked Transformer Encoder blocks (default: 2).
    :param ff_dim: Feed-forward expansion dimensionality (default: 128).
    :param dense_units: Bottleneck classification dense units (default: 16).
    :param attention_dropout: Dropout rate inside MHA and FFN blocks (default: 0.1).
    :param classifier_dropout: Dropout rate before the softmax output (default: 0.5).
    :return: Uncompiled Keras Functional Model.
    """
    sequence_length, num_channels = input_shape

    inp = layers.Input(shape=input_shape, name="input")

    # 1. Linear projection from C channels to d_model dimensions
    x = layers.Dense(d_model, name="projection")(inp)

    # 2. Learnable positional encoding
    x = LearnablePositionalEncoding(
        sequence_length=sequence_length,
        d_model=d_model,
        name="pos_encoding",
    )(x)

    # 3. Stacked Transformer Encoder blocks
    for i in range(num_blocks):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            attention_dropout=attention_dropout,
            ff_dropout=attention_dropout,
            name=f"transformer_block_{i + 1}",
        )(x)

    # 4. Final layer normalization (standard in transformer architectures)
    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    # 5. Global Average Pooling → Classification Head
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", name="classifier_dense")(x)
    x = layers.Dropout(classifier_dropout, name="classifier_dropout")(x)
    output = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)

    model = Model(inputs=inp, outputs=output, name="temporal_transformer")

    logger.info(
        "Temporal Transformer built. Input shape %s, d_model=%d, heads=%d, "
        "blocks=%d, ff_dim=%d, dense_units=%d, classes=%d, total params=%d.",
        input_shape, d_model, num_heads, num_blocks, ff_dim, dense_units,
        num_classes, model.count_params(),
    )
    return model
