# Unified Model Training Pipeline

This document provides comprehensive documentation for the unified model training pipeline that trains, evaluates, and exports all three gesture classification candidate architectures from a single codebase.


---

## Overview

The unified model training pipeline replaces the previous architecture-specific training scripts with a single, configuration-driven system. It supports:

- **Three candidate architectures**: Early Fusion CNN, Late Fusion CNN, Temporal Transformer
- **Architecture-agnostic training loop**: All model-specific branching is isolated to builder dispatch and input routing
- **Bayesian dynamic feature optimization** via Optuna TPE
- **Dynamic temporal jittering** via a custom Keras `Sequence` generator
- **Per-sample 3D rotation augmentation** using Rodrigues' rotation formula
- **Full metadata compliance** with the project's `model_metadata.json` schema
- **Configurable architecture presets** (`standard` and `compact`)

### Entry Point

```bash
python scripts/train.py --model-type <architecture> [options]
```

---

## Architecture

### Directory Structure

```
src/data_fusion_project/training/
├── model_training_pipeline/                 # Unified training loop & data logic
│   ├── __init__.py                          # Exports train_model(), TimeSeriesJitterSequence
│   ├── pipeline.py                          # Core training, evaluation, & metadata saving
│   ├── generator.py                         # TimeSeriesJitterSequence (Keras Sequence)
│   └── augmentation.py                      # 3D rotation augmentation (Rodrigues)
├── early_fusion_single_branch_1d_cnn/       # Early Fusion model package
│   ├── __init__.py                          # Exports build_early_fusion_cnn()
│   └── model.py                             # Single-branch Conv1D model builder
├── late_fusion_multi_branch_1d_cnn/         # Late Fusion model package
│   ├── __init__.py                          # Exports build_late_fusion_cnn()
│   └── model.py                             # Multi-branch Conv1D model builder
├── self_attention_temporal_transformer/      # Transformer model package
│   ├── __init__.py                          # Exports build_temporal_transformer()
│   └── model.py                             # Multi-head self-attention builder
└── late_fusion_multi_branch_cnn_test/       # Existing playground (untouched)

scripts/
├── train.py                                 # Unified CLI entry point
└── train_test_cnn.py                        # Existing playground CLI (untouched)
```
**Key source files:**

| Module | Source File |
|:---|:---|
| Training pipeline | [pipeline.py](../src/data_fusion_project/training/model_training_pipeline/pipeline.py) |
| Jitter generator | [generator.py](../src/data_fusion_project/training/model_training_pipeline/generator.py) |
| Rotation augmentation | [augmentation.py](../src/data_fusion_project/training/model_training_pipeline/augmentation.py) |
| Early Fusion builder | [model.py](../src/data_fusion_project/training/early_fusion_single_branch_1d_cnn/model.py) |
| Late Fusion builder | [model.py](../src/data_fusion_project/training/late_fusion_multi_branch_1d_cnn/model.py) |
| Transformer builder | [model.py](../src/data_fusion_project/training/self_attention_temporal_transformer/model.py) |
| Unified CLI entry point | [train.py](../scripts/train.py) |
| Playground CLI | [train_test_cnn.py](../scripts/train_test_cnn.py) |

### Separation of Concerns

| Module | Responsibility |
|:---|:---|
| `model.py` (per architecture) | Layer graph construction only. No data loading, splitting, or scaling. |
| [pipeline.py](../src/data_fusion_project/training/model_training_pipeline/pipeline.py) | Data splitting, scaling, model dispatch, training loop, evaluation, artifact saving |
| [generator.py](../src/data_fusion_project/training/model_training_pipeline/generator.py) | On-the-fly temporal jitter slicing and rotation augmentation |
| [augmentation.py](../src/data_fusion_project/training/model_training_pipeline/augmentation.py) | Rodrigues' rotation matrix generation and IMU coordinate group detection |
| [train.py](../scripts/train.py) | CLI argument parsing, Optuna orchestration, user-facing output |

---

## Supported Model Types

### 1. Early Fusion Single-Branch Conv1D CNN (`early_fusion_cnn`)

All sensor channels (wrist + finger) are concatenated into a single input tensor of shape `(150, C)` and processed through one Conv1D pipeline.

**Standard Configuration:**
```
Input(150, C) → Conv1D(32, k=5) → BN → ReLU → MaxPool(2)
               → Conv1D(64, k=3) → BN → ReLU → GAP
               → Dense(16) → Dropout(0.5) → Dense(8, softmax)
```

**Compact Configuration:**
```
Input(150, C) → Conv1D(16, k=5) → BN → ReLU → GAP
               → Dense(16) → Dropout(0.5) → Dense(8, softmax)
```

- **Pros:** Minimum parameter count, easiest to implement and run.
- **Cons:** Cannot specialize filters independently for wrist vs. finger IMU dynamics.
- **Scaling:** Single `TimeSeriesScaler` → `scaler_x.joblib`

### 2. Late Fusion Multi-Branch Conv1D CNN (`late_fusion_cnn`)

Independent Conv1D encoders for wrist (IMU1) and finger (IMU2/diff) channels, plus an optional Dense MLP branch for scalar statistical features. Outputs are concatenated late before the classification head.

**Standard Configuration:**
```
Wrist Branch:  Input(150, C_w) → Conv1D(32, k=5) → BN → ReLU → MaxPool(2)
                                → Conv1D(64, k=3) → BN → ReLU → GAP → 64-dim

Finger Branch: Input(150, C_f) → Conv1D(32, k=5) → BN → ReLU → MaxPool(2)
                                → Conv1D(64, k=3) → BN → ReLU → GAP → 64-dim

MLP Branch:    Input(F) → Dense(32, ReLU) → Dropout(0.5) → 32-dim

Fusion:        Concatenate → Dense(16) → Dropout(0.5) → Dense(8, softmax)
```

- **Pros:** Prevents spatial feature dilution; kernel specialization per sensor location.
- **Cons:** Slightly larger parameter footprint.
- **Scaling:** Dual `TimeSeriesScaler` → `scaler_x_wrist.joblib` + `scaler_x_finger.joblib` + optional `scaler_feat.joblib`

**Channel Routing:** Wrist and finger indices are determined dynamically by pattern-matching channel names (`"IMU1"` → wrist, `"IMU2"` / `"diff"` → finger). This is handled by `parse_channel_indices()` in the pipeline, not in the model builder.

**MLP Branch:** The statistical MLP branch is always activated when scalar features (`ds.features`) are available. It receives cross-correlation coefficients and per-channel window statistics.

### 3. Self-Attention Temporal Transformer (`temporal_transformer`)

Multi-head self-attention along the temporal dimension with learnable positional encoding.

**Standard Configuration:**
```
Input(150, C) → Dense(64, linear)               [Projection]
               → LearnablePositionalEncoding(150, 64)
               × 2 blocks:
                   → MultiHeadAttention(heads=4, key_dim=16)
                   → Residual + LayerNorm
                   → Dense(128, ReLU) → Dense(64)
                   → Residual + LayerNorm
               → LayerNorm → GAP
               → Dense(16) → Dropout(0.5) → Dense(8, softmax)
```

- **Pros:** Captures long-range temporal dependencies via global attention.
- **Cons:** Extremely data-hungry; prone to overfitting on small datasets.
- **Scaling:** Single `TimeSeriesScaler` → `scaler_x.joblib`

**Critical Requirement:** Low-pass filtering of magnitude features is **mandatory** before feeding to the transformer. Attention weight softmax is highly sensitive to high-frequency noise spikes. This is a preprocessing concern handled by `PipelineConfig.filters`, not by the model builder.

---

## Pipeline Stages

The unified `train_model()` function executes the following stages in order:

### 1. Feature Slicing

If `feature_toggles` are provided (from Optuna or CLI), the dataset channels are sliced to retain only active features. The slicing uses fuzzy name matching (`matches_feature()`) to handle naming convention differences between the audit feature lists and dataset column headers.

### 2. Channel Index Routing

For `late_fusion_cnn`, dynamically computes `wrist_idx` and `finger_idx` by pattern-matching channel names:
- `"IMU1"` or `"wrist"` → Wrist branch
- `"IMU2"`, `"finger"`, or `"diff"` → Finger branch
- Unrecognized → Wrist (with warning)

For `early_fusion_cnn` and `temporal_transformer`, all channels are concatenated into a single tensor.

### 3. Data Splitting

Three splitting strategies are supported:

| Strategy | Function | Use Case |
|:---|:---|:---|
| `leave-session-out` | `leave_sessions_out_three_way()` | Gold standard for cross-session generalization |
| `stratified` | `stratified_split_three_way()` | Balanced class distribution |
| `chronological` | `chronological_split_three_way()` | Preserves temporal ordering |

**Balanced LSO Auto-Detection:** If `ds.groups` contains session names matching `"test_data"` and `"validation_data"` (V4 dataset), the pipeline automatically uses a manual balanced split instead of random leave-session-out. This prevents the class-exclusion failure discovered in the playground experiments.

**Safety Fallback:** If a gesture class has fewer than 2 unique sessions under LSO, the pipeline warns and falls back to `chronological` to ensure all classes are represented.

### 4. Scaling

Architecture-dependent scaler routing:

| Architecture | Scalers | Serialized Files |
|:---|:---|:---|
| Early Fusion CNN | 1× `TimeSeriesScaler` on all channels | `scaler_x.joblib` |
| Temporal Transformer | 1× `TimeSeriesScaler` on all channels | `scaler_x.joblib` |
| Late Fusion CNN | 2× `TimeSeriesScaler` (wrist, finger) + 1× `StandardScaler` (features) | `scaler_x_wrist.joblib`, `scaler_x_finger.joblib`, `scaler_feat.joblib` |

The `TimeSeriesScaler` is a `StandardScaler` wrapper that reshapes `(N, T, C)` tensors to `(N×T, C)` for fitting, computes per-channel zero-mean unit-variance normalization, and reshapes back.

### 5. Model Building

The pipeline dispatches to the appropriate builder based on `model_type`:

```python
if model_type == "early_fusion_cnn":
    model = build_early_fusion_cnn(input_shape=X_train.shape[1:], ...)
elif model_type == "late_fusion_cnn":
    model = build_late_fusion_cnn(input_shape_wrist=..., input_shape_finger=..., ...)
elif model_type == "temporal_transformer":
    model = build_temporal_transformer(input_shape=X_train.shape[1:], ...)
```

All builders follow the **Dynamic Input Binding Strategy**: the `input_shape` parameter is read from the loaded dataset tensor dimensions at runtime. No channel counts are hardcoded.

### 6. Compilation

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### 7. Callbacks

- `EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)` — halts training when validation loss plateaus and restores the weights from the best epoch.
- `ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.5, min_lr=1e-6)` — halves the learning rate every 10 stagnant epochs.

### 8. Training

Standard `model.fit()` with the (optionally generator-wrapped) training data.

### 9. Evaluation

Computes `classification_report()` with per-class precision, recall, F1, and support.

### 10. Artifact Saving

See [Output Artifacts](#output-artifacts) section.

---

## Dynamic Temporal Jittering

### The Problem

The data processing pipeline ([dataset.py](../src/data_fusion_project/processing/dataset.py)) applies **static jitter**: a single random temporal offset is drawn once during `load_dataset()` and frozen for the entire training run. Every epoch sees the identical 150-sample window.

During real-time inference, however, causal Butterworth filters introduce a 20–40 ms group delay that shifts the gesture peak relative to the window center. A model trained only on perfectly centered windows may fail to trigger because the peak is shifted.

### The Solution: `TimeSeriesJitterSequence`

The `TimeSeriesJitterSequence` (in `generator.py`) implements **dynamic per-epoch jittering**:

1. **Pre-load raw uncropped windows** of shape `(N, T_raw, C)` where `T_raw ≈ 174` (150 target + 12 pre-buffer + 12 post-buffer at 100 Hz).
2. **Pre-scale** the raw windows using the fitted `TimeSeriesScaler`.
3. On each `__getitem__` call, draw a **fresh random offset** within `±jitter_range` for every sample independently.
4. Slice a 150-sample window from the scaled raw data at `center_idx + offset`.

Over 70 epochs, the network sees ~70 unique temporal translations of each gesture sample.

### When To Use

| Jitter Range | Behavior |
|:---|:---|
| `--jitter-range 0` | No jitter. Standard array-based training (backward compatible). |
| `--jitter-range 10` | ±10 sample shifts. Conservative for initial experiments. |
| `--jitter-range 20` | ±20 sample shifts. Recommended for production models to cover causal filter delay. |

---

## 3D Rotation Augmentation

### Rationale

The physical mounting angle of the IMU wrist strap and finger ring varies between donning sessions. Without rotational invariance, the model overfits to the specific sensor orientation used during training.

### Implementation

The `apply_rotation_augmentation()` function:

1. Identifies accelerometer `(accX, accY, accZ)` and gyroscope `(gyrX, gyrY, gyrZ)` coordinate triads per IMU prefix.
2. For each sample in the batch, generates a unique random rotation matrix using Rodrigues' formula via quaternion representation.
3. Applies the rotation: `X_rotated = X @ R.T`

Rotation is applied **per-sample** (not per-batch), so each sample sees a unique orientation. Over 70 epochs, this produces ~70 unique rotational views per gesture.

### CLI Usage

```bash
# Disabled by default
python scripts/train.py --augment-rotation 0.0

# Recommended for production
python scripts/train.py --augment-rotation 25.0
```

---

## Bayesian Feature Optimization (Optuna)

### Feature Categories

Based on the [data quality audit](../data_analysis/data_analysis_data_v2/data_quality_audit.ipynb) (Random Forest Gini importance + Mutual Information), all 37 candidate features are categorized:

| Category | Count | Selection | Criteria |
|:---|:---|:---|:---|
| **Pruned** | 6 | Always `False` | RF < 0.002 AND MI < 0.50 |
| **Mandatory** | 11 | Always `True` | MI > 0.90 AND RF > 0.02 |
| **Dynamic** | 21 | Optimized by Optuna | Remaining features |

### Optimization Procedure

1. **Search Space:** Each of the 21 dynamic features is mapped to a binary `trial.suggest_categorical(feat, [True, False])`.
2. **Sampler:** Tree-structured Parzen Estimator (TPE) with deterministic seeding.
3. **Objective:** For each trial:
   - Slice the dataset to the trial's active features.
   - Build and train the target architecture for `--optuna-epochs` (default: 15).
   - Compute the **Joint Utility Score**:
     $$\text{Utility} = \text{F1} - (w_1 \times \text{Latency ms}) - (w_2 \times \text{Parameters})$$
   - Latency is estimated by averaging 3× `model.predict()` calls on a dummy batch.
4. **Retraining:** After all trials complete, the best feature combination is used for a full `--epochs` training run with artifact saving.

### CLI Usage

```bash
# Standard optimization (50 trials, 15 epochs each)
python scripts/train.py --model-type late_fusion_cnn --optimize

# Custom trial budget
python scripts/train.py --optimize --optuna-trials 100 --optuna-epochs 20

# Custom utility weights
python scripts/train.py --optimize --w1 0.002 --w2 1e-5
```

---

## Architecture Presets

The `--config` flag selects a predefined set of architecture hyperparameters:

| Parameter | `standard` | `compact` |
|:---|:---|:---|
| `conv_filters` | `[32, 64]` | `[16]` |
| `dense_units` | `16` | `16` |
| `d_model` | `64` | `32` |
| `num_heads` | `4` | `2` |
| `num_blocks` | `2` | `1` |
| `ff_dim` | `128` | `64` |

Individual parameters can be overridden explicitly:

```bash
# Use compact preset but override conv_filters
python scripts/train.py --config compact --conv-filters 24 48
```

---

## CLI Reference

```
python scripts/train.py [options]
```

### Architecture & Preset

| Flag | Type | Default | Description |
|:---|:---|:---|:---|
| `--model-type` | choice | `early_fusion_cnn` | `early_fusion_cnn`, `late_fusion_cnn`, `temporal_transformer` |
| `--config` | choice | `standard` | `standard` or `compact` |
| `--model-name` | str | (auto) | Override model folder name |
| `--backend` | choice | auto | `tensorflow`, `torch`, `jax` |

### Training

| Flag | Type | Default | Description |
|:---|:---|:---|:---|
| `--epochs` | int | `70` | Maximum training epochs |
| `--batch-size` | int | `32` | Batch size |
| `--split` | choice | `leave-session-out` | `stratified`, `leave-session-out`, `chronological` |
| `--test-fraction` | float | `0.2` | Test held-out fraction |
| `--val-fraction` | float | `0.1` | Validation held-out fraction |
| `--seed` | int | `42` | Random seed |
| `--augment-rotation` | float | `0.0` | Max 3D rotation (degrees) |
| `--jitter-range` | int | `0` | Temporal jitter range (±samples) |
| `--run-name` | str | None | Custom session folder name |

### CNN Architecture (overrides `--config`)

| Flag | Type | Preset Default |
|:---|:---|:---|
| `--conv-filters` | int[] | `[32, 64]` / `[16]` |
| `--dense-units` | int | `16` |

### Transformer Architecture (overrides `--config`)

| Flag | Type | Preset Default |
|:---|:---|:---|
| `--d-model` | int | `64` / `32` |
| `--num-heads` | int | `4` / `2` |
| `--num-blocks` | int | `2` / `1` |
| `--ff-dim` | int | `128` / `64` |

### Optuna

| Flag | Type | Default |
|:---|:---|:---|
| `--optimize` | flag | `False` |
| `--optuna-trials` | int | `50` |
| `--optuna-epochs` | int | `15` |
| `--w1` | float | `0.001` |
| `--w2` | float | `1e-6` |

### Signal Processing & Features

All flags from the data processing pipeline are supported:

| Flag | Description |
|:---|:---|
| `--no-filter` | Disable Butterworth filtering |
| `--acc-cutoff` | Accelerometer LP cutoff (Hz) |
| `--gyro-cutoff` | Gyroscope LP cutoff (Hz) |
| `--diff` | Inter-IMU difference channels |
| `--linear-jerk` | Linear jerk features |
| `--angular-acceleration` | Angular acceleration features |
| `--relative-yaw` | Relative yaw integration |
| `--acc-magnitude` | Accelerometer magnitude |
| `--gyro-magnitude` | Gyroscope magnitude |
| `--gravity-free-acc` | Gravity-free linear acceleration |
| `--cross-correlation` | Cross-correlation scalar features |
| `--statistics` | Per-channel statistical features |

---

## Output Artifacts

Each training run produces the following directory structure:

```
models/<model_identifier>/
└── training_session_<index>_<timestamp>/
    ├── model.keras              # Full Keras model (structure + weights)
    ├── model.weights.h5         # Raw weights (always saved, backend-safe)
    ├── scaler_x.joblib          # Single-branch scaler (early fusion / transformer)
    ├── scaler_x_wrist.joblib    # Wrist branch scaler (late fusion)
    ├── scaler_x_finger.joblib   # Finger branch scaler (late fusion)
    ├── scaler_feat.joblib       # Scalar feature scaler (late fusion MLP)
    ├── model_metadata.json      # Complete training run metadata
    ├── confusion_matrix.png     # Per-class confusion matrix heatmap
    └── learning_curves.png      # Training/validation accuracy & loss curves
```

### Model Identifier Folders

| Model Type | Folder Name |
|:---|:---|
| `early_fusion_cnn` | `early_fusion_single_branch_1d_cnn` |
| `late_fusion_cnn` | `late_fusion_multi_branch_1d_cnn` |
| `temporal_transformer` | `slef_attention_temporal_transformer` |

### Session Indexing

Training session folders are auto-indexed sequentially (`training_session_0_...`, `training_session_1_...`). The `--run-name` flag allows custom naming (must start with `training_session_`).

### PyTorch Backend Workaround

On macOS with the PyTorch backend, full `.keras` model serialization causes a segfault. The pipeline:
1. Always saves `model.weights.h5` (safe across all backends).
2. If `KERAS_BACKEND == "torch"`: touches `model.keras` as an empty file and logs a warning.
3. Otherwise: attempts full `model.save()`, falling back to weights-only on failure.

### Metadata Schema (`model_metadata.json`)

The `model_metadata.json` file contains training run audit properties. It follows the structure below:

```json
{
  "timestamp": "20260701_181200",
  "model_name": "late_fusion_multi_branch_1d_cnn",
  "training_duration_s": 142.35,
  "epochs_trained": 62,
  "early_stopped": true,
  "classes": [
    "none",
    "swipe_left",
    "swipe_right",
    "circle_cw",
    "circle_ccw",
    "fist",
    "jerk_down",
    "jerk_up"
  ],
  "channels": [
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
    "IMU1_gyr_mag"
  ],
  "wrist_channels": [
    "IMU1_accX",
    "IMU1_accZ",
    "IMU1_gyrX",
    "IMU1_pitch",
    "diff_accX",
    "diff_accZ",
    "IMU1_gyr_mag"
  ],
  "finger_channels": [
    "IMU2_accX",
    "IMU2_accY",
    "IMU2_accZ",
    "IMU2_gyrX"
  ],
  "feature_names": [],
  "feature_toggles": {
    "IMU1_accX": true,
    "IMU1_accZ": true,
    "IMU1_accY": false,
    "IMU1_gyrX": true,
    "IMU1_pitch": true,
    "IMU2_accX": true,
    "IMU2_accY": true,
    "IMU2_accZ": true,
    "IMU2_gyrX": true,
    "diff_accX": true,
    "diff_accZ": true,
    "IMU1_gyr_mag": true
  },
  "features_selection": {
    "default_selected_features": [
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
      "IMU1_gyr_mag"
    ],
    "default_deselected_features": [
      "IMU1_accY"
    ]
  },
  "model_structure": {
    "total_parameters": 24968,
    "layers": [
      {
        "layer_name": "wrist_input",
        "class_name": "InputLayer",
        "output_shape": [null, 150, 7],
        "parameter_count": 0
      },
      {
        "layer_name": "finger_input",
        "class_name": "InputLayer",
        "output_shape": [null, 150, 4],
        "parameter_count": 0
      }
    ]
  },
  "training_parameters": {
    "epochs": 70,
    "batch_size": 32,
    "learning_rate": 0.001,
    "split_type": "leave-session-out",
    "test_fraction": 0.20,
    "val_fraction": 0.10,
    "seed": 42
  },
  "split_info": {
    "strategy": "leave-session-out",
    "total_samples": 1950,
    "train_size_abs": 1365,
    "val_size_abs": 195,
    "test_size_abs": 390,
    "train_fraction_real": 0.70,
    "val_fraction_real": 0.10,
    "test_fraction_real": 0.20,
    "train_sessions": ["session_0", "session_2"],
    "val_sessions": ["session_1"],
    "test_sessions": ["session_3"]
  },
  "performance": {
    "best_epoch": 42,
    "train_accuracy": 0.992,
    "train_loss": 0.021,
    "val_accuracy": 0.985,
    "val_loss": 0.033,
    "val_f1_score": 0.984
  },
  "evaluation": {
    "accuracy": 0.985,
    "macro_avg": {
      "precision": 0.986,
      "recall": 0.984,
      "f1-score": 0.985,
      "support": 390
    },
    "per_class_metrics": {
      "none": { "precision": 0.99, "recall": 1.0, "f1-score": 0.99, "support": 100 },
      "swipe_left": { "precision": 0.98, "recall": 0.97, "f1-score": 0.97, "support": 45 },
      "swipe_right": { "precision": 0.97, "recall": 0.98, "f1-score": 0.97, "support": 45 }
    }
  },
  "pipeline_config": {
    "sample_rate_hz": 100.0,
    "window_size": 150,
    "pad_mode": "edge"
  }
}
```

Key fields inside `model_metadata.json`:
- `model_type`: Architecture identifier (`early_fusion_cnn`, `late_fusion_cnn`, `temporal_transformer`)
- `channels`, `wrist_channels`, `finger_channels`: Dynamic input binding metadata
- `feature_toggles`, `features_selection`: Optuna optimization results
- `model_structure`: Layer-by-layer parameter breakdown
- `machine_info`: Hardware/OS context for reproducibility
- `training_parameters`: All hyperparameters including architecture-specific fields (e.g., `d_model`, `num_heads` for transformers)
- `split_info`: Detailed split statistics with per-partition session lists
- `performance`: Best epoch metrics
- `evaluation`: Per-class precision, recall, F1, and support
- `pipeline_config`: Full `PipelineConfig` serialization


---

## Implementation Details

### Dynamic Input Binding Strategy

Model builders never hardcode channel counts. The `input_shape` parameter is read from the loaded dataset tensor at runtime:

```python
# The first layer's shape is bound dynamically
batch, time_steps, channel_count = X_train.shape
model = build_early_fusion_cnn(input_shape=(time_steps, channel_count), ...)
```

For multi-branch models, column indices are determined by pattern-matching channel names:

```python
wrist_idx = [i for i, name in enumerate(channel_names) if "IMU1" in name.lower()]
finger_idx = [i for i, name in enumerate(channel_names) if "IMU2" in name.lower() or "diff" in name.lower()]
```

This decoupling allows feature engineering (adding/removing channels) to be done entirely through `PipelineConfig` flags without any model code changes.

### Feature Slicing

The `matches_feature()` function handles naming convention differences between the audit feature lists (e.g., `"IMU1_gyr_mag"`) and the dataset column headers (e.g., `"IMU1_gyroscope_magnitude"`) through a set of normalization rules and replacement mappings.

### TimeSeriesScaler

A thin wrapper around `sklearn.preprocessing.StandardScaler` that handles the 3D `(N, T, C)` reshaping:

```python
class TimeSeriesScaler:
    def fit(self, X):
        # Reshape (N, T, C) → (N*T, C) for fitting
        self.scaler.fit(X.reshape(-1, C))
    
    def transform(self, X):
        # Reshape, transform, reshape back
        return self.scaler.transform(X.reshape(-1, C)).reshape(N, T, C)
```

### Transformer Architecture Details

- **Pre-LN (Pre-Layer-Normalization):** Layer normalization is applied before the attention/FFN operations rather than after, producing more stable gradient flow in small-model regimes.
- **Learnable Positional Encoding:** A trainable weight matrix `(150, d_model)` is added to the projected input. Since the sequence length is fixed at 150 and the dataset is small, learnable encodings adapt better than sinusoidal ones.
- **Final LayerNorm:** A normalization layer after the last transformer block stabilizes the representation before pooling.

---

## Recommended Training Commands

The following commands represent the **optimal training configurations** derived from the playground experiments. All use:

- **Kalman filter** orientation fusion (estimates gyroscope bias)
- **No low-pass filter** (disabled Butterworth filtering) on raw inputs
- **Leave-session-out** splitting (cross-session generalization gold standard)
- **70 epochs** (compact models need ≥50 to converge)
- **25° rotation augmentation** (IMU strap mounting angle invariance)
- **±20 sample jitter** (bridges zero-phase ↔ causal filter group delay)
- **Optuna TPE feature optimization** (50 trials, 15 epochs/trial) to select the optimal feature subset

> [!IMPORTANT]
> **When `--optimize` is enabled, the pipeline automatically loads the full feature pool** (all 37 candidate channels including diff, jerk, angular acceleration, magnitudes, yaw, etc.) regardless of individual feature flags like `--diff` or `--linear-jerk`. The Optuna search space then categorizes these into three tiers:
>
> | Category | Count | Selection | Rationale |
> |:---|:---|:---|:---|
> | **Pruned** | 6 | Always `False` (never used) | RF Gini < 0.002 AND MI < 0.50 |
> | **Mandatory** | 11 | Always `True` (always kept) | MI > 0.90 AND RF > 0.02 |
> | **Dynamic** | 21 | Optimized by Optuna TPE | Remaining features — each is a binary trial parameter |
>
> After all trials complete, the best feature combination is used for a **final full-epoch retraining** with artifact saving. Individual feature flags (`--diff`, `--linear-jerk`, etc.) are only relevant when training **without** `--optimize`.

### Early Fusion CNN — Standard

```bash
python scripts/train.py \
    --model-type early_fusion_cnn \
    --config standard \
    --split leave-session-out \
    --epochs 70 \
    --augment-rotation 25 \
    --jitter-range 20 \
    --orientation kalman \
    --no-filter \
    --optimize \
    --optuna-trials 50 \
    --optuna-epochs 15
```

### Early Fusion CNN — Compact

```bash
python scripts/train.py \
    --model-type early_fusion_cnn \
    --config compact \
    --split leave-session-out \
    --epochs 70 \
    --augment-rotation 25 \
    --jitter-range 20 \
    --orientation kalman \
    --no-filter \
    --optimize \
    --optuna-trials 50 \
    --optuna-epochs 15
```

### Late Fusion CNN — Standard

```bash
python scripts/train.py \
    --model-type late_fusion_cnn \
    --config standard \
    --split leave-session-out \
    --epochs 70 \
    --augment-rotation 25 \
    --jitter-range 20 \
    --orientation kalman \
    --no-filter \
    --optimize \
    --optuna-trials 50 \
    --optuna-epochs 15
```

### Late Fusion CNN — Compact

```bash
python scripts/train.py \
    --model-type late_fusion_cnn \
    --config compact \
    --split leave-session-out \
    --epochs 70 \
    --augment-rotation 25 \
    --jitter-range 20 \
    --orientation kalman \
    --no-filter \
    --optimize \
    --optuna-trials 50 \
    --optuna-epochs 15
```

### Temporal Transformer — Standard

```bash
python scripts/train.py \
    --model-type temporal_transformer \
    --config standard \
    --split leave-session-out \
    --epochs 70 \
    --augment-rotation 25 \
    --jitter-range 20 \
    --orientation kalman \
    --no-filter \
    --optimize \
    --optuna-trials 50 \
    --optuna-epochs 15
```

### Temporal Transformer — Compact

```bash
python scripts/train.py \
    --model-type temporal_transformer \
    --config compact \
    --split leave-session-out \
    --epochs 70 \
    --augment-rotation 25 \
    --jitter-range 20 \
    --orientation kalman \
    --no-filter \
    --optimize \
    --optuna-trials 50 \
    --optuna-epochs 15
```

### Full Comparative Run (all six configurations)

Run all architectures × presets sequentially for a direct comparison:

```bash
# 1. Early Fusion CNN — Standard
python scripts/train.py \
    --model-type early_fusion_cnn --config standard \
    --split leave-session-out --epochs 70 \
    --augment-rotation 25 --jitter-range 20 \
    --orientation kalman --no-filter \
    --optimize --optuna-trials 50 --optuna-epochs 15

# 2. Early Fusion CNN — Compact
python scripts/train.py \
    --model-type early_fusion_cnn --config compact \
    --split leave-session-out --epochs 70 \
    --augment-rotation 25 --jitter-range 20 \
    --orientation kalman --no-filter \
    --optimize --optuna-trials 50 --optuna-epochs 15

# 3. Late Fusion CNN — Standard
python scripts/train.py \
    --model-type late_fusion_cnn --config standard \
    --split leave-session-out --epochs 70 \
    --augment-rotation 25 --jitter-range 20 \
    --orientation kalman --no-filter \
    --optimize --optuna-trials 50 --optuna-epochs 15

# 4. Late Fusion CNN — Compact
python scripts/train.py \
    --model-type late_fusion_cnn --config compact \
    --split leave-session-out --epochs 70 \
    --augment-rotation 25 --jitter-range 20 \
    --orientation kalman --no-filter \
    --optimize --optuna-trials 50 --optuna-epochs 15

# 5. Temporal Transformer — Standard
python scripts/train.py \
    --model-type temporal_transformer --config standard \
    --split leave-session-out --epochs 70 \
    --augment-rotation 25 --jitter-range 20 \
    --orientation kalman --no-filter \
    --optimize --optuna-trials 50 --optuna-epochs 15

# 6. Temporal Transformer — Compact
python scripts/train.py \
    --model-type temporal_transformer --config compact \
    --split leave-session-out --epochs 70 \
    --augment-rotation 25 --jitter-range 20 \
    --orientation kalman --no-filter \
    --optimize --optuna-trials 50 --optuna-epochs 15
```

---

## Design Decisions & Rationale

| Decision | Rationale | Evidence Source |
|:---|:---|:---|
| `dense_units=16` default (not 64) | Classifier bottlenecking prevents session-specific memorization | [Playground Experiments](model_architectures/playground_model_experiments.md) |
| `epochs=70` default | Compact models need ≥50 epochs before convergence | [Playground Experiments](model_architectures/playground_model_experiments.md) |
| `patience=20` for EarlyStopping | Prevents premature halting on class-skewed validation | [Playground Experiments](model_architectures/playground_model_experiments.md) |
| Always compute MLP features for late fusion | Scalar features (cross-correlation, statistics) carry complementary information for the late fusion architecture | User design decision |
| Balanced LSO auto-detection | V4 dataset uses `test_data`/`validation_data` naming to guarantee all 8 classes appear in every partition | Test model Experiments 3-6 |
| Per-sample rotation augmentation | Maximizes the diversity of sensor orientations seen during training | Architecture spec §3 |
| Rotation augmentation disabled by default | Spec states "not activated by default" | Test model doc §3 |
| `TimeSeriesJitterSequence` separate from `dataset.py` | Static jitter (single offset at load time) is insufficient for bridging zero-phase/causal filter mismatch | model_training.md §"Signal Filtering" |
| Optuna default: 15 epochs/trial | Sufficient for the TPE sampler to observe meaningful F1 differentiation | Architecture specs |
| Preserve typo `slef_attention_temporal_transformer` in folder naming | Backward compatibility with existing project structure | self_attention_temporal_transformer.md §8 |
| Dual `TimeSeriesScaler` for late fusion | Wrist and finger IMUs have completely different amplitude distributions; a single scaler would distort one branch | Test model capacity experiments |
| `l2_reg=1e-4` on Conv1D kernels | Mild weight decay prevents kernel explosion without dampening learned features | Architecture specs |
