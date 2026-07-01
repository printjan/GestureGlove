# Production Real-Time Inference Pipeline

This document provides comprehensive documentation for the **architecture-agnostic production real-time inference pipeline** that runs live gesture classification using any trained model from the unified training pipeline. Unlike the [playground inference system](asynchronous_real-time_inference.md), which is hardcoded to the `late_fusion_cnn_test` architecture, this production pipeline dynamically detects the model architecture from `model_metadata.json` and dispatches all architecture-specific logic (model building, scaler loading, input routing) transparently.

## System Workflow

The production real-time inference pipeline executes in the following sequence:
1. **Model Loading & Architecture Detection:** Reads `model_metadata.json` from the specified model directory to discover the `model_type`, training configuration, channel lists, and scaler paths. Dispatches to the correct model builder and loads weights.
2. **Sensor Ingestion:** Connects to dual-IMU serial ports (specified in `config/devices.yml`). Alternatively, starts high-frequency simulated streams when `--simulate` is enabled.
3. **Static Calibration:** Prompts the user to hold still for 6.0 seconds. Computes the baseline offset and aligns sensor timestamps.
4. **Asynchronous Slicing:** Asynchronously collects data, extracts sliding windows, and resamples/interpolates sensor signals via the `AsynchronousDataGrabber`.
5. **ZUPT Background Calibration:** Continuously monitors signal standard deviations in the background. If stillness is detected, it recalibrates the gyroscope bias registers on-the-fly via Exponential Moving Average (EMA).
6. **Preprocessing & Pre-Prediction Transform:** Passes the calibrated sliding windows through the architecture-specific model transform callback (channel selection, scaling, and input routing).
7. **Architecture-Agnostic Forward Pass:** Executes `predict_fn(frame)` which wraps `model.predict()` with the correct input format for the detected architecture.
8. **Live Performance Evaluation (Optional):** If `--evaluate` is enabled, the pipeline detects gesture triggers and listens for user objections (corrective hotkeys) to record True Positives (TP), False Positives (FP), and False Negatives (FN).
9. **Action Dispatcher:** Translates classified gestures into keyboard shortcuts using [powerpoint_control.yml](../config/powerpoint_control.yml) and fires key events to control the active PowerPoint window.

---

### Usage Commands

* **Live Mode (Physical Rigs) — Early Fusion CNN:**
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.95
  ```

* **Live Mode — Late Fusion CNN:**
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/late_fusion_multi_branch_1d_cnn --threshold 0.95
  ```

* **Live Mode — Temporal Transformer:**
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/slef_attention_temporal_transformer --threshold 0.95
  ```

* **Simulated Dry-Run (No Hardware Needed):**
  Useful for quick offline verification and pipeline logic checks:
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.95 --no-control --simulate --timeout 20
  ```

* **Disable Background ZUPT Recalibration:**
  Disable the dynamic background stillness recalibration via:
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.95 --no-zupt
  ```

* **Custom ZUPT Stillness Duration:**
  Set the duration in seconds that the hand must remain completely still before recalibrating (default: 2.0s):
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.95 --zupt-duration 3.5
  ```

* **Objection-Based Live Evaluation:**
  Run live inference while evaluating gesture classification accuracy in real-time:
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.90 --evaluate --objection-window 1.8 --eval-out reports/
  ```

* **Optimum Config for Real-Time Testing:**
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.95 --evaluate --objection-window 1.8 --eval-out reports/ --zupt-duration 3.5 --cooldown 2.0
  ```

---

# Architecture-Agnostic Model Loader — Technical Documentation

This section documents the design, architecture, and implementation details of the **Model Loader** (`InferenceBundle` and `load_inference_model()`) implemented in `src/data_fusion_project/inference/model_loader.py`.

---

## 1. Motivation: Why Is It Needed?

The original [playground inference script](asynchronous_real-time_inference.md) (`scripts/run_realtime_inference_test_async.py`) was hardcoded to the `late_fusion_cnn_test` architecture at four critical code locations:

1. **Model Builder Import:** Directly imported `build_multi_branch_cnn` from the test package — completely unusable for Early Fusion CNN or Transformer architectures.
2. **Scaler Class Import:** Imported `TimeSeriesScaler` from the test training module. The production models serialize scalers using the *pipeline* `TimeSeriesScaler` class (from `model_training_pipeline.pipeline`). Joblib deserialization requires the **exact same class** on the import path. Loading production scalers via the test import path causes a `ModuleNotFoundError` or `UnpicklingError`.
3. **Transform Callback:** Hardcoded dual-branch `(wrist, finger)` tensor routing and dual scaler application. This is incorrect for the Early Fusion CNN and Temporal Transformer, which use a single `scaler_x.joblib` and a single concatenated tensor.
4. **`model.predict()` Call:** Hardcoded dict input `{"wrist_input": ..., "finger_input": ...}`. Single-branch architectures expect a single `np.ndarray`, not a named-input dict.

The `model_loader` module resolves this by encapsulating **all architecture-specific dispatch** behind a clean `InferenceBundle` interface, making the inference script and all downstream consumers completely architecture-agnostic.

---

## 2. Architecture Dispatch Table

The model loader uses the `model_type` field from `model_metadata.json` to dynamically dispatch to the correct builder, load the appropriate scalers, and construct the correct tensor routing and prediction closures.

| `model_type` | Builder Function | Scaler Artifacts | `transform_fn` Behavior | `predict_fn` Input Format |
|:---|:---|:---|:---|:---|
| `early_fusion_cnn` | `build_early_fusion_cnn()` | `scaler_x.joblib` | Select `channels`, batch to `(1, T, C)`, scale via single `TimeSeriesScaler` | Single `np.ndarray` |
| `late_fusion_cnn` | `build_late_fusion_cnn()` | `scaler_x_wrist.joblib` + `scaler_x_finger.joblib` + optional `scaler_feat.joblib` | Split by `wrist_channels` / `finger_channels`, batch, scale independently | Dict `{"wrist_input": ..., "finger_input": ...}` |
| `temporal_transformer` | `build_temporal_transformer()` | `scaler_x.joblib` | Select `channels`, batch to `(1, T, C)`, scale via single `TimeSeriesScaler` | Single `np.ndarray` |

---

## 3. Software Architecture & Component Interaction

```mermaid
flowchart TD
    subgraph Model Loader (model_loader.py)
        Meta[model_metadata.json] -->|read model_type| Dispatch{Architecture Dispatch}
        
        Dispatch -->|early_fusion_cnn| EF[build_early_fusion_cnn]
        Dispatch -->|late_fusion_cnn| LF[build_late_fusion_cnn]
        Dispatch -->|temporal_transformer| TF[build_temporal_transformer]
        
        EF --> Weights[Load model.weights.h5]
        LF --> Weights
        TF --> Weights
        
        Dispatch -->|single-branch| SingleScaler[Load scaler_x.joblib]
        Dispatch -->|late-fusion| DualScaler[Load scaler_x_wrist + scaler_x_finger]
        
        SingleScaler --> TransformFn[Build transform_fn closure]
        DualScaler --> TransformFn
        
        Weights --> PredictFn[Build predict_fn closure]
        
        TransformFn --> Bundle[InferenceBundle]
        PredictFn --> Bundle
    end

    subgraph Inference Script (run_realtime_inference.py)
        Bundle -->|transform_fn| Grabber[AsynchronousDataGrabber]
        Grabber -->|preprocessed frame| Predict[bundle.predict_fn]
        Predict --> Dispatcher[GestureDispatcher]
        Predict --> Evaluator[LivePerformanceEvaluator]
    end
```

---

## 4. Implementation Details

### InferenceBundle Dataclass

The `InferenceBundle` is a `@dataclass` that encapsulates all artifacts required for architecture-agnostic inference:

```python
@dataclass
class InferenceBundle:
    model: keras.Model           # Compiled Keras model with loaded weights
    model_type: str              # Architecture identifier from metadata
    class_names: list[str]       # Ordered softmax output class labels
    pipeline_config: PipelineConfig  # Reconstructed signal processing config
    transform_fn: Callable       # (channels, channel_names) → model-ready tensor(s)
    predict_fn: Callable         # (frame) → softmax probability array (1, N_classes)
    metadata: dict               # Full raw model_metadata.json
    model_dir: Path              # Resolved training session directory
```

### Session Directory Auto-Resolution

When the user passes a top-level model identifier directory (e.g., `models/early_fusion_single_branch_1d_cnn`), the loader automatically scans for `training_session_*` subdirectories and resolves to the one with the highest sequential index. This allows the user to point at the model family folder and always get the latest trained session.

### Scaler Class Path Alignment

Joblib deserializes Python objects by reconstructing the class from its fully qualified import path. During training, scalers are serialized as instances of `data_fusion_project.training.model_training_pipeline.pipeline.TimeSeriesScaler`. The model loader module imports this exact class (even though it's not directly used in the loader logic) to ensure the import path is registered before `joblib.load()` is called:

```python
# This import is required for joblib deserialization — do NOT remove
from data_fusion_project.training.model_training_pipeline.pipeline import TimeSeriesScaler  # noqa: F401
```

### Transformer Custom Layer Registration

The Temporal Transformer architecture uses custom Keras `Layer` subclasses (`TransformerEncoderBlock`, `LearnablePositionalEncoding`). For `model.load_weights()` to correctly match weight names to layers, these custom classes must be importable in the current Python session. The model loader handles this by importing them explicitly when `model_type == "temporal_transformer"`.

### PipelineConfig Reconstruction

The `load_pipeline_config()` function reconstructs a `PipelineConfig` instance from the nested dictionary stored in `model_metadata.json["pipeline_config"]`. This includes deserializing enum types (`FilterType`, `OrientationMethod`) and reconstructing the four sub-config dataclasses (`CalibrationConfig`, `FilterConfig`, `OrientationConfig`, `FeatureConfig`). This function was previously inlined in the playground script and is now centralized in the model loader for reuse.

### Transform Function Closures

#### Single-Branch (Early Fusion CNN, Temporal Transformer)
```python
def transform_fn(channels: np.ndarray, channel_names: list[str]) -> np.ndarray:
    # 1. Select only the channels listed in metadata["channels"]
    idx = [channel_names.index(ch) for ch in expected_channels]
    # 2. Add batch dimension: (T, C) → (1, T, C)
    X = channels[:, idx][np.newaxis, :, :]
    # 3. Apply TimeSeriesScaler normalization
    return scaler_x.transform(X)
```

#### Late Fusion Multi-Branch CNN
```python
def transform_fn(channels: np.ndarray, channel_names: list[str]) -> tuple:
    # 1. Split channels by wrist_channels and finger_channels from metadata
    wrist_idx = [channel_names.index(ch) for ch in wrist_channels]
    finger_idx = [channel_names.index(ch) for ch in finger_channels]
    # 2. Batch and scale independently
    X_wrist = scaler_wrist.transform(channels[:, wrist_idx][np.newaxis, :, :])
    X_finger = scaler_finger.transform(channels[:, finger_idx][np.newaxis, :, :])
    return (X_wrist, X_finger)
```

### Predict Function Closures

#### Single-Branch
```python
def predict_fn(frame: np.ndarray) -> np.ndarray:
    return model.predict(frame, verbose=0)
```

#### Late Fusion Multi-Branch
```python
def predict_fn(frame: tuple) -> np.ndarray:
    input_dict = {name: frame[i] for i, name in enumerate(input_names)}
    return model.predict(input_dict, verbose=0)
```

The named input dict keys (`"wrist_input"`, `"finger_input"`, `"feat_input"`) are read dynamically from `model.inputs`, not hardcoded, maintaining full forward compatibility with model builder changes.

---

## 5. Design Decisions & Rationale

| Decision | Rationale | Evidence Source |
|:---|:---|:---|
| `--model-dir` is a required argument (no default) | Production models span 3 architectures × 2 presets. Explicit model selection prevents accidentally running the wrong model | Playground experiments showed architecture-specific failure modes |
| `transform_fn` and `predict_fn` are closures, not methods | Closures capture the loaded scaler references and channel lists at construction time, avoiding repeated metadata lookups per frame | Performance: 10–30 Hz inference loop cannot tolerate per-frame dict accesses |
| `TimeSeriesScaler` is imported even though not directly used | Joblib deserialization requires the class to be importable from the same module path used during training-time serialization | Python pickle module specification |
| `predict_fn` reads `model.inputs` for named keys | Avoids hardcoding `"wrist_input"` / `"finger_input"` — future model builder changes (e.g., adding a third branch) are automatically supported | Separation of concerns: builder owns layer names, loader reads them |
| Session auto-resolution picks highest index | Users typically want the latest training run. `training_session_2_*` is prioritized over `training_session_1_*` | Project convention from `model_training_pipeline.md` |
| Playground script left completely untouched | Project convention: `late_fusion_cnn_test` is an isolated playground and must not be modified | `model_training_pipeline.md` directory structure |
| Custom Keras layers imported for transformer | `model.load_weights()` requires matching layer classes; without the import, weight loading raises `ValueError: Unknown layer` | Keras serialization documentation |

---

# Relationship Between Playground and Production Inference Pipelines

The project maintains **two separate inference pipelines** that share the same async infrastructure but differ in architecture coupling:

| Aspect | Playground Pipeline | Production Pipeline |
|:---|:---|:---|
| **Script** | `scripts/run_realtime_inference_test_async.py` | `scripts/run_realtime_inference.py` |
| **Documentation** | [asynchronous_real-time_inference.md](asynchronous_real-time_inference.md) | This document |
| **Supported Architectures** | `late_fusion_cnn_test` only (hardcoded) | All 3 production architectures (auto-detected) |
| **Model Builder Import** | `build_multi_branch_cnn` from test package | Dynamic dispatch via `model_loader` |
| **Scaler Class** | `TimeSeriesScaler` from test `train.py` | `TimeSeriesScaler` from `model_training_pipeline.pipeline` |
| **Transform Callback** | Hardcoded dual-branch wrist/finger routing | Architecture-specific closure from `InferenceBundle` |
| **`model.predict()` Format** | Hardcoded `{"wrist_input": ..., "finger_input": ...}` | Dynamic `predict_fn` closure |
| **Default `--model-dir`** | `models/late_fusion_cnn_test` | Required argument (no default) |

Both pipelines share these components **verbatim** (zero code duplication):
- `AsynchronousDataGrabber` (producer-consumer threading)
- `LivePerformanceEvaluator` (objection-based accuracy assessment)
- `TriggerDetector` (de-bounced fire event detection)
- `GestureDispatcher` + `PowerPointController` (action dispatch)
- `MockIMU` (simulated sensor streams for dry-runs)
- ZUPT background recalibration (gyroscope bias drift compensation)
- Static calibration procedure (6-second stillness baseline)

---

# CLI Reference

```
python scripts/run_realtime_inference.py [options]
```

### Required

| Flag | Type | Description |
|:---|:---|:---|
| `--model-dir` | str | Path to the model directory. Accepts either a model identifier folder (e.g., `models/early_fusion_single_branch_1d_cnn`) or a specific training session. |

### Inference Control

| Flag | Type | Default | Description |
|:---|:---|:---|:---|
| `--threshold` | float | `0.80` | Confidence threshold to trigger a gesture (0.0 to 1.0). |
| `--cooldown` | float | `1.0` | Minimum seconds between two fired actions (de-bounce cool-down). |
| `--config` | str | `config/powerpoint_control.yml` | Path to a PowerPoint control config file. |
| `--dry-run` | flag | `False` | Do not send real key presses; only log shortcuts. |
| `--no-control` | flag | `False` | Disable PowerPoint control (predictions are only displayed). |

### Simulation & Timeout

| Flag | Type | Default | Description |
|:---|:---|:---|:---|
| `--simulate` | flag | `False` | Simulate IMU data streaming instead of connecting to real serial hardware. |
| `--timeout` | float | `None` | Automated exit timeout in seconds (useful for headless verification). |

### ZUPT Calibration

| Flag | Type | Default | Description |
|:---|:---|:---|:---|
| `--no-zupt` | flag | `False` | Disable background Zero-Velocity Updates (ZUPT) recalibration. |
| `--zupt-duration` | float | `2.0` | Sustained stillness duration (seconds) required for ZUPT recalibration. |

### Live Evaluation

| Flag | Type | Default | Description |
|:---|:---|:---|:---|
| `--evaluate` | flag | `False` | Enable the objection-based live-performance evaluator. |
| `--objection-window` | float | `1.5` | Seconds to wait for a correcting keypress before committing a fire as TP. |
| `--eval-out` | str | Model session dir | Directory for the evaluation report output. |

---

# API Reference

### `load_inference_model()`
```python
def load_inference_model(
    model_dir: str | Path,
) -> InferenceBundle:
```
Loads a trained gesture classification model and returns an `InferenceBundle`. Auto-resolves to the latest training session if a top-level model directory is provided.

### `InferenceBundle`
```python
@dataclass
class InferenceBundle:
    model: keras.Model
    model_type: str
    class_names: list[str]
    pipeline_config: PipelineConfig
    transform_fn: Callable[[np.ndarray, list[str]], Any]
    predict_fn: Callable[[Any], np.ndarray]
    metadata: dict
    model_dir: Path
```

### `load_pipeline_config()`
```python
def load_pipeline_config(
    metadata: dict,
) -> PipelineConfig:
```
Reconstructs a `PipelineConfig` instance from the `"pipeline_config"` nested dictionary in `model_metadata.json`.

---

# Directory Structure

```
src/data_fusion_project/inference/
├── __init__.py              # Exports AsynchronousDataGrabber, LivePerformanceEvaluator,
│                            # TriggerDetector, InferenceBundle, load_inference_model
├── data_grabber.py          # AsynchronousDataGrabber (producer-consumer threading)
├── live_evaluation.py       # LivePerformanceEvaluator, TriggerDetector
└── model_loader.py          # InferenceBundle, load_inference_model, load_pipeline_config

scripts/
├── run_realtime_inference.py            # Production inference (architecture-agnostic)
└── run_realtime_inference_test_async.py # Playground inference (late_fusion_cnn_test only)
```
