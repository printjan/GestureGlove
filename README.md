# Data Fusion Project


## Team

- Lucas Horn: `hornlu95907@th-nuernberg.de`
- Jan Tischner: `tischnerja95752@th-nuernberg.de`


---


## Project description

### IMU Sensor boards:

- We are using two `XIAOML Kit` devices: Designed by Professor Vijay Janapa Reddi (Harvard University), author of the Machine Learning Systems textbook. One at the wrist and one at the index finger.
- What's inside: XIAO ESP32-S3 Sense, CAM • IMU • SD Toolkit
- Advertising: Build keyword detection, image classification, motion detection, object detection, and more
- Links: For ...
  - Learners: mlsysbook.ai
  - Builders: mlsysbook.ai/kits
  - Developers: github.com/mlsysbook

### Hardware Setup:

- The two XIAOML Kits are directly conncted to the computer via USB-C.
- IMU Data will be streamed unprocessed via USB-C-Serial to the computer.
- All processing, fusion, filtering, and ML will run on the Computer. 

### Hardware Mounting:
- One XIAOML Kit on the wrist (IMU Data).
- One XIAOML Kit on the tip of the index finger (Camera Data).
- Orientation usb-c-plug downward and backward.
- Mounted on right hand.
  
### Goal:

- Recognize arm- and hand-gestures with wrist worn IMU Sensor.
- Demonstation: Control power point with hand gestures.

### Possible Extension:

- Use finger as an air mouse to interact with the computer.
- Demonstration: Cotrol the power point laser pointer by hand movement.


---


## Guestures

### Very important:

- Discrete Movement (Recognizable Start and Stop of the movement with a stationary moment before and after to differentiate the geusture from natural movement)!
- Calibration: 
  - At the beginning of each recording:
    - 3 seconds still pose:
      - wrist mounted normally
      - index finger extended or relaxed in defined pose

### Arm gestures

**Swipe Right / Swipe Left:**
- Movement: Horizonal movement of the hand with dedicated still moment at the end and the beginning of the gesture.
- Demonstration: Next / Previous slide in powerpoint.
**Jerk Up / Jerk Down:**
- Movement: Vertical movement of the hand with dedicated still moment at the end and the beginning of the gesture.
- Demonstration: Volume Up / Volume Down.
**Circle Clockwise / Circle Counter Clockwise:**
- Movement: Clockwise / Counter Clockwise movement of the wrist with dedicated still moment at the end and the beginning of the gesture.
- Demonstration: Toggle Laser Pointer Mode.

### Hand gestures

**Make fist:**
- Movement: Close hand (make fist) and immediately open it again twice. Hand celarly open at the end and beginning of the gesture. During the gesture arm stays still
- Demonstration: Toggle Laser Pointer Mode.

### None

**None class:**
- Movement: Idle: Hold still or slightly move indiscriminateley.
- Demonstration: Speaking and moving naturally.

### Naming scheme

In the dataset an classifiers the naming scheme will be as follows:

```
[
  "none",
  "swipe_left",
  "swipe_right",
  "circle_cw",
  "circle_ccw",
  "fist",
  "jerk_down",
  "jerk_up"
]
```


---


## Project Structure 

### Data & Model Structure

```
data_fusion_project/
├── data/
│   ├── <gesture name>/
│   │   │   ├── <recording_session>/
│   │   │   │   ├── recording_session.json           # properties of the particular recording session
│   │   │   │   ├── calibration_<index>.csv          # 5 second recording of no movement to establish sensor drift
│   │   │   │   ├── calibration_<index>.png          # plot of the static calibration recording
│   │   │   │   ├── energy_distribution_<index>.csv  # motion energy distribution stats (centered 150 samples)
│   │   │   │   ├── centered_energy_distribution_<index>.png # plot of the centered 150-sample energy band
│   │   │   │   ├── overall_energy_distribution_<index>.png  # plot of the raw 1.74s energy band with average bounds
│   │   │   │   ├── 00001.csv                        # raw 1.74-second gesture recording (~174 samples)
│   │   │   │   ├── 00001.txt                        # start index of the centered 150-sample gesture window
│   │   │   │   ├── 00001.png                        # plot of raw recording with vertical start/end markers
│   │   │   │   └── ...
├── models/
│   ├── <model_name>/
│   │   ├── training_session_<index>_<timestamp>/           # one particular training session for that model
│   │   │   ├── confusion_matrix.png                        # confusion matrix of the trained model
│   │   │   ├── learning_curves.png                         # learning curves of the trained model
│   │   │   ├── model_metadata.json                         # JSON file containing training run audit properties
│   │   │   ├── model.keras                                 # saved trained Keras model structure & weights
│   │   │   ├── model.weights.h5                            # saved raw weights
│   │   │   └── scaler_x.joblib                             # StandardScaler instance (or scaler_x_wrist.joblib & scaler_x_finger.joblib for multi-branch)
│   │   ├── training_session_<name>/                        # alternative naming scheme using name instead of index
```

### Data set structure

Column structure of `<id>.csv` or `calibration_<index>.csv` files (they only contain raw data):

```csv
IMU1_accX,IMU1_accY,IMU1_accZ,IMU1_gyrX,IMU1_gyrY,IMU1_gyrZ,IMU2_accX,IMU2_accY,IMU2_accZ,IMU2_gyrX,IMU2_gyrY,IMU2_gyrZ
```

### Recording Session Properties Structure

```json
{
  "baudrate": 115200,
  "record_duration_s": 1.5,
  "target_samples": 150,
  "max_samples_before_recalibration": 25,
  "pre_buffer_s": 0.12,
  "post_buffer_s": 0.12,
  "recalibrations": [
    {
      "file": "calibration_<index>.csv", # IMU static recording used to calculate sensor drift
      "sample_index": <sample_count_at_calibration>
    },
    ...
  ],
  "energy_distributions": [
    {
      "file": "energy_distribution_<index>.csv", # Calculated motion energy distribution stats
      "sample_index": <sample_count_at_distribution>
    },
    ...
  ]
}
```

### Model Metadata Properties Structure (`model_metadata.json`)

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

---

## Data Processing Pipeline

Extracts trainings data and OPTIONALLY preprocesses it providing different preprocessing options or OPTIONALLY calculates features:

- Calibration: 
  - Periodic static calibration (every `MAX_SAMPLES_BEFORE_RECALIBRATION` samples):
    - 5 seconds still pose:
      - wrist mounted normally
      - index finger extended or relaxed in defined pose
  - Estimate:
    - accelerometer bias / gravity direction
    - gyroscope zero bias
    - axis orientation sanity check
  - Normalize:
    - `gyro_corrected = gyro_raw - gyro_bias`
    - `acc_norm = acc_raw / g`

- Filtering:
  - Low Pass filtering (for noise reduction)
  - High pass filtering
  - Gravity Removal

- Features:
  - `index_acc - wrist_acc`
  - `index_gyro - wrist_gyro`
  - `cross-correlation features`

### Implementation: `data_fusion_project.processing`

The pipeline above is implemented as a configurable package in
`src/data_fusion_project/processing/`. It reads the `data/` tree, processes every window
through four declarative stages and returns CNN-ready NumPy arrays.

| Stage | Module | What it does |
|-------|--------|--------------|
| Calibration | `calibration.py` | Estimates gyro zero-bias, gravity magnitude/direction and acc bias from each session's `calibration_<index>.csv`, then applies `gyro - bias` and `acc / g`. |
| Filtering | `filters.py` | Zero-phase Butterworth low-/high-/band-pass (scipy `sosfiltfilt`) plus gravity removal (low-pass split into gravity + linear acceleration). |
| Orientation | `orientation.py` | Computes roll/pitch from the *pre-filtered* signals and refines them with a fusion filter: `accel`, `gyro`, **`complementary`** or **`kalman`** (2-state, estimates gyro bias). |
| Features | `features.py` | Assembles the `(T, C)` channel matrix (raw acc/gyro, inter-IMU differences, magnitudes, roll/pitch) and optional scalar features (cross-correlation, statistics). |

Everything is driven by `PipelineConfig` (and its stage configs), so feature experiments
are a one-line change — see [documentation/pipeline_configuration.md](documentation/pipeline_configuration.md)
for every setting, its default and what it does. The result is a `GestureDataset` container:

- `X` — `float32` time-series tensor of shape `(N, T, C)` (directly feeds `Conv1D`)
- `y` — `int` labels `(N,)`; `groups` — session id per window `(N,)` for leave-session-out splits
- `features` — optional scalar-feature matrix `(N, F)`; plus `class_names` / `channel_names` / `feature_names`

```python
from data_fusion_project.processing import (
    load_dataset, PipelineConfig, OrientationConfig, FeatureConfig,
    OrientationMethod, leave_sessions_out,
)

# Default configuration: calibrated + low-pass filtered + complementary roll/pitch.
ds = load_dataset()
print(ds.summary())                 # shapes, channels, per-class counts

# Custom experiment: Kalman orientation for both IMUs + inter-IMU diff + cross-correlation.
cfg = PipelineConfig(
    orientation=OrientationConfig(method=OrientationMethod.KALMAN, imus=("IMU1", "IMU2")),
    features=FeatureConfig(include_diff_acc=True, include_diff_gyro=True, cross_correlation=True),
)
ds = load_dataset(cfg)

# Honest evaluation: no session appears in both train and test.
train_idx, test_idx = leave_sessions_out(ds.groups, test_fraction=0.2)
X_train, y_train = ds.X[train_idx], ds.y[train_idx]   # -> CNN
```

CLI / cached export:

```bash
python scripts/build_dataset.py --orientation kalman --diff --cross-correlation
python scripts/build_dataset.py --save data/cache/dataset.npz   # reload via GestureDataset.load(...)
```


---


## CNN Training



---

## Real time classification pipeline

### sliding window setup:

- sampling rate: 100 Hz
- window length: 1.5 s
- stride: 100–200 ms
- prediction rate: 5–10 Hz

### Runtime

1. Receive samples from wrist and index.
2. Resample both to common 100 Hz time grid.
3. Append to synchronized ring buffer.
4. Every 100–200 ms:
   - extract last 1.0 s window
   - normalize using training scaler
   - classify
   - smooth probabilities
   - pass to state machine
5. If action is triggered:
   - send keyboard/mouse event


---



## `data_fusion_project` python module

Codebase: `src/data_fusion_project/`

Capabilities:
- path resolution
- logging
- cli ui
- data processing & feature extraction (`data_fusion_project.processing`): calibration, filtering, roll/pitch sensor fusion (complementary & Kalman), configurable feature assembly, CNN-ready dataset loading