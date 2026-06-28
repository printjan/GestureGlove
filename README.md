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
│   ├── <model_name>_<timestamp>.keras               # saved trained Keras model structure & weights
│   └── <model_name>_<timestamp>_metadata.json       # JSON file containing training run audit properties
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

### Model Metadata Properties Structure (`<model_name>_<timestamp>_metadata.json`)

```json
{
  "model_name": "late_fusion_multi_branch_1d_cnn",
  "timestamp": "20260628_220600",
  "machine_info": {
    "hostname": "MacBook-Pro",
    "os": "macOS-14.5",
    "cpu": "Apple M3 Max",
    "gpu": "Apple M3 Max (Unified Memory)"
  },
  "training_parameters": {
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "jitter_range": 10,
    "filters": {
      "acc_cutoff_hz": 8.0,
      "gyro_cutoff_hz": 12.0
    }
  },
  "dataset_info": {
    "total_samples": 450,
    "per_class_count": {
      "none": 120,
      "swipe_left": 50,
      "swipe_right": 50,
      "circle_cw": 50,
      "circle_ccw": 50,
      "fist": 45,
      "jerk_down": 45,
      "jerk_up": 40
    },
    "sessions_used": ["session_0", "session_1"]
  },
  "performance": {
    "best_epoch": 42,
    "train_accuracy": 0.985,
    "val_accuracy": 0.962,
    "val_loss": 0.125,
    "val_f1_score": 0.961
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