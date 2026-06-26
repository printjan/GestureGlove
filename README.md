# Data Fusion Project


## Team

- Lucas Horn: `hornlu95907@th-nuernberg.de`
- Jan Tichner: `tischnerja95752@th-nuernberg.de`


---


## Hardware Setup

### Sensor board:

- We are using two `XIAOML Kit` devices: Designed by Professor Vijay Janapa Reddi (Harvard University), author of the Machine Learning Systems textbook. One at the wrist and one at the index finger.
- What's inside: XIAO ESP32-S3 Sense, CAM • IMU • SD Toolkit
- Advertising: Build keyword detection, image classification, motion detection, object detection, and more
- Links: For ...
  - Learners: mlsysbook.ai
  - Builders: mlsysbook.ai/kits
  - Developers: github.com/mlsysbook

### Setup:

- The two XIAOML Kits are directly conncted to the computer via USB-C.
- IMU Data will be streamed unprocessed via USB-C-Serial to the computer.
- All processing, fusion, filtering, and ML will run on the Computer. 


---


## Project description

**Setup:**
- One XIAOML Kit on the wrist (IMU Data).
- One XIAOML Kit on the tip of the index finger (Camera Data).
- Orientation usb-c-plug downward and backward.
- Mounted on right hand.
  
**Goal:**
- Recognize arm- and hand-gestures with wrist worn IMU Sensor.
- Demonstation: Control power point with hand gestures.

**Possible Extension:**
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


## Project Strucure 

```
data_fusion_project/
├── data/
│   ├── <guesture name>/
│   │   │   ├── <recording_session>/
│   │   │   │   ├── calibration.csv # 5 second recording of no movement to establish sensor drift
│   │   │   │   ├── 00001.csv # first recording of the gesture
│   │   │   │   ├── 00002.csv # second recording of the gesture
│   │   │   │   └── 
```


---


## Data set structure

Column structure of `<id>.csv` or `calibration.csv` files (they only contain raw data):

```csv
IMU1_accX,IMU1_accY,IMU1_accZ,IMU1_gyrX,IMU1_gyrY,IMU1_gyrZ,IMU2_accX,IMU2_accY,IMU2_accZ,IMU2_gyrX,IMU2_gyrY,IMU2_gyrZ
```


---

## Data Processing Pipeline

Extracts trainings data and OPTIONALLY preprocesses it providing different preprocessing options or OPTIONALLY calculates features:

- Calibration: 
  - At the beginning of each recording:
    - 3 seconds still pose:
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
| Calibration | `calibration.py` | Estimates gyro zero-bias, gravity magnitude/direction and acc bias from each session's `calibration.csv`, then applies `gyro - bias` and `acc / g`. |
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


## CNN Experiments


---


## Pitch

- 5 min (auf keinen Fall mehr).
- 4 Folien.
  - 1. Team und Problem.
  - 2./3. Folie Implementierungsdetails.
  - 4. Folie: Gefilmte Demo (keine Live Demo).
- Schöne Animationen sind wichtig!
- Code in Git Repo der Fak Inf ablegen mit Axenie als Maintainer.
- Folien in Repo ablegen (als .pdf).
- Alle Medien (also auch Präsi Videos) auf Git ablegen.
- Mündlich darauf vorbereiten, Fragen zum Projekt zu beantworten (auch kritische).
- Abgabe: 1. Juli 23:59. Kein Commit mehr danach.



---


## `data_fusion_project` python module

Codebase: `src/data_fusion_project/`

Capabilities:
- path resolution
- logging
- cli ui
- data processing & feature extraction (`data_fusion_project.processing`): calibration, filtering, roll/pitch sensor fusion (complementary & Kalman), configurable feature assembly, CNN-ready dataset loading