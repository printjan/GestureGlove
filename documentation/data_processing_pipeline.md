# Pipeline Configuration

This document is a reference for **`PipelineConfig`** — the object that controls every stage
of the data-processing / feature-extraction pipeline in
[`processing/config.py`](../src/data_fusion_project/processing/config.py). It lists every
setting, its type, default, allowed values and what it does, so feature experiments can be
configured without reading the source.

The pipeline runs four stages per window, in this order:

```
load data -> [1] Calibration -> [2] Filtering -> [3] Orientation -> [4] Features -> (X, y, ...)
```

Each stage has its own config dataclass, all bundled in `PipelineConfig`. Every option has a
sensible default, so `PipelineConfig()` alone already produces a usable dataset.

---

## 0. How to set it

Configuration is plain Python dataclasses — set only what you want to change, defaults fill the rest.

```python
from data_fusion_project.processing import (
    load_dataset, PipelineConfig, CalibrationConfig, FilterConfig,
    OrientationConfig, FeatureConfig, FilterType, OrientationMethod,
)

config = PipelineConfig(
    window_size=150,
    filters=FilterConfig(acc_filter=FilterType.LOWPASS, acc_cutoff_hz=8.0),
    orientation=OrientationConfig(method=OrientationMethod.KALMAN, imus=("IMU1", "IMU2")),
    features=FeatureConfig(include_diff_acc=True, cross_correlation=True),
)

ds = load_dataset(config)        # -> GestureDataset (X, y, groups, channel_names, ...)
print(config.to_dict())          # inspect the full effective configuration
```

A subset of these options is also exposed on the command line via
[`scripts/build_dataset.py`](../scripts/build_dataset.py) (run with `--help`).

---

## 1. `PipelineConfig` — global parameters

Top-level acquisition parameters plus the four nested stage configs.

| Field | Type | Default | Options / Range | What it does |
|-------|------|---------|-----------------|--------------|
| `sample_rate_hz` | float | `100.0` | > 0 | Sampling rate of the recordings. Used for filter cutoffs and angle integration. Must match the recording rate (100 Hz). |
| `window_size` | int | `150` | > 0 | Fixed number of time steps `T` per window. For active gestures, a 150-sample window is sliced from the raw 1.74-second CSV starting at the index defined in the companion `.txt` file. Defines the `T` axis of the output tensor `X`. |
| `pad_mode` | str | `"edge"` | `"edge"`, `"zero"` | How windows shorter than `window_size` are padded. Note that under the strict start-index companion scheme, gesture samples are cropped to exactly 150 samples without needing padding. |
| `jitter_range` | int | `0` | ≥ 0 | Maximum sample shift for translation jitter augmentation during dataset loading. Shifting `start_idx` by a random offset in `[-jitter_range, jitter_range]` clipped to raw CSV limits. |
| `calibration` | `CalibrationConfig` | defaults | see §2 | Calibration stage settings. |
| `filters` | `FilterConfig` | defaults | see §3 | Filtering stage settings. |
| `orientation` | `OrientationConfig` | defaults | see §4 | Orientation (roll/pitch) stage settings. |
| `features` | `FeatureConfig` | defaults | see §5 | Feature-selection settings. |

---

## 2. `CalibrationConfig` — bias removal & normalization

Uses the session's calibration files (e.g. `calibration_0.csv`, `calibration_1.csv`, etc., as resolved via `recording_session.json`) to correct the raw signals.

| Field | Type | Default | Options / Range | What it does |
|-------|------|---------|-----------------|--------------|
| `enabled` | bool | `True` | `True` / `False` | Master switch. If `False`, signals are passed through uncorrected. |
| `remove_gyro_bias` | bool | `True` | `True` / `False` | Subtracts the gyroscope zero bias estimated during stillness (`gyro - bias`). Removes constant drift; recommended on. |
| `normalize_acc_to_g` | bool | `True` | `True` / `False` | Divides the accelerometer by the measured gravity magnitude `g` so 1 g maps to 1.0, correcting scale errors. |
| `remove_acc_bias` | bool | `False` | `True` / `False` | Subtracts the full accelerometer bias vector. This also removes the gravity component, so leave it off unless you intend to strip gravity here instead of in the filter stage. |
| `anchor_samples` | int | `0` | ≥ 0 | Number of leading calibration samples to skip before estimating biases, to avoid settling transients at the start of the still pose. |

---

## 3. `FilterConfig` — digital filters

Zero-phase Butterworth filters (`scipy.signal.sosfiltfilt`, no time shift) applied to the
calibrated signals before orientation and feature extraction. Accelerometer and gyroscope
are filtered independently.

| Field | Type | Default | Options / Range | What it does |
|-------|------|---------|-----------------|--------------|
| `enabled` | bool | `True` | `True` / `False` | Master switch for all filtering. |
| `acc_filter` | `FilterType` | `LOWPASS` | `none`, `lowpass`, `highpass`, `bandpass` | Filter type for the 3 accelerometer axes. See §6 for the enum. |
| `acc_cutoff_hz` | float \| (float, float) | `8.0` | 0 < f < `sample_rate_hz`/2 | Accelerometer cutoff in Hz. A single value for low-/high-pass; a `(low, high)` tuple for band-pass. |
| `gyro_filter` | `FilterType` | `LOWPASS` | `none`, `lowpass`, `highpass`, `bandpass` | Filter type for the 3 gyroscope axes. |
| `gyro_cutoff_hz` | float \| (float, float) | `12.0` | 0 < f < `sample_rate_hz`/2 | Gyroscope cutoff in Hz (single value, or `(low, high)` for band-pass). Gyro often keeps a higher cutoff than acc to preserve fast rotation. |
| `order` | int | `2` | ≥ 1 | Butterworth filter order. Higher = sharper roll-off but more ringing on short windows. 2–4 is typical. |
| `remove_gravity` | bool | `False` | `True` / `False` | Additionally splits the accelerometer into a gravity component (low-pass at `gravity_cutoff_hz`) and a gravity-free linear-acceleration component. |
| `gravity_cutoff_hz` | float | `0.5` | 0 < f < `sample_rate_hz`/2 | Low-pass cutoff used to estimate the slowly-varying gravity component. Typically 0.3–0.8 Hz. |
| `replace_acc_with_linear` | bool | `False` | `True` / `False` | When gravity removal is on, replaces the accelerometer channels with the linear (gravity-free) acceleration instead of keeping the raw filtered values. |

> **Note on cutoffs:** the cutoff must stay below the Nyquist frequency (`sample_rate_hz / 2` = 50 Hz here). For `bandpass`, pass a tuple, e.g. `acc_cutoff_hz=(0.5, 20.0)`.

---

## 4. `OrientationConfig` — roll/pitch sensor fusion

Computes roll and pitch from the **pre-filtered** signals (gyro bias already removed), then
refines them with the chosen fusion filter. Yaw is not estimated (unobservable without a
magnetometer).

| Field | Type | Default | Options / Range | What it does |
|-------|------|---------|-----------------|--------------|
| `enabled` | bool | `True` | `True` / `False` | Master switch for the orientation stage. If `False`, no roll/pitch channels are produced. |
| `method` | `OrientationMethod` | `COMPLEMENTARY` | `none`, `accel`, `gyro`, `complementary`, `kalman` | Fusion algorithm. See §6 for behavior of each. |
| `imus` | tuple of str | `("IMU1",)` | `("IMU1",)`, `("IMU2",)`, `("IMU1", "IMU2")` | Which IMUs get roll/pitch computed. Each adds 2 channels (`<imu>_roll`, `<imu>_pitch`) when included in features. |
| `alpha` | float | `0.98` | 0.0 – 1.0 | Complementary-filter weight on the gyro path. Higher trusts the gyro more short-term; the accelerometer anchors long-term. Typical 0.95–0.99. Only used by `complementary`. |
| `degrees` | bool | `True` | `True` / `False` | Output angles in degrees (`True`) or radians (`False`). |
| `kalman_q_angle` | float | `0.001` | > 0 | Kalman process noise of the angle state. Larger = trusts the gyro prediction less / reacts faster to the accelerometer. Only used by `kalman`. |
| `kalman_q_bias` | float | `0.003` | > 0 | Kalman process noise of the gyro-bias state. Controls how quickly the estimated bias can change. Only used by `kalman`. |
| `kalman_r_measure` | float | `0.03` | > 0 | Kalman measurement noise of the accelerometer angle. Larger = trusts the (noisy) accelerometer less, giving smoother but slower tracking. Only used by `kalman`. |

---

## 5. `FeatureConfig` — channel & feature selection

Decides which **time-series channels** form the `(T, C)` tensor `X` and which **scalar
features** form the optional `(F,)` vector `features`.

| Field | Type | Default | Options / Range | What it does |
|-------|------|---------|-----------------|--------------|
| `imus` | tuple of str | `("IMU1", "IMU2")` | `("IMU1",)`, `("IMU2",)`, `("IMU1", "IMU2")` | Which IMUs contribute raw channels. |
| `include_acc` | bool | `True` | `True` / `False` | Include the 3 accelerometer axes per selected IMU (`<imu>_accX/Y/Z`). |
| `include_gyro` | bool | `True` | `True` / `False` | Include the 3 gyroscope axes per selected IMU (`<imu>_gyrX/Y/Z`). |
| `include_acc_magnitude` | bool | `False` | `True` / `False` | Add the accelerometer magnitude `‖acc‖` as one channel per IMU (`<imu>_acc_mag`). Orientation-invariant motion intensity. |
| `include_gyro_magnitude` | bool | `False` | `True` / `False` | Add the gyroscope magnitude `‖gyr‖` as one channel per IMU (`<imu>_gyr_mag`). |
| `include_diff_acc` | bool | `False` | `True` / `False` | Add the inter-IMU acc difference `IMU2_acc − IMU1_acc` (3 channels, `diff_accX/Y/Z`). Captures finger-relative-to-wrist motion. Needs both IMUs present. |
| `include_diff_gyro` | bool | `False` | `True` / `False` | Add the inter-IMU gyro difference `IMU2_gyr − IMU1_gyr` (3 channels, `diff_gyrX/Y/Z`). Needs both IMUs present. |
| `include_orientation` | bool | `True` | `True` / `False` | Include the roll/pitch channels produced by the orientation stage (§4). |
| `cross_correlation` | bool | `False` | `True` / `False` | Add scalar cross-correlation features between corresponding wrist/finger axes (zero-lag correlation, peak correlation, lag). Goes into `features`, not `X`. |
| `statistics` | bool | `False` | `True` / `False` | Add per-channel scalar statistics (mean/std/min/max/rms) over each window. Goes into `features`, not `X`. |
| `include_linear_jerk` | bool | `False` | `True` / `False` | Add lowpass-filtered linear jerk (3 channels, `<imu>_linear_jerkX/Y/Z`). |
| `include_angular_acceleration` | bool | `False` | `True` / `False` | Add angular acceleration (3 channels, `<imu>_angular_accelerationX/Y/Z`). |
| `include_relative_acceleration` | bool | `False` | `True` / `False` | Add relative acceleration `IMU2_acc − IMU1_acc` (3 channels, `relative_accelerationX/Y/Z`). |
| `include_relative_rotation` | bool | `False` | `True` / `False` | Add relative rotation `IMU2_gyr − IMU1_gyr` (3 channels, `relative_rotationX/Y/Z`). |
| `include_relative_yaw` | bool | `False` | `True` / `False` | Add relative yaw integrated over active window (`<imu>_relative_yaw`). |
| `include_accelerometer_magnitude` | bool | `False` | `True` / `False` | Add accelerometer magnitude (`<imu>_accelerometer_magnitude`). |
| `include_gyroscope_magnitude` | bool | `False` | `True` / `False` | Add gyroscope magnitude (`<imu>_gyroscope_magnitude`). |
| `include_gravity_free_linear_acceleration` | bool | `False` | `True` / `False` | Add gravity-free linear acceleration using pitch/roll (3 channels, `<imu>_gravity_free_linear_accelerationX/Y/Z`). |

### How settings map to the output channels

The number of channels `C` in `X` is the sum of everything enabled above. Channel order is
deterministic: raw channels (per IMU: acc, gyro, magnitudes) → inter-IMU differences →
orientation. Example with defaults (`imus=("IMU1","IMU2")`, acc+gyro on, orientation on for
`IMU1` only):

```
IMU1_accX, IMU1_accY, IMU1_accZ, IMU1_gyrX, IMU1_gyrY, IMU1_gyrZ,
IMU2_accX, IMU2_accY, IMU2_accZ, IMU2_gyrX, IMU2_gyrY, IMU2_gyrZ,
IMU1_roll, IMU1_pitch                                              ->  C = 14
```

`ds.channel_names` and `ds.feature_names` always list the exact produced order.

---

## 6. Enumerations

### `FilterType`

| Value | Meaning |
|-------|---------|
| `none` | No filtering (pass-through). |
| `lowpass` | Keeps frequencies below the cutoff. Reduces high-frequency noise. |
| `highpass` | Keeps frequencies above the cutoff. Removes slow drift / constant offset. |
| `bandpass` | Keeps a frequency band; requires a `(low, high)` cutoff tuple. |

### `OrientationMethod`

| Value | Meaning | Characteristics |
|-------|---------|-----------------|
| `none` | Do not compute orientation. | No roll/pitch channels. |
| `accel` | Roll/pitch from the accelerometer only. | No drift, but noisy and corrupted by linear acceleration. |
| `gyro` | Integrated gyroscope only. | Smooth, but drifts over time. |
| `complementary` | Blends accel (long-term) and gyro (short-term) via `alpha`. | Smooth and drift-free; one tuning knob. |
| `kalman` | 2-state Kalman filter (angle + gyro bias). | Smooth and drift-free; also tracks/removes a slow gyro bias; three noise knobs. |

---

## 7. Common recipes

```python
# A) Raw signals only (no filtering, no orientation) — baseline / debugging
PipelineConfig(
    filters=FilterConfig(enabled=False),
    orientation=OrientationConfig(enabled=False),
    features=FeatureConfig(include_orientation=False),
)

# B) Gravity-free linear acceleration + Kalman orientation on both IMUs
PipelineConfig(
    filters=FilterConfig(remove_gravity=True, replace_acc_with_linear=True),
    orientation=OrientationConfig(method=OrientationMethod.KALMAN, imus=("IMU1", "IMU2")),
)

# C) Feature-rich: differences, magnitudes, cross-correlation + statistics
PipelineConfig(
    features=FeatureConfig(
        include_diff_acc=True, include_diff_gyro=True,
        include_acc_magnitude=True, include_gyro_magnitude=True,
        cross_correlation=True, statistics=True,
    ),
)

# D) Band-pass the gyro to isolate gesture-band rotation (0.5–20 Hz)
PipelineConfig(
    filters=FilterConfig(gyro_filter=FilterType.BANDPASS, gyro_cutoff_hz=(0.5, 20.0)),
)
```
