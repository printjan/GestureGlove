# Real-Time Asynchronous Inference System

To validate our model's robustness in a realistic scenario, we implement a real-time inference system that either ingests data from physical IMU sensors or simulates high-frequency input streams.

## System Workflow

The real-time inference system executes in the following sequence:
1. **Sensor Ingestion:** Connects to dual-IMU serial ports (specified in `config/devices.yml`). Alternatively, starts high-frequency simulated streams when `--simulate` is enabled.
2. **Static Calibration:** Prompts the user to hold still for 6.0 seconds. Computes the baseline offset and aligns sensor timestamps.
3. **Asynchronous Slicing:** Asynchronously collects data, extracts sliding windows, and resamples/interpolates sensor signals.
4. **ZUPT Background Calibration:** Continuously monitors signal standard deviations in the background. If stillness is detected, it recalibrates the gyroscope bias registers on-the-fly.
5. **Preprocessing & Pre-Prediction Transform:** Passes the calibrated sliding windows through the custom model transform callback (scaling and feature selection).
6. **Action Dispatcher:** Translates classified gestures into keyboard shortcuts using [powerpoint_control.yml](file:///Users/jantischner/Library/CloudStorage/OneDrive-Personal/TH_OHM_B.Sc.Inf/Th-Ohm_B.Sc.Inf_Sem6/DatFus_Sem6_Axenie/DataFusionProject/config/powerpoint_control.yml) and fires key events to control the active PowerPoint window.

### Usage Commands

* **Live Mode (Physical Rigs):**
  ```bash
  python scripts/run_realtime_inference_test_async.py --model-dir models/late_fusion_cnn_test --threshold 0.95
  ```

* **Simulated Dry-Run (No Hardware Needed):**
  Useful for quick offline verification and pipeline logic checks:
  ```bash
  python scripts/run_realtime_inference_test_async.py --model-dir models/late_fusion_cnn_test --threshold 0.95 --no-control --simulate --timeout 20
  ```

* **Disable Background ZUPT Recalibration:**
  Disable the dynamic background stillness recalibration via:
  ```bash
  python scripts/run_realtime_inference_test_async.py --model-dir models/late_fusion_cnn_test --threshold 0.95 --no-zupt
  ```

---

# Asynchronous Data Grabber - Technical Documentation

This document describes the design, architecture, implementation, and usage of the **Asynchronous Data Grabber** (`AsynchronousDataGrabber`) implemented in `src/data_fusion_project/inference/data_grabber.py`.

---

## 1. Motivation: Why is it Needed?

Real-time gesture recognition systems must process high-frequency sensor streams (100 Hz per IMU, dual IMU setup) and deliver low-latency control commands. In a synchronous design:
1. Ingestion of serial data, synchronization of streams, window preprocessing, scaling, neural network forward pass, and shortcut dispatching are all executed sequentially in a single main-thread loop.
2. The neural network forward pass (executing TensorFlow/PyTorch via Keras on CPU) is computationally intensive, introducing variable latency depending on hardware.
3. When the execution loop is blocked by the inference forward pass, the serial input buffers continue to fill up. 

This synchronous bottleneck results in:
*   **Buffer Backlogs**: The physical inputs accumulate, and the system falls behind real time.
*   **Latency Accumulation**: The end-to-end delay between the physical gesture and slide transition grows continuously over time (lag accumulation).
*   **Packet Loss/Skew**: Alignment logic becomes computationally expensive because it must search and process increasingly larger pandas dataframes to align timestamps.

The `AsynchronousDataGrabber` resolves this by decoupling the **Data Ingestion and Preprocessing** (100 Hz, I/O-bound and lightweight CPU-bound math) from the **Model Inference** (10–30 Hz, CPU/GPU-bound heavy tensor math) using a multi-threaded Producer-Consumer architecture.

---

## 2. Software Architecture & Component Interaction

The real-time inference pipeline uses a three-tier concurrent threading model:

```mermaid
flowchart TD
    subgraph Serial Ingestion (I/O Bound)
        IMU1_Serial[ESP32 IMU1 Serial Stream] -->|readline| IMU1_Thread[IMUDataInput Thread 1]
        IMU2_Serial[ESP32 IMU2 Serial Stream] -->|readline| IMU2_Thread[IMUDataInput Thread 2]
    end

    subgraph Data Grabber (I/O & Preprocessing)
        IMU1_Thread -->|Queue.put| Queue1[(IMU1 Raw Queue)]
        IMU2_Thread -->|Queue.put| Queue2[(IMU2 Raw Queue)]
        
        Queue1 -->|Queue.get_nowait| DG_Thread[AsynchronousDataGrabber Thread]
        Queue2 -->|Queue.get_nowait| DG_Thread
        
        DG_Thread -->|1. process_stream| Align[Timestamp Alignment & Interpolation]
        DG_Thread -->|2. ZUPT Check| ZUPT[Stillness recalibration / EMA update]
        DG_Thread -->|3. process_window| Preprocess[Calibration, Filters & Orientation]
        DG_Thread -->|4. transform_fn| Transform[Slice Channels & TimeSeriesScaler]
        
        Transform -->|Atomic Write| SafeSlot((Shared Thread-Safe Slot))
        Transform -->|Signaling| Event[threading.Event]
    end

    subgraph Inference System (Compute Bound)
        SafeSlot -->|get_newest_frame| Main_Thread[Inference Main Thread]
        Event -->|wait| Main_Thread
        
        Main_Thread -->|5. predict| CNN[Multi-Branch 1D CNN Model]
        Main_Thread -->|6. feed| Dispatcher[GestureDispatcher]
        Dispatcher -->|7. trigger| Controller[PowerPointController / Keyboard OS Event]
    end
```

### Component Roles & Interaction:
1.  **Ingestion (Producer Part 1)**: Two independent `IMUDataInput` background threads continuously read serial packets at 100 Hz, placing parsed raw dictionaries into thread-safe `queue.Queue` buffers.
2.  **Grabber Loop (Producer Part 2)**: The `AsynchronousDataGrabber` background thread runs at a high frequency (e.g. 100 Hz, `poll_interval_s=0.01`). It drains the queues, updates its internal sliding windows, aligns the dual IMU streams, performs stillness detection (ZUPT), updates calibration profiles, and runs a custom model-specific tensor transformation callback. It writes the result into a shared slot.
3.  **Inference Consumer**: The main application thread runs a loop. When it is ready to execute a forward pass, it queries the shared slot of the grabber. If no new frame is available, it blocks/waits. Once it retrieves a frame, it performs inference, feeds predictions to the `GestureDispatcher`, and starts the next iteration.

---

## 3. Implementation Details

The `AsynchronousDataGrabber` class leverages several synchronization and memory management techniques:

### Thread-Safe Slot (Lossy Queue of Capacity 1)
Instead of placing preprocessed tensors in a standard queue, the grabber uses a lock-protected shared variable slot (`_latest_frame`). 
*   If the Grabber thread finishes preprocessing a new window while the Inference thread is busy, the grabber overwrites the slot with the newer data.
*   This represents a **lossy FIFO of size 1**, guaranteeing that the model always gets the most up-to-date data. It completely avoids backlog lag.
*   Writes are atomic and protected by a `threading.Lock`:
    ```python
    with self._frame_lock:
        self._latest_frame = frame
        self._new_frame_event.set()
    ```

### Inter-Thread Signaling
The grabber uses a `threading.Event` (`_new_frame_event`) to avoid CPU spinning in the consumer thread. When a new frame is processed, the grabber calls `.set()`. The consumer's `get_newest_frame(block=True, timeout=...)` calls `.wait(timeout)` to sleep efficiently until data is available.

### Memory Optimization: Buffer Trimming
To keep memory footprint and computational overhead of alignment constant, the grabber maintains a sliding buffer and regularly prunes old packets.
```python
def _trim_before(self, buf: list, cutoff_us: int) -> None:
    i = 0
    while i < len(buf) and buf[i]['pc_timestamp_us'] < cutoff_us:
        i += 1
    if i:
        del buf[:i]
```
This trims raw buffers after sliding forward, keeping the length of dataframes passed to scipy/numpy interpolation filters strictly bounded to the necessary 1.5 seconds.

### Sensor Health Monitoring
If either `IMUDataInput` background thread crashes due to serial connection loss, the grabber thread logs the error, terminates, and `get_newest_frame` raises a `RuntimeError`. This enables the main application to immediately fail fast, disconnect serial ports, and exit cleanly.

### Zero-Velocity Updates (ZUPT) Background Calibration
To eliminate gyroscope bias drift over time (which skew integrated orientation estimation), `AsynchronousDataGrabber` monitors the standard deviation of raw accelerometer and gyroscope signals in `window_df` (1.5s / 150 samples).
*   **Stillness Conditions**: $\sigma_{\text{gyro}} < 3.0$ dps and $\sigma_{\text{acc}} < 0.025$ g.
*   **Update Rule**: When stillness is met for an IMU, the gyroscope bias is updated via an Exponential Moving Average (EMA) with $\beta = 0.1$:
    $$\mathbf{b}_{new} = (1 - \beta) \cdot \mathbf{b}_{current} + \beta \cdot \mathbf{b}_{measured}$$
*   **CLI Control**: ZUPT can be toggled via the `enable_zupt` parameter in the grabber's constructor, or using the `--no-zupt` flag at the CLI script level.
*   **Output Logging**: Recalibration events are printed to terminal (stdout) in blue text, rate-limited to at most once per second to prevent clutter.

---

## 4. API Reference & Usage Guide

### Class Constructor
```python
def __init__(
    self,
    imu1: IMUDataInput,
    imu2: IMUDataInput,
    pipeline_config: PipelineConfig,
    calibration_profile: CalibrationProfile,
    window_size_samples: int = 150,
    advance_samples: int = 10,
    freq_hz: float = 100.0,
    max_diff_us: int = 10000,
    transform_fn: Optional[Callable[[np.ndarray, list[str]], Any]] = None,
    poll_interval_s: float = 0.01,
    enable_zupt: bool = True,
)
```

### Integration Recipe
```python
import joblib
import numpy as np
from data_fusion_project.recording.input_data import IMUDataInput
from data_fusion_project.processing import PipelineConfig, estimate_calibration
from data_fusion_project.inference import AsynchronousDataGrabber

# 1. Initialize sensors
imu1 = IMUDataInput(port="/dev/cu.usbserial-1", name="IMU1")
imu2 = IMUDataInput(port="/dev/cu.usbserial-2", name="IMU2")
imu1.start()
imu2.start()

# 2. Load scalers and metadata
scaler_wrist = joblib.load("models/scaler_x_wrist.joblib")
scaler_finger = joblib.load("models/scaler_x_finger.joblib")
metadata = {"wrist_channels": [...], "finger_channels": [...]}

# 3. Define the transform callback
def transform_fn(channels, channel_names):
    # Slice channels as expected by model branches
    wrist_idx = [channel_names.index(ch) for ch in metadata["wrist_channels"]]
    finger_idx = [channel_names.index(ch) for ch in metadata["finger_channels"]]
    
    # Reshape to (1, T, C)
    X_wrist = channels[:, wrist_idx][np.newaxis, :, :]
    X_finger = channels[:, finger_idx][np.newaxis, :, :]
    
    # Scale features
    return scaler_wrist.transform(X_wrist), scaler_finger.transform(X_finger)

# 4. Instantiate and start Grabber (ZUPT enabled by default)
grabber = AsynchronousDataGrabber(
    imu1=imu1,
    imu2=imu2,
    pipeline_config=PipelineConfig(),
    calibration_profile=estimate_calibration(...),
    transform_fn=transform_fn,
    enable_zupt=True
)
grabber.start()

# 5. Core Consumer Loop
try:
    while True:
        # Blocks up to 100ms waiting for the next sliding window preprocessed frame
        frame = grabber.get_newest_frame(block=True, timeout=0.1)
        if frame is None:
            continue
            
        X_wrist_scaled, X_finger_scaled = frame
        
        # Forward pass on GPU/CPU
        preds = model.predict({"wrist_input": X_wrist_scaled, "finger_input": X_finger_scaled}, verbose=0)
        # Dispatch gesture...
finally:
    grabber.stop()
    imu1.stop()
    imu2.stop()
```
