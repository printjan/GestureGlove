# IMU Gesture Glove

![IMU Gesture Glove](gesture_glove_presentation/gesture-glove.png)

## Team

- Lucas Horn: `hornlu95907@th-nuernberg.de`
- Jan Tischner: `tischnerja95752@th-nuernberg.de`


---


## Project description

### Hardware - IMU Sensor boards:

- We are using two `XIAOML Kit` devices: Designed by Professor Vijay Janapa Reddi (Harvard University), author of the Machine Learning Systems textbook. One at the wrist and one at the index finger.
- What's inside: XIAO ESP32-S3 Sense, CAM • IMU • SD Toolkit
- Advertising: Build keyword detection, image classification, motion detection, object detection, and more
- Links: For ...
  - Learners: mlsysbook.ai
  - Builders: mlsysbook.ai/kits
  - Developers: github.com/mlsysbook

### Hardware - Setup:

- The two XIAOML Kits are directly conncted to the computer via USB-C.
- IMU Data will be streamed unprocessed via USB-C-Serial to the computer.
- All processing, fusion, filtering, and ML will run on the Computer. 

### Hardware - Mounting:

- One XIAOML Kit on the wrist (IMU Data).
- One XIAOML Kit on the tip of the index finger (Camera Data).
- Orientation usb-c-plug downward and backward.
- Mounted on right hand.
- For consistency we are using a modified garden glove: We stuck one `XIAOML Kit` on the upper side of the palm and the other one on the upper side of the index finger.
  
### ProjectGoal:

- Recognize arm- and hand-gestures with wrist worn IMU Sensor.
- Demonstation: Control power point with hand gestures.

#### Demonstration Video

<video src="gesture_glove_presentation/gesture-glove_presentation-demonstration.mp4" controls width="100%"></video>

*If the video player above does not render, you can view the video file directly here: [gesture-glove_presentation-demonstration.mp4](gesture_glove_presentation/gesture-glove_presentation-demonstration.mp4)*

### Possible Future Extension:

- Use finger as an air mouse to interact with the computer.
- Demonstration: Cotrol the power point laser pointer by hand movement.


---


## Guestures

### Very important:

> - **Discrete Movement:** Recognizable Start and Stop of the movement with a stationary moment before and after to differentiate the geusture from natural movement!
> - **Calibration:** 
>   - 3 seconds still pose:
>   - wrist positioned naturally
>   - index finger extended or relaxed in defined pose

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

In the dataset and classifiers the naming scheme is implemented as follows:

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


# Project Structure

For detailed documentation on the project data structures and pipelines, please refer to:
- **Recorded Data & Directory Structure**: [documentation/data_recording_pipeline.md](documentation/data_recording_pipeline.md)
- **Model Training & Metadata Schema**: [documentation/model_training_pipeline.md](documentation/model_training_pipeline.md)

---

# Project Components

## Data Recording Pipeline

- [Data Recording Pipeline Documentation](documentation/data_recording_pipeline.md) — Details on the asynchronous multi-threaded data acquisition, hidden buffering, and grid resampling.
- [Dataset Analysis Notebook](data_analysis/dataset_analysis.ipynb) — Interactive Jupyter notebook providing comparative visualizations of dataset versions, class balance analysis, session density distributions, calibration frequency audits, centroid start index histograms, and motion energy centering case studies.

---

## Data Processing Pipeline

- [Data Processing Pipeline Documentation](documentation/data_processing_pipeline.md) — Documentation of the processing pipeline, calibration logic, Butterworth filtering, and dynamic feature calculations.

---

## Model Training Pipeline

- [Model Training Pipeline Documentation](documentation/model_training_pipeline.md) — Comprehensive overview of the unified configuration-driven training pipeline.
- [Model Training Strategies](documentation/model_training_strategies.md) — Details on baseline comparisons, pipeline evolution, and validation metrics.
- [Model Training Runs & Evaluation Summary](documentation/models.md) — Registration and comparison of the final model training runs.
- **Architectures**:
  - [Early Fusion Single-Branch 1D CNN](documentation/model_architectures/early_fusion_single_branch_1d_cnn.md)
  - [Late Fusion Multi-Branch 1D CNN](documentation/model_architectures/late_fusion_multi_branch_1d_cnn.md)
  - [Self-Attention Temporal Transformer](documentation/model_architectures/self_attention_temporal_transformer.md)
  - [Playground Model Experiments](documentation/model_architectures/playground_model_experiments.md)

---

## Real Time Inference Pipeline

- [Real-Time Inference Pipeline](documentation/real_time_inference_pipeline.md) — Architecture-agnostic inference pipeline supporting all three production architectures with model loader, scaler dispatch, dynamic input routing, ZUPT calibration, and live evaluation.
- [PowerPoint Control Interface](documentation/powerpoint_control_interface.md) — Specifications for the real-time presentation controller.

---

## Installable `python` Module `data_fusion_project`

- [Core Module Documentation](documentation/data_fusion_project_core.md) — Reference for CLI helpers, path resolution, logging configurations, and testing.
- [Model Architecture Definition Files (model.py)](src/data_fusion_project/training/) — Source files defining the structural layout of each architecture (Early Fusion, Late Fusion, and Temporal Transformer).

---

## Code Guidelines

- [Code Guidelines](documentation/codeguidelines.md) — Standards and coding guidelines for the project.

---

## Getting Started: Environment Setup & Execution

### 1. Environment Setup

The python dependencies of this project can be installed in a local `.env/` virtual environment directory (which is automatically excluded from git tracking via `.gitignore`).

We provide automated setup notebooks for different operating systems in the `.setup/` directory:
- **Windows**: [setup_windows_env.ipynb](.setup/setup_windows_env.ipynb)
- **macOS**: [setup_mac_env.ipynb](.setup/setup_mac_env.ipynb)
- **Linux**: [setup_linux_env.ipynb](.setup/setup_linux_env.ipynb)

#### How to use the setup notebooks:
1. Open the notebook matching your OS for example in VS Code or Cursor.
2. Select your default system Python interpreter (Python 3.8+) in the top-right corner to run the cells.
3. Run the cells sequentially to build the environment and register the `data_fusion_env_1` Jupyter kernel.

#### Alternative: Manual Command Line Setup
If you prefer standard CLI tools:
```bash
# 1. Create the virtual environment
python -m venv .env

# 2. Activate the environment
# On Windows (PowerShell / Command Prompt):
.env\Scripts\Activate.ps1   # PowerShell
.env\Scripts\activate.bat   # CMD
# On macOS / Linux:
source .env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch (with CUDA for Windows/Linux)
# Windows:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# macOS / Linux:
pip install torch torchvision torchaudio

# 5. Install TensorFlow & other dependencies
pip install tensorflow
pip install numpy pandas scipy scikit-learn matplotlib pyserial tqdm joblib ipykernel jupyterlab notebook pyyaml h5py pyarrow filterpy tensorboard opencv-python mediapipe pyautogui pynput

# 6. Install the project module in editable mode with PowerPoint control features
pip install -e .[control]
```

---

### 2. Generating Your Own Training Data

1. **Record Gesture Samples**: Connect the physical dual-IMU gesture glove via USB and start the interactive data recording CLI:
   ```bash
   python scripts/record_data.py
   ```
   Follow the on-screen prompts to input the session name, perform calibration (static stillness), and record gestures sample by sample (or continuously for `none` class).

2. **Process and Build Dataset**: Once data is recorded under `data/`, process it and assemble the CNN-ready dataset:
   ```bash
   # Build with complementary/Kalman orientation and save to cache
   python scripts/build_dataset.py --orientation kalman --diff --cross-correlation --save data/cache/dataset.npz
   ```

3. **Archive/Version the Dataset**: Archive your current processed active dataset folder into a versioned release folder:
   ```bash
   python scripts/build_dataset_version.py
   ```

4. **Train a Model**: Train one of the production candidate architectures on your compiled dataset:
   ```bash
   # Train Early Fusion CNN (Standard preset)
   python scripts/train.py --model-type early_fusion_cnn --epochs 70
   
   # Train Late Fusion CNN (with rotation augmentation and Optuna feature sweep)
   python scripts/train.py --model-type late_fusion_cnn --optimize --optuna-trials 25
   ```

---

### 3. Running Real-Time Inference

Run live gesture recognition using any trained model. The pipeline dynamically resolves configurations, applies calibrations, and controls PowerPoint:

* **Live Mode (Physical Glove) — Early Fusion CNN**:
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.95
  ```

* **Live Mode (Physical Glove) — Late Fusion CNN**:
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/late_fusion_multi_branch_1d_cnn --threshold 0.95
  ```

* **Simulated Dry-Run (No Hardware Needed)**:
  Run offline simulated testing with high-frequency stream playback (timeout exit in 20 seconds):
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.95 --no-control --simulate --timeout 20
  ```

* **Live Evaluation with Objection Window**:
  Evaluate live gesture classification and save performance metrics (`reports/`):
  ```bash
  python scripts/run_realtime_inference.py --model-dir models/early_fusion_single_branch_1d_cnn --threshold 0.95 --evaluate --objection-window 1.8 --eval-out reports/
  ```

