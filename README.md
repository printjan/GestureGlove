# IMU Gesture Glove


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
- [Code Guidelines](documentation/codeguidelines.md) — Standards and coding guidelines for the project.

