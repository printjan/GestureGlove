# Implementation Plan: Late Fusion Multi-Branch Conv1D CNN

This document details the architecture design, layers, and engineering justifications for the **Late Fusion Multi-Branch Conv1D CNN** candidate.

## 1. Network Architecture Diagram

```mermaid
graph TD
    Input["Input Window (150, Channels)"] --> W_Split["Wrist Channels"]
    Input --> F_Split["Finger Channels"]
    Input --> S_Split["Statistical / Handcrafted Features"]

    W_Split --> Branch1["Wrist Conv1D Branch<br/>- Conv1D (kernel=5, filters=32)<br/>- BatchNorm & ReLU<br/>- MaxPool1D<br/>- Conv1D (kernel=3, filters=64)<br/>- GlobalAveragePooling1D"]
    
    F_Split --> Branch2["Finger Conv1D Branch<br/>- Conv1D (kernel=5, filters=32)<br/>- BatchNorm & ReLU<br/>- MaxPool1D<br/>- Conv1D (kernel=3, filters=64)<br/>- GlobalAveragePooling1D"]

    S_Split --> Branch3["Dense Feature MLP<br/>- Dense (32, ReLU)<br/>- Dropout (0.5)"]

    Branch1 --> Concatenate["Concatenate Layer"]
    Branch2 --> Concatenate
    Branch3 --> Concatenate

    Concatenate --> Classifier["FC Dense (64, ReLU)"]
    Classifier --> Dropout["Dropout (0.3)"]
    Dropout --> Softmax["Softmax Layer (8 Classes)"]
```

---

## 2. Detailed Layer Specifications

### A. Temporal Branches (Wrist & Finger)
Each of the two parallel Conv1D branches is constructed as follows:

| Layer Type | Specifications | Output Shape | Activation / Purpose |
|---|---|---|---|
| **Input Branch** | Dynamic sliced channels `(150, C_sub)` | `(None, 150, C_sub)` | Input binding |
| **Conv1D** | 32 filters, kernel=5, padding="same" | `(None, 150, 32)` | ReLU |
| **Batch Normalization** | Normalizes activations along channels | `(None, 150, 32)` | Stability |
| **MaxPool1D** | pool_size=2, stride=2 | `(None, 75, 32)` | Downsampling |
| **Conv1D** | 64 filters, kernel=3, padding="same" | `(None, 75, 64)` | ReLU |
| **Batch Normalization** | Normalizes activations along channels | `(None, 75, 64)` | Stability |
| **GlobalAveragePooling1D** | Average pooling along time axis | `(None, 64)` | Temporal extraction |

### B. Statistical Summary Branch (MLP)
For scalar features that summarize the entire window (e.g. cross-correlation coefficients and statistics):

| Layer Type | Specifications | Output Shape | Activation / Purpose |
|---|---|---|---|
| **Input Branch** | Flat scalar features `(F,)` | `(None, F)` | Summary input |
| **Dense** | 32 hidden units | `(None, 32)` | ReLU |
| **Dropout** | Dropout rate = 50% | `(None, 32)` | Regularization |

### C. Late Fusion & Classifier Layers

| Layer Type | Specifications | Output Shape | Activation / Purpose |
|---|---|---|---|
| **Concatenate** | Merges `[Branch1, Branch2, Branch3]` outputs | `(None, 160)` | Late Fusion (64 + 64 + 32) |
| **Dense** | 64 hidden units | `(None, 64)` | ReLU |
| **Dropout** | Dropout rate = 30% | `(None, 64)` | Regularization |
| **Dense (Softmax)** | 8 outputs (one per gesture class) | `(None, 8)` | Softmax activation |

---

## 3. Design Justifications & Precedents

### A. Late Fusion Concept
* **Justification:** Human Activity Recognition (HAR) research shows that separating sensor clusters in early layers performs significantly better than early fusion. The wrist and finger sensors capture different scales of motion (arm translation vs. hand posture). Decoupling their layers allows the filters of Branch 1 to optimize for wrist dynamics, while Branch 2 specializes in fine finger trajectories.

### B. MLP Statistical Branch
* **Justification:** Some features (like cross-correlation or window statistics) are scalar values rather than sequential time-series waveforms. We cannot feed these scalars directly into Conv1D layers. This separate Dense MLP branch embeds these scalar metrics into a `32`-dimensional space before fusing them with the temporal features.
* **Regularization:** The MLP branch uses a high `50% Dropout` rate to prevent the classifier from over-relying on simple statistics (which leads to overfitting on the training user) and forcing it to prioritize the temporal motion shapes.

### C. Dynamic Binding Strategy
* **Justification:** Instead of hardcoding channels, the model uses dynamic column index maps:
  ```python
  wrist_indices = [i for i, name in enumerate(dataset.channel_names) if "IMU1" in name]
  finger_indices = [i for i, name in enumerate(dataset.channel_names) if "IMU2" in name]
  ```
  This ensures that if we configure our features to exclude specific channels, the routing remains correct without requiring a rewrite of the model architecture code.

### D. Spatial-Kinematic Decoupling (Post-Audit Synthesis)
* **Justification:** Random Forest Gini ranking in [feature_filter_analysis_results.json](file:///Users/jantischner/Library/CloudStorage/OneDrive-Personal/TH_OHM_B.Sc.Inf/Th-Ohm_B.Sc.Inf_Sem6/DatFus_Sem6_Axenie/DataFusionProject/data_analysis/feature_filter_analysis_results.json) revealed that inter-IMU difference features (`diff_accZ`, `diff_accY`) hold over **30%** of decision boundary splitting weight. This confirms that arm translation (wrist IMU1) and hand posture (finger relative to wrist) are kinematically decoupled. The late fusion multi-branch model is uniquely suited for this: Branch 1 is fed wrist-only dynamics (arm sweeps), Branch 2 processes finger-relative differences, and the MLP receives short-term relative yaw (highpass-filtered at 0.5 Hz prior to integration to prevent linear drift). This decodes arm vs. hand dynamics in parallel pathways prior to late fusion.

### E. Output Classification Layer (Explicit 8-Class Setup)
* **Justification:** The output Dense classification layer utilizes a Softmax activation over 8 distinct classes (comprising the 7 active gestures and the `none`/idle class). Since continuous PowerPoint control runs continuously in sliding windows, the system is in an idle state 95% of the time. Training the network explicitly on `none` samples forces the convolutional filters to outline explicit decision boundaries in latent space separating noise and idle motion (like keyboard usage) from gesture profiles. A thresholded 7-class system, by contrast, suffers from out-of-distribution extrapolation, projecting random movements confidently into active gesture classes due to Softmax probability saturation. Explicitly modeling `none` is crucial to maintaining a zero false-positive rate.

### F. Input Feature Configuration & Dynamic Selection (Post-Audit Synthesis)
* **Justification:** Based on the feature filter analysis and data quality audit, we classify our features into three tiers:
  * **Pruned (Dismissed):** We completely discard 6 derivative features (such as `IMU1_linear_jerkX/Z` and `IMU1/2_angular_accelerationY/Z`) because they satisfy `RF Gini < 0.002` and `Mutual Information < 0.5`, indicating they only introduce high-frequency noise without adding any discriminatory information.
  * **Mandatory (Kept):** We permanently bind 11 high-yield features (including `IMU1_accX/Z`, `IMU2_accX/Y/Z`, `IMU2_gyrX`, `diff_accX/Z`, `IMU1_pitch`, and `IMU1_gyr_mag`) because they satisfy `Mutual Information > 0.9` and `RF Gini > 0.02`, carrying major motion shape information.
  * **Dynamic Selection via Optuna:** The remaining 21 helper features are selected dynamically during training using a Bayesian Optuna search wrapper. The search wrapper evaluates different candidate feature combinations directly on the Late Fusion Multi-Branch Conv1D CNN architecture over multiple training trials, selecting the configuration that maximizes the Joint Utility Score. This lets the pipeline automatically optimize inputs specifically for the Multi-Branch model.

---

## 4. Experiment Directory & Saving Structure

Every training session for this model must be saved in accordance with the project's experiment directory structure defined in the `README.md`:

```
models/
└── late_fusion_multi_branch_1d_cnn/                 # Model identifier folder
    └── training_session_<index>_<timestamp>/        # Sequential session (e.g., training_session_0_20260629_020000)
        ├── model.keras                              # Saved trained Keras model weights and architecture
        ├── model_metadata.json                      # JSON file containing training run audit properties
        ├── confusion_matrix.png                     # Validation split confusion matrix plot
        └── learning_curves.png                      # Training/validation loss and accuracy curves
```

* **Sequential Indexing**: The training script must dynamically query existing directories under `models/late_fusion_multi_branch_1d_cnn/` to determine the next available sequential integer `<index>` (starting at `0` for the first run).
* **Metadata Logging**: The `model_metadata.json` file must capture system info, hyperparameters, training dataset stats, and per-class precision, recall, and F1-score evaluation metrics.


