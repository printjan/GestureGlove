# Implementation Plan: Self-Attention Temporal Transformer

This document details the architecture design, layers, and engineering justifications for the **Self-Attention Temporal Transformer** candidate, incorporating our empirical learnings from the playground baseline experiments (`late_fusion_cnn_test`).

---

## 1. Network Architecture Diagram

Based on our capacity-bottlenecking and regularization learnings, the temporal transformer is updated to restrict classification capacity and enforce strict inputs filtering to prevent overfitting.

```mermaid
graph TD
    Input["Input Window (150, Channels)"] --> Project["Dense Linear Projection<br/>- Projects C channels to d_model=64"]
    Project --> PosEnc["Add Positional Encoding<br/>- Shape-matched learnable encoding vector"]
    PosEnc --> TransEncoder1["Transformer Block 1<br/>- Multi-Head Self-Attention (4 heads, key_dim=16)<br/>- Layer Normalization & Dropout (0.1)<br/>- MLP Feed-Forward (128 units -> 64 units)"]
    TransEncoder1 --> TransEncoder2["Transformer Block 2<br/>- Multi-Head Self-Attention (4 heads, key_dim=16)<br/>- Layer Normalization & Dropout (0.1)<br/>- MLP Feed-Forward (128 units -> 64 units)"]
    TransEncoder2 --> GAP["GlobalAveragePooling1D"]
    GAP --> Classifier["Fully Connected Bottleneck<br/>- Dense (16, ReLU)<br/>- Dropout (0.5)"]
    Classifier --> Softmax["Softmax Classification<br/>- Dense (8 classes)"]
```

---

## 2. Detailed Layer Specifications

The classification head is constrained to a 16-unit bottleneck with 50% dropout (derived from Experiment D, our best performer) to prevent the network from exploiting session-specific coordinate shifts.

| Layer # | Layer Type | Specifications | Output Shape | Parameters / Activation |
|---|---|---|---|---|
| **0** | **Input** | Dynamic channel count `(150, C)` | `(None, 150, C)` | Input sequence |
| **1** | **Dense (Projection)** | Linear mapping to `d_model=64` | `(None, 150, 64)` | Linear (no activation) |
| **2** | **Positional Add** | Adds learnable vector of shape `(150, 64)` | `(None, 150, 64)` | Temporal ordering |
| **3** | **MultiHeadAttention** | 4 heads, `key_dim=16`, dropout=0.1 | `(None, 150, 64)` | Query, Key, Value extraction |
| **4** | **Layer Normalization** | Applied along `d_model` dimension | `(None, 150, 64)` | Normalization |
| **5** | **Feed-Forward Block** | Dense (128, ReLU) -> Dense (64, Linear) | `(None, 150, 64)` | Dense layer expansions |
| **6** | **Layer Normalization** | Applied along `d_model` dimension | `(None, 150, 64)` | Normalization |
| **7** | **GlobalAveragePooling1D** | Average pooling along time axis | `(None, 64)` | Sequence summary |
| **8** | **Dense (FC)** | **16 hidden units** (reduced from 32) | `(None, 16)` | ReLU activation |
| **9** | **Dropout** | **Dropout rate = 50%** (increased from 30%) | `(None, 16)` | Overfitting mitigation |
| **10** | **Dense (Softmax)** | 8 outputs (one per gesture class) | `(None, 8)` | Softmax activation |

---

## 3. Design Justifications & Baseline Learnings

### A. Temporal Attention vs. Convolutions
* **Justification:** Convolutions extract shift-invariant *local* features. In contrast, self-attention calculates pairwise relationships between **any two time steps in the entire window** directly, capturing global temporal flow and tempo variations without relying on recurrent LSTM states.

### B. Positional Encoding
* **Justification:** The self-attention operation is permutation-invariant. To preserve time ordering, we add a **learnable Positional Encoding vector** to the projected input, embedding temporal index coordinates directly.

### C. Multi-Head Attention Configuration
* **Justification:** Using `4 heads` with a key dimension of `16` keeps the parameters small. Each head can learn to focus on different temporal stages of a gesture (e.g., initial acceleration trigger vs. deceleration landing phase).

### D. Low-Pass Pre-filtering Dependency (Crucial Requirement)
* **Justification:** Attention mechanisms are highly sensitive to noise outliers because the dot-product exponentials in the softmax calculation scale exponentially with magnitudes. A single high-frequency noise spike can dominate the entire attention matrix. Thus, **low-pass filtering accelerometer (8.0 Hz) and gyroscope (12.0 Hz) inputs is critical** for Transformer convergence.

### E. Signal Envelope Smoothing (Post-Audit Synthesis)
* **Justification:** Magnitudes ($a_{mag}$, $g_{mag}$) are calculated by squaring and rectifying, which amplifies noise peaks. To prevent these peaks from corrupting the self-attention projections, the magnitude features must be lowpass-filtered (8.0 Hz Butterworth) directly upon calculation. This delivers a smooth, clean motion envelope for the transformer attention layers.

### F. Classifier Capacity Bottlenecking (Major Learning)
* **Justification:** In baseline experiments, high-capacity models (`64` dense units) quickly memorized session-specific coordinate offsets and baseline sensor orientations, causing validation loss to diverge after epoch 8. By reducing classification dense units to **16** and increasing dropout to **50%**, we introduce a structural bottleneck. This makes it mathematically impossible for the dense head to memorize specific high-frequency baseline shifts, forcing it to make decisions based on generalized, scale-invariant spatial-temporal patterns. This resulted in a **negative generalization gap** (test performance higher than training, with stable validation progress up to epoch 42).

### G. Output Classification Layer (Explicit 8-Class Setup)
* **Justification:** The output Dense classification layer utilizes a Softmax activation over 8 distinct classes (comprising the 7 active gestures and the `none`/idle class). Since continuous PowerPoint control runs continuously in sliding windows, training the network explicitly on `none` samples forces the attention weights to outline explicit boundaries in latent space separating noise and idle motion (like keyboard usage) from gesture profiles. A thresholded 7-class system, by contrast, suffers from out-of-distribution extrapolation, projecting random movements confidently into active gesture classes due to Softmax probability saturation.

### H. Input Feature Configuration & Dynamic Selection (Post-Audit Synthesis)
* **Justification:** Based on the feature filter analysis and data quality audit, we classify our features into three tiers:
  * **Pruned (Dismissed):** We completely discard 6 derivative features (such as `IMU1_linear_jerkX/Z` and `IMU1/2_angular_accelerationY/Z`) because they satisfy `RF Gini < 0.002` and `Mutual Information < 0.5`, indicating they only introduce high-frequency noise without adding any discriminatory information.
  * **Mandatory (Kept):** We permanently bind 11 high-yield features (including `IMU1_accX/Z`, `IMU2_accX/Y/Z`, `IMU2_gyrX`, `diff_accX/Z`, `IMU1_pitch`, and `IMU1_gyr_mag`) because they satisfy `Mutual Information > 0.9` and `RF Gini > 0.02`, carrying major motion shape information.
  * **Dynamic Selection via Optuna:** The remaining 21 helper features are selected dynamically during training using a Bayesian Optuna search wrapper. The search wrapper evaluates different candidate feature combinations directly on the Self-Attention Temporal Transformer architecture over multiple training trials, selecting the configuration that maximizes the Joint Utility Score. This lets the pipeline automatically optimize inputs specifically for the Transformer model.

---

## 4. Training Pipeline & Hyperparameters

Developers must implement the training loop in code using the following configurations:

* **Optimizer:** Adam with an initial learning rate of `0.001`.
* **Loss Function:** `categorical_crossentropy` (with one-hot label encoding).
* **Epoch Budget:** `70` epochs (with callbacks activated to allow full convergence).
* **Batch Size:** `32`.
* **Callbacks:**
  * **Early Stopping (`EarlyStopping`):** Monitor `val_loss`, patience = `20` epochs, `restore_best_weights=True` to retrieve weights from the epoch with the lowest validation loss.
  * **Learning Rate Decay (`ReduceLROnPlateau`):** Monitor `val_loss`, patience = `10` epochs, learning rate reduction `factor=0.5`, minimum learning rate clamped at `min_lr=1e-6`.
* **Bayesian Optimization wrapper (Optuna):** Runs hyperparameter and dynamic feature sweeps over a set number of trials (e.g., 30-50 trials, with a trial-specific epoch limit of 10-15). The search selects optimal features using the **Joint Utility Score**:
  $$\text{Utility} = \text{Validation F1} - (0.001 \times \text{Latency ms}) - (10^{-6} \times \text{Parameter Count})$$
  This utility function penalizes model size and inference latency, directing the search toward simpler, less overfitted models.

---

## 5. Data Splitting & Leakage Prevention

To ensure honest model evaluation, the training pipeline supports index-based splitting methods. Developers must understand and configure splits as follows:

1. **Stratified Split (`stratified`):** Splits indices randomly while maintaining class balance ratios.
   * *Pitfall:* Sliding windows overlap heavily. Randomly splitting overlapping windows between Train, Val, and Test subsets leads to **severe information leakage**, yielding a deceptive 99% accuracy on paper but failing in real life.
2. **Chronological Split (`chronological`):** Splits indices sequentially per class (e.g., 70% Train / 10% Val / 20% Test) to isolate test data in time.
   * *Pitfall:* While it prevents temporal overlap leakage, it still leaks session-specific characteristics (sensor mounting, baseline drift) if Train and Test data come from the same physical session.
3. **Leave-Session-Out (`leave-session-out`):** Groups indices by session, holding out whole sessions for Test/Val.
   * *Pitfall:* Under the initial V3 dataset, sessions only contained recordings of a *single gesture class*. अल्फाबेटically permuting and partitioning sessions (70/10/20) mathematically guaranteed that entire classes were completely excluded from splits (e.g., val set containing only `none` and `fist`). Since `fist` was OOD for train, validation loss spiked at Epoch 1, triggering premature early stopping.
   * *Resolution (Balanced Leave-Session-Out Split):* Developers must run evaluations using a multi-session setup (e.g. V4 dataset) containing validation and test sessions where **all classes are represented**, and where the sensors were physically repositioned between sessions. This isolates mounting and fatigue variances without introducing class exclusion or validation early stopping failure.

---

## 6. Data Augmentation (Regularization)

To mitigate overfitting on small, single-subject datasets, two dynamic, on-the-fly augmentation techniques must be implemented during batch loading:

1. **3D Random Rotation (Spatial Regularization):**
   * *Mechanism:* Applies random 3D rotations to the raw accelerometer and gyroscope vector coordinates ($X, Y, Z$) using **Rodrigues' rotation formula**.
   * *Justification:* Simulates variations in sensor mounting angles and wrist/finger alignments, teaching the network rotation-invariant representations instead of absolute coordinate biases.
   * *Configuration:* Parameterized via `--augment-rotation <degrees>` (recommend `15` to `25` degrees).
2. **Temporal Jittering / Shift (Temporal Regularization):**
   * *Mechanism:* Dynamically offsets the start index of the sliding window during dataset loading by a random offset.
   * *Justification:* Prevents the network from relying on absolute gesture alignments or assuming the movement always starts in the exact center of the window.
   * *Configuration:* Parameterized via `--jitter-range <samples>` (recommend `20` to `25` samples).

---

## 7. Real-Time Inference Integration

The real-time sliding window inference script must consume the trained model package under the following constraints:

* **Sliding Window:** Size = `150` samples (1.5 seconds at a constant `100 Hz` sampling rate).
* **Normalization:** Load the serialized standardization scalers (`scaler.pkl` or `StandardScaler`) generated during training, applying scaling parameters per-channel online.
* **Startup Calibration:** Implement a static calibration step. At startup, the user holds their hand still for `6.0` seconds. The script calculates the static accelerometer and gyroscope offsets and subtracts these baseline biases from the stream to minimize domain shift before inputs enter the model.
* **Thresholding & Cooldown:** Gestures are dispatched only if the output Softmax probability exceeds a strict threshold (default `0.95` or `0.85` depending on noise environment). To prevent double execution of slides, a post-trigger cooldown lock (default `1.5` seconds) must be enforced.

---

## 8. Experiment Directory & Saving Structure

Every training session for this model must be saved in accordance with the project's experiment directory structure:

```
models/
└── slef_attention_temporal_transformer/             # Model identifier folder (respecting folder name typo)
    └── training_session_<index>_<timestamp>/        # Sequential session (e.g., training_session_0_20260629_020000)
        ├── model.keras                              # Saved trained Keras model weights and architecture
        ├── model.weights.h5                         # Serialized weights file
        ├── scaler_x.joblib                          # Serialized StandardScaler instance
        ├── model_metadata.json                      # JSON file containing training run audit properties
        ├── confusion_matrix.png                     # Validation split confusion matrix plot
        └── learning_curves.png                      # Training/validation loss and accuracy curves
```

* **Sequential Indexing**: The training script must dynamically query existing directories under `models/slef_attention_temporal_transformer/` to determine the next available sequential integer `<index>` (starting at `0` for the first run).
* **Metadata Logging**: The `model_metadata.json` file must capture system info, hyperparameters, training dataset stats, and per-class precision, recall, and F1-score evaluation metrics.
