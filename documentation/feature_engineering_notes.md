# Gesture Separability Analysis and Real-Time Feature Engineering

This document outlines statistical methodologies to estimate gesture separability from recorded training data and defines real-time feature engineering strategies to optimize CNN classification performance.



---



## Data Quality Auditing and Estimating Gesture Separability

Before training a CNN, we can estimate how well gestures will differentiate from one another and from `none` using statistical and information-theoretic metrics on the training data.


### Distance Metrics & Silhouette Analysis on DTW

* **Methodology**: Compute pairwise **Dynamic Time Warping (DTW)** distances between all samples. Since gestures are time-series sequences that can vary in speed, standard Euclidean distance is highly sensitive to slight temporal shifts. 
* **Separability Evaluation**: For any two gesture classes $C_A$ and $C_B$, compute the **Fisher Criterion / Silhouette Score**:
  $$S(C_A, C_B) = \frac{\mu_{inter} - \mu_{intra}}{\sigma_{intra}}$$
  where $\mu_{inter}$ is the mean distance between samples of different classes, and $\mu_{intra}$ is the mean distance between samples of the same class. A high ratio indicates that the two gestures are highly distinct and easily separable.
* **Why**: Low inter-class distance warns you beforehand that the CNN is highly likely to confuse those two gestures.
* **Visualization:** Plot a pairwise distance matrix heatmap of all recorded gesture windows. Group samples by class along both axes.
* **Good Signs:**
  * **Diagonal Block Structure:** Clear, dark square blocks along the diagonal showing low DTW distances within the same class ($D_{intra} \to 0$).
  * **Clean Inter-Class Margins:** Highly contrasting light regions off-diagonal, indicating high distance between different classes ($D_{inter} \gg D_{intra}$).
  * **Fisher Separability Ratio:** $S(C_A, C_B) \ge 1.5$.
* **Bad Signs:**
  * **Inter-Class Smearing:** Faint block boundaries (e.g., `swipe_left` and `swipe_right` showing low pairwise distance), warning that the trajectory profiles are too similar.
  * **Fisher Separability Ratio:** $S(C_A, C_B) < 1.0$, indicating overlapping clusters.
  * **Fisher Separability Ratio:** $S(C_A, C_B) < 1.0$, indicating overlapping clusters.


### Intra-class Variance

* **Gesture Consistency**: 
  * **What to calculate**: The average Dynamic Time Warping (DTW) distance between all pairs of samples within a single gesture category.
  * **Why**: High intra-class variance means you perform the same gesture very differently each time, which will make it harder for the CNN to learn a stable representation.

* **Trajectory Consistency**:
  * **Visualization:** Plot the mean time-series curve for each sensor channel with a $\pm 1$ standard deviation shaded band.
  * **Good Signs:** Narrow, compact standard deviation bands. This indicates high user consistency (performing the gesture at a similar speed and path each time).
  * **Bad Signs:** Wide, ballooning bands, or bimodal peaks. This suggests high intra-class variance, meaning the gesture is performed in inconsistent tempos or directions.


### Confusion Mapping

* **Dimensionality Reduction (PCA, UMAP, t-SNE)**
  1. Flatten each window of shape $(T=150, C=14)$ into a single vector of size $2100$.
  2. Perform Dimensionality Reduction (e.g., PCA to 50 components, followed by t-SNE or UMAP to 2D).

* **Visualization:** Project the flattened $150 \times 12$ windows down to 2D using UMAP or t-SNE and plot them color-coded by class.
  * **Good Signs:** Compact, well-separated island clusters with clear decision boundaries.
  * **Bad Signs:** A single massive blob in the center, or gesture classes heavily overlapping with the `none` (idle) class cluster.

* **Confusion Mapping via KNN/SVM on Flat Windows**: An excellent proxy for CNN classifier capabilities is to fit a lightweight statistical model on flattened window vectors.
  * **Methodology**: Train a simple $K$-Nearest Neighbors ($K=3$) or linear Support Vector Machine (SVM) classifier using Leave-One-Session-Out Cross-Validation.
  * **Evaluation**: Plot the resulting **confusion matrix**. If two gestures (e.g. `fist` and `none`) mix up in the KNN predictions, the CNN is likely to struggle on them as well unless temporal filters are highly optimized.


### Jensen-Shannon (JS) Divergence from `none` (Idle)
To ensure gestures can be distinguished from random idle movement (`none`):

* **Methodology**: Compute the distribution of statistical features (e.g. peak energy, variance of gyroscope signals) for the `none` class and the target gesture class.
* **Evaluation**: Compute the **JS Divergence** (symmetrized version of Kullback-Leibler divergence):
  $$D_{JS}(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$
  where $M = \frac{1}{2}(P + Q)$. Gestures with low JS divergence from the `none` distribution are at risk of triggering false positives during idle movement.
* **Visualization:** Plot overlapping histograms of motion energy peaks for `none` versus active gestures.
* **Good Signs:** $D_{JS} \ge 0.8$. Complete separation between stillness noise and the gesture envelope.
* **Bad Signs:** $D_{JS} < 0.4$. This indicates the gesture is too subtle or contains too much stillness, making real-time trigger detection highly prone to false positives.


### Motion Energy Separability Analysis

Motion Energy provides a powerful, low-cost preview of model feasibility before initiating full CNN training:

* **Methodology**: For each gesture class and the `none` class:
  1. Calculate the **Peak Motion Energy** ($E_{peak} = \max_{t} E_t$) and **Integrated Motion Energy** ($E_{total} = \sum_{t} E_t$) over the aligned 150-sample windows.
  2. Plot overlapping histograms and mean trajectory curves of the motion energy profile across all gesture categories.
* **Separability Evaluation**:
  * **Good Signs**:
    * **High Separability from `none`:** Active gestures show $E_{total} \ge 10 \times E_{total,\text{none}}$. This guarantees low false-trigger rates during idle states.
    * **Distinct Energy Signatures:** Different gesture categories exhibit clear temporal profile shapes (e.g. sharp, short-lived impulse peaks for `jerk_down` vs. flat, extended plateaus for `circle_cw`).
  * **Bad Signs**:
    * **Energy Overlap:** Low separation between an active gesture's energy distribution and `none`, signaling the user performed the gesture too weakly or slowly.
    * **Inconsistent Envelopes:** High standard deviation bands in a gesture's energy curves, showing that speed or execution path varied too much between repetitions.

---



## Translating Audits to Engineering Decisions

Once the dataset is audited, use the metrics to drive engineering choices:

| Audit Finding | Pipeline Level | Operational Action |
|---------------|----------------|--------------------|
| **Low Separability** ($S(C_A, C_B) < 1.0$) between directionals | **Feature Engineering** | Compute first-order time derivatives (jerk) and integrate Z-gyroscope (relative yaw) to isolate direction vectors. |
| **High Intra-class Variance** (inconsistent speed) | **Preprocessing / Augmentation** | Apply **Time Warping Augmentation** (rescaling the time-axis by $\pm 20\%$) to teach rate invariance. |
| **Low JS Divergence** ($D_{JS} < 0.4$) for subtle gestures | **Real-Time Pipeline** | Implement a two-stage classification: (1) high-sensitivity energy trigger, (2) multi-class CNN inference gating. |
| **Real-time Latency Mismatch** | **Model Training** | Set `jitter_range` in `PipelineConfig` to randomly shift slices, training the model to recognize off-center boundaries. |



---



## Feature Selection Matrix

To feed the CNN model, we structure features into a multi-channel tensor of shape `(Batch, Time=150, Channels)`:

| Feature Channel Group | Math Formulation | Purpose |
|----------------------|------------------|---------|
| **Raw Signals** (12 channels) | $a_x, a_y, a_z, g_x, g_y, g_z$ (IMU1 & IMU2) | Baseline motion dynamics. |
| **Kinematic Invariants** (4 channels) | $a_{mag} = \sqrt{\sum a_i^2}$, $g_{mag} = \sqrt{\sum g_i^2}$ | Orientation-invariant magnitudes. Excellent for segment bounds and triggering. |
| **Linear Acceleration** (6 channels) | $\mathbf{a}_{linear} = \mathbf{a}_{raw} - \mathbf{g}_{rotated}$ | Removes static gravity vectors using Kalman orientation (roll/pitch). |
| **Differential Dynamics** (6 channels) | $\Delta a = a_{finger} - a_{wrist}$, $\Delta g = g_{finger} - g_{wrist}$ | Separates arm movement (rigid rotation) from isolated finger gestures. |
| **Relative Yaw** (2 channels) | $\Delta \psi = \sum g_z \cdot \Delta t$ | Resolves left/right and clockwise/counter-clockwise trajectories. |



---



## CNN Model Architecture Design: Late Fusion Multi-Branch CNN

Since we are fusing two distinct physical nodes (Wrist vs. Finger) and handcrafted statistical features, a **Late Fusion Multi-Branch Conv1D CNN** is the optimal setup:

```mermaid
graph TD
    Input["Input Window (150, Channels)"] --> W_Split["Wrist Channels"]
    Input --> F_Split["Finger Channels"]
    Input --> S_Split["Statistical / Handcrafted Features"]

    W_Split --> Branch1["Wrist Conv1D Branch<br/>- Conv1D (kernel=5, filters=32)<br/>- BatchNorm & ReLU<br/>- MaxPool1D<br/>- Conv1D (kernel=3, filters=64)<br/>- GlobalAveragePooling1D"]
    
    F_Split --> Branch2["Finger Conv1D Branch<br/>- Conv1D (kernel=5, filters=32)<br/>- BatchNorm & ReLU<br/>- MaxPool1D<br/>- Conv1D (kernel=3, filters=64)<br/>- GlobalAveragePooling1D"]

    S_Split --> Branch3["Dense Feature MLP<br/>- Dense (32, ReLU)<br/>- Dropout (0.2)"]

    Branch1 --> Concatenate["Concatenate Layer"]
    Branch2 --> Concatenate
    Branch3 --> Concatenate

    Concatenate --> Classifier["FC Dense (64, ReLU)"]
    Classifier --> Dropout["Dropout (0.3)"]
    Dropout --> Softmax["Softmax Layer (8 Classes)"]
```

### Key Architectural Choices:
1. **Parallel Temporal Branches (Late Fusion):**
   * Keeping the Wrist and Finger networks separate allows each branch to build local spatial features independently before merging. Arm gestures are dominated by wrist dynamics, whereas hand gestures are dominated by finger-to-wrist deltas.
2. **Conv1D for Temporal Learning:**
   * 1D convolutions extract shift-invariant local features along the timeline. This helps handle slight temporal misalignments during real-time sliding window inference.
3. **Global Average Pooling (GAP) vs. Flattening:**
   * Replacing flat outputs with `GlobalAveragePooling1D` reduces the parameter footprint drastically, preventing overfitting on small training sets.
4. **Regularization:**
   * Batch Normalization is applied after each Conv1D layer to stabilize training.
   * Dropout ($30\%$) is added before the final classifier to ensure generalization.
5. **Loss & Optimization:**
   * **Loss:** `categorical_crossentropy` (with one-hot label encoding).
   * **Optimizer:** `Adam(learning_rate=0.001)` paired with a learning rate decay schedule (`ReduceLROnPlateau`).


---


## Real-Time Pre-Computed Features

Feeding raw sensor data ($a_x, a_y, a_z, g_x, g_y, g_z$) directly into a CNN is possible, but pre-calculating physical invariants and kinematic features dramatically reduces the required network complexity, speeds up convergence, and increases robustness to sensor orientation deviations.

### A. Kinematic Invariants (Magnitudes)
* **Accelerometer Magnitude ($a_{mag}$)**:
  $$a_{mag} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$
  * *Benefit*: Independent of sensor orientation. Under static conditions (no motion), $a_{mag} \approx 1.0g$. Any deviation ($|a_{mag} - 1.0| > \epsilon$) indicates linear acceleration, which is a perfect trigger for separating active gestures from `none`.
* **Gyroscope Magnitude ($g_{mag}$)**:
  $$g_{mag} = \sqrt{g_x^2 + g_y^2 + g_z^2}$$
  * *Benefit*: Captures net angular velocity. Distinguishes rotational movements (like `circle_cw`) from purely linear movements (like `jerk_down`).

### B. Gravity-Free Linear Acceleration ($a_{linear}$)
The raw accelerometer measures both physical acceleration and the static gravity vector ($1.0g$).
* **Methodology**: Use the calculated pitch ($\theta$) and roll ($\phi$) from your sensor fusion filter (Kalman/Complementary) to project gravity out:
  $$\mathbf{a}_{linear} = \mathbf{a}_{raw} - \mathbf{g}_{rotated}$$
* **Benefit**: Isolates the true user-generated translation force. Essential for separating directional gestures (like `swipe_left` vs `swipe_right`) from simple posture rotations.



## Real-Time Pre-Computed Features & Selection Rationale

### A. Feature Selection Strategy: Overengineering vs. Real-Time Efficiency

* **The Theoretical Reality:** In modern deep learning, a sufficiently deep CNN can theoretically learn to extract optimal spatial-temporal representations directly from raw multi-channel IMU data ($a_x, a_y, a_z, g_x, g_y, g_z$). Gradient descent naturally suppresses useless or noisy features by decaying their weight parameters, meaning pre-training feature selection is not strictly *necessary* for the model to work.
* **The Practical Constraints (Real-Time Hardware):** In our target application (real-time PowerPoint control running continuously), feeding raw data and relying entirely on the network to figure it out has three severe drawbacks:
  1. **Computational Overhead & Latency:** Real-time inference on a host machine or embedded system runs a continuous sliding window (e.g. at 100 Hz). Pre-calculating complex features (like Kalman-filtered Euler angles or sliding derivatives) consumes precious floating-point CPU cycles, adding latency and battery drain.
  2. **Curse of Dimensionality:** With small training sets, adding too many noisy or redundant input channels increases model parameter size, raising the risk of overfitting.
  3. **Sensor-Defect Sensitivity (Yaw Drift):** Standard IMUs (without magnetometers) suffer from continuous yaw drift. If the absolute yaw is fed to a model, the CNN will overfit to the absolute angle of the room, failing after just a few minutes of use as drift accumulates.
* **The Goal:** We want a **minimalist, highly discriminative feature set** where every channel is physically justified and computed in $O(1)$ time complexity per sample step.



### C. First-Order Time Derivatives (Jerk & Angular Acceleration)
* **Linear Jerk ($J_t$)**:
  $$J_t = \frac{\mathbf{a}_t - \mathbf{a}_{t-1}}{\Delta t}$$
* **Angular Acceleration ($\alpha_t$)**:
  $$\alpha_t = \frac{\mathbf{g}_t - \mathbf{g}_{t-1}}{\Delta t}$$
* **Benefit**: Jerk is highly discriminative for "snappy" vs. "smooth" gestures. For example, `jerk_up` has an extremely high transient jerk peak, whereas `circle_cw` has a low, continuous jerk profile.

### D. Inter-IMU Differential Features (Finger vs. Wrist)
* **Relative Acceleration ($\Delta a$) & Relative Rotation ($\Delta g$)**:
  $$\Delta a = \mathbf{a}_{finger} - \mathbf{a}_{wrist}, \quad \Delta g = \mathbf{g}_{finger} - \mathbf{g}_{wrist}$$
* **Benefit**: Distinguishes **arm gestures** from **hand gestures**.
  * During a `swipe_right`, the whole arm moves as a rigid body: $\Delta a \approx 0$ and $\Delta g \approx 0$.
  * During a `fist` or finger gesture, only the finger moves relative to the wrist: $\Delta a \gg 0$.

### E. Short-Term Window Integrals (Relative Yaw)
While absolute Yaw cannot be anchored without a magnetometer, relative yaw change ($\Delta \psi$) over the short gesture window ($1.5\text{ s}$) is highly accurate and drift-free.
* **Methodology**: Integrate the z-gyroscope channel over the active window:
  $$\Delta \psi = \sum_{t \in W} g_{z,t} \cdot \Delta t$$
* **Benefit**: Provides clear directional mapping in the horizontal plane (distinguishing clock rotations and horizontal sweeps).



### B. How to Analyze Feature Utility Quantitatively
To avoid subjective overengineering, we can audit candidate features before settling on the final set:
1. **Mutual Information (MI) Regression/Classification:** Calculate the Mutual Information score between each candidate feature channel and the target gesture labels. Channels with MI $\to 0$ are candidates for exclusion.
2. **Random Forest Gini Importance:** Flatten the windows and train a lightweight Random Forest. Review the feature importance weights to identify channels that do not contribute to splitting decision boundaries.
3. **Ablation Studies:** During early training runs, systematically train the CNN with and without specific feature groups (e.g. Raw + Differentials vs. Raw + Differentials + Jerk) and check if validation accuracy changes significantly.

---

### C. Individual Feature Arguments Matrix

The table below outlines our engineering rationale for including or excluding each candidate feature:

| Feature Candidate | Computation | Status | Engineering Arguments |
|-------------------|-------------|--------|----------------------|
| **Raw Signals** | $a_x, a_y, a_z, g_x, g_y, g_z$ | **INCLUDE** | Baseline motion dynamics. Low cost, no preprocessing latency. |
| **Kinematic Invariants** (Magnitudes) | $a_{mag} = \sqrt{a_x^2+a_y^2+a_z^2}$, $g_{mag} = \sqrt{g_x^2+g_y^2+g_z^2}$ | **INCLUDE** | **Orientation Invariant:** If the user mounts the sensor rotated, magnitudes remain identical. Highly discriminative for separating active motion from idle `none` ($a_{mag} \approx 1.0g$ when still). |
| **Pitch ($\theta$) & Roll ($\phi$)** | Kalman / Complementary Filter | **INCLUDE** | Observes absolute posture tilt relative to gravity. Essential for pose changes (e.g., finger orientation, wrist elevation). |
| **Absolute Yaw ($\psi$)** | Integrated Z-Gyro (Long-term) | **EXCLUDE** | **Yaw Drift:** Accumulates continuous non-linear drift. Feeding absolute yaw will corrupt the classifier decision boundary over time. |
| **Relative Yaw ($\Delta\psi$)** | Integrated Z-Gyro over window ($1.5\text{s}$) | **INCLUDE** | Integrated strictly within the current local 150-sample window. Drift-free over short windows and highly discriminative for direction (`swipe_left` vs `swipe_right` and circular rotations). |
| **Linear Jerk** | $\frac{d(a_i)}{dt} \approx \frac{a_t - a_{t-1}}{\Delta t}$ | **INCLUDE (Filtered)** | Distinguishes "snappy" gestures (like `fist` or `jerk_up` which have massive jerk spikes) from "smooth" gestures (like `circle_cw`). *Warning:* Amplifies high-frequency noise; requires lowpass filtering. |
| **Gravity-Free Linear Acc** | $\mathbf{a}_{linear} = \mathbf{a}_{raw} - \mathbf{g}_{rotated}$ | **INCLUDE** | Subtracts static gravity using pitch/roll, isolating pure translation force. Needed for directional sweeps. |
| **Differential Dynamics** | $\Delta a = a_{\text{finger}} - a_{\text{wrist}}$ | **INCLUDE** | Key indicator for skeletal constraints: during arm gestures (`swipe`), wrist and finger move together ($\Delta a \approx 0$). During hand gestures (`fist`), only the finger moves ($\Delta a \gg 0$). |

---



---



## Model Training Discussion: Alignment vs. Translation Jitter

When designing the CNN training pipeline, we must investigate a core trade-off enabled by our new 1.74s raw boundary data:

### The Hypothesis
* **Centered-Only Training**: By using the `.txt` start indices to extract perfectly centered gestures, the CNN maps the gesture's peak acceleration/velocity to the exact middle of the 150-sample sequence. This maximizes peak class boundary separation and reduces class confusion during offline validation.
* **The Real-Time Mismatch**: During continuous real-time sliding-window inference, the gesture slides dynamically across the input window. A model trained *only* on centered data might fail to trigger because the peak is shifted to the edges.
* **Proposed Solution**: 
  - Add a hidden recording window before and after the actual recording window during which the gesture is performed increasing the overall number of recorded datapoints. Instead of cutting 150 datapoints out of the recording sample and deleting the rest we will keep all datapoints in the recording sample - also those of the hidden windows. The start and end point of the window in that the guestures are perfectly centered are marked.
  - During a first training round only use the perfectly centered guestures by selecting the relevant 150 datapoints out of the sample using the <index>.txt files. We can then analyze how the model will react if the guesture is not perfectly centered in the processed window by cutting the 150 datapoints out of our recoring windows closer to the edges and adjust our training strategy moving forward.
* **The Augmentation Concern:** 
  - We only need our model to give a positive result for a gesture on two or three processesd windwos - that is enough to trigger the accoring event. So it is not important for us, that our model perfectly predicts during which period a gesture accured, but it is important, that our model confidently predicts if a gesture was performed and what gesture was performed.
  - If we apply temporal jitter (e.g., slicing training inputs at $s \pm \text{jitter}$ samples) to force translation invariance, does the model lose its class-discrimination capability? There is a risk that by making the model robust to temporal shift, the decision boundary between classes (such as `swipe_right` vs `circle_cw`) degrades—meaning the model gets better at predicting *that* a gesture happened, but less certain about *which* gesture it was.

### Evaluation Workflow
Using the raw boundaries saved in our CSVs, we should systematically evaluate:
1. **Model A (No Jitter)**: Train on centered windows (sliced exactly at the companion `.txt` index).
2. **Model B (Jittered)**: Train with a random offset shift (e.g., $s \pm 10$ samples) introduced during batch generation.
3. **Comparison**: Compare both models' validation confusion matrices, precision/recall per class, and classification confidence distributions. If Model B shows a significant increase in inter-class confusion, it will confirm the risk, suggesting we should favor a low-latency stillness trigger loop (which centers the real-time frame before feeding it to Model A) rather than relying on translation invariant training.
