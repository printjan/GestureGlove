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


### Translating Audit Results to Engineering Decisions

Once the dataset is audited, use the metrics to drive engineering choices:

| Audit Finding | Pipeline Level | Operational Action |
|---------------|----------------|--------------------|
| **Low Separability** ($S(C_A, C_B) < 1.0$) between directionals | **Feature Engineering** | Compute first-order time derivatives (jerk) and integrate Z-gyroscope (relative yaw) to isolate direction vectors. |
| **High Intra-class Variance** (inconsistent speed) | **Preprocessing / Augmentation** | Apply **Time Warping Augmentation** (rescaling the time-axis by $\pm 20\%$) to teach rate invariance. |
| **Low JS Divergence** ($D_{JS} < 0.4$) for subtle gestures | **Real-Time Pipeline** | Implement a two-stage classification: (1) high-sensitivity energy trigger, (2) multi-class CNN inference gating. |
| **Real-time Latency Mismatch** | **Model Training** | Set `jitter_range` in `PipelineConfig` to randomly shift slices, training the model to recognize off-center boundaries. |



---



## Real-Time Pre-Computed Features & Selection Rationale


### A. Feature Selection Strategy: Overengineering vs. Real-Time Efficiency

* **The Theoretical Reality:** 
  * In modern deep learning, a sufficiently deep CNN can theoretically learn to extract optimal spatial-temporal representations directly from raw multi-channel IMU data ($a_x, a_y, a_z, g_x, g_y, g_z$). 
  * But pre-calculating physical invariants and kinematic features can dramatically reduce the required network complexity, speed up convergence, and increase robustness to sensor orientation deviations.
  * Gradient descent naturally suppresses useless or noisy features by decaying their weight parameters, meaning pre-training feature selection is not strictly *necessary* for the model to work.

* **The Practical Constraints of Real-Time:** In our target application (real-time PowerPoint control running continuously), feeding raw data and relying entirely on the network to figure it out has three severe drawbacks:
  1. **Computational Overhead & Latency:** Real-time inference on a host machine runs a continuous sliding window (in our case at 100 Hz). Pre-calculating complex features (like Kalman-filtered Euler angles or sliding derivatives) consumes precious floating-point CPU cycles, adding latency.
  2. **Curse of Dimensionality:** With small training sets, adding too many noisy or redundant input channels increases model parameter size, raising the risk of overfitting.
  3. **Sensor-Defect Sensitivity (Yaw Drift):** Standard IMUs (without magnetometers) suffer from continuous yaw drift. If the absolute yaw is fed to a model, the CNN will overfit to the absolute angle of the room, failing after just a few minutes of use as drift accumulates.

* **The Goal:** We want a **minimalist, highly discriminative feature set** where every channel is physically justified and computed in $O(1)$ time complexity per sample step.



### How to Analyze Feature Utility Quantitatively

To avoid subjective overengineering, we can audit candidate features before settling on the final set:

1. **Mutual Information (MI) Regression/Classification:** Calculate the Mutual Information score between each candidate feature channel and the target gesture labels. Channels with MI $\to 0$ are candidates for exclusion.
2. **Random Forest Gini Importance:** Flatten the windows and train a lightweight Random Forest. Review the feature importance weights to identify channels that do not contribute to splitting decision boundaries.
3. **Ablation Studies:** During early training runs, systematically train the CNN with and without specific feature groups (e.g. Raw + Differentials vs. Raw + Differentials + Jerk) and check if validation accuracy changes significantly.



## Real-Time Pre-Computed Features & Filtering Requirements


### First-Order Time Derivatives (Jerk & Angular Acceleration)

* **Linear Jerk ($J_t$)**:
  $$J_t = \frac{\mathbf{a}_t - \mathbf{a}_{t-1}}{\Delta t}$$
* **Angular Acceleration ($\alpha_t$)**:
  $$\alpha_t = \frac{\mathbf{g}_t - \mathbf{g}_{t-1}}{\Delta t}$$
* **Benefit**: Jerk is highly discriminative for "snappy" vs. "smooth" gestures. For example, `jerk_up` has an extremely high transient movement-energy peak, whereas `circle_cw` has a low, continuous movement-energy profile.
* **Filtering Strategy:** Differentiation acts as a high-pass operation that multiplies high-frequency noise. Jerk **must be low-pass filtered** (either by smoothing raw accelerometer inputs at 8.0 Hz prior to differentiation, or by smoothing the computed jerk channel itself) to prevent complete noise saturation.


### Inter-IMU Differential Features (Finger vs. Wrist)

* **Relative Acceleration ($\Delta a$) & Relative Rotation ($\Delta g$)**:
  $$\Delta a = \mathbf{a}_{finger} - \mathbf{a}_{wrist}, \quad \Delta g = \mathbf{g}_{finger} - \mathbf{g}_{wrist}$$
* **Benefit**: Distinguishes **arm gestures** from **hand gestures**.
  * During a `swipe_right`, the whole arm moves as a rigid body: $\Delta a \approx 0$ and $\Delta g \approx 0$.
  * During a `fist` or finger gesture, only the finger moves relative to the wrist: $\Delta a \gg 0$.
* **Filtering Strategy:** Relies on low-pass filtered raw inputs (8.0 Hz for accelerometer, 12.0 Hz for gyroscope) to keep differentiation clean and prevent noise propagation between the two sensors.
* **Mounting Alignment Assumption:** This simple axis-wise subtraction assumes that the two IMUs (Wrist and Finger) are mounted with **parallel coordinate axes** (specifically, both devices must have their USB-C ports pointing backward and downward). If the sensors were mounted with different orientations, the subtraction would yield arbitrary vector directions. This strict physical setup constraint replaces the need for expensive real-time coordinate transformation matrix multiplications.


### Short-Term Window Integrals (Relative Yaw)

While absolute Yaw cannot be anchored without a magnetometer, relative yaw change ($\Delta \psi$) over the short gesture window ($1.5\text{ s}$) is highly accurate and drift-free.

* **Methodology**: Integrate the z-gyroscope channel over the active window:
  $$\Delta \psi = \sum_{t \in W} g_{z,t} \cdot \Delta t$$
* **Benefit**: Provides clear directional mapping in the horizontal plane (distinguishing clock rotations and horizontal sweeps).
* **Note**: Absolute Yaw accumulates continuous non-linear drift. Will corrupt decision boundaries over time without a magnetometer.
* **Filtering Strategy:** The input gyroscope signal must be zero-bias calibrated and high-pass filtered (0.5 Hz cutoff) prior to integration to remove the DC offset and prevent linear drift over the 150-sample window.


### Kinematic Invariants (Magnitudes)

* **Accelerometer Magnitude ($a_{mag}$)**:
  $$a_{mag} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$
  * *Benefit*: Independent of sensor orientation. Under static conditions (no motion), $a_{mag} \approx 1.0g$. Any deviation ($|a_{mag} - 1.0| > \epsilon$) indicates linear acceleration, which is a perfect trigger for separating active gestures from `none`.
* **Gyroscope Magnitude ($g_{mag}$)**:
  $$g_{mag} = \sqrt{g_x^2 + g_y^2 + g_z^2}$$
  * *Benefit*: Captures net angular velocity. Distinguishes rotational movements (like `circle_cw`) from purely linear movements (like `jerk_down`).
* **Filtering Strategy:** Squaring rectifies and amplifies high-frequency noise peaks. Magnitude outputs should be low-pass filtered to create a smooth, clean envelope representing physical motion energy.


### Gravity-Free Linear Acceleration ($a_{linear}$)

The raw accelerometer measures both physical acceleration and the static gravity vector ($1.0g$).

* **Methodology**: Use the calculated pitch ($\theta$) and roll ($\phi$) from your sensor fusion filter (Kalman/Complementary) to project gravity out:
  $$\mathbf{a}_{linear} = \mathbf{a}_{raw} - \mathbf{g}_{rotated}$$
* **Benefit**: Distinguishes directional gestures (like `swipe_left` vs `swipe_right`) from simple posture rotations.
* **Filtering Strategy:** Raw accelerometer signals and the estimated orientation angles must be low-pass filtered to align their phase before performing gravity projection, preventing artifacts from temporal lag.



---



## Sensor Bias Calibration & Offset Correction

To correct physical sensor defects and offsets, the pipeline runs a **static calibration correction stage** before any filtering or feature calculations:

1. **Gyroscope Zero-Bias Subtraction:** Raw gyroscopes read small non-zero values even when held completely still. During static calibration (recorded for 5 seconds), the pipeline calculates the mean value ($\bar{\mathbf{g}}_{bias}$) for each axis. During subsequent dynamic recordings, this static offset is subtracted: $\mathbf{g}_{corrected} = \mathbf{g}_{raw} - \bar{\mathbf{g}}_{bias}$. This removes constant offsets, which is critical to preventing relative yaw integration from drifting.
2. **Accelerometer Scale Normalization:** The gravity vector magnitude $g$ is estimated from the resting accelerometer values. The accelerometer channels are divided by this scale factor: $\mathbf{a}_{normalized} = \mathbf{a}_{raw} / g$, mapping 1 g of gravity to exactly 1.0 units.
3. **Dynamic Calibration Mapping:** A single recording session has multiple static calibrations (recorded every `max_samples_before_recalibration` gestures to adjust for thermal drift). The processing pipeline dynamically reads the `"recalibrations"` log in `recording_session.json` and associates each gesture sample (e.g., sample $15$) with the **closest prior calibration file** (e.g., the calibration recorded at sample index $0$). This ensures the calibration offsets are always locally fresh.


## Integration of Calibration: Training vs. Real-Time Inference

It is critical that the calibration pipeline used during offline model training matches the real-time classification runtime:

* **Offline Training Pipeline:**
  * When loading the dataset via `load_dataset`, the pipeline reads the `.csv` files for calibration.
  * It computes the calibration parameters (bias values and scaling factors) from each static file.
  * It subtracts the gyro bias and scales the accelerometer values for all samples associated with that calibration run.
  * This ensures the CNN models are trained on **perfectly clean, zero-bias, gravity-normalized waveforms**, allowing the model to focus purely on the shape and trajectory of the gesture, decoupled from hardware-specific variances.
* **Real-Time Inference Pipeline:**
  * In the continuous classification runtime (e.g., running sliding-window predictions to control PowerPoint), the incoming sensor stream contains raw, uncorrected offsets.
  * At startup, the user is prompted to run a **5-second static calibration pose** (holding still). The system computes the current real-time gyro bias and gravity scale factor.
  * These real-time calibration offsets are continuously subtracted/scaled from every incoming sliding window *before* the window is passed to the CNN.
  * This guarantees that the inputs entering the model at runtime match the mathematical distribution it saw during training, preventing prediction degradation due to sensor drift or placement variations.



---



## Model Training Discussion: Alignment vs. Translation Jitter

When designing the CNN training pipeline, we must investigate a core trade-off of our data recording solution:


### The Hypothesis

* **Centered-Only Training**: By using the `.txt` start indices to extract perfectly centered gestures, the CNN maps the gesture's peak acceleration/velocity to the exact middle of the 150-sample sequence. This maximizes peak class boundary separation and reduces class confusion during offline validation.
* **The Real-Time Mismatch**: During continuous real-time sliding-window inference, the gesture slides dynamically across the input window. A model trained *only* on centered data might fail to trigger because the peak is shifted to the edges.


### Discussion

* **Proposed Solution**: 
  - Add a hidden recording window before and after the actual recording window during which the gesture is performed increasing the overall number of recorded datapoints. Instead of cutting 150 datapoints out of the recording sample and deleting the rest we will keep all datapoints in the recording sample - also those of the hidden windows. The start and end point of the window in that the gestures are perfectly centered are marked.
  - During a first training round only use the perfectly centered gestures by selecting the relevant 150 datapoints out of the sample using the <index>.txt files. We can then analyze how the model will react if the gesture is not perfectly centered in the processed window by cutting the 150 datapoints out of our recording windows closer to the edges and adjust our training strategy moving forward.
* **The Augmentation Concern:** 
  - We only need our model to give a positive result for a gesture on two or three processed windows - that is enough to trigger the according event. So it is not important for us, that our model perfectly predicts during which period a gesture occurred, but it is important, that our model confidently predicts if a gesture was performed and what gesture was performed.
  - If we apply temporal jitter (e.g., slicing training inputs at $s \pm \text{jitter}$ samples) to force translation invariance, does the model lose its class-discrimination capability? There is a risk that by making the model robust to temporal shift, the decision boundary between classes (such as `swipe_right` vs `circle_cw`) degrades—meaning the model gets better at predicting *that* a gesture happened, but less certain about *which* gesture it was.


### Evaluation Workflow

Using the raw boundaries saved in our CSVs, we should systematically evaluate:
1. **Model A (No Jitter)**: Train on centered windows (sliced exactly at the companion `.txt` index).
2. **Model B (Jittered)**: Train with a random offset shift (e.g., $s \pm 10$ samples) introduced during batch generation.
3. **Comparison**: Compare both models' validation confusion matrices, precision/recall per class, and classification confidence distributions. If Model B shows a significant increase in inter-class confusion, it will confirm the risk, suggesting we should favor a low-latency stillness trigger loop (which centers the real-time frame before feeding it to Model A) rather than relying on translation invariant training.



---



## Signal Filtering & Preprocessing: Rationale & Model Impact

To build a high-fidelity classification system, we must establish a clear strategy for signal filtering. Raw IMU signals are physically noisy, and feeding raw waveforms blindly into machine learning models without understanding the physical and model constraints leads to poor generalization.


### How Do We Know We Need Filters in the First Place?

We rely on three empirical and physical indicators to justify filtering:

1. **Physiological & Environmental Noise:** IMU sensors pick up high-frequency electrical noise, mechanical vibrations from the mounting straps, and natural human micro-tremors (physiological hand shivers). Human gesture intent is a low-frequency signal; everything else is noise.
2. **Spectral Inspection (FFT):** If we perform a Fast Fourier Transform (FFT) on static (resting) calibration recordings, we observe significant energy spikes in the **15–50 Hz** spectrum. Since active voluntary human hand movement cannot physically oscillate at 50 Hz, this spectrum is purely non-motion noise and must be removed.
3. **Derivative Amplification:** First-order derivatives (like jerk) computed on raw signals are completely dominated by high-frequency noise. Without filtering, jerk features contain zero readable gesture structure.


### What Filters Do We Need?

We implement digital **Butterworth Filters** (2nd to 4th order) inside our processing configuration:

* **Low-Pass Filter (Noise Reduction):**
  * *Cutoff Frequencies:* **8.0 Hz** for accelerometers and **12.0 Hz** for gyroscopes.
  * *Rationale:* Human voluntary gesture dynamics reside almost entirely in the **0.5 to 8.0 Hz** band. Removing frequencies above 12 Hz strips away jitter without dampening active gesture sweeps.
* **High-Pass Filter (Drift / DC Offset Removal):**
  * *Cutoff Frequency:* **0.5 Hz** (very low).
  * *Rationale:* Strips away constant gravity components and slow sensor bias drift, keeping only the dynamic, transient changes of the movement.
* **Phase Alignment (Zero-Phase vs. Causal):**
  * **Offline Training (Zero-Phase):** Double-filtering (`scipy.signal.sosfiltfilt`) runs the filter forward and then backward. This completely cancels out any phase shift, keeping Wrist and Finger channels perfectly synchronized.
  * **Real-Time Inference (Causal):** Real-time streams cannot inspect the future. They run causal filters, which introduce a **group delay** (e.g. 20–40 ms phase lag).
  * **Bridging the Mismatch (Training Strategy):** Your intuition is correct—to keep the model as close to the real system as possible, training directly on causally filtered signals is ideal. However, doing so introduces variable phase delays between different sensors if their sampling ticks align differently. The standard strategy is to train using **Zero-Phase (for pristine synchronization) augmented with random temporal jitter (using the `jitter_range` parameter)**. Since the model is trained to recognize gestures shifted by $\pm 10$ or $\pm 15$ samples, it naturally handles the constant 20–40 ms delay introduced by causal filters in production.


### Do Different Features Need Different Filters?

Regardless of the machine learning model used, certain pre-computed features **mathematically and physically require filtering** to be usable at all:

* **Raw Signals ($a_x, g_x$):** Low-pass filtered (8 Hz/12 Hz) to strip out ambient high-frequency noise before feeding into feature-extraction filters.
* **Linear Jerk ($J_t = \frac{da}{dt}$):** Differentiation is a high-pass operation that multiplies high-frequency noise. Without low-pass filtering, linear jerk is dominated by sensor jitter and is completely unusable.
* **Relative Yaw (Gyro Integral):** Integration accumulates DC offsets and bias drift over time. The gyroscope signals must be zero-bias calibrated and high-pass filtered before integration to prevent the relative yaw from ramping up linearly.
* **Vector Magnitudes ($a_{mag}$):** Squaring and summing rectifies and amplifies noise peaks. Low-pass filtering the magnitude output is required to create a smooth motion energy envelope.


### Do Different Models Need Different Filtering Configurations?

While the individual feature calculations have mathematical filtering requirements that are fixed, the **global signal configurations** passed as input to the models can be optimized for the model's architecture:

1. **Conv1D CNNs (Single & Multi-Branch):**
   * *Sensitivity:* **Moderate.** CNNs are robust and can theoretically learn to ignore high-frequency noise if given a massive dataset.
   * *Filter Impact:* Filtering speeds up gradient convergence and prevents the CNN from overfitting to high-frequency noise signatures unique to a specific hardware device.
2. **DeepConvLSTM (CNN + RNN):**
  * *Sensitivity:* **High.** LSTMs/GRUs model sequential dependencies over time and are extremely sensitive to low-frequency offset drift (causes LSTM cell state saturation over continuous sliding windows, leading to activation decay).
  * *Filtering Strategy:* High-pass filtering (DC bias removal) is mandatory for recurrent layers to function in continuous real-time sliding windows.
3. **Lightweight Transformers (Self-Attention):**
   * *Sensitivity:* **Very High.** Attention mechanisms compute softmax weights based on dot-product comparisons. Spikey, high-frequency noise outliers cause attention weights to saturate on noise spikes rather than the gesture trajectory. **Low-pass smoothing is essential** to stabilize self-attention maps.
4. **Lightweight Statistical Classifiers (KNN / SVM):**
   * *Sensitivity:* **Very High.** Distance metrics degrade rapidly in high-dimensional noise. **Global low-pass filtering** is critical to reducing feature space variance.



---



## Model Architecture Selection: Validation & Precedents

As we approach the training phase, we must work out the optimal real-time classification setup for our dual-IMU system.


### Scientific Precedent for Multi-Branch Sensor Fusion

In wearable Human Activity Recognition (HAR) and multi-sensor fusion literature (e.g., *Ordóñez & Roggen, 2016*), **late fusion** multi-branch CNN based architectures are the established standard. The rationale is physically and biologically motivated:

* **Spatial Independence:** 
  - The wrist sensor captures arm-level, low-frequency, high-amplitude translation forces. 
  - The finger sensor captures fine-grained, high-frequency, relative rotational movements. 
* **Kernel Specialization:** Feeding both sensors into a single flat convolutional layer (early fusion) forces the convolution kernels to try and find joint spatial patterns across sensors with entirely different scales and reference frames. This leads to feature dilution and noise propagation. Parallel branches allow each Conv1D network to extract specialized spatial-temporal signatures independently before concatenation.


### Candidate Architectures to Benchmark

We will implement and empirically compare three distinct architectures to determine the best balance between classification accuracy, memory footprint, and real-time execution speed:

#### 1. Early Fusion Single-Branch Conv1D CNN (see [early_fusion_single_branch_1d_cnn.md](file:///Users/jantischner/Library/CloudStorage/OneDrive-Personal/TH_OHM_B.Sc.Inf/Th-Ohm_B.Sc.Inf_Sem6/DatFus_Sem6_Axenie/DataFusionProject/documentation/early_fusion_single_branch_1d_cnn.md))
* **Structure:** Stacks all input channels into a single tensor of shape `(150, C)` (where `C` is the optimized 18-channel feature set rather than the raw 28 coordinate channels) and feeds it to a single Conv1D pipeline.
* **Pros:** Minimum parameter count, easiest to implement and run.
* **Cons:** Prone to noise propagation; filters cannot optimize independently for finger vs. wrist coordinates.
* **Features:** Auditing has shown that using the **18 optimized orientation-invariant features** (excluding raw coordinate baselines) improves generalization accuracy on unseen sessions by preventing session-to-session baseline shifts.
* **Filters:** Jerk derivatives are lowpass-filtered (8 Hz), and relative yaw is integrated on 0.5 Hz highpass-filtered gyroscopes to prevent linear integration drift.

#### 2. Late Fusion Multi-Branch Conv1D CNN (see [late_fusion_multi_branch_1d_cnn.md](file:///Users/jantischner/Library/CloudStorage/OneDrive-Personal/TH_OHM_B.Sc.Inf/Th-Ohm_B.Sc.Inf_Sem6/DatFus_Sem6_Axenie/DataFusionProject/documentation/late_fusion_multi_branch_1d_cnn.md))
* **Structure:** Parallel Conv1D encoders for the wrist channels and finger channels, and a separate Dense MLP for statistical summary features, concatenated late before classification layers.
* **Pros:** Highly accurate; prevents spatial feature dilution; structurally matches the physical setup.
* **Cons:** Slightly larger parameter footprint.
* **Features:** 
  * Based on our Random Forest Gini importance audit, wrist-finger differences carry over **30%** of decision boundary splitting weight. We route wrist-only dynamics to Branch 1, finger-relative differences to Branch 2, and short-term relative yaw to the MLP branch.
* **Filters:** All inputs are pre-filtered to remove noise prior to differential subtraction and integration to prevent noise propagation.

#### 3. Lightweight Temporal Transformer (Self-Attention) (see [slef_attention_temporal_transformer.md](file:///Users/jantischner/Library/CloudStorage/OneDrive-Personal/TH_OHM_B.Sc.Inf/Th-Ohm_B.Sc.Inf_Sem6/DatFus_Sem6_Axenie/DataFusionProject/documentation/slef_attention_temporal_transformer.md))
* **Structure:** Multi-head self-attention layers applied along the time dimension to capture long-range temporal dependencies.
* **Pros:** State-of-the-art capability for sequence classification.
* **Cons:** Extremely data-hungry; prone to severe overfitting on small datasets.
* **Features:** Attention mechanisms lack sequential order by design, requiring us to add positional encodings to the time dimension. It prefers highly normalized and scaled input distributions.
* **Filters:** **Low-pass smoothing of magnitudes is mandatory.** Magnitude features ($a_{mag}$, $g_{mag}$) are lowpass-filtered (8.0 Hz Butterworth) directly upon calculation to prevent noise peaks from corrupting the self-attention projections.



### Validation Protocol

* **Validation Strategy:** We must use **Leave-One-Session-Out (LOSO) Cross-Validation**. If we validate on data from the same recording session, the model will overfit to the specific sensor alignment and user fatigue state of that day. Testing on an entirely unseen session is the only true test of real-time generalization.


### Dynamic Input Binding Strategy

Hardcoding the feature shapes or indices in model code is a major anti-pattern. Instead, we decouple feature engineering from model code using a **Dynamic Input Binding Strategy**:

1. **Shape-Agnostic Models:** The model creation functions should inspect the loaded dataset tensor dimensions dynamically:
   ```python
   # The first Conv1D layer input shape is bound dynamically at runtime
   batch, time_steps, channel_count = X_train.shape
   inputs = layers.Input(shape=(time_steps, channel_count))
   ```
2. **Column-Based Group Slicing (for Multi-Branch):** For the multi-branch setup, the branches should query the column names in the dataset to extract their relevant indices dynamically, rather than using hardcoded array slices:
   ```python
   # Dynamic channel mapping for Multi-Branch late fusion
   wrist_indices = [i for i, name in enumerate(dataset.channel_names) if "IMU1" in name]
   finger_indices = [i for i, name in enumerate(dataset.channel_names) if "IMU2" in name]
   ```
3. **Decoupled Configuration:** We can configure the features via `PipelineConfig` parameters (e.g. `filters.enabled=True`, `features.include_diff_acc=True`). The training loop loads the configured dataset, inspects its channels, and passes it to the model.

This decoupling allows us to iterate on feature engineering (e.g. adding relative yaw, lowpass filters, or jerk derivatives) and benchmark different architectures (Single-Branch, Multi-Branch, LSTM) completely independently without rewriting any model code.


### Feature Selection Timing: Before vs. After Pipeline Development

We can structure our workflow in two ways: deciding on features and filters **before** writing the training script, or doing so **after** the pipeline is built.

#### Option A: Deciding Features & Filters Beforehand (Static Setup)
* **Description:** Hardcode the target features, index coordinates, and filter configurations directly in the dataset loader, model input layers, and training loop.
* **Implications:**
  * **Rigidity & Churn:** If we decide to add relative yaw or remove jerk, we must rewrite the indices, the model input shapes, the training data loader, and coordinate checks.
  * **High Risk of Silent Mismatches:** If dataset columns shift (e.g., column index 6 changes from pitch to raw accZ), hardcoded index slices like `X[:, :, 6]` will silently train on the wrong data, producing garbage predictions without throwing exceptions.
  * **No Fast Feature Ablation:** Running feature ablation experiments (e.g. comparing model accuracy with vs. without differential features) requires creating and maintaining separate code branches.

#### Option B: Deciding Features & Filters After Pipeline Development (Dynamic Setup)
* **Description:** Build a configuration-driven training pipeline where model input sizes, index slices, and spatial routing are determined **dynamically at runtime** by inspecting the loaded dataset metadata.
* **Implications:**
  * **Config-Only Iteration:** We can change feature channels, filter cutoffs, and padding modes by simply modifying the CLI arguments or JSON configs (e.g., `python scripts/build_dataset.py --diff --jitter-range 10`). The model and training script adapt automatically with zero code changes.
  * **Pristine Spatial Routing:** For multi-branch architectures, the training script parses the dataset column header labels (e.g., matching `"IMU1"` or `"IMU2"`) to dynamically build the indexing maps, guaranteeing that wrist and finger channels are routed correctly even if columns are reordered.
  * **Rapid Prototyping:** Makes it trivial to run continuous grid search or ablation sweeps over different configurations.

**Our Choice:** We are implementing **Option B (Dynamic Setup)**. Our `GestureDataset` container encapsulates column names and metadata, enabling us to safely postpone the final feature selection until we have systematically benchmarked all options.



### Benchmark Metrics

The winning model will be selected based on a joint utility score:
$$\text{Utility} = \text{F1-Score} - w_1 \cdot \text{Inference Latency (ms)} - w_2 \cdot \text{Parameter Count}$$
This ensures we do not overengineer a heavy model that performs well offline but introduces unacceptable lag in our real-time PowerPoint control loops.

---


## Output Class Strategy: Explicit 8-Class vs. 7-Class + Threshold

In our real-time PowerPoint control system, sliding-window inference runs continuously. Since the user is idle 95% of the time, mapping non-gesture movements correctly to the `none` state is critical. We evaluated two architectures for output classification:

### Option A: Explicit 8-Class Setup (Softmax = 8, Including `none`)
The model is trained explicitly on both the 7 active gestures and recorded background idle movements (`none`). The final layer is an 8-way Softmax classifier.

* **PROS:**
  * **Explicit Decision Boundary for Background Noise:** Forces the network's convolutional filters to learn features that distinguish structured gesture trajectories from unstructured activities (e.g. typing, resting, hand scratching).
  * **Low False-Positive Trigger Rate:** Background activities map directly to the `none` class with high probability, minimizing accidental active gesture triggers.
  * **Aligned with HAR Literature:** Established wearable Human Activity Recognition (HAR) models (e.g. *Ordóñez & Roggen, 2016*) explicitly model the "null/idle" class to prevent false positives.
* **CONS:**
  * **Class Diversity Demands:** The variety of background movements is infinite. A poorly representative training set for the `none` class can lead to leakage into other active gesture categories.

### Option B: Implicit 7-Class Setup + Confidence Thresholding
The model is trained only on the 7 active gestures. During real-time inference, if the maximum probability among the 7 Softmax outputs falls below a threshold $\tau$ (e.g. $\tau = 0.75$), the system classifies the window as `none`.

* **PROS:**
  * **Simpler Classifier Structure:** Eliminates the need to model the infinite variance of background noise, reducing the output dense layer by one unit.
* **CONS:**
  * **No Latent Space Boundaries for Noise:** Because the model never sees "idle" data during training, the network's decision boundaries for the active gestures are unbounded. 
  * **Confident Extrapolation on Noise (OOD):** Deep neural networks tend to make highly confident predictions on Out-of-Distribution (OOD) data. Pure sensor noise or random arm movements can easily project into high-confidence regions ($> \tau$) of a gesture (e.g., $95\%$ probability for `circle_cw` on a hand scratch), leading to catastrophic false-trigger rates.
  * **Softmax Sum-to-One Saturation:** The Softmax function forces outputs to sum to 1.0. Even under static rest (pure background noise), the model *must* distribute $100\%$ probability across the 7 active gestures. Any minor skew will push one class above the threshold.

| Evaluation Metric | Option A: Explicit 8-Class Setup | Option B: 7-Class + Thresholding |
|---|---|---|
| **Real-time False-Positive Rate** | **Extremely Low** (Explicitly trained decision boundaries) | **High** (Confident extrapolation on noise) |
| **Tuning Sensitivity** | **None** (Dynamic probability argmax) | **Very High** (Requires tuning hyperparameter $\tau$) |
| **System Compatibility** | **High** (Enforces single-class-only prediction) | **Medium** (Vulnerable to threshold boundary leakage) |

### Final Engineering Decision
We will **maintain the Explicit 8-Class Setup (Option A)**. Since our system is designed to predict exactly one class at a time (no multiple overlapping gestures), establishing an explicit `none` class in the Softmax layer is the only mathematically sound way to guarantee a robust, noise-tolerant real-time classifier.
