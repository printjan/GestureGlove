# Model Training Runs & Evaluation Summary

This document registers and evaluates the performance of the three deep learning architectures trained on the balanced dual-IMU dataset (V4) using our [optimized configuration-driven pipeline](model_training_pipeline.md). 

---

## 1. Experimental Setup & Configuration

All three models were trained using identical hyperparameters and pre-processing pipeline configurations to ensure a direct, scientific comparison:

*   **Split Strategy:** [`leave-session-out`](model_training_pipeline.md#3-data-splitting) (deterministic split with `seed=42`)
*   **Data Directory:** Balanced gesture dataset V4 (`data/dataset_current`)
*   **Maximum Epochs:** 70 (guided by a learning rate reduction scheduler and dynamic early stopping)
*   **Data Augmentation:** Rotation range of $\pm 25^{\circ}$, temporal window jitter of $\pm 20$ samples
*   **Orientation Filter:** [`kalman`](data_processing_pipeline.md#orientationmethod) (Kalman-filtered quaternion-derived roll and pitch angles)
*   **Signal Filters:** `no-filter` (Butterworth raw accelerometer and gyroscope low-pass filters disabled to feed raw high-frequency features directly into the network)
*   **Hyperparameter Search:** [Optuna dynamic feature optimization](model_training_pipeline.md#bayesian-feature-optimization-optuna) enabled (25 trials, 10 epochs per trial)

---

## 2. Overall Performance Comparison

### 2.1 Offline Test-Set Metrics

| Model Architecture | Preset | Parameter Count | Test Accuracy | Macro F1-Score | Best Val Loss | Stopped Epoch (Best Epoch) | Target Run Subdirectory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Early Fusion CNN** | Standard | 10,840 | **98.80%** | **0.9900** | 0.0422 | 70 (57) | [training_session_2_20260701_223131](../models/early_fusion_single_branch_1d_cnn/training_session_2_20260701_223131/) |
| **Temporal Transformer** | Standard | 79,256 | **98.40%** | **0.9866** | 0.0758 | 36 (16) | [training_session_0_20260701_223842](../models/slef_attention_temporal_transformer/training_session_0_20260701_223842/) |
| **Late Fusion CNN** | Standard | 18,968 | **99.20%** | **0.9933** | 0.0540 | 44 (24) | [training_session_0_20260701_225609](../models/late_fusion_multi_branch_1d_cnn/training_session_0_20260701_225609/) |

### 2.2 Real-Time Live Evaluation

Each model was additionally evaluated in a real-time, on-hardware inference session using the [objection-based live evaluation pipeline](real_time_inference_pipeline.md) (`--evaluate`). The operator performed live gestures with the physical glove and had a configurable objection window to flag false positives (FP) or signal missed gestures (FN). All sessions used a confidence threshold of $0.95$ and a cooldown of $2.0$ seconds. Full evaluation reports are stored in the [`reports/`](../reports/) directory.

#### Overall Live Metrics

| Model Architecture | Duration | Gestures Fired | TP | FP | FN | Idle False Triggers | Live Precision | Live Recall | Live F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Early Fusion CNN** | 448.7 s | 131 | 130 | 1 | 3 | 1 | **99.24%** | **97.74%** | **98.48%** |
| **Late Fusion CNN** | 436.3 s | 76 | 76 | 0 | 8 | 0 | **100.00%** | **90.48%** | **95.00%** |
| **Temporal Transformer** | 511.3 s | 66 | 62 | 4 | 10 | 4 | **93.94%** | **86.11%** | **89.86%** |

#### Per-Class Live Recall Comparison

| Gesture Class | Early Fusion CNN | Late Fusion CNN | Temporal Transformer | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **swipe_left** | 100.0% (26/26) | 58.3% (7/12) | 69.2% (9/13) | Late Fusion and Transformer both struggle; gestures fall below confidence threshold and are classified as `none`. |
| **swipe_right** | 95.7% (22/23) | 100.0% (12/12) | 92.9% (13/14) | Reliable across all models. |
| **circle_cw** | 100.0% (12/12) | 100.0% (9/9) | 100.0% (9/9) | Perfect live recall across the board. |
| **circle_ccw** | 92.9% (13/14) | 100.0% (5/5) | 100.0% (8/8) | Near-perfect; Early Fusion had 1 miss. |
| **fist** | 94.7% (18/19) | 100.0% (15/15) | 90.0% (9/10) | Good overall; finger-contraction dynamics translate well. |
| **jerk_down** | 100.0% (19/19) | 100.0% (14/14) | 100.0% (14/14) | Perfect live recall across the board. |
| **jerk_up** | 100.0% (20/20) | 82.4% (14/17) | 0.0% (0/4) | **Critical Transformer failure** — all `jerk_up` attempts were missed. Late Fusion also shows notable drop. |

#### Live Confusion Matrices

The confusion matrices below visualize the actual (ground truth) versus predicted (fired) gesture classifications from each live evaluation session:

| Early Fusion CNN | Late Fusion CNN | Temporal Transformer |
| :---: | :---: | :---: |
| ![Early Fusion CNN Live Confusion Matrix](../reports/early_fusion/live_confusion_matrix.png) | ![Late Fusion CNN Live Confusion Matrix](../reports/late_fusion/live_confusion_matrix.png) | ![Temporal Transformer Live Confusion Matrix](../reports/transformer/live_confusion_matrix.png) |

#### Analysis: Offline vs. Real-Time Performance Gap

1.  **Early Fusion CNN dominates in real-time.** Despite being the smallest model (10,840 parameters), the Early Fusion CNN achieved the best live F1-score (**98.48%**) with only 1 false positive and 3 missed gestures across 131 fired events. Its high recall (97.74%) combined with near-perfect precision (99.24%) makes it the most reliable architecture for production deployment. The offline-to-live performance gap is minimal (~0.4 pp accuracy drop).
2.  **Late Fusion CNN trades recall for zero false positives.** The Late Fusion CNN fired zero false positives and had no idle false triggers, yielding a perfect 100% live precision. However, it exhibited a significant recall drop to 90.48%, primarily driven by `swipe_left` (58.3% recall — 5 of 12 attempts were missed). This suggests that the dual-branch architecture, while excellent at avoiding false activations, applies stricter confidence thresholds that cause some gestures to fall below the 0.95 detection boundary under real sensor noise.
3.  **Temporal Transformer suffers the largest real-world degradation.** The Transformer dropped from 98.40% offline accuracy to only 89.86% live F1-score. It produced 4 idle false triggers (all misclassified as `jerk_down`) and completely failed to detect `jerk_up` (0% recall). The `swipe_left` class also degraded to 69.2% recall. This confirms that the Transformer's global attention mechanism, while powerful on clean windowed test data, is more sensitive to the noisy, asynchronous conditions of real-time IMU streaming.

---

## 3. Comparison with Baseline Playground Model

The baseline playground model (`late_fusion_cnn_test`) was analyzed in [Playground Model Experiments](model_architectures/playground_model_experiments.md). Below we compare our new optimized runs against those baseline results:

1.  **Late Fusion CNN Standard vs. Playground Leave-Session-Out (Repeat):**
    *   The baseline playground model run `Leave-Session-Out (Repeat)` (Experiment D architecture with `conv_filters=[32, 64]`, `dense_units=16`) achieved a **99.00%** Test Accuracy and **99.18%** F1-Score with **18,808** parameters.
    *   Our newly trained `Late Fusion CNN (Standard)` achieved a **99.20%** Test Accuracy and **99.33%** F1-score with **18,968** parameters.
    *   For implementation details of this network, see [Late Fusion Multi-Branch 1D CNN Documentation](model_architectures/late_fusion_multi_branch_1d_cnn.md).
    *   **Improvement:** The combination of the **Kalman filter** for orientation estimation and **disabling Butterworth lowpass filters** (`--no-filter`) allowed the model to extract richer, high-frequency physical dynamics from the raw signals, yielding higher accuracy and a cleaner test margin.
2.  **Early Fusion CNN vs. Playground Baseline:**
    *   The `Early Fusion CNN (Standard)` merges all input channels at the input layer before passing them through a single 1D convolutional branch.
    *   For implementation details of this network, see [Early Fusion Single-Branch 1D CNN Documentation](model_architectures/early_fusion_single_branch_1d_cnn.md).
    *   With only **10,840** parameters (nearly half of the Late Fusion model's capacity), it achieved a remarkable **98.80%** Test Accuracy and **0.9900** Macro F1-score. This demonstrates that early spatial coupling is highly effective when combined with Optuna feature optimization.
3.  **Temporal Transformer vs. Playground Baseline:**
    *   The `Temporal Transformer (Standard)` utilizes a multi-head self-attention mechanism to learn global temporal relations across the window.
    *   For implementation details of this network, see [Self-Attention Temporal Transformer Documentation](model_architectures/self_attention_temporal_transformer.md).
    *   It contains the largest parameter footprint (**79,256** parameters). It achieved **98.40%** Test Accuracy and **0.9866** F1-score, but stopped early at epoch 36. The slightly lower accuracy and higher validation loss (0.0758) compared to the CNNs suggests that on small temporal window sizes (150 samples), the inductive bias of 1D convolutions (translation invariance and local pooling) generalizes slightly better than the non-local attention maps of transformers.

---

## 4. Class-by-Class Performance Analysis

The class-by-class F1-scores across the three models are detailed below:

| Gesture Class | Early Fusion CNN F1 | Temporal Transformer F1 | Late Fusion CNN F1 | Performance Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **none** (idle) | 0.980 | 0.974 | 0.987 | Excellent separation across all models, preventing false activations. |
| **swipe_left** | 0.990 | 0.990 | 0.990 | Very clean distinction, with only minor confusion with swipe_right. |
| **swipe_right** | 1.000 | 1.000 | 1.000 | Perfect classification ($F1 = 1.00$) in all three runs. |
| **circle_cw** | 0.970 | 0.969 | 0.980 | The most challenging class, occasionally confused with circle_ccw. |
| **circle_ccw** | 1.000 | 0.990 | 1.000 | Near-perfect separation. |
| **fist** | 1.000 | 0.980 | 1.000 | Clean detection of finger contraction dynamics. |
| **jerk_down** | 1.000 | 1.000 | 1.000 | Perfect classification ($F1 = 1.00$) in all three runs. |
| **jerk_up** | 0.980 | 0.990 | 0.990 | Near-perfect separation. |

---

## 5. Architectural Conclusions

1.  **Offline Winner: Late Fusion CNN:**
    The **Late Fusion CNN (Standard)** remains the best-performing model on the held-out test set (99.20% accuracy / 0.9933 F1). Keeping the Wrist and Finger branches independent before the dense classification head mirrors the physical layout of the gesture glove (decoupling hand/finger gesture dynamics from overall arm translation movement) and generalizes best on clean, pre-segmented windows.
2.  **Real-Time Winner: Early Fusion CNN:**
    In live on-hardware evaluation, the **Early Fusion CNN** delivered the highest live F1-score (**98.48%**) — outperforming the Late Fusion CNN (95.00%) and the Transformer (89.86%). With only 1 false positive and 3 misses across 131 gesture events, it is the recommended architecture for production deployment. Its compact size (10,840 parameters) also makes it the most efficient choice for resource-constrained inference loops.
3.  **Optuna and Kalman Filter Utility:**
    The Optuna dynamic feature sweep successfully selected optimal subsets of feature channels (typically including relative yaw, gravity-free linear acceleration, and differential values), which combined with the Kalman filter to elevate performance above the baseline playground experiments.
4.  **Transformer Generalization:**
    While the Self-Attention Transformer is highly expressive, it has a larger memory footprint, slower training speed, and is prone to slight overfitting on smaller sequence datasets compared to the spatial weight-sharing of 1D CNNs. The real-time evaluation further revealed critical failure modes (0% `jerk_up` recall, 4 idle false triggers), confirming that its global attention mechanism is more sensitive to the noisy, asynchronous conditions of live IMU streaming than the translation-invariant convolution kernels of the CNN architectures.
