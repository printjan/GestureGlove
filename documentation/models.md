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

| Model Architecture | Preset | Parameter Count | Test Accuracy | Macro F1-Score | Best Val Loss | Stopped Epoch (Best Epoch) | Target Run Subdirectory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Early Fusion CNN** | Standard | 10,840 | **98.80%** | **0.9900** | 0.0422 | 70 (57) | [training_session_2_20260701_223131](../models/early_fusion_single_branch_1d_cnn/training_session_2_20260701_223131/) |
| **Temporal Transformer** | Standard | 79,256 | **98.40%** | **0.9866** | 0.0758 | 36 (16) | [training_session_0_20260701_223842](../models/slef_attention_temporal_transformer/training_session_0_20260701_223842/) |
| **Late Fusion CNN** | Standard | 18,968 | **99.20%** | **0.9933** | 0.0540 | 44 (24) | [training_session_0_20260701_225609](../models/late_fusion_multi_branch_1d_cnn/training_session_0_20260701_225609/) |

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

1.  **Winner: Late Fusion CNN:**
    The **Late Fusion CNN (Standard)** remains the best-performing model (99.20% accuracy / 0.9933 F1). Keeping the Wrist and Finger branches independent before the dense classification head mirrors the physical layout of the gesture glove (decoupling hand/finger gesture dynamics from overall arm translation movement) and generalizes best.
2.  **Optuna and Kalman Filter Utility:**
    The Optuna dynamic feature sweep successfully selected optimal subsets of feature channels (typically including relative yaw, gravity-free linear acceleration, and differential values), which combined with the Kalman filter to elevate performance above the baseline playground experiments.
3.  **Transformer Generalization:**
    While the Self-Attention Transformer is highly expressive, it has a larger memory footprint, slower training speed, and is prone to slight overfitting on smaller sequence datasets compared to the spatial weight-sharing of 1D CNNs.
