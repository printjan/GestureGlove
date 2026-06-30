# scripts/train_cnn.py
"""
CLI script to train and evaluate the Late Fusion Multi-Branch Conv1D CNN.

Exposes parameters for configuring the processing pipeline, dataset split,
epochs, batch size, and outputs. Saves all results (weights, scalers, plots, metadata)
in a dedicated model folder in the project's models/ directory.

Example:
    python scripts/train_cnn.py --epochs 20 --diff --cross-correlation
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project src/ directory to the python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

# Configure Keras backend before importing Keras modules
import os
import platform
temp_p = argparse.ArgumentParser(add_help=False)
temp_p.add_argument("--backend", default=None)
temp_args, _ = temp_p.parse_known_args()

if temp_args.backend:
    os.environ["KERAS_BACKEND"] = temp_args.backend
elif "KERAS_BACKEND" not in os.environ:
    if platform.system() == "Darwin":
        # Default to PyTorch on Mac due to local tensorflow-metal dynamic linker bugs
        os.environ["KERAS_BACKEND"] = "torch"
    else:
        os.environ["KERAS_BACKEND"] = "tensorflow"

from data_fusion_project.core.cli_ui import ui, Style
from data_fusion_project.core.paths import DATA_DIR, MODELS_DIR, get_model_run_dir
from data_fusion_project.processing import (
    load_dataset,
    PipelineConfig,
    FilterConfig,
    OrientationConfig,
    FeatureConfig,
    FilterType,
    OrientationMethod,
)
from data_fusion_project.training.late_fusion_multi_branch_cnn_test.train import train_model


# ======================================================================================================================
# Config Builders
# ======================================================================================================================
def build_pipeline_config(args) -> PipelineConfig:
    """
    Translates CLI arguments into a PipelineConfig.
    """
    return PipelineConfig(
        jitter_range=args.jitter_range,
        filters=FilterConfig(
            enabled=not args.no_filter,
            acc_filter=FilterType(args.acc_filter),
            acc_cutoff_hz=args.acc_cutoff,
            gyro_filter=FilterType(args.gyro_filter),
            gyro_cutoff_hz=args.gyro_cutoff,
            remove_gravity=args.gravity_removal,
            replace_acc_with_linear=args.gravity_removal,
        ),
        orientation=OrientationConfig(
            method=OrientationMethod(args.orientation),
            imus=tuple(args.orientation_imus),
        ),
        features=FeatureConfig(
            include_diff_acc=args.diff or args.relative_acceleration,
            include_diff_gyro=args.diff or args.relative_rotation,
            cross_correlation=args.cross_correlation,
            statistics=args.statistics,
            include_linear_jerk=args.linear_jerk,
            include_angular_acceleration=args.angular_acceleration,
            include_relative_acceleration=args.relative_acceleration,
            include_relative_rotation=args.relative_rotation,
            include_relative_yaw=args.relative_yaw,
            include_accelerometer_magnitude=args.acc_magnitude,
            include_gyroscope_magnitude=args.gyro_magnitude,
            include_gravity_free_linear_acceleration=args.gravity_free_acc,
        ),
    )


# ======================================================================================================================
# CLI Parser
# ======================================================================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate a Late Fusion CNN on gesture data.")
    
    # Core directories & naming
    p.add_argument("--data-dir", default=None, help="Data root directory (defaults to project data/).")
    p.add_argument("--model-name", default="late_fusion_cnn_test", help="Target model subfolder inside models/.")
    p.add_argument("--backend", default=None, choices=["tensorflow", "torch", "jax"],
                   help="Keras backend to use (automatically defaults to 'torch' on macOS and 'tensorflow' elsewhere).")
    
    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p.add_argument("--split", default="chronological", choices=["stratified", "leave-session-out", "chronological"],
                   help="Evaluation splitting strategy (leave-session-out, stratified, or chronological).")
    p.add_argument("--test-fraction", type=float, default=0.2, help="Validation held-out fraction (default: 0.2).")
    p.add_argument("--seed", type=int, default=42, help="Random split seed.")
    p.add_argument("--augment-rotation", type=float, default=0.0,
                   help="Maximum random 3D rotation angle in degrees for IMU data augmentation (default: 0.0, disabled).")
    
    # Bayesian dynamic feature optimization
    p.add_argument("--optimize", action="store_true", help="Enable Bayesian optimization of dynamic features using Optuna.")
    p.add_argument("--optuna-trials", type=int, default=20, help="Number of Optuna trials (default: 20).")
    p.add_argument("--optuna-epochs", type=int, default=3, help="Number of training epochs per trial (default: 3).")
    p.add_argument("--w1", type=float, default=0.001, help="Latency weight for Joint Utility Score (default: 0.001).")
    p.add_argument("--w2", type=float, default=1e-6, help="Parameter count weight for Joint Utility Score (default: 1e-6).")
    
    # Pipeline parameters (matches build_dataset.py)
    p.add_argument("--jitter-range", type=int, default=0, help="Jitter range for translation augmentation.")
    p.add_argument("--no-filter", action="store_true", help="Disable digital filtering.")
    p.add_argument("--acc-filter", default="lowpass", choices=[f.value for f in FilterType])
    p.add_argument("--acc-cutoff", type=float, default=8.0, help="Accelerometer cutoff in Hz.")
    p.add_argument("--gyro-filter", default="lowpass", choices=[f.value for f in FilterType])
    p.add_argument("--gyro-cutoff", type=float, default=12.0, help="Gyroscope cutoff in Hz.")
    p.add_argument("--gravity-removal", action="store_true", help="Replace raw acc with gravity-free linear acc.")
    p.add_argument("--orientation", default="complementary", choices=[m.value for m in OrientationMethod])
    p.add_argument("--orientation-imus", nargs="+", default=["IMU1", "IMU2"], help="IMUs to compute roll/pitch for.")
    p.add_argument("--diff", action="store_true", help="Add inter-IMU (finger-wrist) difference channels.")
    p.add_argument("--cross-correlation", action="store_true", help="Add cross-correlation scalar features.")
    p.add_argument("--statistics", action="store_true", help="Add per-channel statistical scalar features.")
    
    # Pre-computed features
    p.add_argument("--linear-jerk", action="store_true", help="Include low-pass filtered linear jerk.")
    p.add_argument("--angular-acceleration", action="store_true", help="Include angular acceleration.")
    p.add_argument("--relative-acceleration", action="store_true", help="Include relative acceleration (finger - wrist).")
    p.add_argument("--relative-rotation", action="store_true", help="Include relative rotation (finger - wrist).")
    p.add_argument("--relative-yaw", action="store_true", help="Include high-pass filtered short-term relative yaw integration.")
    p.add_argument("--acc-magnitude", action="store_true", help="Include low-pass filtered accelerometer magnitude.")
    p.add_argument("--gyro-magnitude", action="store_true", help="Include low-pass filtered gyroscope magnitude.")
    p.add_argument("--gravity-free-acc", action="store_true", help="Include gravity-free linear acceleration projected by complementary pitch/roll.")
    
    return p.parse_args(argv)


# ======================================================================================================================
# Main Execution
# ======================================================================================================================
def main(argv=None) -> int:
    args = parse_args(argv)
    
    # Generate run timestamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    ui.hr(title="Multi-Branch CNN Training & Optimization Pipeline")
    
    # Define feature lists
    PRUNED = [
        "IMU1_linear_jerkX", "IMU1_linear_jerkZ", "IMU2_linear_jerkZ",
        "IMU1_angular_accelerationY", "IMU1_angular_accelerationZ", "IMU2_angular_accelerationY"
    ]
    MANDATORY = [
        "IMU1_accX", "IMU1_accZ", "IMU1_gyrX", "IMU1_pitch",
        "IMU2_accX", "IMU2_accY", "IMU2_accZ", "IMU2_gyrX",
        "diff_accX", "diff_accZ", "IMU1_gyr_mag"
    ]
    DYNAMIC = [
        "IMU1_accY", "IMU1_gyrY", "IMU1_gyrZ", "IMU1_acc_mag", "IMU1_roll",
        "IMU1_relative_yaw", "IMU1_linear_jerkY", "IMU1_angular_accelerationX",
        "IMU2_gyrY", "IMU2_gyrZ", "IMU2_gyr_mag", "IMU2_acc_mag", "IMU2_relative_yaw",
        "IMU2_linear_jerkX", "IMU2_linear_jerkY", "IMU2_angular_accelerationX",
        "IMU2_angular_accelerationZ", "diff_accY", "diff_gyrX", "diff_gyrY", "diff_gyrZ"
    ]
    
    feature_toggles = None
    
    if args.optimize:
        ui.info("Running under dynamic feature optimization mode.")
        ui.info(f"Target trials: {args.optuna_trials} | Target epochs/trial: {args.optuna_epochs}")
        
        # Build search-space config containing all possible features
        search_pipeline_cfg = PipelineConfig(
            jitter_range=args.jitter_range,
            filters=FilterConfig(
                enabled=not args.no_filter,
                acc_filter=FilterType(args.acc_filter),
                acc_cutoff_hz=args.acc_cutoff,
                gyro_filter=FilterType(args.gyro_filter),
                gyro_cutoff_hz=args.gyro_cutoff,
                remove_gravity=args.gravity_removal,
                replace_acc_with_linear=args.gravity_removal,
            ),
            orientation=OrientationConfig(
                method=OrientationMethod(args.orientation),
                imus=tuple(args.orientation_imus),
            ),
            features=FeatureConfig(
                imus=("IMU1", "IMU2"),
                include_acc=True,
                include_gyro=True,
                include_acc_magnitude=True,
                include_gyro_magnitude=True,
                include_diff_acc=True,
                include_diff_gyro=True,
                include_orientation=True,
                include_linear_jerk=True,
                include_angular_acceleration=True,
                include_relative_yaw=True,
                include_gravity_free_linear_acceleration=True,
            ),
        )
        
        with ui.spinner("Loading complete search-space dataset..."):
            try:
                ds = load_dataset(search_pipeline_cfg, data_dir=args.data_dir)
            except Exception as exc:
                ui.error(f"Failed to load dataset: {exc}")
                return 1
    else:
        # Load standard user config dataset
        pipeline_cfg = build_pipeline_config(args)
        with ui.spinner("Loading gesture dataset..."):
            try:
                ds = load_dataset(pipeline_cfg, data_dir=args.data_dir)
            except Exception as exc:
                ui.error(f"Failed to load dataset: {exc}")
                return 1
                
    ui.info(ds.summary())
    
    # Sanity checks on leave-session-out splitting
    if args.split == "leave-session-out":
        class_session_counts = {}
        for label in range(ds.n_classes):
            class_mask = (ds.y == label)
            class_sessions = set(ds.groups[class_mask].tolist())
            class_session_counts[label] = len(class_sessions)
            
        insufficient_classes = [ds.class_names[lbl] for lbl, cnt in class_session_counts.items() if cnt < 2]
        if insufficient_classes:
            ui.warning(f"Leave-session-out split requested, but some classes have only 1 unique session: {insufficient_classes}")
            ui.warning("Falling back to 'chronological' split to ensure all classes are present in train and test splits!")
            args.split = "chronological"

    if args.optimize:
        import optuna
        import logging
        from data_fusion_project.training.late_fusion_multi_branch_cnn_test.train import parse_channel_indices, matches_feature, ALL_37_FEATURES
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            toggles = {}
            for feat in PRUNED:
                toggles[feat] = False
            for feat in MANDATORY:
                toggles[feat] = True
            for feat in DYNAMIC:
                toggles[feat] = trial.suggest_categorical(feat, [True, False])
                
            try:
                # Silence stdout logs during trials
                logging.getLogger("data_fusion_project").setLevel(logging.WARNING)
                
                model, history, report = train_model(
                    ds=ds,
                    split_type=args.split,
                    test_fraction=args.test_fraction,
                    epochs=args.optuna_epochs,
                    batch_size=args.batch_size,
                    model_name=None, # do not save trial folders
                    seed=args.seed,
                    augment_rotation=args.augment_rotation,
                    feature_toggles=toggles
                )
                
                f1 = float(report["macro avg"]["f1-score"])
                
                # Estimate latency
                inputs_dummies = []
                for inp in model.inputs:
                    inputs_dummies.append(np.zeros((1, *inp.shape[1:]), dtype=np.float32))
                
                # Quick 3 runs (as latency optimization is non-critical for asynchronous queue grabber)
                import time
                start_l = time.perf_counter()
                for _ in range(3):
                    model.predict(inputs_dummies, verbose=0)
                latency = ((time.perf_counter() - start_l) / 3.0) * 1000.0
                
                param_count = model.count_params()
                
                # Utility score: f1 penalty based on w1 * latency (ms) and w2 * parameters
                utility = f1 - args.w1 * latency - args.w2 * param_count
                
                # Restore log level to print trial result
                logging.getLogger("data_fusion_project").setLevel(logging.INFO)
                ui.info(f"Trial {trial.number:2d} | F1: {f1:.4f} | Latency: {latency:5.2f}ms | Params: {param_count:6d} | Utility: {utility:6.3f}")
                return utility
            except Exception as e:
                logging.getLogger("data_fusion_project").setLevel(logging.INFO)
                ui.error(f"Trial {trial.number:2d} failed: {e}")
                return -999.0

        ui.hr(title="Optuna Dynamic Feature Sweep")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna_trials)
        
        feature_toggles = {}
        for feat in PRUNED:
            feature_toggles[feat] = False
        for feat in MANDATORY:
            feature_toggles[feat] = True
        for feat in DYNAMIC:
            feature_toggles[feat] = study.best_params[feat]
            
        ui.hr()
        ui.success("Bayesian dynamic feature selection search complete!")
        ui.box([
            f"Best Joint Utility Score: {study.best_value:.4f}",
            f"Optimized Dynamic Features: {[f for f in DYNAMIC if feature_toggles[f]]}"
        ], title="Tree-structured Parzen Estimator (TPE) Study Output")
        
        # final retraining
        ui.info("Retraining final model on the optimal feature subset for full epochs...")
        logging.getLogger("data_fusion_project").setLevel(logging.INFO)

    run_dir = get_model_run_dir(args.model_name, timestamp)
    ui.info(f"Target model run save location: {run_dir}")
    
    # Execute training loop
    ui.info(f"Initiating training loop using split: {args.split}...")
    model, history, report = train_model(
        ds=ds,
        split_type=args.split,
        test_fraction=args.test_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        timestamp=timestamp,
        seed=args.seed,
        augment_rotation=args.augment_rotation,
        feature_toggles=feature_toggles
    )
    
    # Print metrics summary
    ui.hr(title="CNN Evaluation Summary")
    ui.kv([
        ("Accuracy", f"{report['accuracy'] * 100:.2f}%"),
        ("Validation Loss", f"{history['val_loss'][-1]:.4f}"),
        ("Macro F1-Score", f"{report['macro avg']['f1-score']:.4f}")
    ])
    
    report_lines = []
    for label in ds.class_names:
        metrics = report.get(label)
        if metrics:
            report_lines.append(f"{label:<14} -> Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
    ui.box(report_lines, title="Classwise Classification Report")
    
    ui.success("All outputs saved successfully inside the model package directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
