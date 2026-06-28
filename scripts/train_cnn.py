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

from data_fusion_project.core.paths import DATA_DIR, MODELS_DIR
from data_fusion_project.processing import (
    load_dataset,
    PipelineConfig,
    FilterConfig,
    OrientationConfig,
    FeatureConfig,
    FilterType,
    OrientationMethod,
)
from data_fusion_project.training.train import train_model


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
            include_diff_acc=args.diff,
            include_diff_gyro=args.diff,
            cross_correlation=args.cross_correlation,
            statistics=args.statistics,
        ),
    )


# ======================================================================================================================
# CLI Parser
# ======================================================================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate a Late Fusion CNN on gesture data.")
    
    # Core directories & naming
    p.add_argument("--data-dir", default=None, help="Data root directory (defaults to project data/).")
    p.add_argument("--model-name", default="late_fusion_cnn_v1", help="Target model subfolder inside models/.")
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
    
    # Pipeline parameters (matches build_dataset.py)
    p.add_argument("--jitter-range", type=int, default=0, help="Jitter range for translation augmentation.")
    p.add_argument("--no-filter", action="store_true", help="Disable digital filtering.")
    p.add_argument("--acc-filter", default="lowpass", choices=[f.value for f in FilterType])
    p.add_argument("--acc-cutoff", type=float, default=8.0, help="Accelerometer cutoff in Hz.")
    p.add_argument("--gyro-filter", default="lowpass", choices=[f.value for f in FilterType])
    p.add_argument("--gyro-cutoff", type=float, default=12.0, help="Gyroscope cutoff in Hz.")
    p.add_argument("--gravity-removal", action="store_true", help="Replace raw acc with gravity-free linear acc.")
    p.add_argument("--orientation", default="complementary", choices=[m.value for m in OrientationMethod])
    p.add_argument("--orientation-imus", nargs="+", default=["IMU1"], help="IMUs to compute roll/pitch for.")
    p.add_argument("--diff", action="store_true", help="Add inter-IMU (finger-wrist) difference channels.")
    p.add_argument("--cross-correlation", action="store_true", help="Add cross-correlation scalar features.")
    p.add_argument("--statistics", action="store_true", help="Add per-channel statistical scalar features.")
    
    return p.parse_args(argv)


# ======================================================================================================================
# Main Execution
# ======================================================================================================================
def main(argv=None) -> int:
    args = parse_args(argv)
    pipeline_cfg = build_pipeline_config(args)
    
    # Define output folder inside central project models/ directory
    output_dir = MODELS_DIR / args.model_name
    print(f"Target model save location: {output_dir}\n")
    
    # Load dataset
    print("Loading gesture dataset...")
    try:
        ds = load_dataset(pipeline_cfg, data_dir=args.data_dir)
    except Exception as exc:
        print(f"[ERROR] Failed to build dataset: {exc}")
        return 1
        
    print(ds.summary())
    
    # Sanity checks on leave-session-out splitting
    unique_sessions = len(set(ds.groups.tolist()))
    if args.split == "leave-session-out":
        if unique_sessions <= ds.n_classes:
            print("\n" + "!" * 80)
            print("[WARNING] Leave-session-out split requested, but session count matches or is less than class count.")
            print(f"There are only {unique_sessions} sessions available on disk for {ds.n_classes} classes.")
            print("Under this configuration, train and test splits will contain disjoint classes.")
            print("Falling back to 'stratified' split to prevent training failure!")
            print("!" * 80 + "\n")
            args.split = "stratified"
            
    # Execute training loop
    print(f"Initiating training loop using split: {args.split}...")
    model, history, report = train_model(
        ds=ds,
        split_type=args.split,
        test_fraction=args.test_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=output_dir,
        seed=args.seed,
        augment_rotation=args.augment_rotation
    )
    
    # Print metrics summary
    print("\n" + "=" * 60)
    print("                 CNN Evaluation Summary")
    print("=" * 60)
    print(f"Accuracy: {report['accuracy'] * 100:.2f}%")
    print("\nClasswise Classification Report:")
    for label in ds.class_names:
        metrics = report.get(label)
        if metrics:
            print(f"  {label:<15} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
    print("=" * 60)
    print("All outputs saved successfully inside the model package directory.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
