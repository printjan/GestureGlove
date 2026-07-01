# scripts/train.py
"""
Unified CLI entry point for training all three gesture classification
candidate architectures:

1. Early Fusion Single-Branch Conv1D CNN
2. Late Fusion Multi-Branch Conv1D CNN
3. Self-Attention Temporal Transformer

Supports Bayesian dynamic feature optimization via Optuna, architecture-
specific hyperparameters, and the full PipelineConfig feature engineering
configuration.

Example usage:
    # Standard training (early fusion, leave-session-out, 70 epochs)
    python scripts/train.py --model-type early_fusion_cnn --split leave-session-out --epochs 70

    # Compact configuration
    python scripts/train.py --model-type early_fusion_cnn --config compact

    # Late fusion with rotation augmentation and jitter
    python scripts/train.py --model-type late_fusion_cnn --augment-rotation 25 --jitter-range 20

    # Transformer with custom attention params
    python scripts/train.py --model-type temporal_transformer --d-model 64 --num-heads 4

    # Optuna feature optimization (50 trials, 15 epochs per trial)
    python scripts/train.py --model-type late_fusion_cnn --optimize --optuna-trials 50
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================
import sys
import argparse
import time
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
        os.environ["KERAS_BACKEND"] = "torch"
    else:
        os.environ["KERAS_BACKEND"] = "tensorflow"

from data_fusion_project.core.cli_ui import ui, Style
from data_fusion_project.core.paths import MODELS_DIR, get_model_run_dir
from data_fusion_project.processing import (
    load_dataset,
    PipelineConfig,
    FilterConfig,
    OrientationConfig,
    FeatureConfig,
    FilterType,
    OrientationMethod,
)
from data_fusion_project.training.model_training_pipeline.pipeline import (
    train_model,
    MODEL_TYPE_TO_FOLDER,
    matches_feature,
    ALL_37_FEATURES,
)


# ======================================================================================================================
# Architecture Presets
# ======================================================================================================================
PRESETS = {
    "standard": {
        "conv_filters": [32, 64],
        "dense_units": 16,
        "d_model": 64,
        "num_heads": 4,
        "num_blocks": 2,
        "ff_dim": 128,
    },
    "compact": {
        "conv_filters": [16],
        "dense_units": 16,
        "d_model": 32,
        "num_heads": 2,
        "num_blocks": 1,
        "ff_dim": 64,
    },
}


# ======================================================================================================================
# Feature Category Lists (from data quality audit)
# ======================================================================================================================
PRUNED = [
    "IMU1_linear_jerkX", "IMU1_linear_jerkZ", "IMU2_linear_jerkZ",
    "IMU1_angular_accelerationY", "IMU1_angular_accelerationZ", "IMU2_angular_accelerationY",
]

MANDATORY = [
    "IMU1_accX", "IMU1_accZ", "IMU1_gyrX", "IMU1_pitch",
    "IMU2_accX", "IMU2_accY", "IMU2_accZ", "IMU2_gyrX",
    "diff_accX", "diff_accZ", "IMU1_gyr_mag",
]

DYNAMIC = [
    "IMU1_accY", "IMU1_gyrY", "IMU1_gyrZ", "IMU1_acc_mag", "IMU1_roll",
    "IMU1_relative_yaw", "IMU1_linear_jerkY", "IMU1_angular_accelerationX",
    "IMU2_gyrY", "IMU2_gyrZ", "IMU2_gyr_mag", "IMU2_acc_mag", "IMU2_relative_yaw",
    "IMU2_linear_jerkX", "IMU2_linear_jerkY", "IMU2_angular_accelerationX",
    "IMU2_angular_accelerationZ", "diff_accY", "diff_gyrX", "diff_gyrY", "diff_gyrZ",
]


# ======================================================================================================================
# Config Builder
# ======================================================================================================================
def build_pipeline_config(args) -> PipelineConfig:
    """Translates CLI arguments into a PipelineConfig."""
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
    p = argparse.ArgumentParser(
        description="Unified training pipeline for gesture classification models."
    )

    # Architecture selection
    p.add_argument(
        "--model-type", default="early_fusion_cnn",
        choices=["early_fusion_cnn", "late_fusion_cnn", "temporal_transformer"],
        help="Architecture to train.",
    )
    p.add_argument(
        "--config", default="standard", choices=["standard", "compact"],
        help="Architecture preset: 'standard' (2-layer CNN, 2-block transformer) "
             "or 'compact' (1-layer CNN, 1-block transformer).",
    )

    # Core directories & naming
    p.add_argument("--data-dir", default=None, help="Data root directory.")
    p.add_argument(
        "--model-name", default=None,
        help="Override model folder name (defaults to architecture-specific name).",
    )
    p.add_argument(
        "--backend", default=None, choices=["tensorflow", "torch", "jax"],
        help="Keras backend.",
    )

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=70, help="Training epochs (default: 70).")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p.add_argument(
        "--split", default="leave-session-out",
        choices=["stratified", "leave-session-out", "chronological"],
        help="Split strategy.",
    )
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--run-name", default=None,
                   help="Custom session folder name (must start with 'training_session_').")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment-rotation", type=float, default=0.0,
                   help="Max random 3D rotation angle in degrees.")

    # CNN architecture params (overrides --config preset)
    p.add_argument("--conv-filters", type=int, nargs="+", default=None,
                   help="Conv1D filter list (overrides preset).")
    p.add_argument("--dense-units", type=int, default=None,
                   help="Classification head dense units (overrides preset).")

    # Transformer architecture params (overrides --config preset)
    p.add_argument("--d-model", type=int, default=None, help="Projection dimensionality.")
    p.add_argument("--num-heads", type=int, default=None, help="Attention heads.")
    p.add_argument("--num-blocks", type=int, default=None, help="Transformer blocks.")
    p.add_argument("--ff-dim", type=int, default=None, help="Feed-forward expansion dim.")

    # Optuna feature optimization
    p.add_argument("--optimize", action="store_true", help="Enable Bayesian feature sweep.")
    p.add_argument("--optuna-trials", type=int, default=50, help="Number of Optuna trials.")
    p.add_argument("--optuna-epochs", type=int, default=15, help="Epochs per trial.")
    p.add_argument("--w1", type=float, default=0.001, help="Latency penalty weight.")
    p.add_argument("--w2", type=float, default=1e-6, help="Parameter count penalty weight.")

    # Pipeline config (signal filtering & feature engineering)
    p.add_argument("--jitter-range", type=int, default=0)
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--acc-filter", default="lowpass", choices=[f.value for f in FilterType])
    p.add_argument("--acc-cutoff", type=float, default=8.0)
    p.add_argument("--gyro-filter", default="lowpass", choices=[f.value for f in FilterType])
    p.add_argument("--gyro-cutoff", type=float, default=12.0)
    p.add_argument("--gravity-removal", action="store_true")
    p.add_argument("--orientation", default="complementary",
                   choices=[m.value for m in OrientationMethod])
    p.add_argument("--orientation-imus", nargs="+", default=["IMU1", "IMU2"])
    p.add_argument("--diff", action="store_true", help="Add inter-IMU difference channels.")
    p.add_argument("--cross-correlation", action="store_true")
    p.add_argument("--statistics", action="store_true")

    # Pre-computed feature flags
    p.add_argument("--linear-jerk", action="store_true")
    p.add_argument("--angular-acceleration", action="store_true")
    p.add_argument("--relative-acceleration", action="store_true")
    p.add_argument("--relative-rotation", action="store_true")
    p.add_argument("--relative-yaw", action="store_true")
    p.add_argument("--acc-magnitude", action="store_true")
    p.add_argument("--gyro-magnitude", action="store_true")
    p.add_argument("--gravity-free-acc", action="store_true")

    return p.parse_args(argv)


# ======================================================================================================================
# Main
# ======================================================================================================================
def main(argv=None) -> int:
    args = parse_args(argv)

    # Resolve architecture preset and apply explicit overrides
    preset = PRESETS[args.config]
    conv_filters = args.conv_filters or preset["conv_filters"]
    dense_units = args.dense_units if args.dense_units is not None else preset["dense_units"]
    d_model = args.d_model if args.d_model is not None else preset["d_model"]
    num_heads = args.num_heads if args.num_heads is not None else preset["num_heads"]
    num_blocks = args.num_blocks if args.num_blocks is not None else preset["num_blocks"]
    ff_dim = args.ff_dim if args.ff_dim is not None else preset["ff_dim"]

    # Resolve model name (folder in models/)
    model_name = args.model_name or MODEL_TYPE_TO_FOLDER.get(args.model_type, args.model_type)

    # Generate timestamp or use custom run name
    if args.run_name:
        if not args.run_name.startswith("training_session_"):
            ui.error("--run-name must start with 'training_session_' prefix!")
            return 1
        timestamp = args.run_name
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    ui.hr(title=f"Unified Training Pipeline — {args.model_type} ({args.config})")
    ui.info(f"Config preset: {args.config}")
    ui.info(f"Conv filters: {conv_filters}, Dense units: {dense_units}")
    if args.model_type == "temporal_transformer":
        ui.info(f"Transformer: d_model={d_model}, heads={num_heads}, blocks={num_blocks}, ff_dim={ff_dim}")

    # ──────────────────────────────────────────────────────────────────
    # Dataset Loading
    # ──────────────────────────────────────────────────────────────────
    feature_toggles = None

    if args.optimize:
        ui.info("Optuna dynamic feature optimization enabled.")
        ui.info(f"Trials: {args.optuna_trials} | Epochs/trial: {args.optuna_epochs}")

        # Load dataset with ALL features enabled for search-space coverage
        search_cfg = PipelineConfig(
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
                ds = load_dataset(search_cfg, data_dir=args.data_dir)
            except Exception as exc:
                ui.error(f"Failed to load dataset: {exc}")
                return 1
    else:
        pipeline_cfg = build_pipeline_config(args)
        with ui.spinner("Loading gesture dataset..."):
            try:
                ds = load_dataset(pipeline_cfg, data_dir=args.data_dir)
            except Exception as exc:
                ui.error(f"Failed to load dataset: {exc}")
                return 1

    ui.info(ds.summary())

    # ──────────────────────────────────────────────────────────────────
    # Sanity Checks
    # ──────────────────────────────────────────────────────────────────
    if args.split == "leave-session-out":
        class_session_counts = {}
        for label in range(ds.n_classes):
            class_mask = ds.y == label
            class_session_counts[label] = len(set(ds.groups[class_mask].tolist()))

        insufficient = [
            ds.class_names[lbl] for lbl, cnt in class_session_counts.items() if cnt < 2
        ]
        if insufficient:
            ui.warning(
                f"LSO split: classes with only 1 session: {insufficient}. "
                "Falling back to 'chronological'."
            )
            args.split = "chronological"

    # ──────────────────────────────────────────────────────────────────
    # Optuna Feature Optimization
    # ──────────────────────────────────────────────────────────────────
    if args.optimize:
        import optuna
        import logging

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
                logging.getLogger("data_fusion_project").setLevel(logging.WARNING)

                model, history, report = train_model(
                    ds=ds,
                    model_type=args.model_type,
                    split_type=args.split,
                    test_fraction=args.test_fraction,
                    val_fraction=args.val_fraction,
                    epochs=args.optuna_epochs,
                    batch_size=args.batch_size,
                    model_name=None,  # Don't save trial artifacts
                    timestamp=None,
                    seed=args.seed,
                    augment_rotation=args.augment_rotation,
                    feature_toggles=toggles,
                    conv_filters=conv_filters,
                    dense_units=dense_units,
                    d_model=d_model,
                    num_heads=num_heads,
                    num_blocks=num_blocks,
                    ff_dim=ff_dim,
                )

                f1 = float(report["macro avg"]["f1-score"])

                # Estimate inference latency
                dummy_inputs = []
                for inp in model.inputs:
                    dummy_inputs.append(np.zeros((1, *inp.shape[1:]), dtype=np.float32))

                start_l = time.perf_counter()
                for _ in range(3):
                    model.predict(dummy_inputs, verbose=0)
                latency = ((time.perf_counter() - start_l) / 3.0) * 1000.0

                param_count = model.count_params()
                utility = f1 - args.w1 * latency - args.w2 * param_count

                logging.getLogger("data_fusion_project").setLevel(logging.INFO)
                ui.info(
                    f"Trial {trial.number:2d} | F1: {f1:.4f} | "
                    f"Latency: {latency:5.2f}ms | Params: {param_count:6d} | "
                    f"Utility: {utility:6.3f}"
                )
                return utility
            except Exception as e:
                logging.getLogger("data_fusion_project").setLevel(logging.INFO)
                ui.error(f"Trial {trial.number:2d} failed: {e}")
                return -999.0

        ui.hr(title="Optuna Dynamic Feature Sweep (TPE)")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.optuna_trials)

        # Extract best feature toggles
        feature_toggles = {}
        for feat in PRUNED:
            feature_toggles[feat] = False
        for feat in MANDATORY:
            feature_toggles[feat] = True
        for feat in DYNAMIC:
            feature_toggles[feat] = study.best_params[feat]

        ui.hr()
        ui.success("Bayesian feature selection complete!")
        ui.box([
            f"Best Joint Utility Score: {study.best_value:.4f}",
            f"Optimized Dynamic Features: {[f for f in DYNAMIC if feature_toggles[f]]}",
        ], title="TPE Study Results")

        ui.info("Retraining final model on optimal feature subset for full epochs...")
        import logging
        logging.getLogger("data_fusion_project").setLevel(logging.INFO)

    # ──────────────────────────────────────────────────────────────────
    # Final Training
    # ──────────────────────────────────────────────────────────────────
    run_dir = get_model_run_dir(model_name, timestamp)
    ui.info(f"Output: {run_dir}")
    ui.info(f"Training {args.model_type} | split: {args.split} | epochs: {args.epochs}")

    model, history, report = train_model(
        ds=ds,
        model_type=args.model_type,
        split_type=args.split,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=model_name,
        timestamp=timestamp,
        seed=args.seed,
        augment_rotation=args.augment_rotation,
        feature_toggles=feature_toggles,
        conv_filters=conv_filters,
        dense_units=dense_units,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ff_dim=ff_dim,
    )

    # ──────────────────────────────────────────────────────────────────
    # Results Summary
    # ──────────────────────────────────────────────────────────────────
    ui.hr(title=f"{args.model_type} — Evaluation Summary")
    ui.kv([
        ("Accuracy", f"{report['accuracy'] * 100:.2f}%"),
        ("Val Loss", f"{history['val_loss'][-1]:.4f}"),
        ("Macro F1", f"{report['macro avg']['f1-score']:.4f}"),
    ])

    report_lines = []
    for label in ds.class_names:
        metrics = report.get(label)
        if metrics:
            report_lines.append(
                f"{label:<14} -> P: {metrics['precision']:.3f} | "
                f"R: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}"
            )
    ui.box(report_lines, title="Per-Class Metrics")

    ui.success("All outputs saved to model package directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
