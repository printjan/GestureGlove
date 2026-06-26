# scripts/build_dataset.py
"""
Example / CLI for the data processing interface (``data_fusion_project.processing``).

Loads the recorded ``data/`` tree, runs the configurable pipeline (calibration -> filtering
-> roll/pitch fusion -> feature assembly) and reports the resulting CNN-ready arrays.
Optionally caches the dataset to a compressed ``.npz`` for repeated training runs.

Examples:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --orientation kalman --diff --cross-correlation
    python scripts/build_dataset.py --save data/cache/dataset.npz
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================
import sys
import argparse
from pathlib import Path

# Add the project src/ directory to the python path (matches scripts/check_samples.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_fusion_project.processing import (
    load_dataset,
    PipelineConfig,
    FilterConfig,
    OrientationConfig,
    FeatureConfig,
    FilterType,
    OrientationMethod,
    leave_sessions_out,
)


# ======================================================================================================================
# Configuration assembly
# ======================================================================================================================
def build_config(args) -> PipelineConfig:
    """
    Translates CLI arguments into a :class:`PipelineConfig`.
    :param: args (argparse.Namespace): parsed command line arguments.
    :return: config (PipelineConfig): assembled pipeline configuration.
    """
    return PipelineConfig(
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


def parse_args(argv=None) -> argparse.Namespace:
    """
    Defines and parses the command line interface.
    :param: argv (list | None): argument list (defaults to ``sys.argv``).
    :return: args (argparse.Namespace): parsed arguments.
    """
    p = argparse.ArgumentParser(description="Build a CNN-ready gesture dataset from recorded CSVs.")
    p.add_argument("--data-dir", default=None, help="Data root (defaults to the project data/ directory).")
    p.add_argument("--save", default=None, help="Optional .npz output path to cache the dataset.")

    p.add_argument("--no-filter", action="store_true", help="Disable digital filtering.")
    p.add_argument("--acc-filter", default="lowpass", choices=[f.value for f in FilterType])
    p.add_argument("--acc-cutoff", type=float, default=8.0, help="Accelerometer cutoff in Hz.")
    p.add_argument("--gyro-filter", default="lowpass", choices=[f.value for f in FilterType])
    p.add_argument("--gyro-cutoff", type=float, default=12.0, help="Gyroscope cutoff in Hz.")
    p.add_argument("--gravity-removal", action="store_true", help="Replace acc with gravity-free linear acc.")

    p.add_argument("--orientation", default="complementary", choices=[m.value for m in OrientationMethod])
    p.add_argument("--orientation-imus", nargs="+", default=["IMU1"], help="IMUs to compute roll/pitch for.")

    p.add_argument("--diff", action="store_true", help="Add inter-IMU (finger-wrist) difference channels.")
    p.add_argument("--cross-correlation", action="store_true", help="Add cross-correlation scalar features.")
    p.add_argument("--statistics", action="store_true", help="Add per-channel statistic scalar features.")
    return p.parse_args(argv)


# ======================================================================================================================
# Main
# ======================================================================================================================
def main(argv=None) -> int:
    """
    Builds the dataset from CLI arguments and prints a summary.
    :param: argv (list | None): argument list (defaults to ``sys.argv``).
    :return: exit_code (int): 0 on success, 1 if no data was found.
    """
    args = parse_args(argv)
    config = build_config(args)

    print("Pipeline configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print()

    try:
        ds = load_dataset(config, data_dir=args.data_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        print("Record some gestures first (scripts/record_data.py) so data/ is populated.")
        return 1

    print(ds.summary())

    # Demonstrate a leave-session-out split (the recommended evaluation protocol).
    if len(set(ds.groups.tolist())) > 1:
        train_idx, test_idx = leave_sessions_out(ds.groups, test_fraction=0.2)
        print(f"\nLeave-session-out split: {len(train_idx)} train / {len(test_idx)} test windows")
        print(f"CNN input shape (T, C): {ds.input_shape}   |   classes: {ds.n_classes}")

    if args.save:
        ds.save(args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
