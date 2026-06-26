# scripts/visualize_processing.py
"""
Quality-assurance diagrams for the processing pipeline (``data_fusion_project.processing``).

Renders the diagnostic plots that let you judge pipeline quality at a glance:

  1. Filtering            raw vs. low-pass filtered acc/gyro (noise reduction)
  2. Gravity removal      raw acc vs. estimated gravity vs. gravity-free linear acc
  3. Orientation methods  roll/pitch from accel / gyro / complementary / Kalman overlaid
                          (the key sensor-fusion quality plot)
  4. Calibration          gyro before vs. after bias removal
  5. Dataset overview     class distribution + per-class channel mean (real data only)

It runs against a recorded sample from ``data/`` or, with ``--synthetic``, against a
generated window with known ground truth so you can validate the diagrams immediately
(before any real recordings exist).

Examples:
    python scripts/visualize_processing.py --synthetic
    python scripts/visualize_processing.py --gesture swipe_left
    python scripts/visualize_processing.py --gesture swipe_left --session session_a --sample 00003.csv
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (matches scripts/record_data.py).
import matplotlib.pyplot as plt

# Add the project src/ directory to the python path (matches scripts/check_samples.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_fusion_project.core.paths import DATA_DIR, GESTURES
from data_fusion_project.processing import (
    PipelineConfig, OrientationConfig, OrientationMethod, FilterType,
    estimate_calibration, identity_profile, load_dataset,
)
from data_fusion_project.processing import calibration as calib
from data_fusion_project.processing import filters as filt
from data_fusion_project.processing.orientation import estimate_orientation

COLUMNS = [
    "IMU1_accX", "IMU1_accY", "IMU1_accZ", "IMU1_gyrX", "IMU1_gyrY", "IMU1_gyrZ",
    "IMU2_accX", "IMU2_accY", "IMU2_accZ", "IMU2_gyrX", "IMU2_gyrY", "IMU2_gyrZ",
]
_ACC_AXES = ("accX", "accY", "accZ")
_GYR_AXES = ("gyrX", "gyrY", "gyrZ")
_METHOD_STYLE = {
    OrientationMethod.ACCEL: ("Accel only", "#bbbbbb", 1.0, "-"),
    OrientationMethod.GYRO: ("Gyro integ.", "#ff7f0e", 1.2, "-"),
    OrientationMethod.COMPLEMENTARY: ("Complementary", "#1f77b4", 1.8, "-"),
    OrientationMethod.KALMAN: ("Kalman", "#2ca02c", 1.8, "-"),
}


# ======================================================================================================================
# Synthetic ground-truth window (demo without real recordings)
# ======================================================================================================================
def synthetic_window(fs: float = 100.0, n: int = 150, seed: int = 0):
    """
    Generates a window with a known roll/pitch trajectory for validating the diagrams.
    :param: fs (float): sampling rate in Hz.
    :param: n (int): number of samples.
    :param: seed (int): RNG seed.
    :return: result (tuple): (dataframe, truth) where truth holds true roll/pitch in degrees.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs

    # True roll: a smooth raised-cosine swing 0 -> 35 deg -> 0. Pitch: a small 0 -> -15 deg ramp.
    roll_true = 35.0 * 0.5 * (1 - np.cos(2 * np.pi * t / t[-1]))
    pitch_true = -15.0 * (t / t[-1])

    roll_rad = np.deg2rad(roll_true)
    pitch_rad = np.deg2rad(pitch_true)

    gyro_bias = np.array([1.5, -2.0, 0.5])  # dps, recovered by calibration below

    data = np.zeros((n, 12))
    for imu_off, phase in ((0, 0.0), (6, 0.3)):
        # Gravity direction for roll about X then pitch about Y (acc in g).
        ax = -np.sin(pitch_rad)
        ay = np.sin(roll_rad) * np.cos(pitch_rad)
        az = np.cos(roll_rad) * np.cos(pitch_rad)
        acc = np.column_stack([ax, ay, az])
        # Accelerometer corruption: noise + a linear-acceleration burst mid-window.
        burst = np.zeros(n)
        burst[n // 2 - 10:n // 2 + 10] = 0.35
        acc[:, 1] += burst + rng.normal(0, 0.05, n)
        acc[:, 0] += rng.normal(0, 0.05, n)

        # Gyro rates = derivative of true angles (deg/s) + bias + noise.
        roll_rate = np.gradient(roll_true, t)
        pitch_rate = np.gradient(pitch_true, t)
        gyr = np.zeros((n, 3))
        gyr[:, 0] = roll_rate
        gyr[:, 1] = pitch_rate
        gyr = gyr + gyro_bias + rng.normal(0, 0.8, (n, 3))

        data[:, imu_off:imu_off + 3] = acc
        data[:, imu_off + 3:imu_off + 6] = gyr

    df = pd.DataFrame(data, columns=COLUMNS)
    # Calibration frame: pure stillness at flat pose with the same gyro bias.
    cal = np.zeros((500, 12))
    cal[:, 0:3] = [0, 0, 1]; cal[:, 6:9] = [0, 0, 1]
    cal[:, 3:6] = gyro_bias; cal[:, 9:12] = gyro_bias
    cal_df = pd.DataFrame(cal + rng.normal(0, 0.01, (500, 12)), columns=COLUMNS)
    truth = {"roll": roll_true, "pitch": pitch_true}
    return df, cal_df, truth


# ======================================================================================================================
# Sample selection from recorded data
# ======================================================================================================================
def pick_sample(data_dir: Path, gesture: str | None, session: str | None, sample: str | None):
    """
    Selects a recorded window and its session calibration frame.
    :param: data_dir (Path): data root.
    :param: gesture (str | None): gesture name; first non-empty gesture if None.
    :param: session (str | None): session name; first session if None.
    :param: sample (str | None): sample file name; first sample if None.
    :return: result (tuple): (window_df, calibration_df_or_None, label_str).
    :raises: FileNotFoundError: if no suitable sample can be located.
    """
    candidates = [gesture] if gesture else list(GESTURES)
    for g in candidates:
        gdir = data_dir / g
        if not gdir.is_dir():
            continue
        sessions = [session] if session else [p.name for p in sorted(gdir.iterdir()) if p.is_dir()]
        for s in sessions:
            sdir = gdir / s
            if not sdir.is_dir():
                continue
            files = sorted(p for p in sdir.glob("*.csv")
                           if p.name not in ("calibration.csv", "energy_distribution.csv") and p.stem.isdigit())
            if sample:
                files = [sdir / sample] if (sdir / sample).exists() else []
            if not files:
                continue
            df = pd.read_csv(files[0])
            cal_file = sdir / "calibration.csv"
            cal_df = pd.read_csv(cal_file) if cal_file.exists() else None
            return df, cal_df, f"{g}/{s}/{files[0].name}"
    raise FileNotFoundError("No recorded sample found. Record data first or use --synthetic.")


# ======================================================================================================================
# Signal preparation
# ======================================================================================================================
def prepare(df: pd.DataFrame, cal_df, imu: str, config: PipelineConfig):
    """
    Extracts and processes one IMU's blocks at each pipeline stage for plotting.
    :param: df (pd.DataFrame): raw window.
    :param: cal_df (pd.DataFrame | None): calibration frame for the session.
    :param: imu (str): IMU prefix, e.g. "IMU1".
    :param: config (PipelineConfig): pipeline configuration.
    :return: stages (dict): raw/calibrated/filtered acc & gyr blocks plus the sample period.
    """
    acc_raw = df[[f"{imu}_{ax}" for ax in _ACC_AXES]].to_numpy(float)
    gyr_raw = df[[f"{imu}_{ax}" for ax in _GYR_AXES]].to_numpy(float)

    if cal_df is not None:
        profile = estimate_calibration(cal_df, config.calibration)
    else:
        profile = identity_profile()
    acc_cal, gyr_cal = calib.apply_calibration(acc_raw, gyr_raw, profile.get(imu), config.calibration)

    fs = config.sample_rate_hz
    acc_filt = filt.apply_filter(acc_cal, config.filters.acc_filter, config.filters.acc_cutoff_hz, fs, config.filters.order)
    gyr_filt = filt.apply_filter(gyr_cal, config.filters.gyro_filter, config.filters.gyro_cutoff_hz, fs, config.filters.order)

    return {
        "fs": fs, "t": np.arange(len(df)) / fs,
        "acc_raw": acc_raw, "gyr_raw": gyr_raw,
        "acc_cal": acc_cal, "gyr_cal": gyr_cal,
        "acc_filt": acc_filt, "gyr_filt": gyr_filt,
        "profile": profile,
    }


# ======================================================================================================================
# Plots
# ======================================================================================================================
def plot_filtering(st, imu, out_dir):
    """Saves a raw-vs-filtered comparison for acc and gyro. Returns the file path."""
    t = st["t"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    fig.suptitle(f"Filtering — {imu}: raw vs. low-pass filtered", fontweight="bold")
    labels = ("X", "Y", "Z")
    colors = ("#d62728", "#2ca02c", "#1f77b4")

    for j in range(3):
        axs[0, 0].plot(t, st["acc_cal"][:, j], color=colors[j], alpha=0.35, lw=0.8)
        axs[0, 1].plot(t, st["acc_filt"][:, j], color=colors[j], lw=1.4, label=f"acc{labels[j]}")
        axs[1, 0].plot(t, st["gyr_cal"][:, j], color=colors[j], alpha=0.35, lw=0.8)
        axs[1, 1].plot(t, st["gyr_filt"][:, j], color=colors[j], lw=1.4, label=f"gyr{labels[j]}")

    axs[0, 0].set_title("Accelerometer (raw, calibrated)"); axs[0, 0].set_ylabel("g")
    axs[0, 1].set_title("Accelerometer (filtered)"); axs[0, 1].legend(loc="upper right", fontsize=8)
    axs[1, 0].set_title("Gyroscope (raw, calibrated)"); axs[1, 0].set_ylabel("dps"); axs[1, 0].set_xlabel("t [s]")
    axs[1, 1].set_title("Gyroscope (filtered)"); axs[1, 1].set_xlabel("t [s]"); axs[1, 1].legend(loc="upper right", fontsize=8)
    for ax in axs.ravel():
        ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    path = out_dir / f"01_filtering_{imu}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_gravity_removal(st, imu, out_dir, config):
    """Saves raw acc vs. estimated gravity vs. gravity-free linear acc. Returns the file path."""
    t = st["t"]
    linear, gravity = filt.remove_gravity(st["acc_filt"], st["fs"], config.filters.gravity_cutoff_hz, config.filters.order)
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(f"Gravity removal — {imu} (cutoff {config.filters.gravity_cutoff_hz} Hz)", fontweight="bold")
    labels = ("X", "Y", "Z")
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for j in range(3):
        axs[0].plot(t, st["acc_filt"][:, j], color=colors[j], lw=1.2, label=f"acc{labels[j]}")
        axs[0].plot(t, gravity[:, j], color=colors[j], ls="--", lw=1.0, alpha=0.7)
        axs[1].plot(t, linear[:, j], color=colors[j], lw=1.3, label=f"lin{labels[j]}")
    axs[0].set_title("Filtered acc (solid) vs. estimated gravity (dashed)"); axs[0].set_ylabel("g")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].set_title("Linear acceleration (gravity removed)"); axs[1].set_ylabel("g"); axs[1].set_xlabel("t [s]")
    axs[1].legend(loc="upper right", fontsize=8)
    for ax in axs:
        ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    path = out_dir / f"02_gravity_removal_{imu}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_orientation_methods(st, imu, out_dir, config, truth=None):
    """Saves roll/pitch from all fusion methods overlaid. Returns the file path."""
    t = st["t"]
    acc, gyr, fs = st["acc_filt"], st["gyr_filt"], st["fs"]
    angles = {}
    for method in _METHOD_STYLE:
        oc = OrientationConfig(method=method, alpha=config.orientation.alpha, degrees=True,
                               kalman_q_angle=config.orientation.kalman_q_angle,
                               kalman_q_bias=config.orientation.kalman_q_bias,
                               kalman_r_measure=config.orientation.kalman_r_measure)
        angles[method] = estimate_orientation(acc, gyr, fs, oc)

    fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"Orientation fusion — {imu}: accel / gyro / complementary / Kalman", fontweight="bold")
    for key, ax in (("roll", axs[0]), ("pitch", axs[1])):
        if truth is not None:
            ax.plot(t, truth[key], color="black", ls=":", lw=2.2, label="ground truth")
        for method, (name, color, lw, ls) in _METHOD_STYLE.items():
            ax.plot(t, angles[method][key], color=color, lw=lw, ls=ls, label=name)
        ax.set_title(f"{key.capitalize()} [deg]"); ax.set_ylabel("deg")
        ax.grid(True, ls="--", alpha=0.4); ax.legend(loc="upper right", fontsize=8, ncol=2)
    axs[1].set_xlabel("t [s]")
    plt.tight_layout()
    path = out_dir / f"03_orientation_methods_{imu}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_calibration(st, imu, out_dir):
    """Saves gyro before vs. after bias removal. Returns the file path."""
    t = st["t"]
    bias = st["profile"].get(imu).gyro_bias
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(f"Calibration — {imu}: gyro bias removal "
                 f"(bias = [{bias[0]:.2f}, {bias[1]:.2f}, {bias[2]:.2f}] dps)", fontweight="bold")
    labels = ("X", "Y", "Z"); colors = ("#d62728", "#2ca02c", "#1f77b4")
    for j in range(3):
        axs[0].plot(t, st["gyr_raw"][:, j], color=colors[j], lw=1.2, label=f"gyr{labels[j]}")
        axs[1].plot(t, st["gyr_cal"][:, j], color=colors[j], lw=1.2, label=f"gyr{labels[j]}")
    for k in (0, 1):
        axs[k].axhline(0, color="black", lw=0.8, alpha=0.5)
        axs[k].grid(True, ls="--", alpha=0.4); axs[k].set_xlabel("t [s]"); axs[k].legend(fontsize=8)
    axs[0].set_title("Raw gyro"); axs[0].set_ylabel("dps"); axs[1].set_title("Bias removed")
    plt.tight_layout()
    path = out_dir / f"04_calibration_{imu}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_dataset_overview(data_dir, out_dir):
    """Saves class distribution + per-class IMU1 acc magnitude mean. Returns the path or None."""
    try:
        ds = load_dataset(data_dir=data_dir)
    except (FileNotFoundError, RuntimeError):
        return None

    dist = ds.class_distribution()
    present = {k: v for k, v in dist.items() if v > 0}
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset overview", fontweight="bold")
    axs[0].bar(list(present.keys()), list(present.values()), color="#1f77b4")
    axs[0].set_title("Samples per class"); axs[0].set_ylabel("count")
    axs[0].tick_params(axis="x", rotation=45)

    # Per-class mean of the first accelerometer channel over time.
    ch = 0
    t = np.arange(ds.X.shape[1]) / (ds.config.sample_rate_hz if ds.config else 100.0)
    for label in np.unique(ds.y):
        name = ds.class_names[int(label)]
        mean = ds.X[ds.y == label][:, :, ch].mean(axis=0)
        axs[1].plot(t, mean, label=name, lw=1.3)
    axs[1].set_title(f"Per-class mean of '{ds.channel_names[ch]}'")
    axs[1].set_xlabel("t [s]"); axs[1].grid(True, ls="--", alpha=0.4); axs[1].legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / "05_dataset_overview.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


# ======================================================================================================================
# Main
# ======================================================================================================================
def main(argv=None) -> int:
    """
    Parses arguments, prepares signals and writes the diagnostic PNGs.
    :param: argv (list | None): argument list (defaults to ``sys.argv``).
    :return: exit_code (int): 0 on success, 1 if no data is available.
    """
    p = argparse.ArgumentParser(description="Render QA diagrams for the processing pipeline.")
    p.add_argument("--synthetic", action="store_true", help="Use a generated window with ground truth.")
    p.add_argument("--data-dir", default=None, help="Data root (defaults to project data/).")
    p.add_argument("--gesture", default=None, help="Gesture to pick a sample from.")
    p.add_argument("--session", default=None, help="Session name.")
    p.add_argument("--sample", default=None, help="Sample file name, e.g. 00003.csv.")
    p.add_argument("--imu", default="IMU1", help="IMU to visualize (IMU1=wrist, IMU2=finger).")
    p.add_argument("--out", default=str(PROJECT_ROOT / "diagnostics"), help="Output directory.")
    p.add_argument("--orientation-cutoff", type=float, default=8.0, help="Acc/gyro low-pass cutoff (Hz).")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Low-pass config so the filtering/orientation diagrams are meaningful.
    config = PipelineConfig()
    config.filters.acc_filter = FilterType.LOWPASS
    config.filters.gyro_filter = FilterType.LOWPASS
    config.filters.acc_cutoff_hz = args.orientation_cutoff
    config.filters.gyro_cutoff_hz = args.orientation_cutoff

    truth = None
    if args.synthetic:
        df, cal_df, truth = synthetic_window(fs=config.sample_rate_hz, n=config.window_size)
        label = "synthetic window (with ground truth)"
    else:
        try:
            df, cal_df, label = pick_sample(data_dir, args.gesture, args.session, args.sample)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return 1

    print(f"Source : {label}")
    print(f"IMU    : {args.imu}")
    print(f"Output : {out_dir}")

    st = prepare(df, cal_df, args.imu, config)

    written = [
        plot_filtering(st, args.imu, out_dir),
        plot_gravity_removal(st, args.imu, out_dir, config),
        plot_orientation_methods(st, args.imu, out_dir, config, truth=truth),
        plot_calibration(st, args.imu, out_dir),
    ]
    overview = plot_dataset_overview(data_dir, out_dir)
    if overview is not None:
        written.append(overview)

    print("\nWrote diagrams:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
