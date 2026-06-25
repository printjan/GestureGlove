# code/sync.py
"""
Data synchronization module for merging and windowing dual IMU sensor streams.
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from data_fusion_project.core.logger_setup import get_logger

logger = get_logger("IMU_Sync")

_META_COLS = frozenset({'sensor_id', 'pc_timestamp_us', 'esp_timestamp_us', 'sync_time_us'})


# ======================================================================================================================
# Sync Diagnostics
# ======================================================================================================================
@dataclass
class SyncDiagnostics:
    """
    Diagnostics results for evaluation of the sync pipeline.
    """
    # -- Interpolation --
    imu1_samples: int = 0
    imu2_samples: int = 0
    imu1_max_gap_us: float = 0.0
    imu2_max_gap_us: float = 0.0
    imu1_mean_interval_us: float = 0.0
    imu2_mean_interval_us: float = 0.0
    overlap_duration_us: int = 0
    resampled_samples: int = 0

    # -- Window Validation --
    total_windows: int = 0
    valid_windows: int = 0
    max_diff_threshold_us: float = 5000.0
    window_max_diffs_us: list = field(default_factory=list)

    @property
    def discarded_windows(self) -> int:
        """
        Returns the number of discarded windows.
        :return: count (int): number of discarded windows.
        """
        return self.total_windows - self.valid_windows

    def summary(self) -> str:
        """
        Generates a summary string of the sync diagnostics.
        :return: report (str): formatted diagnostic report.
        """
        lines = [
            "─── Synchronization Diagnostics ───────────────────────",
            "  Interpolation:",
            f"    IMU1  {self.imu1_samples} Samples"
            f"  │  Ø {self.imu1_mean_interval_us/1000:.2f} ms"
            f"  │  max gap {self.imu1_max_gap_us/1000:.2f} ms",
            f"    IMU2  {self.imu2_samples} Samples"
            f"  │  Ø {self.imu2_mean_interval_us/1000:.2f} ms"
            f"  │  max gap {self.imu2_max_gap_us/1000:.2f} ms",
            f"    Overlap {self.overlap_duration_us/1000:.1f} ms"
            f"  →  {self.resampled_samples} Grid points",
            "  Window Validation:",
            f"    {self.valid_windows}/{self.total_windows} valid"
            f"  ({self.discarded_windows} discarded)",
        ]
        finite = [d for d in self.window_max_diffs_us if np.isfinite(d)]
        if finite:
            lines.append(
                f"    max dt  Ø {np.mean(finite)/1000:.2f} ms"
                f"  │  worst {max(finite)/1000:.2f} ms"
            )
        for i, d in enumerate(self.window_max_diffs_us):
            ok = np.isfinite(d) and d <= self.max_diff_threshold_us
            d_str = f"{d/1000:.2f} ms" if np.isfinite(d) else "inf"
            lines.append(f"    [{'✓' if ok else '✗'}] Window {i:>2}: {d_str:>9}")
        lines.append("──────────────────────────────────────────────────────")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """
        Prints the diagnostics summary report to standard output.
        :return: None:
        """
        print(self.summary())


# ======================================================================================================================
# Alignment & Interpolation Functions
# ======================================================================================================================
def _sensor_cols(df) -> list:
    """
    Filters numeric sensor reading columns from metadata.
    :param: df (DataFrame): input data.
    :return: cols (list): list of sensor reading column names.
    """
    return [c for c in df.columns if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def align_timestamps(df1, df2, n_anchor=10) -> tuple:
    """
    Calculates synced microsecond timestamps normalized to a common zero anchor point.
    :param: df1 (DataFrame): data of first IMU.
    :param: df2 (DataFrame): data of second IMU.
    :param: n_anchor (int): number of initial samples to calculate median offset.
    :return: synced_dfs (tuple): tuple containing aligned df1 and df2.
    """
    if df1.empty or df2.empty:
        return df1, df2

    def _offset(df):
        n = min(n_anchor, len(df))
        return int(np.median(df['pc_timestamp_us'].iloc[:n].values
                             - df['esp_timestamp_us'].iloc[:n].values))

    t1 = df1['esp_timestamp_us'] + _offset(df1)
    t2 = df2['esp_timestamp_us'] + _offset(df2)
    anchor = min(t1.iloc[0], t2.iloc[0])

    out1, out2 = df1.copy(), df2.copy()
    out1['sync_time_us'] = t1 - anchor
    out2['sync_time_us'] = t2 - anchor
    return out1, out2


def interpolate_and_merge(df1, df2, freq_hz=100) -> pd.DataFrame:
    """
    Resamples both sensor streams using np.interp onto a common timestamp grid.
    :param: df1 (DataFrame): aligned data of first IMU.
    :param: df2 (DataFrame): aligned data of second IMU.
    :param: freq_hz (int): target grid frequency in Hertz.
    :return: merged_df (DataFrame): merged dataframe containing prefixed IMU1_ and IMU2_ columns.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame()

    period_us = int(1e6 / freq_hz)
    t_start = int(max(df1['sync_time_us'].iloc[0], df2['sync_time_us'].iloc[0]))
    t_end   = int(min(df1['sync_time_us'].iloc[-1], df2['sync_time_us'].iloc[-1]))

    if t_start >= t_end:
        return pd.DataFrame()

    common_time = np.arange(t_start, t_end, period_us, dtype=np.int64)

    def _resample(df, prefix):
        t = df['sync_time_us'].values.astype(np.float64)
        cols = {'sync_time_us': common_time}
        for col in _sensor_cols(df):
            cols[f'{prefix}{col}'] = np.interp(common_time, t, df[col].values)
        return pd.DataFrame(cols)

    return _resample(df1, 'IMU1_').merge(_resample(df2, 'IMU2_'), on='sync_time_us')


# ======================================================================================================================
# Windowing & Synchronization Functions
# ======================================================================================================================
def _nn_max_diff(t1, t2) -> float:
    """
    Computes the maximum nearest neighbor distance between timestamps using searchsorted.
    :param: t1 (NDArray): timestamps list 1.
    :param: t2 (NDArray): timestamps list 2.
    :return: max_diff (float): max Nearest Neighbor timestamp discrepancy.
    """
    if len(t1) == 0 or len(t2) == 0:
        return np.inf

    t2s = np.sort(t2)
    idx = np.searchsorted(t2s, t1)
    n = len(t2s)

    right_diff = np.abs(t1 - t2s[np.clip(idx,     0, n - 1)])
    left_diff  = np.abs(t1 - t2s[np.clip(idx - 1, 0, n - 1)])

    right_diff = np.where(idx >= n, np.inf, right_diff)
    left_diff  = np.where(idx == 0, np.inf, left_diff)

    return float(np.minimum(left_diff, right_diff).max())


def window_data(merged_df, df1_aligned, df2_aligned, window_size_samples,
                max_time_diff_us=5000, diagnostics=False) -> list | tuple:
    """
    Splits merged data into fixed size windows, discarding those exceeding max time difference.
    :param: merged_df (DataFrame): merged sensor streams.
    :param: df1_aligned (DataFrame): aligned IMU1 stream.
    :param: df2_aligned (DataFrame): aligned IMU2 stream.
    :param: window_size_samples (int): size of the window in number of samples.
    :param: max_time_diff_us (int): maximum allowed Nearest Neighbor offset.
    :param: diagnostics (bool): whether to return sync diagnostics.
    :return: windows (list | tuple): list of valid window DataFrames, optionally with discrepancies.
    """
    if merged_df.empty:
        return ([], []) if diagnostics else []

    t1 = df1_aligned['sync_time_us'].values.astype(np.int64)
    t2 = df2_aligned['sync_time_us'].values.astype(np.int64)
    sync_times = merged_df['sync_time_us'].values
    windows = []
    diffs_per_window = []

    for start in range(0, len(merged_df) - window_size_samples + 1, window_size_samples):
        end = start + window_size_samples
        t_start, t_end = sync_times[start], sync_times[end - 1]

        mask1 = (t1 >= t_start) & (t1 <= t_end)
        mask2 = (t2 >= t_start) & (t2 <= t_end)
        max_diff = _nn_max_diff(t1[mask1], t2[mask2])

        if diagnostics:
            diffs_per_window.append(max_diff)

        valid = max_diff <= max_time_diff_us
        logger.info(f"Window {len(windows)}: max diff {max_diff:.0f} us -> {'OK' if valid else 'discarded'}")

        if valid:
            windows.append(merged_df.iloc[start:end])

    return (windows, diffs_per_window) if diagnostics else windows


def extract_centered_window(merged_df, df1_aligned, df2_aligned, window_sz, max_time_diff_us=10000) -> pd.DataFrame | None:
    """
    Finds the center of gravity of motion energy, extracts a window of size window_sz
    centered around it, and validates its synchronization.
    :param: merged_df (DataFrame): merged sensor streams.
    :param: df1_aligned (DataFrame): aligned IMU1 stream.
    :param: df2_aligned (DataFrame): aligned IMU2 stream.
    :param: window_sz (int): size of the window in number of samples.
    :param: max_time_diff_us (int): maximum allowed Nearest Neighbor offset.
    :return: centered_window (DataFrame | None): synchronized and centered window, or None if invalid.
    """
    if len(merged_df) < window_sz:
        logger.error(f"Merged dataframe size {len(merged_df)} is smaller than target window size {window_sz}.")
        return None

    # Calculate motion energy using accelerometer deviations and gyroscope magnitudes
    acc1 = np.sqrt(merged_df['IMU1_accX']**2 + merged_df['IMU1_accY']**2 + merged_df['IMU1_accZ']**2)
    acc2 = np.sqrt(merged_df['IMU2_accX']**2 + merged_df['IMU2_accY']**2 + merged_df['IMU2_accZ']**2)
    gyr1 = np.sqrt(merged_df['IMU1_gyrX']**2 + merged_df['IMU1_gyrY']**2 + merged_df['IMU1_gyrZ']**2)
    gyr2 = np.sqrt(merged_df['IMU2_gyrX']**2 + merged_df['IMU2_gyrY']**2 + merged_df['IMU2_gyrZ']**2)

    # Scale gyro to match acc scale (roughly 100 dps = 1g deviation)
    energy = np.abs(acc1 - 1.0) + np.abs(acc2 - 1.0) + 0.01 * (gyr1 + gyr2)

    total_energy = energy.sum()
    if total_energy < 1e-5:
        mu = len(merged_df) / 2.0
    else:
        indices = np.arange(len(merged_df))
        mu = float(np.sum(indices * energy) / total_energy)

    start_idx = int(round(mu - window_sz / 2.0))
    start_idx = max(0, min(start_idx, len(merged_df) - window_sz))
    end_idx = start_idx + window_sz

    window = merged_df.iloc[start_idx:end_idx]

    # Validate nearest-neighbor timestamp discrepancy for this specific window
    t1 = df1_aligned['sync_time_us'].values.astype(np.int64)
    t2 = df2_aligned['sync_time_us'].values.astype(np.int64)
    sync_times = merged_df['sync_time_us'].values
    t_start, t_end = sync_times[start_idx], sync_times[end_idx - 1]

    mask1 = (t1 >= t_start) & (t1 <= t_end)
    mask2 = (t2 >= t_start) & (t2 <= t_end)
    max_diff = _nn_max_diff(t1[mask1], t2[mask2])

    valid = max_diff <= max_time_diff_us
    logger.info(f"Centered window: start={start_idx}, mu={mu:.1f}, max diff {max_diff:.0f} us -> {'OK' if valid else 'discarded'}")

    return window if valid else None


def process_stream(df1, df2, window_sz=50, max_diff_us=5000, freq_hz=100, diagnostics=False, center_gesture=False) -> tuple:
    """
    Runs the synchronization pipeline: alignment, resampling, and windowing.
    :param: df1 (DataFrame): IMU1 raw packets.
    :param: df2 (DataFrame): IMU2 raw packets.
    :param: window_sz (int): window size in samples.
    :param: max_diff_us (int): maximum Nearest Neighbor timestamp discrepancy.
    :param: freq_hz (int): target resampling grid frequency.
    :param: diagnostics (bool): whether to return detailed SyncDiagnostics object.
    :param: center_gesture (bool): whether to extract the window centered around the gesture's centroid.
    :return: result (tuple): tuple containing merged df, valid windows, and optional diagnostics.
    """
    df1_a, df2_a = align_timestamps(df1, df2)
    merged = interpolate_and_merge(df1_a, df2_a, freq_hz)

    if center_gesture:
        win = extract_centered_window(merged, df1_a, df2_a, window_sz, max_diff_us)
        windows = [win] if win is not None else []
        if not diagnostics:
            return merged, windows
        diag = SyncDiagnostics(
            imu1_samples=len(df1_a),
            imu2_samples=len(df2_a),
            imu1_max_gap_us=0.0,
            imu2_max_gap_us=0.0,
            imu1_mean_interval_us=0.0,
            imu2_mean_interval_us=0.0,
            overlap_duration_us=max(0, int(merged['sync_time_us'].iloc[-1] - merged['sync_time_us'].iloc[0])) if not merged.empty else 0,
            resampled_samples=len(merged),
            total_windows=1,
            valid_windows=len(windows),
            max_diff_threshold_us=float(max_diff_us),
            window_max_diffs_us=[0.0] if windows else [float(max_diff_us + 1.0)],
        )
        return merged, windows, diag

    if not diagnostics:
        windows = window_data(merged, df1_a, df2_a, window_sz, max_diff_us)
        return merged, windows

    windows, diffs = window_data(merged, df1_a, df2_a, window_sz, max_diff_us, diagnostics=True)

    def _interval_stats(df):
        ts = df['sync_time_us'].values
        if len(ts) < 2:
            return 0.0, 0.0
        d = np.diff(ts).astype(float)
        return float(d.max()), float(d.mean())

    imu1_max_gap, imu1_mean = _interval_stats(df1_a)
    imu2_max_gap, imu2_mean = _interval_stats(df2_a)

    t_start = int(max(df1_a['sync_time_us'].iloc[0], df2_a['sync_time_us'].iloc[0]))
    t_end   = int(min(df1_a['sync_time_us'].iloc[-1], df2_a['sync_time_us'].iloc[-1]))

    diag = SyncDiagnostics(
        imu1_samples=len(df1_a),
        imu2_samples=len(df2_a),
        imu1_max_gap_us=imu1_max_gap,
        imu2_max_gap_us=imu2_max_gap,
        imu1_mean_interval_us=imu1_mean,
        imu2_mean_interval_us=imu2_mean,
        overlap_duration_us=max(0, t_end - t_start),
        resampled_samples=len(merged),
        total_windows=len(diffs),
        valid_windows=len(windows),
        max_diff_threshold_us=float(max_diff_us),
        window_max_diffs_us=diffs,
    )
    return merged, windows, diag

