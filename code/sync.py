import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from data_fusion_project.core.logger_setup import get_logger

logger = get_logger("IMU_Sync")

_META_COLS = frozenset({'sensor_id', 'pc_timestamp_us', 'esp_timestamp_us', 'sync_time_us'})


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostik-Datenklasse
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SyncDiagnostics:
    """
    Auswertung der Synchronisationspipeline.
    Wird von process_stream(..., diagnostics=True) als drittes Rückgabeelement geliefert.

    Verwendung:
        merged, windows, diag = process_stream(df1, df2, diagnostics=True)
        diag.print_summary()
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

    # -- Fenstervalidierung --
    total_windows: int = 0
    valid_windows: int = 0
    max_diff_threshold_us: float = 5000.0
    window_max_diffs_us: list = field(default_factory=list)

    @property
    def discarded_windows(self) -> int:
        return self.total_windows - self.valid_windows

    def summary(self) -> str:
        lines = [
            "─── Synchronisation Diagnostik ───────────────────────",
            "  Interpolation:",
            f"    IMU1  {self.imu1_samples} Samples"
            f"  │  Ø {self.imu1_mean_interval_us/1000:.2f} ms"
            f"  │  max Lücke {self.imu1_max_gap_us/1000:.2f} ms",
            f"    IMU2  {self.imu2_samples} Samples"
            f"  │  Ø {self.imu2_mean_interval_us/1000:.2f} ms"
            f"  │  max Lücke {self.imu2_max_gap_us/1000:.2f} ms",
            f"    Überlappung {self.overlap_duration_us/1000:.1f} ms"
            f"  →  {self.resampled_samples} Gitterpunkte",
            "  Fenstervalidierung:",
            f"    {self.valid_windows}/{self.total_windows} valide"
            f"  ({self.discarded_windows} verworfen)",
        ]
        finite = [d for d in self.window_max_diffs_us if np.isfinite(d)]
        if finite:
            lines.append(
                f"    max Δt  Ø {np.mean(finite)/1000:.2f} ms"
                f"  │  worst {max(finite)/1000:.2f} ms"
            )
        for i, d in enumerate(self.window_max_diffs_us):
            ok = np.isfinite(d) and d <= self.max_diff_threshold_us
            d_str = f"{d/1000:.2f} ms" if np.isfinite(d) else "∞"
            lines.append(f"    [{'✓' if ok else '✗'}] Fenster {i:>2}: {d_str:>9}")
        lines.append("──────────────────────────────────────────────────────")
        return "\n".join(lines)

    def print_summary(self):
        print(self.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-Funktionen
# ─────────────────────────────────────────────────────────────────────────────

def _sensor_cols(df):
    return [c for c in df.columns if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def align_timestamps(df1, df2, n_anchor=10):
    """
    Berechnet sync_time_us für beide DataFrames.

    Der PC-ESP-Offset wird als Median der ersten n_anchor Samples geschätzt —
    robuster gegen verzögerte erste Pakete als ein einzelner iloc[0]-Wert.
    Beide Sensoren werden auf einen gemeinsamen Nullpunkt normiert.
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


def interpolate_and_merge(df1, df2, freq_hz=100):
    """
    Resamplet beide Sensoren via np.interp auf ein gemeinsames Zeitgitter
    und gibt einen DataFrame mit Präfix IMU1_/IMU2_ zurück.
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


def _nn_max_diff(t1, t2):
    """
    Gibt den maximalen Nearest-Neighbor-Abstand zwischen t1 und t2 zurück.
    O(n log n) via searchsorted — statt O(n²) Brute-Force.
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
                max_time_diff_us=5000, diagnostics=False):
    """
    Unterteilt merged_df in nicht-überlappende Fenster fester Größe.
    Verwirft Fenster, in denen die maximale zeitliche Lücke zwischen den
    Originalmessungen beider Sensoren max_time_diff_us überschreitet.

    diagnostics=True: gibt (windows, diffs_per_window) zurück statt nur windows.
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
        logger.info(f"Fenster {len(windows)}: max Diff {max_diff:.0f} µs -> {'OK' if valid else 'verworfen'}")

        if valid:
            windows.append(merged_df.iloc[start:end])

    return (windows, diffs_per_window) if diagnostics else windows


def process_stream(df1, df2, window_sz=50, max_diff_us=5000, freq_hz=100, diagnostics=False):
    """
    Synchronisierungs-Pipeline:
    1. Timestamps ausrichten  (robuster Median-Offset)
    2. Auf gemeinsames Zeitgitter resamplen  (np.interp)
    3. In valide Fenster unterteilen  (O(n log n) Qualitätsprüfung)

    Rückgabe:
        diagnostics=False  →  (merged_df, windows)
        diagnostics=True   →  (merged_df, windows, SyncDiagnostics)
    """
    df1_a, df2_a = align_timestamps(df1, df2)
    merged = interpolate_and_merge(df1_a, df2_a, freq_hz)

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
