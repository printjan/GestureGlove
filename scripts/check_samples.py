# scripts/check_samples.py
"""
Script to recursively check all recorded gesture samples in the data directory 
and identify any samples that do not contain exactly 150 datapoints (too short or too long).

Input:
data/
└── <gesture_name>/
    └── <session_name>/
        └── #####.csv (gesture samples)
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================
import sys
from pathlib import Path
import pandas as pd

# Add the project src/ directory to python path for importing project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

try:
    from data_fusion_project.core.paths import DATA_DIR
except ImportError:
    # Fallback to resolving relative path if project package is not installed/discoverable
    DATA_DIR = PROJECT_ROOT / "data"


# ======================================================================================================================
# Analysis Function
# ======================================================================================================================
def check_samples():
    """
    Scans the DATA_DIR recursively and reports any CSV files not containing exactly 150 rows.
    :return: None:
    """
    print(f"Scanning data directory: {DATA_DIR}")
    if not DATA_DIR.exists():
        print("Error: Data directory does not exist.")
        return

    csv_files = list(DATA_DIR.glob("**/*.csv"))
    checked_count = 0
    anomaly_count = 0

    for csv_file in csv_files:
        # Skip calibration and distribution files
        if csv_file.name in ("calibration.csv", "energy_distribution.csv"):
            continue

        try:
            # Read CSV and count rows (excluding header)
            df = pd.read_csv(csv_file)
            row_count = len(df)
            checked_count += 1

            if row_count != 150:
                anomaly_count += 1
                # Output relative path for clean display
                relative_path = csv_file.relative_to(DATA_DIR)
                print(f"[ANOMALY] {relative_path} has {row_count} datapoints (Expected: exactly 150).")

        except Exception as e:
            print(f"[ERROR] Failed to read {csv_file.name}: {e}")

    print("\n--- Summary ---")
    print(f"Total gesture samples checked: {checked_count}")
    print(f"Anomalous samples found (not exactly 150 datapoints): {anomaly_count}")


if __name__ == "__main__":
    check_samples()
