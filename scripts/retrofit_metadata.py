#!/usr/bin/env python3
# scripts/retrofit_metadata.py
"""
Retrofits existing model metadata files on disk to include 'feature_toggles'.
"""

import sys
import json
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_fusion_project.core.cli_ui import ui

ALL_37_FEATURES = [
    "IMU1_linear_jerkX",
    "IMU1_linear_jerkZ",
    "IMU2_linear_jerkZ",
    "IMU1_angular_accelerationY",
    "IMU1_angular_accelerationZ",
    "IMU2_angular_accelerationY",
    "IMU1_accX",
    "IMU1_accZ",
    "IMU1_gyrX",
    "IMU1_pitch",
    "IMU2_accX",
    "IMU2_accY",
    "IMU2_accZ",
    "IMU2_gyrX",
    "diff_accX",
    "diff_accZ",
    "IMU1_gyr_mag",
    "IMU1_accY",
    "IMU1_gyrY",
    "IMU1_gyrZ",
    "IMU1_acc_mag",
    "IMU1_roll",
    "IMU1_relative_yaw",
    "IMU1_linear_jerkY",
    "IMU1_angular_accelerationX",
    "IMU2_gyrY",
    "IMU2_gyrZ",
    "IMU2_gyr_mag",
    "IMU2_acc_mag",
    "IMU2_relative_yaw",
    "IMU2_linear_jerkX",
    "IMU2_linear_jerkY",
    "IMU2_angular_accelerationX",
    "IMU2_angular_accelerationZ",
    "diff_accY",
    "diff_gyrX",
    "diff_gyrY",
    "diff_gyrZ",
]


def matches_feature(feature_name: str, channel_name: str) -> bool:
    norm_feat = feature_name.lower().replace("_", "")
    norm_chan = channel_name.lower().replace("_", "")
    if norm_feat == norm_chan:
        return True
    
    replacements = {
        "gyrmag": "gyroscopemagnitude",
        "accmag": "accelerometermagnitude",
        "diffacc": "relativeacceleration",
        "diffgyr": "relativerotation",
    }
    for k, v in replacements.items():
        if norm_feat.replace(k, v) == norm_chan or norm_chan.replace(k, v) == norm_feat:
            return True
            
    return False


def main() -> int:
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.is_dir():
        ui.error(f"Models directory not found at: {models_dir}")
        return 1

    ui.hr(title="Retrofitting Model Metadata")
    
    metadata_files = sorted(models_dir.glob("**/model_metadata.json")) + sorted(models_dir.glob("**/metadata.json"))
    metadata_files = list(set(metadata_files)) # deduplicate
    
    if not metadata_files:
        ui.info("No metadata files found on disk.")
        return 0

    retrofitted_count = 0
    for meta_path in metadata_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            ui.error(f"Failed to read {meta_path.relative_to(PROJECT_ROOT)}: {exc}")
            continue

        channels = data.get("channels", [])
        
        # Build features toggles dict
        feature_toggles = {}
        for feat in ALL_37_FEATURES:
            feature_toggles[feat] = any(matches_feature(feat, chan) for chan in channels)

        # Update and save
        data["feature_toggles"] = feature_toggles
        
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            ui.success(f"Retrofitted: {meta_path.relative_to(models_dir)}")
            retrofitted_count += 1
        except Exception as exc:
            ui.error(f"Failed to write {meta_path.relative_to(PROJECT_ROOT)}: {exc}")

    ui.hr()
    ui.success(f"Successfully retrofitted {retrofitted_count} metadata files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
