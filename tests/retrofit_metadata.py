#!/usr/bin/env python3
# tests/retrofit_metadata.py
"""
Retrofits existing model metadata files on disk to include:
- Early stopping status (early_stopped)
- Actual split sizes and real fractions (decimal and absolute)
- Default selected/deselected feature lists
- Programmatic model structure, layer outputs, and parameter counts
"""

import os
import sys
import json
import platform
from pathlib import Path

# Prevent OpenMP runtime clash termination on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set default Keras backend to match workspace execution (torch on Darwin)
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch" if platform.system() == "Darwin" else "tensorflow"

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_fusion_project.core.cli_ui import ui
from data_fusion_project.processing import (
    load_dataset,
    leave_sessions_out_three_way,
    stratified_split_three_way,
    chronological_split_three_way
)
from data_fusion_project.training.late_fusion_multi_branch_cnn_test.model import build_multi_branch_cnn

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

    # Load dataset once to compute exact split distributions
    ui.info("Loading dataset to compute real split sizes...")
    try:
        ds = load_dataset()
        ui.success(f"Dataset loaded. Total samples available: {len(ds)}")
    except Exception as exc:
        ui.error(f"Failed to load dataset: {exc}. Proceeding without split calculations.")
        ds = None
    
    metadata_files = sorted(models_dir.glob("**/model_metadata.json")) + sorted(models_dir.glob("**/metadata.json"))
    metadata_files = list(set(metadata_files))  # deduplicate
    
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
        
        # 1. Build feature toggles dict if missing
        if "feature_toggles" not in data:
            feature_toggles = {}
            for feat in ALL_37_FEATURES:
                feature_toggles[feat] = any(matches_feature(feat, chan) for chan in channels)
            data["feature_toggles"] = feature_toggles
        else:
            feature_toggles = data["feature_toggles"]

        # 2. Selects / Deselects list
        data["features_selection"] = {
            "default_selected_features": [f for f, v in feature_toggles.items() if v],
            "default_deselected_features": [f for f, v in feature_toggles.items() if not v]
        }

        # 3. Early stopping verification
        epochs_trained = data.get("epochs_trained")
        training_params = data.get("training_parameters", {})
        epochs_target = training_params.get("epochs")
        if epochs_trained is not None and epochs_target is not None:
            data["early_stopped"] = bool(epochs_trained < epochs_target)
        else:
            data["early_stopped"] = False

        # 4. Model component layer dimensions and parameters
        wrist_channels = data.get("wrist_channels", [])
        finger_channels = data.get("finger_channels", [])
        classes = data.get("classes", ["none", "swipe_left", "swipe_right", "circle_cw", "circle_ccw", "fist", "jerk_down", "jerk_up"])
        
        conv_filters = training_params.get("conv_filters", [32, 64])
        dense_units = training_params.get("dense_units", 64)
        
        input_shape_wrist = (150, len(wrist_channels)) if wrist_channels else None
        input_shape_finger = (150, len(finger_channels)) if finger_channels else None
        
        try:
            model = build_multi_branch_cnn(
                input_shape_wrist=input_shape_wrist,
                input_shape_finger=input_shape_finger,
                num_classes=len(classes),
                conv_filters=conv_filters,
                dense_units=dense_units
            )
            
            def format_shape(shape):
                if shape is None:
                    return None
                if isinstance(shape, list):
                    return [format_shape(s) for s in shape]
                if isinstance(shape, tuple):
                    return [int(dim) if dim is not None else None for dim in shape]
                return str(shape)

            model_layers_info = []
            for layer in model.layers:
                try:
                    out_shape = layer.output.shape
                except AttributeError:
                    try:
                        out_shape = layer.output_shape
                    except AttributeError:
                        try:
                            out_shape = layer.input_shape
                        except AttributeError:
                            out_shape = None

                model_layers_info.append({
                    "layer_name": layer.name,
                    "class_name": layer.__class__.__name__,
                    "output_shape": format_shape(out_shape),
                    "parameter_count": int(layer.count_params())
                })
            
            data["model_structure"] = {
                "total_parameters": int(model.count_params()),
                "layers": model_layers_info
            }
        except Exception as exc:
            ui.error(f"Failed to build model for {meta_path.name}: {exc}")

        # 5. Real split counts and fractions
        if ds is not None:
            split_type = training_params.get("split_type") or data.get("split_info", {}).get("strategy") or "chronological"
            test_fraction = training_params.get("test_fraction", 0.2)
            val_fraction = training_params.get("val_fraction", 0.1)
            seed = training_params.get("seed", 42)
            
            try:
                if split_type in ("leave_session_out", "leave-session-out"):
                    train_idx, val_idx, test_idx = leave_sessions_out_three_way(
                        ds.groups, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
                    )
                elif split_type == "chronological":
                    train_idx, val_idx, test_idx = chronological_split_three_way(
                        ds.y, test_fraction=test_fraction, val_fraction=val_fraction
                    )
                else:
                    train_idx, val_idx, test_idx = stratified_split_three_way(
                        ds.y, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
                    )
                
                total_samples = len(train_idx) + len(val_idx) + len(test_idx)
                
                if "split_info" not in data or not isinstance(data["split_info"], dict):
                    data["split_info"] = {}
                    
                data["split_info"].update({
                    "strategy": split_type,
                    "total_samples": total_samples,
                    "train_size_abs": len(train_idx),
                    "val_size_abs": len(val_idx),
                    "test_size_abs": len(test_idx),
                    "train_fraction_real": float(len(train_idx)) / max(1, total_samples),
                    "val_fraction_real": float(len(val_idx)) / max(1, total_samples),
                    "test_fraction_real": float(len(test_idx)) / max(1, total_samples),
                })
            except Exception as exc:
                ui.error(f"Failed to calculate splits for {meta_path.name}: {exc}")

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
