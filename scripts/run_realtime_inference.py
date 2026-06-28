#!/usr/bin/env python
# scripts/run_realtime_inference.py
"""
Prototype script to run real-time inference using the trained multi-branch CNN.
Performs a 5-second initial calibration, connects to both IMU sensors, and
runs continuous sliding window predictions.
"""

import sys
import os
import time
import argparse
import platform
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json

# Add project src/ directory to the python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

# Configure Keras backend before importing Keras modules
if "KERAS_BACKEND" not in os.environ:
    if platform.system() == "Darwin":
        os.environ["KERAS_BACKEND"] = "torch"
    else:
        os.environ["KERAS_BACKEND"] = "tensorflow"

from data_fusion_project.core.cli_ui import ui, Style
from data_fusion_project.recording.input_data import IMUDataInput
from data_fusion_project.recording.sync import process_stream
from data_fusion_project.recording import device_resolution
from data_fusion_project.processing import (
    process_window,
    PipelineConfig,
    CalibrationConfig,
    FilterConfig,
    OrientationConfig,
    FeatureConfig,
    estimate_calibration
)
from data_fusion_project.processing.config import FilterType, OrientationMethod
from data_fusion_project.training.model import build_multi_branch_cnn
# Keep TimeSeriesScaler class loaded for joblib deserialization
from data_fusion_project.training.train import TimeSeriesScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time CNN Inference Prototype")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "late_fusion_cnn_v1"),
        help="Path to the trained model directory."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Confidence threshold to trigger a gesture (0.0 to 1.0)."
    )
    return parser.parse_args()


def load_pipeline_config(metadata: dict) -> PipelineConfig:
    """
    Reconstructs the PipelineConfig object from metadata dictionary.
    """
    p_cfg_dict = metadata["pipeline_config"]

    cal_cfg = CalibrationConfig(**p_cfg_dict["calibration"])

    fil_cfg_dict = p_cfg_dict["filters"].copy()
    if "acc_filter" in fil_cfg_dict:
        fil_cfg_dict["acc_filter"] = FilterType(fil_cfg_dict["acc_filter"])
    if "gyro_filter" in fil_cfg_dict:
        fil_cfg_dict["gyro_filter"] = FilterType(fil_cfg_dict["gyro_filter"])
    fil_cfg = FilterConfig(**fil_cfg_dict)

    ori_cfg_dict = p_cfg_dict["orientation"].copy()
    if "method" in ori_cfg_dict:
        ori_cfg_dict["method"] = OrientationMethod(ori_cfg_dict["method"])
    ori_cfg = OrientationConfig(**ori_cfg_dict)

    feat_cfg = FeatureConfig(**p_cfg_dict["features"])

    return PipelineConfig(
        sample_rate_hz=p_cfg_dict.get("sample_rate_hz", 100.0),
        window_size=p_cfg_dict.get("window_size", 150),
        pad_mode=p_cfg_dict.get("pad_mode", "edge"),
        jitter_range=p_cfg_dict.get("jitter_range", 0),
        calibration=cal_cfg,
        filters=fil_cfg,
        orientation=ori_cfg,
        features=feat_cfg
    )


def record_calibration(imu1: IMUDataInput, imu2: IMUDataInput, config: CalibrationConfig) -> pd.DataFrame:
    """
    Records 5 seconds of stillness and aligns the streams to generate a calibration dataset.
    """
    ui.wait_for_enter("Ready? Press [Enter] to start 5s static calibration...")
    ui.info("Please hold the sensors absolutely still...")
    
    # Drain queues to start fresh
    imu1.get_data()
    imu2.get_data()
    
    # Simple countdown using progress bar
    ui.progress_bar(5.0, label="Calibrating: ", color=Style.SUCCESS)
    
    snapshot1 = imu1.get_data()
    snapshot2 = imu2.get_data()
    
    if not snapshot1 or not snapshot2:
        raise RuntimeError("No sensor data received during calibration.")
        
    df1 = pd.DataFrame(snapshot1)
    df2 = pd.DataFrame(snapshot2)
    
    # Synchronize streams on a larger window (500 samples = 5s)
    merged, valid_windows = process_stream(df1, df2, window_sz=500, max_diff_us=10000, freq_hz=100)
    if not valid_windows:
        raise RuntimeError("Failed to align sensors during calibration. Please hold them still.")
        
    return merged


def _trim_before(buf, cutoff_us):
    i = 0
    while i < len(buf) and buf[i]['pc_timestamp_us'] < cutoff_us:
        i += 1
    if i:
        del buf[:i]


def print_prediction(class_name, prob, threshold):
    """
    Prints a formatted, color-coded prediction bar on the current terminal line.
    """
    bar_width = 20
    filled = int(round(prob * bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)
    
    # Color-coding based on active status and confidence threshold
    if prob >= threshold and class_name != "none":
        # Green highlighted for successfully triggered gesture
        color_start = "\033[92m\033[1m"
        color_end = "\033[0m"
    elif class_name == "none":
        # Gray for stillness
        color_start = "\033[90m"
        color_end = "\033[0m"
    else:
        # Yellow for gesture predicted but below threshold
        color_start = "\033[93m"
        color_end = "\033[0m"
        
    sys.stdout.write(f"\rGesture: {color_start}{class_name:<12}{color_end} | Conf: {prob*100:5.1f}% [{bar}]")
    sys.stdout.flush()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute() and not model_dir.is_dir():
        # Fallback to checking relative to the project root
        project_relative = PROJECT_ROOT / model_dir
        if project_relative.is_dir():
            model_dir = project_relative

    ui.hr(title="Real-Time CNN Inference Prototype")
    ui.info(f"Target model directory: {model_dir}")
    
    if not model_dir.is_dir():
        ui.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    # 1. Load Model Metadata and Config
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    classes = metadata["classes"]
    config = load_pipeline_config(metadata)
    
    ui.success("Model metadata and pipeline configuration loaded successfully.")

    # 2. Build model and load weights
    ui.info("Building model and loading saved weights...")
    wrist_shape = (config.window_size, len(metadata["wrist_channels"]))
    finger_shape = (config.window_size, len(metadata["finger_channels"]))
    
    model = build_multi_branch_cnn(
        input_shape_wrist=wrist_shape,
        input_shape_finger=finger_shape,
        num_classes=len(classes),
        input_shape_feat=None
    )
    
    weights_path = model_dir / "model.weights.h5"
    model.load_weights(weights_path)
    ui.success("Weights loaded successfully.")

    # 3. Load Scalers
    scaler_wrist_path = model_dir / "scaler_x_wrist.joblib"
    scaler_finger_path = model_dir / "scaler_x_finger.joblib"
    
    scaler_wrist = joblib.load(scaler_wrist_path)
    scaler_finger = joblib.load(scaler_finger_path)
    ui.success("Scalers deserialized.")

    # 4. Resolve Device Ports & Initialize
    device_resolution.print_available_serial_ports()
    port_imu1 = device_resolution.resolve_device_port("imu1")
    port_imu2 = device_resolution.resolve_device_port("imu2")

    imu1 = IMUDataInput(port=port_imu1, name="IMU1")
    imu2 = IMUDataInput(port=port_imu2, name="IMU2")

    ui.info("Connecting to sensors...")
    imu1.start()
    imu2.start()
    time.sleep(2.0)
    
    if not imu1.running or not imu2.running:
        ui.error("Unable to connect to one or both sensors. Aborting.")
        imu1.stop()
        imu2.stop()
        sys.exit(1)

    try:
        # 5. Run Initial Calibration
        cal_df = record_calibration(imu1, imu2, config.calibration)
        profile = estimate_calibration(cal_df, config.calibration)
        ui.success("Static calibration completed and estimated.")
        
        # Silence frequent background sync/packet logs during the streaming loop
        from data_fusion_project.core.logger_setup import set_log_level
        set_log_level("WARNING")
        
        ui.hr(title="Live Inference Active")
        ui.info("Perform gestures to see real-time classifications. Press Ctrl+C to exit.\n")
        
        # 6. Sliding Window Inference Loop
        local_buf1 = []
        local_buf2 = []
        
        window_us = int(1.5 * 1e6)  # 1.5 seconds window size
        advance_us = 100000        # Evaluate every 10 samples (100 ms)
        next_start_us = None
        
        # Clear data queues to start prediction cleanly
        imu1.get_data()
        imu2.get_data()
        
        while True:
            if not imu1.running or not imu2.running:
                ui.error("\nConnection lost to sensor.")
                break
                
            time.sleep(0.05)
            
            # Fetch new packets
            local_buf1.extend(imu1.get_data())
            local_buf2.extend(imu2.get_data())
            
            if not local_buf1 or not local_buf2:
                continue
                
            if next_start_us is None:
                next_start_us = max(local_buf1[0]['pc_timestamp_us'], local_buf2[0]['pc_timestamp_us'])
                
            latest_us = min(local_buf1[-1]['pc_timestamp_us'], local_buf2[-1]['pc_timestamp_us'])
            if latest_us - next_start_us < window_us:
                continue
                
            # Align and extract window
            df1 = pd.DataFrame(local_buf1)
            df2 = pd.DataFrame(local_buf2)
            merged_win, valid_windows = process_stream(
                df1, df2, window_sz=config.window_size, max_diff_us=10000, freq_hz=100
            )
            
            if valid_windows:
                window_df = valid_windows[0]
                
                # Preprocess features (Filter + Complementary roll/pitch orientation)
                channels, channel_names, _, _ = process_window(window_df, profile, config)
                
                # Slice and reshape inputs
                wrist_idx = [channel_names.index(ch) for ch in metadata["wrist_channels"]]
                finger_idx = [channel_names.index(ch) for ch in metadata["finger_channels"]]
                
                X_wrist = channels[:, wrist_idx][np.newaxis, :, :]
                X_finger = channels[:, finger_idx][np.newaxis, :, :]
                
                # Apply time-series scaling
                X_wrist_scaled = scaler_wrist.transform(X_wrist)
                X_finger_scaled = scaler_finger.transform(X_finger)
                
                # Predict
                preds = model.predict(
                    {"wrist_input": X_wrist_scaled, "finger_input": X_finger_scaled},
                    verbose=0
                )
                
                pred_idx = np.argmax(preds[0])
                prob = preds[0][pred_idx]
                class_name = classes[pred_idx]
                
                # Display output
                print_prediction(class_name, prob, args.threshold)
                
            # Slide window forward
            next_start_us += advance_us
            _trim_before(local_buf1, next_start_us)
            _trim_before(local_buf2, next_start_us)

    except KeyboardInterrupt:
        ui.info("\nExiting real-time inference loop...")
    finally:
        imu1.stop()
        imu2.stop()
        ui.success("Sensors disconnected.")


if __name__ == "__main__":
    main()
