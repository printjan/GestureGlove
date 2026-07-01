#!/usr/bin/env python
# scripts/run_realtime_inference_async.py
"""
Script to run real-time inference using the trained multi-branch CNN,
leveraging the AsynchronousDataGrabber to decouple serial ingestion and preprocessing
from the model execution thread.
"""

import sys
import os
import time
import argparse
import platform
import threading
import contextlib
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
from data_fusion_project.training.late_fusion_multi_branch_cnn_test.model import build_multi_branch_cnn
# Keep TimeSeriesScaler class loaded for joblib deserialization
from data_fusion_project.training.late_fusion_multi_branch_cnn_test.train import TimeSeriesScaler
from data_fusion_project.control import (
    PowerPointController,
    GestureDispatcher,
    ControlConfig,
    DryRunBackend,
    PyAutoGuiBackend,
)
from data_fusion_project.inference import (
    AsynchronousDataGrabber,
    TriggerDetector,
    LivePerformanceEvaluator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Async Real-Time CNN Inference")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "late_fusion_cnn_test"),
        help="Path to the trained model directory."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Confidence threshold to trigger a gesture (0.0 to 1.0)."
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="Minimum seconds between two fired actions (de-bounce cool-down)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a powerpoint_control.yml config file (defaults to config/powerpoint_control.yml)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send real key presses; only log the shortcuts that would be sent."
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Disable the PowerPoint control interface (predictions are only displayed)."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate IMU data streaming instead of connecting to real serial hardware."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Automated exit timeout in seconds (useful for headless verification)."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable the objection-based live-performance evaluator (records TP/FP/FN via hotkeys)."
    )
    parser.add_argument(
        "--objection-window",
        type=float,
        default=1.5,
        help="Seconds a fired gesture waits for a correcting keypress before it counts as correct (default: 1.5)."
    )
    parser.add_argument(
        "--eval-out",
        type=str,
        default=None,
        help="Directory for the evaluation report (defaults to the resolved model session directory)."
    )
    return parser.parse_args()


def build_dispatcher(args):
    """
    Build the gesture dispatcher wired to a PowerPoint controller.
    """
    if args.no_control:
        return None, False

    config_path = Path(args.config) if args.config else None
    config = ControlConfig.load(config_path)

    if args.dry_run:
        backend = DryRunBackend()
        live = False
    else:
        try:
            backend = PyAutoGuiBackend(pause_s=config.send_pause_s)
            live = True
        except ImportError:
            ui.warning("pyautogui is not installed - falling back to DRY-RUN (no keys will be sent).")
            ui.hint("Install it with:  pip install -e .[control]    (or:  pip install pyautogui)")
            backend = DryRunBackend()
            live = False

    controller = PowerPointController(config=config, backend=backend)
    dispatcher = GestureDispatcher(
        controller,
        confidence_threshold=args.threshold,
        cooldown_s=args.cooldown,
        require_release=True,
    )
    return dispatcher, live


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
    if sys.stdin.isatty():
        ui.wait_for_enter("Ready? Press [Enter] to start 5s static calibration...")
        ui.info("Please hold the sensors absolutely still...")
    else:
        ui.info("Non-interactive run: Starting 5s static calibration...")
    
    # Drain queues to start fresh
    imu1.get_data()
    imu2.get_data()
    
    # Simple countdown using progress bar
    ui.progress_bar(6.0, label="Calibrating: ", color=Style.SUCCESS)
    
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


def print_prediction(class_name, prob, threshold):
    """
    Prints a formatted, color-coded prediction bar on the current terminal line.
    """
    bar_width = 20
    filled = int(round(prob * bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)
    
    if prob >= threshold and class_name != "none":
        color_start = "\033[92m\033[1m"
        color_end = "\033[0m"
    elif class_name == "none":
        color_start = "\033[90m"
        color_end = "\033[0m"
    else:
        color_start = "\033[93m"
        color_end = "\033[0m"
        
    sys.stdout.write(f"\rGesture: {color_start}{class_name:<12}{color_end} | Conf: {prob*100:5.1f}% [{bar}]")
    sys.stdout.flush()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    
    if not model_dir.is_absolute() and not model_dir.exists():
        fallback_path = PROJECT_ROOT / model_dir
        if fallback_path.exists():
            model_dir = fallback_path

    ui.hr(title="Async Real-Time CNN Inference")
    
    if not model_dir.exists():
        ui.error(f"Model directory not found: {model_dir}")
        sys.exit(1)

    def get_session_num(p):
        parts = p.name.split("_")
        try:
            return int(parts[2])
        except (IndexError, ValueError):
            return 0

    sessions = sorted(
        [p for p in model_dir.glob("training_session_*") if p.is_dir()],
        key=get_session_num
    )
    if sessions:
        model_dir = sessions[-1]
        ui.info(f"Resolved to latest training run session directory: {model_dir}")
    else:
        ui.info(f"Target model directory: {model_dir}")

    # 1. Load Model Metadata and Config
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        metadata_path = model_dir / "metadata.json"
        
    if not metadata_path.exists():
        ui.error(f"Model metadata file (model_metadata.json or metadata.json) not found in {model_dir}")
        sys.exit(1)
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    classes = metadata["classes"]
    config = load_pipeline_config(metadata)
    
    ui.success("Model metadata and pipeline configuration loaded successfully.")

    # 2. Build model and load weights
    ui.info("Building model and loading saved weights...")
    wrist_shape = (config.window_size, len(metadata["wrist_channels"]))
    finger_shape = (config.window_size, len(metadata["finger_channels"]))

    train_params = metadata.get("training_parameters", {})
    conv_filters = train_params.get("conv_filters", [32, 64])
    dense_units = train_params.get("dense_units", 64)

    model = build_multi_branch_cnn(
        input_shape_wrist=wrist_shape,
        input_shape_finger=finger_shape,
        num_classes=len(classes),
        input_shape_feat=None,
        conv_filters=conv_filters,
        dense_units=dense_units
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

    # 3b. Build the gesture -> PowerPoint control dispatcher
    dispatcher, control_live = build_dispatcher(args)
    if dispatcher is None:
        ui.info("PowerPoint control disabled (--no-control); predictions are only displayed.")
    elif control_live:
        ui.success("PowerPoint control active (LIVE - real key presses will be sent).")
    else:
        ui.warning("PowerPoint control active (DRY-RUN - shortcuts are only logged, no keys sent).")

    # 3c. Build the objection-based live-performance evaluator (optional)
    detector = None
    evaluator = None
    eval_out_dir = None
    if args.evaluate:
        detector = TriggerDetector(
            confidence_threshold=args.threshold,
            cooldown_s=args.cooldown,
            require_release=True,
        )
        evaluator = LivePerformanceEvaluator(
            classes,
            objection_window_s=args.objection_window,
            enable_fn=True,
            session_meta={
                "model_dir": str(model_dir),
                "threshold": args.threshold,
                "cooldown_s": args.cooldown,
            },
        )
        eval_out_dir = Path(args.eval_out) if args.eval_out else model_dir
        ui.success("Live-performance evaluation enabled (objection-based).")

    # 4. Resolve Device Ports & Initialize
    if args.simulate:
        class MockIMU:
            def __init__(self, name):
                self.name = name
                self.running = False
                self._thread = None
                self._data = []
                self.lock = threading.Lock()
                self.is_mock = True
                
            def start(self):
                self.running = True
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
                
            def stop(self):
                self.running = False
                if self._thread:
                    self._thread.join()
                
            def get_data(self):
                with self.lock:
                    data = list(self._data)
                    self._data.clear()
                    return data
                    
            def _run(self):
                last_t = 0
                while self.running:
                    time.sleep(0.01) # 100 Hz
                    t_now_us = int(time.time() * 100) * 10000
                    if t_now_us <= last_t:
                        t_now_us = last_t + 10000
                    last_t = t_now_us
                    
                    packet = {
                        'pc_timestamp_us': t_now_us,
                        'esp_timestamp_us': t_now_us,
                        'imu_timestamp_ms': int(t_now_us / 1000),
                        'accX': np.random.uniform(-0.02, 0.02),
                        'accY': np.random.uniform(-0.02, 0.02),
                        'accZ': np.random.uniform(0.98, 1.02),
                        'gyrX': np.random.uniform(-0.01, 0.01),
                        'gyrY': np.random.uniform(-0.01, 0.01),
                        'gyrZ': np.random.uniform(-0.01, 0.01),
                    }
                    with self.lock:
                        self._data.append(packet)

        ui.info("Initializing simulated IMU sensor data streams...")
        imu1 = MockIMU("IMU1")
        imu2 = MockIMU("IMU2")
        imu1.start()
        imu2.start()
    else:
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

    # 5. Define transform callback for Asynchronous Frame Grabber
    def transform_fn(channels, channel_names):
        wrist_idx = [channel_names.index(ch) for ch in metadata["wrist_channels"]]
        finger_idx = [channel_names.index(ch) for ch in metadata["finger_channels"]]

        X_wrist = channels[:, wrist_idx][np.newaxis, :, :]
        X_finger = channels[:, finger_idx][np.newaxis, :, :]

        X_wrist_scaled = scaler_wrist.transform(X_wrist)
        X_finger_scaled = scaler_finger.transform(X_finger)
        
        return X_wrist_scaled, X_finger_scaled

    grabber = None
    try:
        # Run Initial Calibration
        cal_df = record_calibration(imu1, imu2, config.calibration)
        profile = estimate_calibration(cal_df, config.calibration)
        ui.success("Static calibration completed and estimated.")
        
        # Initialize and start Data Grabber
        grabber = AsynchronousDataGrabber(
            imu1=imu1,
            imu2=imu2,
            pipeline_config=config,
            calibration_profile=profile,
            window_size_samples=config.window_size,
            advance_samples=10,
            freq_hz=100.0,
            max_diff_us=10000,
            transform_fn=transform_fn,
            poll_interval_s=0.01
        )
        
        # Silence frequent background sync/packet logs during the streaming loop
        from data_fusion_project.core.logger_setup import set_log_level
        set_log_level("WARNING")
        
        ui.hr(title="Live Inference Active (Asynchronous)")
        if evaluator is not None:
            ui.info("Evaluating live performance. Object to wrong fires via hotkeys; press [q] or Ctrl+C to finish.\n")
            ui.box(evaluator.hotkey_legend(), title="Evaluation Hotkeys")
        elif dispatcher is not None:
            ui.info("Perform gestures to control PowerPoint in real time. Press Ctrl+C to exit.\n")
        else:
            ui.info("Perform gestures to see real-time classifications. Press Ctrl+C to exit.\n")

        grabber.start()

        # Enter non-canonical keyboard mode only while evaluating, so objection keys are read live.
        input_ctx = ui.non_blocking_input() if evaluator is not None else contextlib.nullcontext()
        with input_ctx:
            start_time_loop = time.time()
            while True:
                if args.timeout is not None and (time.time() - start_time_loop) > args.timeout:
                    ui.info(f"\nSimulated run timeout of {args.timeout}s reached. Exiting cleanly.")
                    break

                # Drain objection keystrokes and commit expired fires every iteration (also while idle).
                if evaluator is not None:
                    key = ui.get_key()
                    while key:
                        evaluator.poll(key)
                        key = ui.get_key()
                    evaluator.tick()
                    if evaluator.quit_requested:
                        ui.info("\nEvaluation finished by user ([q]).")
                        break

                # Block up to 100ms for a new preprocessed frame
                frame = grabber.get_newest_frame(block=True, timeout=0.1)
                if frame is None:
                    continue

                X_wrist_scaled, X_finger_scaled = frame

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

                # Feed the prediction to the live-performance evaluator (de-bounced fire events).
                if evaluator is not None:
                    event = detector.feed(class_name, prob)
                    if event is not None:
                        evaluator.on_fire(*event)

                # Feed the prediction to the control dispatcher (de-bounced).
                if dispatcher is not None:
                    fired_action = dispatcher.feed(class_name, prob)
                    if fired_action is not None:
                        sys.stdout.write(f"\n\033[96m\033[1m  --> Action triggered: {fired_action}\033[0m\n")
                        sys.stdout.flush()

    except KeyboardInterrupt:
        ui.info("\nExiting real-time inference loop...")
    except Exception as e:
        ui.error(f"\nError occurred during live inference: {e}")
    finally:
        if grabber:
            grabber.stop()
        imu1.stop()
        imu2.stop()
        ui.success("Sensors disconnected.")
        if evaluator is not None:
            try:
                evaluator.finalize(eval_out_dir)
            except Exception as e:
                ui.error(f"Failed to finalize live evaluation: {e}")


if __name__ == "__main__":
    main()
