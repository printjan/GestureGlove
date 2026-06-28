# scripts/record_data.py
"""
Data recording script to stream, synchronize, and save IMU datasets for gestures.

Input:
data/
└── <gesture_name>/
    └── <session_name>/
        └── #####.csv

Output:
data/
└── <gesture_name>/
    └── <session_name>/
        ├── calibration.csv
        ├── calibration.png
        ├── #####.csv
        ├── #####.png
        ├── energy_distribution.csv
        └── energy_distribution.png

**Columns (#####.csv / calibration.csv):**
| Column name | Description |
|-------------|-------------|
| IMU1_accX   | Accelerometer X-axis reading from IMU1 (Wrist) in g |
| IMU1_accY   | Accelerometer Y-axis reading from IMU1 (Wrist) in g |
| IMU1_accZ   | Accelerometer Z-axis reading from IMU1 (Wrist) in g |
| IMU1_gyrX   | Gyroscope X-axis reading from IMU1 (Wrist) in dps |
| IMU1_gyrY   | Gyroscope Y-axis reading from IMU1 (Wrist) in dps |
| IMU1_gyrZ   | Gyroscope Z-axis reading from IMU1 (Wrist) in dps |
| IMU2_accX   | Accelerometer X-axis reading from IMU2 (Finger) in g |
| IMU2_accY   | Accelerometer Y-axis reading from IMU2 (Finger) in g |
| IMU2_accZ   | Accelerometer Z-axis reading from IMU2 (Finger) in g |
| IMU2_gyrX   | Gyroscope X-axis reading from IMU2 (Finger) in dps |
| IMU2_gyrY   | Gyroscope Y-axis reading from IMU2 (Finger) in dps |
| IMU2_gyrZ   | Gyroscope Z-axis reading from IMU2 (Finger) in dps |

**Columns (energy_distribution.csv):**
| Column name | Description |
|-------------|-------------|
| sample_index| Sample index inside the 1.5s window (0 to 149) |
| IMU1_acc_mean| Mean accelerometer magnitude of IMU1 across all recorded samples |
| IMU1_acc_std | Standard deviation of accelerometer magnitude of IMU1 |
| IMU1_gyr_mean| Mean gyroscope magnitude of IMU1 across all recorded samples |
| IMU1_gyr_std | Standard deviation of gyroscope magnitude of IMU1 |
| IMU2_acc_mean| Mean accelerometer magnitude of IMU2 across all recorded samples |
| IMU2_acc_std | Standard deviation of accelerometer magnitude of IMU2 |
| IMU2_gyr_mean| Mean gyroscope magnitude of IMU2 across all recorded samples |
| IMU2_gyr_std | Standard deviation of gyroscope magnitude of IMU2 |
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================
import time
import os
import random
import threading
import queue
import platform
import json
from pathlib import Path
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI/threading crashes
import matplotlib.pyplot as plt
from data_fusion_project.recording.input_data import IMUDataInput
from data_fusion_project.recording.sync import process_stream
from data_fusion_project.recording import device_resolution
from data_fusion_project.core.paths import DATA_DIR, GESTURES, get_calibration_file, get_next_recording_file, get_session_metadata_file
from data_fusion_project.core.logger_setup import get_logger, set_log_level
from data_fusion_project.core.cli_ui import ui, Style

logger = get_logger("IMU_Record")
set_log_level("WARNING")



# ======================================================================================================================
# Configuration & State
# ======================================================================================================================
BAUDRATE = 115200
RECORD_DURATION_S = 1.5
TARGET_SAMPLES = 150  # = RECORD_DURATION_S * 100 Hz
PLOT_EVERY_SAMPLE = False
PLOT_MOVEMENT_DISTRIBUTION = True
PLOT_CALIBRATION_RECORDING = True
MAX_DEVIATION_OF_TARGET_SAMPLE_RATE = 30  # Max permitted sample count deviation percentage (30%)
MAX_SAMPLES_PER_SESSION = 25  # Number of samples recorded before automatically starting a new session

PRE_BUFFER_S = 0.15
POST_BUFFER_S = 0.15

# Geste, die kontinuierlich (überlappend) statt sample-weise aufgenommen wird.
NONE_GESTURE_NAME = "none"
# Überlappung aufeinanderfolgender 'none'-Fenster (0 = keine, 0.5 = 50 %).
OVERLAP_RATIO = 0.5

# Pause zwischen einzelnen Samples bei allen Gesten außer 'none'.
PAUSE_DURATION_S = 1

# Maximal erlaubter Zeitversatz zwischen IMU1 und IMU2 innerhalb eines Fensters.
MAX_SYNC_DIFF_US = 10000

GESTURE_LABELS = {i: name for i, name in enumerate(GESTURES)}

# Thread-Steuerung über Events
running = threading.Event()
running.set()
recording = threading.Event()
stop_session = threading.Event()

# Puffer + Lock für thread-sicheren Zugriff
_buffer_lock = threading.Lock()
imu1_data_buffer = []
imu2_data_buffer = []
received_counts = {'IMU1': 0, 'IMU2': 0}
current_gesture = 0


session_metadata = {}


def load_or_init_metadata(gesture_name: str, session_name: str):
    """
    Loads existing session metadata or initializes a new one.
    """
    global session_metadata
    meta_file = get_session_metadata_file(gesture_name, session_name)
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                session_metadata = json.load(f)
        except Exception as e:
            logger.error("Failed to load session metadata: %s. Re-initializing.", e)
            init_new_metadata()
    else:
        init_new_metadata()


def init_new_metadata():
    """
    Initializes a clean session metadata dictionary.
    """
    global session_metadata
    session_metadata = {
        "baudrate": BAUDRATE,
        "record_duration_s": RECORD_DURATION_S,
        "target_samples": TARGET_SAMPLES,
        "max_samples_per_session": MAX_SAMPLES_PER_SESSION,
        "pre_buffer_s": PRE_BUFFER_S,
        "post_buffer_s": POST_BUFFER_S,
        "recalibrations": [],
        "energy_distributions": []
    }


def save_metadata(gesture_name: str, session_name: str):
    """
    Saves current session metadata to recording_session.json.
    """
    meta_file = get_session_metadata_file(gesture_name, session_name)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(session_metadata, f, indent=2, ensure_ascii=False)


def get_next_block_session_name(gesture_name: str, base_session_name: str) -> str:
    """
    Returns a new, unused session name for the next sample block of a gesture.

    Block sessions are derived from the base session name by appending a part suffix
    ('<base>_p2', '<base>_p3', ...). Directories already present on disk are skipped so
    reruns never overwrite previously recorded blocks.
    """
    from data_fusion_project.core.paths import get_gesture_dir
    gesture_dir = get_gesture_dir(gesture_name)
    part = 2
    while (gesture_dir / f"{base_session_name}_p{part}").exists():
        part += 1
    return f"{base_session_name}_p{part}"


def start_new_session(imu1, imu2, gesture_name: str, base_session_name: str) -> str:
    """
    Begins a fresh recording session for the gesture after a full sample block.

    Creates a new session directory, (re)initializes the module session metadata, and
    records the mandatory initial static calibration at sample_index 0. Replaces the old
    in-session re-calibration step. Returns the new session name.
    """
    new_session = get_next_block_session_name(gesture_name, base_session_name)
    ui.hr(title=f"New session: {new_session}")
    ui.info(f"Beginning a new session for gesture '{gesture_name}'.")

    # Fresh metadata for the new session.
    init_new_metadata()

    # Every session must start with a static calibration at sample_index 0.
    cal_file = get_calibration_file(gesture_name, new_session, index=0)
    record_calibration_with_redo(imu1, imu2, cal_file)
    session_metadata["recalibrations"].append({
        "file": cal_file.name,
        "sample_index": 0,
    })
    save_metadata(gesture_name, new_session)
    ui.success("Calibration successfully saved!")
    return new_session


def get_next_energy_distribution_filepath(gesture_name: str, session_name: str) -> tuple[Path, int]:
    """
    Returns the path and index of the next sequential energy distribution file.
    """
    from data_fusion_project.core.paths import get_session_dir
    idx = len(session_metadata.get("energy_distributions", []))
    session_dir = get_session_dir(gesture_name, session_name)
    dist_file = session_dir / f"energy_distribution_{idx}.csv"
    return dist_file, idx



# ======================================================================================================================
# Recording Helpers
# ======================================================================================================================
def run_single_recording(imu1, imu2, duration_s, target_samples, filename):
    """
    Records a single data sample for a given duration.
    :param: imu1 (IMUDataInput): first IMU reader.
    :param: imu2 (IMUDataInput): second IMU reader.
    :param: duration_s (float): duration of recording in seconds.
    :param: target_samples (int): expected number of samples.
    :param: filename (str | Path): destination path for CSV file.
    :return: success (bool): True if sample recorded successfully.
    :raises: RuntimeError: if sensor threads stopped, data missing, or sync failed.
    """
    if not imu1.running or not imu2.running:
        logger.error("Sensor reading thread stopped prior to single recording.")
        raise RuntimeError("Sensor reading thread stopped prior to single recording.")

    # Clear reader queues before starting
    imu1.get_data()
    imu2.get_data()

    recording.set()

    # Pre-buffer recording silently
    time.sleep(PRE_BUFFER_S)

    completed_status = ui.progress_bar(duration_s, label="Recording: ", color=Style.SUCCESS, stop_session=stop_session)

    # Post-buffer recording silently
    time.sleep(POST_BUFFER_S)

    recording.clear()

    if completed_status != "completed":
        logger.warning("Recording was manually cancelled.")
        return False

    if not imu1.running or not imu2.running:
        logger.error("Sensor reading thread stopped during single recording.")
        raise RuntimeError("Sensor reading thread stopped during single recording.")

    snapshot1 = imu1.get_data()
    snapshot2 = imu2.get_data()

    if not snapshot1 or not snapshot2:
        logger.error("No sensor data received from one or both IMUs.")
        raise RuntimeError("No sensor data received.")

    actual_duration = duration_s + PRE_BUFFER_S + POST_BUFFER_S
    recorded_target_samples = int(actual_duration * 100)
    allowed_deviation = int(recorded_target_samples * (MAX_DEVIATION_OF_TARGET_SAMPLE_RATE / 100.0))
    min_samples = recorded_target_samples - allowed_deviation
    max_samples = recorded_target_samples + allowed_deviation

    if len(snapshot1) < min_samples or len(snapshot1) > max_samples:
        logger.error(f"IMU1 sample count {len(snapshot1)} deviated from target {recorded_target_samples} (allowed: {min_samples}-{max_samples}).")
        raise RuntimeError(f"IMU1 sample count deviation too high: {len(snapshot1)}.")

    if len(snapshot2) < min_samples or len(snapshot2) > max_samples:
        logger.error(f"IMU2 sample count {len(snapshot2)} deviated from target {recorded_target_samples} (allowed: {min_samples}-{max_samples}).")
        raise RuntimeError(f"IMU2 sample count deviation too high: {len(snapshot2)}.")

    # Convert to DataFrames
    df1 = pd.DataFrame(snapshot1)
    df2 = pd.DataFrame(snapshot2)

    with ui.spinner("Processing and synchronizing sensor data..."):
        _merged, valid_windows = process_stream(df1, df2, window_sz=target_samples, max_diff_us=MAX_SYNC_DIFF_US, freq_hz=100, center_gesture=True)

    if not valid_windows:
        logger.error("Synchronization failed (deviation too high).")
        raise RuntimeError("Synchronization failed.")

    # Update global counts for metrics
    global received_counts
    received_counts['IMU1'] += len(snapshot1)
    received_counts['IMU2'] += len(snapshot2)

    # Save the full merged resampled overlap
    save_df = _merged.copy()
    save_df = save_df.drop(columns=['sync_time_us'], errors='ignore')

    if len(save_df) < target_samples:
        logger.error(f"Sample contains too few rows: {len(save_df)} (expected at least {target_samples}).")
        raise RuntimeError(f"Invalid row count: {len(save_df)}.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(filename, index=False)
    ui.success(f"File saved successfully: {filename.name}")

    is_calibration = "calibration" in filename.name
    start_idx = None
    if not is_calibration:
        # Find start index in _merged
        start_time_us = valid_windows[0]['sync_time_us'].iloc[0]
        start_idx = int(np.where(_merged['sync_time_us'].values == start_time_us)[0][0])
        # Write start index to companion .txt file
        txt_filename = filename.with_suffix('.txt')
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(str(start_idx))

    # Save plot of full overlap with vertical bounds lines if start_idx is available
    should_plot = PLOT_CALIBRATION_RECORDING if is_calibration else PLOT_EVERY_SAMPLE
    if should_plot:
        plot_data(save_df, save_path=filename.with_suffix('.png'), start_idx=start_idx)
    return True


def record_calibration_with_redo(imu1, imu2, cal_file: Path) -> bool:
    """
    Handles recording a 5-second calibration with the option to redo/discard.
    Returns True when calibration is successfully recorded and accepted.
    """
    while True:
        ui.wait_for_enter("Ready? Press [Enter] to start 5s static calibration...")
        ui.info("Please hold the sensors absolutely still for 5 seconds...")
        
        success = run_single_recording(imu1, imu2, duration_s=5.0, target_samples=500, filename=cal_file)
        if not success:
            ui.error("Calibration failed. Please try again.")
            continue
            
        ui.info("\nCalibration recording complete.")
        ui.info("Press [Enter] to accept/keep this calibration, or [d] to delete and redo...")
        
        ui.flush_input()
        accepted = None
        while accepted is None:
            key = ui.get_key()
            if key in ("\r", "\n"):
                accepted = True
                ui.success("Calibration accepted!")
            elif key == "d":
                accepted = False
                ui.warning("Calibration discarded. Deleting files and preparing to re-record...")
                try:
                    if cal_file.exists():
                        cal_file.unlink()
                    png_file = cal_file.with_suffix('.png')
                    if png_file.exists():
                        png_file.unlink()
                except Exception as e:
                    ui.error(f"Failed to delete calibration files: {e}")
            time.sleep(0.05)
            
        if accepted:
            return True



# ======================================================================================================================
# UI & State Helpers
# ======================================================================================================================
def handle_pause(last_csv, last_png, paused_during_recording):
    """
    Handles keyboard choices during the pause state.
    :param: last_csv (Path | None): path of last saved csv.
    :param: last_png (Path | None): path of last saved png.
    :param: paused_during_recording (bool): flag if paused mid-run.
    :return: action (str): selected action ('resume', 'stop', or 'deleted').
    """
    ui.info("\n=== RECORDING PAUSED ===")
    if paused_during_recording:
        ui.info("Current (incomplete) sample was discarded.")
        
    if last_csv and last_csv.exists():
        ui.info(f"Last saved sample: {last_csv.name}")
        ui.info("Press [d] to delete this sample.")
    else:
        ui.info("No saved sample available to delete.")
        
    ui.info("Press [Space] to resume or [Enter] to stop.")
    
    ui.flush_input()
    
    while True:
        key = ui.get_key()
        if key == " ":
            ui.success("Resuming recording...")
            return "resume"
        elif key in ("\r", "\n"):
            return "stop"
        elif key == "d":
            if last_csv and last_csv.exists():
                try:
                    last_csv.unlink()
                    if last_png and last_png.exists():
                        last_png.unlink()
                    txt_file = last_csv.with_suffix('.txt')
                    if txt_file.exists():
                        txt_file.unlink()
                    ui.success(f"Deleted: {last_csv.name} and associated files.")
                    return "deleted"
                except Exception as e:
                    ui.error(f"Error deleting file: {e}")
            else:
                ui.warning("No sample available to delete.")
        time.sleep(0.05)


def handle_stop(last_csv, last_png):
    """
    Offers the user a 3-second countdown to delete the last saved sample.
    :param: last_csv (Path | None): path of last saved csv.
    :param: last_png (Path | None): path of last saved png.
    :return: None:
    """
    ui.info("\n=== RECORDING STOPPED ===")
    if last_csv and last_csv.exists():
        ui.info(f"Last saved sample: {last_csv.name}")
        ui.warning("Do you want to delete this last sample?")
        ui.info("Press [d] within 3 seconds to delete, or any other key to skip...")
        
        ui.flush_input()
        start_time = time.time()
        deleted = False
        while time.time() - start_time < 3.0:
            remaining = 3.0 - (time.time() - start_time)
            print(f"\r  Time remaining: {remaining:.1f}s ... ", end="", flush=True)
            
            key = ui.get_key()
            if key == "d":
                try:
                    last_csv.unlink()
                    if last_png and last_png.exists():
                        last_png.unlink()
                    txt_file = last_csv.with_suffix('.txt')
                    if txt_file.exists():
                        txt_file.unlink()
                    print()
                    ui.success(f"Deleted: {last_csv.name} and associated files.")
                    deleted = True
                    break
                except Exception as e:
                    print()
                    ui.error(f"Error deleting: {e}")
                    break
            elif key is not None:
                print()
                ui.info("Deletion skipped.")
                break
            time.sleep(0.05)
        if not deleted:
            print()
    else:
        ui.info("No saved sample available to delete.")


def input_thread(imu1, imu2, session_name):
    """
    Controls user interaction menu thread for choosing gestures and starting loops.
    :param: imu1 (IMUDataInput): first IMU reader.
    :param: imu2 (IMUDataInput): second IMU reader.
    :param: session_name (str): current session directory name.
    :return: None:
    """
    global current_gesture

    ui.banner("Recording Controller", subtitle=f"Session: {session_name}")
    _print_dataset_counts()

    choices = list(GESTURES) + ["[Exit]"]

    while running.is_set():
        selection = ui.ask_choice("\nSelect gesture:", choices)

        if selection is None or selection == len(choices) - 1:
            logger.info("Exiting recording loop...")
            running.clear()
            break

        current_gesture = selection
        gesture_name = GESTURES[current_gesture]
        ui.hr(title=f"Selection: {gesture_name}")

        # Load session metadata
        load_or_init_metadata(gesture_name, session_name)

        # Check Calibration
        cal_file = get_calibration_file(gesture_name, session_name, index=0)
        if not cal_file.exists():
            ui.warning(f"No calibration found for gesture '{gesture_name}' in session '{session_name}'.")
            with ui.non_blocking_input():
                record_calibration_with_redo(imu1, imu2, cal_file)
            
            # Record in metadata
            session_metadata["recalibrations"].append({
                "file": cal_file.name,
                "sample_index": 0
            })
            save_metadata(gesture_name, session_name)
            ui.success("Calibration successfully saved!")
        else:
            # Ensure it is registered in metadata if it is on disk
            if not any(r["file"] == cal_file.name for r in session_metadata.get("recalibrations", [])):
                session_metadata.setdefault("recalibrations", []).append({
                    "file": cal_file.name,
                    "sample_index": 0
                })
                save_metadata(gesture_name, session_name)

        # Determine recording mode
        if gesture_name == NONE_GESTURE_NAME:
            ui.wait_for_enter("Ready? Press [Enter] to start continuous recording...")
            stop_session.clear()
            record_continuous(imu1, imu2, session_name)
        else:
            ui.wait_for_enter("Ready? Press [Enter] to start sample recording loop...")
            stop_session.clear()
            record_samples_loop(imu1, imu2, session_name)
            
        recording.clear()
        ui.success("Recording session finished.")


def _trim_before(buf, cutoff_us):
    """
    Trims packets from the buffer that are older than the cutoff timestamp.
    :param: buf (list): packet list.
    :param: cutoff_us (int): cutoff timestamp in microseconds.
    :return: None:
    """
    i = 0
    while i < len(buf) and buf[i]['pc_timestamp_us'] < cutoff_us:
        i += 1
    if i:
        del buf[:i]


# ======================================================================================================================
# Recording Loops
# ======================================================================================================================
def record_continuous(imu1, imu2, session_name):
    """
    Records the 'none' gesture continuously with overlapping windows.
    :param: imu1 (IMUDataInput): first IMU reader.
    :param: imu2 (IMUDataInput): second IMU reader.
    :param: session_name (str): current session directory name.
    :return: None:
    :raises: RuntimeError: if sensor reading fails during the session.
    """
    gesture_name = GESTURES[current_gesture]
    current_session = session_name
    window_us = int(RECORD_DURATION_S * 1e6)
    advance_us = max(1, int(round(RECORD_DURATION_S * (1 - OVERLAP_RATIO) * 1e6)))
    poll_s = 0.05

    # Clear reader queues and start with fresh local buffers
    imu1.get_data()
    imu2.get_data()
    local_buf1 = []
    local_buf2 = []
    local_cnt1 = 0
    local_cnt2 = 0

    logger.info("CONTINUOUS RECORDING '%s' (window %ss, overlap %s%%)",
                gesture_name, RECORD_DURATION_S, OVERLAP_RATIO*100)
    ui.info("Press [Space] or [Enter] to stop...")

    saved = 0
    next_start_us = None
    
    with ui.non_blocking_input():
        while not stop_session.is_set():
            if not imu1.running or not imu2.running:
                logger.error("IMU thread stopped during continuous recording. Aborting.")
                raise RuntimeError("IMU thread stopped during continuous recording.")

            key = ui.get_key()
            if key in (" ", "\r", "\n"):
                break
                
            time.sleep(poll_s)

            # Retrieve new samples and append to local buffers
            data1 = imu1.get_data()
            data2 = imu2.get_data()
            local_buf1.extend(data1)
            local_buf2.extend(data2)
            local_cnt1 += len(data1)
            local_cnt2 += len(data2)

            if not local_buf1 or not local_buf2:
                continue

            if next_start_us is None:
                next_start_us = max(local_buf1[0]['pc_timestamp_us'], local_buf2[0]['pc_timestamp_us'])

            latest_us = min(local_buf1[-1]['pc_timestamp_us'], local_buf2[-1]['pc_timestamp_us'])
            if latest_us - next_start_us < window_us:
                continue

            window = _extract_window(local_buf1, local_buf2)
            if window is not None:
                rec_file = get_next_recording_file(gesture_name, current_session)
                _save_window(window, rec_file, verbose=False, plot=False)
                saved += 1
                print(f"\r  Overlapping 'none' windows saved: {saved}", end="", flush=True)

                if saved > 0 and saved % MAX_SAMPLES_PER_SESSION == 0:
                    print()  # Clear the carriage return line
                    save_and_plot_energy_distribution(gesture_name, current_session, sample_index=saved)
                    ui.warning(f"\n{saved} samples recorded. Starting a new session!")
                    current_session = start_new_session(imu1, imu2, gesture_name, session_name)
                    saved = 0

                    # Purge buffers and queues to prevent using calibration data as 'none' samples
                    imu1.get_data()
                    imu2.get_data()
                    local_buf1.clear()
                    local_buf2.clear()
                    next_start_us = None
                    continue

            next_start_us += advance_us
            _trim_before(local_buf1, next_start_us)
            _trim_before(local_buf2, next_start_us)

    print()
    global received_counts
    received_counts['IMU1'] += local_cnt1
    received_counts['IMU2'] += local_cnt2
    logger.info("Saved: %d overlapping windows for '%s'", saved, gesture_name)
    _print_received_counts()
    _print_dataset_counts()


def record_samples_loop(imu1, imu2, session_name):
    """
    Loop for recording individual gesture samples with pause intervals.
    :param: imu1 (IMUDataInput): first IMU reader.
    :param: imu2 (IMUDataInput): second IMU reader.
    :param: session_name (str): current session directory name.
    :return: None:
    :raises: RuntimeError: if sensor error or sample rate deviation is too high.
    """
    gesture_name = GESTURES[current_gesture]
    current_session = session_name
    saved = 0
    sample_idx = 0

    last_saved_csv = None
    last_saved_png = None
    
    with ui.non_blocking_input():
        # First progress bar after starting loop (or after calibration) should be a Pause to give the user time to set up
        pause_status = ui.progress_bar(PAUSE_DURATION_S, label="Pause:     ", color=Style.ERROR, stop_session=stop_session)
        if pause_status == "aborted":
            return
        elif pause_status == "space":
            action = handle_pause(None, None, paused_during_recording=False)
            while action == "deleted":
                action = handle_pause(None, None, paused_during_recording=False)
            if action == "stop":
                return
        elif pause_status == "enter":
            handle_stop(None, None)
            return

        while not stop_session.is_set():
            if not imu1.running or not imu2.running:
                logger.error("IMU thread stopped during sample loop. Aborting.")
                raise RuntimeError("IMU thread stopped during sample loop.")

            sample_idx += 1
            rec_file = get_next_recording_file(gesture_name, current_session)

            logger.info("Sample %d ('%s') — Starting recording...", sample_idx, gesture_name)
            
            # Clear reader queues before starting
            imu1.get_data()
            imu2.get_data()

            recording.set()

            # Pre-buffer recording silently
            time.sleep(PRE_BUFFER_S)

            rec_status = ui.progress_bar(RECORD_DURATION_S, label="Recording: ", color=Style.SUCCESS, stop_session=stop_session)

            # Post-buffer recording silently
            time.sleep(POST_BUFFER_S)

            recording.clear()

            if rec_status == "aborted":
                break
            elif rec_status == "space":
                action = handle_pause(last_saved_csv, last_saved_png, paused_during_recording=True)
                if action == "deleted":
                    last_saved_csv = None
                    last_saved_png = None
                    saved = max(0, saved - 1)
                    while action == "deleted":
                        action = handle_pause(None, None, paused_during_recording=False)
                
                if action == "stop":
                    handle_stop(last_saved_csv, last_saved_png)
                    break
                
                sample_idx -= 1
                continue
            elif rec_status == "enter":
                handle_stop(last_saved_csv, last_saved_png)
                break

            if not imu1.running or not imu2.running:
                logger.error("IMU thread stopped during sample loop. Aborting.")
                raise RuntimeError("IMU thread stopped during sample loop.")

            snapshot1 = imu1.get_data()
            snapshot2 = imu2.get_data()

            if not snapshot1 or not snapshot2:
                logger.error("No sensor data received from IMUs in sample loop. Aborting pipeline.")
                raise RuntimeError("No sensor data received from IMUs in sample loop.")

            actual_duration = RECORD_DURATION_S + PRE_BUFFER_S + POST_BUFFER_S
            recorded_target_samples = int(actual_duration * 100)
            allowed_deviation = int(recorded_target_samples * (MAX_DEVIATION_OF_TARGET_SAMPLE_RATE / 100.0))
            min_samples = recorded_target_samples - allowed_deviation
            max_samples = recorded_target_samples + allowed_deviation

            if len(snapshot1) < min_samples or len(snapshot1) > max_samples:
                logger.error(f"IMU1 sample count {len(snapshot1)} deviated from target {recorded_target_samples} (allowed: {min_samples}-{max_samples}).")
                raise RuntimeError(f"IMU1 sample count deviation too high in loop: {len(snapshot1)}.")

            if len(snapshot2) < min_samples or len(snapshot2) > max_samples:
                logger.error(f"IMU2 sample count {len(snapshot2)} deviated from target {recorded_target_samples} (allowed: {min_samples}-{max_samples}).")
                raise RuntimeError(f"IMU2 sample count deviation too high in loop: {len(snapshot2)}.")

            df1 = pd.DataFrame(snapshot1)
            df2 = pd.DataFrame(snapshot2)

            with ui.spinner("Processing and synchronizing sensor data..."):
                _merged, valid_windows = process_stream(df1, df2, window_sz=TARGET_SAMPLES, max_diff_us=MAX_SYNC_DIFF_US, freq_hz=100, center_gesture=True)

            if not valid_windows:
                logger.error("Synchronization failed for sample (deviation too high). Aborting pipeline.")
                raise RuntimeError("Synchronization failed for sample.")

            # Update global counts for metrics
            global received_counts
            received_counts['IMU1'] += len(snapshot1)
            received_counts['IMU2'] += len(snapshot2)

            # Save the full merged resampled overlap
            save_df = _merged.copy()
            save_df = save_df.drop(columns=['sync_time_us'], errors='ignore')

            if len(save_df) < TARGET_SAMPLES:
                logger.error(f"Sample contains too few rows: {len(save_df)} (expected at least {TARGET_SAMPLES}).")
                raise RuntimeError(f"Invalid row count: {len(save_df)}.")

            rec_file = Path(rec_file)
            rec_file.parent.mkdir(parents=True, exist_ok=True)
            save_df.to_csv(rec_file, index=False)
            ui.success(f"File saved successfully: {rec_file.name}")

            # Find start index in _merged
            start_time_us = valid_windows[0]['sync_time_us'].iloc[0]
            start_idx = int(np.where(_merged['sync_time_us'].values == start_time_us)[0][0])
            # Write start index to companion .txt file
            txt_filename = rec_file.with_suffix('.txt')
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(str(start_idx))

            last_saved_csv = rec_file
            last_saved_png = rec_file.with_suffix('.png') if PLOT_EVERY_SAMPLE else None
            
            if PLOT_EVERY_SAMPLE:
                plot_data(save_df, save_path=last_saved_png, start_idx=start_idx)
                
            saved += 1

            # After a full block of MAX_SAMPLES_PER_SESSION samples, start a new session.
            if saved > 0 and saved % MAX_SAMPLES_PER_SESSION == 0:
                save_and_plot_energy_distribution(gesture_name, current_session, sample_index=saved)
                ui.warning(f"\n{saved} samples recorded. Starting a new session!")
                current_session = start_new_session(imu1, imu2, gesture_name, session_name)
                saved = 0
                sample_idx = 0
                last_saved_csv = None
                last_saved_png = None

            # Pause Phase
            pause_status = ui.progress_bar(PAUSE_DURATION_S, label="Pause:     ", color=Style.ERROR, stop_session=stop_session)
            
            if pause_status == "aborted":
                break
            elif pause_status == "space":
                action = handle_pause(last_saved_csv, last_saved_png, paused_during_recording=False)
                if action == "deleted":
                    last_saved_csv = None
                    last_saved_png = None
                    saved = max(0, saved - 1)
                    while action == "deleted":
                        action = handle_pause(None, None, paused_during_recording=False)
                        
                if action == "stop":
                    handle_stop(last_saved_csv, last_saved_png)
                    break
                continue
            elif pause_status == "enter":
                handle_stop(last_saved_csv, last_saved_png)
                break

    # Calculate and plot motion energy distribution on loop finish
    logger.info("Saved: %d samples for '%s'", saved, gesture_name)
    save_and_plot_energy_distribution(gesture_name, current_session, sample_index=saved)
    _print_received_counts()
    _print_dataset_counts()


def _extract_window(snapshot1, snapshot2):
    """
    Synchronizes and extracts the first valid window from continuous buffer.
    :param: snapshot1 (list): buffer data from IMU1.
    :param: snapshot2 (list): buffer data from IMU2.
    :return: window_df (DataFrame | None): first synchronized window or None.
    """
    if len(snapshot1) == 0 and len(snapshot2) == 0:
        return None

    df1 = pd.DataFrame(snapshot1)
    df2 = pd.DataFrame(snapshot2)

    _merged_df, valid_windows = process_stream(
        df1, df2, window_sz=TARGET_SAMPLES, max_diff_us=MAX_SYNC_DIFF_US, freq_hz=100)

    if not valid_windows:
        return None
    return valid_windows[0]


def _save_window(window_df, filename, verbose=True, plot=False):
    """
    Saves a synchronized data window as CSV and generates optional PNG plot.
    :param: window_df (DataFrame): data to save.
    :param: filename (str | Path): destination path.
    :param: verbose (bool): whether to display UI success message.
    :param: plot (bool): whether to plot the saved data.
    :return: filename (Path): path to saved CSV.
    """
    save_df = window_df.copy()
    save_df = save_df.drop(columns=['sync_time_us'], errors='ignore')

    if len(save_df) != TARGET_SAMPLES:
        logger.error(f"Sample contains invalid row count: {len(save_df)} (expected exactly {TARGET_SAMPLES}).")
        raise RuntimeError(f"Invalid row count: {len(save_df)}.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(filename, index=False)
    if verbose:
        ui.success(f"Dataset saved to: {filename.name}")
    if plot and PLOT_EVERY_SAMPLE:
        plot_data(save_df, save_path=filename.with_suffix('.png'))
    return filename


def _print_received_counts():
    """
    Logs the total packet count received from both sensors in the session.
    :return: None:
    """
    with _buffer_lock:
        n1 = received_counts['IMU1']
        n2 = received_counts['IMU2']
    logger.info("Data packets received in this session: IMU1 = %d, IMU2 = %d", n1, n2)


def _print_dataset_counts():
    """
    Prints to UI the number of saved samples per gesture.
    :return: None:
    """
    ui.hr(title="Saved datasets per gesture")
    total = 0
    for label in GESTURES:
        gesture_dir = DATA_DIR / label
        count = 0
        if gesture_dir.is_dir():
            for p in gesture_dir.glob("**/[0-9][0-9][0-9][0-9][0-9].csv"):
                count += 1
        total += count
        ui.kv([(label, str(count))])
    ui.hr()
    logger.info("Total number of valid datasets: %d", total)


# ======================================================================================================================
# Analysis & Plotting
# ======================================================================================================================
def save_and_plot_energy_distribution(gesture_name, session_name, sample_index=None):
    """
    Computes statistical motion energy, saves it as CSV, and plots centered & overall distributions.
    :param: gesture_name (str): name of gesture.
    :param: session_name (str): current session directory name.
    :param: sample_index (int | None): current sample index.
    :return: None:
    """
    if not PLOT_MOVEMENT_DISTRIBUTION:
        return

    from data_fusion_project.core.paths import get_session_dir
    session_dir = get_session_dir(gesture_name, session_name)
    
    # Find all sample CSVs (excluding calibration and energy_distribution)
    files = sorted([f for f in session_dir.glob("*.csv") if f.stem.isdigit() and len(f.stem) == 5])
    if not files:
        logger.warning("No samples found for energy distribution analysis in %s.", session_dir)
        return

    logger.info("Calculating motion energy distribution for '%s' (%d files)...", gesture_name, len(files))

    import numpy as np
    import pandas as pd

    # Center-sliced variables (exactly TARGET_SAMPLES = 150)
    imu1_acc_centered = []
    imu1_gyr_centered = []
    imu2_acc_centered = []
    imu2_gyr_centered = []

    # Overall raw variables (standardized to overall_length)
    imu1_acc_overall = []
    imu1_gyr_overall = []
    imu2_acc_overall = []
    imu2_gyr_overall = []

    start_indices = []
    overall_length = int((RECORD_DURATION_S + PRE_BUFFER_S + POST_BUFFER_S) * 100)

    for f in files:
        try:
            df = pd.read_csv(f)
            txt_path = f.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, "r", encoding="utf-8") as tf:
                    start_idx = int(tf.read().strip())
            else:
                start_idx = 0
            
            start_indices.append(start_idx)

            # Center-sliced data (exactly TARGET_SAMPLES = 150)
            df_centered = df.iloc[start_idx : start_idx + TARGET_SAMPLES]
            if len(df_centered) < TARGET_SAMPLES:
                pad_size = TARGET_SAMPLES - len(df_centered)
                last_row = df_centered.iloc[-1:] if not df_centered.empty else df.iloc[-1:]
                df_centered = pd.concat([df_centered, pd.concat([last_row] * pad_size, ignore_index=True)], ignore_index=True)
            
            imu1_acc_c = np.sqrt(df_centered['IMU1_accX']**2 + df_centered['IMU1_accY']**2 + df_centered['IMU1_accZ']**2)
            imu1_gyr_c = np.sqrt(df_centered['IMU1_gyrX']**2 + df_centered['IMU1_gyrY']**2 + df_centered['IMU1_gyrZ']**2)
            imu2_acc_c = np.sqrt(df_centered['IMU2_accX']**2 + df_centered['IMU2_accY']**2 + df_centered['IMU2_accZ']**2)
            imu2_gyr_c = np.sqrt(df_centered['IMU2_gyrX']**2 + df_centered['IMU2_gyrY']**2 + df_centered['IMU2_gyrZ']**2)

            imu1_acc_centered.append(imu1_acc_c)
            imu1_gyr_centered.append(imu1_gyr_c)
            imu2_acc_centered.append(imu2_acc_c)
            imu2_gyr_centered.append(imu2_gyr_c)

            # Overall raw data (exactly overall_length)
            df_overall = df.copy()
            if len(df_overall) < overall_length:
                pad_size = overall_length - len(df_overall)
                last_row = df_overall.iloc[-1:]
                df_overall = pd.concat([df_overall, pd.concat([last_row] * pad_size, ignore_index=True)], ignore_index=True)
            elif len(df_overall) > overall_length:
                df_overall = df_overall.iloc[:overall_length]

            imu1_acc_o = np.sqrt(df_overall['IMU1_accX']**2 + df_overall['IMU1_accY']**2 + df_overall['IMU1_accZ']**2)
            imu1_gyr_o = np.sqrt(df_overall['IMU1_gyrX']**2 + df_overall['IMU1_gyrY']**2 + df_overall['IMU1_gyrZ']**2)
            imu2_acc_o = np.sqrt(df_overall['IMU2_accX']**2 + df_overall['IMU2_accY']**2 + df_overall['IMU2_accZ']**2)
            imu2_gyr_o = np.sqrt(df_overall['IMU2_gyrX']**2 + df_overall['IMU2_gyrY']**2 + df_overall['IMU2_gyrZ']**2)

            imu1_acc_overall.append(imu1_acc_o)
            imu1_gyr_overall.append(imu1_gyr_o)
            imu2_acc_overall.append(imu2_acc_o)
            imu2_gyr_overall.append(imu2_gyr_o)
        except Exception as e:
            logger.error("Error reading %s: %s", f.name, e)

    if not imu1_acc_centered:
        return

    # Process centered data statistics
    imu1_acc_centered = np.array(imu1_acc_centered)
    imu1_gyr_centered = np.array(imu1_gyr_centered)
    imu2_acc_centered = np.array(imu2_acc_centered)
    imu2_gyr_centered = np.array(imu2_gyr_centered)

    dist_df = pd.DataFrame({
        'sample_index': range(TARGET_SAMPLES),
        'IMU1_acc_mean': np.mean(imu1_acc_centered, axis=0),
        'IMU1_acc_std': np.std(imu1_acc_centered, axis=0),
        'IMU1_gyr_mean': np.mean(imu1_gyr_centered, axis=0),
        'IMU1_gyr_std': np.std(imu1_gyr_centered, axis=0),
        'IMU2_acc_mean': np.mean(imu2_acc_centered, axis=0),
        'IMU2_acc_std': np.std(imu2_acc_centered, axis=0),
        'IMU2_gyr_mean': np.mean(imu2_gyr_centered, axis=0),
        'IMU2_gyr_std': np.std(imu2_gyr_centered, axis=0),
    })

    # Get sequential filepath
    dist_csv, dist_idx = get_next_energy_distribution_filepath(gesture_name, session_name)
    dist_df.to_csv(dist_csv, index=False)
    ui.success(f"Motion energy distribution saved: {dist_csv.name}")

    # Plot 1: Centered energy distribution (exactly TARGET_SAMPLES = 150)
    plt.close('all')
    t_c = np.linspace(0, RECORD_DURATION_S, TARGET_SAMPLES)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Centered Motion Energy Distribution: '{gesture_name}' ({len(files)} Samples)", fontsize=14, fontweight='bold')

    axs[0, 0].plot(t_c, dist_df['IMU1_acc_mean'], color='#1f77b4', label='Mean')
    axs[0, 0].fill_between(t_c, dist_df['IMU1_acc_mean'] - dist_df['IMU1_acc_std'],
                           dist_df['IMU1_acc_mean'] + dist_df['IMU1_acc_std'], color='#1f77b4', alpha=0.2)
    axs[0, 0].set_title("IMU1 (Wrist) Accelerometer Magnitude")
    axs[0, 0].set_ylabel("Acceleration (g)")
    axs[0, 0].grid(True, linestyle='--')
    axs[0, 0].legend()

    axs[1, 0].plot(t_c, dist_df['IMU1_gyr_mean'], color='#ff7f0e', label='Mean')
    axs[1, 0].fill_between(t_c, dist_df['IMU1_gyr_mean'] - dist_df['IMU1_gyr_std'],
                           dist_df['IMU1_gyr_mean'] + dist_df['IMU1_gyr_std'], color='#ff7f0e', alpha=0.2)
    axs[1, 0].set_title("IMU1 (Wrist) Gyroscope Magnitude")
    axs[1, 0].set_ylabel("Angular Velocity (dps)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].grid(True, linestyle='--')
    axs[1, 0].legend()

    axs[0, 1].plot(t_c, dist_df['IMU2_acc_mean'], color='#2ca02c', label='Mean')
    axs[0, 1].fill_between(t_c, dist_df['IMU2_acc_mean'] - dist_df['IMU2_acc_std'],
                           dist_df['IMU2_acc_mean'] + dist_df['IMU2_acc_std'], color='#2ca02c', alpha=0.2)
    axs[0, 1].set_title("IMU2 (Finger) Accelerometer Magnitude")
    axs[0, 1].grid(True, linestyle='--')
    axs[0, 1].legend()

    axs[1, 1].plot(t_c, dist_df['IMU2_gyr_mean'], color='#d62728', label='Mean')
    axs[1, 1].fill_between(t_c, dist_df['IMU2_gyr_mean'] - dist_df['IMU2_gyr_std'],
                           dist_df['IMU2_gyr_mean'] + dist_df['IMU2_gyr_std'], color='#d62728', alpha=0.2)
    axs[1, 1].set_title("IMU2 (Finger) Gyroscope Magnitude")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].grid(True, linestyle='--')
    axs[1, 1].legend()

    plt.tight_layout()
    centered_png = dist_csv.parent / f"centered_energy_distribution_{dist_idx}.png"
    plt.savefig(centered_png)
    plt.close(fig)
    ui.success(f"Centered plot saved: {centered_png.name}")

    # Plot 2: Overall energy distribution (overall_length)
    imu1_acc_overall = np.array(imu1_acc_overall)
    imu1_gyr_overall = np.array(imu1_gyr_overall)
    imu2_acc_overall = np.array(imu2_acc_overall)
    imu2_gyr_overall = np.array(imu2_gyr_overall)

    mean_start_idx = np.mean(start_indices)
    mean_end_idx = mean_start_idx + TARGET_SAMPLES

    # Convert start indices to time coordinates (indices / 100)
    mean_start_t = mean_start_idx / 100.0
    mean_end_t = mean_end_idx / 100.0

    t_o = np.linspace(0, RECORD_DURATION_S + PRE_BUFFER_S + POST_BUFFER_S, overall_length)
    fig_o, axs_o = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig_o.suptitle(f"Overall Motion Energy Distribution: '{gesture_name}' ({len(files)} Samples)", fontsize=14, fontweight='bold')

    # Overall statistics
    o_df = pd.DataFrame({
        'IMU1_acc_mean': np.mean(imu1_acc_overall, axis=0),
        'IMU1_acc_std': np.std(imu1_acc_overall, axis=0),
        'IMU1_gyr_mean': np.mean(imu1_gyr_overall, axis=0),
        'IMU1_gyr_std': np.std(imu1_gyr_overall, axis=0),
        'IMU2_acc_mean': np.mean(imu2_acc_overall, axis=0),
        'IMU2_acc_std': np.std(imu2_acc_overall, axis=0),
        'IMU2_gyr_mean': np.mean(imu2_gyr_overall, axis=0),
        'IMU2_gyr_std': np.std(imu2_gyr_overall, axis=0),
    })

    # Helper function to plot subplot
    def plot_overall_subplot(ax, mean_col, std_col, color, title, ylabel, xlabel=None):
        ax.plot(t_o, o_df[mean_col], color=color, label='Mean')
        ax.fill_between(t_o, o_df[mean_col] - o_df[std_col], o_df[mean_col] + o_df[std_col], color=color, alpha=0.2)
        ax.axvline(x=mean_start_t, color='black', linestyle='--', linewidth=1.5, label='Avg Start')
        ax.axvline(x=mean_end_t, color='black', linestyle='--', linewidth=1.5, label='Avg End')
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(True, linestyle='--')
        ax.legend()

    plot_overall_subplot(axs_o[0, 0], 'IMU1_acc_mean', 'IMU1_acc_std', '#1f77b4', "IMU1 (Wrist) Accelerometer Magnitude", "Acceleration (g)")
    plot_overall_subplot(axs_o[1, 0], 'IMU1_gyr_mean', 'IMU1_gyr_std', '#ff7f0e', "IMU1 (Wrist) Gyroscope Magnitude", "Angular Velocity (dps)", "Time (s)")
    plot_overall_subplot(axs_o[0, 1], 'IMU2_acc_mean', 'IMU2_acc_std', '#2ca02c', "IMU2 (Finger) Accelerometer Magnitude", "")
    plot_overall_subplot(axs_o[1, 1], 'IMU2_gyr_mean', 'IMU2_gyr_std', '#d62728', "IMU2 (Finger) Gyroscope Magnitude", "", "Time (s)")

    plt.tight_layout()
    overall_png = dist_csv.parent / f"overall_energy_distribution_{dist_idx}.png"
    plt.savefig(overall_png)
    plt.close(fig_o)
    ui.success(f"Overall plot saved: {overall_png.name}")

    # Record in metadata
    session_metadata["energy_distributions"].append({
        "file": dist_csv.name,
        "sample_index": sample_index if sample_index is not None else len(files)
    })
    save_metadata(gesture_name, session_name)


def plot_data(df, save_path=None, start_idx=None):
    """
    Plots IMU accelerometer and gyroscope signals to a PNG file.
    :param: df (DataFrame): data to plot.
    :param: save_path (str | Path | None): save destination path.
    :param: start_idx (int | None): start index of the selected gesture window.
    :return: None:
    """
    if df.empty:
        return

    plt.close('all')
    x = range(len(df))
    fig = plt.figure(figsize=(12, 6))

    # --- Accelerometer Plot ---
    plt.subplot(2, 1, 1)
    if 'IMU1_accX' in df.columns:
        plt.plot(x, df['IMU1_accX'], label='IMU1 AccX', color='r')
        plt.plot(x, df['IMU1_accY'], label='IMU1 AccY', color='g')
        plt.plot(x, df['IMU1_accZ'], label='IMU1 AccZ', color='b')
    if 'IMU2_accX' in df.columns:
        plt.plot(x, df['IMU2_accX'], label='IMU2 AccX', linestyle='--', color='r')
        plt.plot(x, df['IMU2_accY'], label='IMU2 AccY', linestyle='--', color='g')
        plt.plot(x, df['IMU2_accZ'], label='IMU2 AccZ', linestyle='--', color='b')
    if start_idx is not None:
        plt.axvline(x=start_idx, color='black', linestyle='--', linewidth=1.5, label='Gesture Start')
        plt.axvline(x=start_idx + 150, color='black', linestyle='--', linewidth=1.5, label='Gesture End')
    plt.title('Accelerometer Data')
    plt.xlabel('Sample')
    plt.ylabel('g')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # --- Gyroscope Plot ---
    plt.subplot(2, 1, 2)
    if 'IMU1_gyrX' in df.columns:
        plt.plot(x, df['IMU1_gyrX'], label='IMU1 GyrX', color='r')
        plt.plot(x, df['IMU1_gyrY'], label='IMU1 GyrY', color='g')
        plt.plot(x, df['IMU1_gyrZ'], label='IMU1 GyrZ', color='b')
    if 'IMU2_gyrX' in df.columns:
        plt.plot(x, df['IMU2_gyrX'], label='IMU2 GyrX', linestyle='--', color='r')
        plt.plot(x, df['IMU2_gyrY'], label='IMU2 GyrY', linestyle='--', color='g')
        plt.plot(x, df['IMU2_gyrZ'], label='IMU2 GyrZ', linestyle='--', color='b')
    if start_idx is not None:
        plt.axvline(x=start_idx, color='black', linestyle='--', linewidth=1.5, label='Gesture Start')
        plt.axvline(x=start_idx + 150, color='black', linestyle='--', linewidth=1.5, label='Gesture End')
    plt.title('Gyroscope Data')
    plt.xlabel('Sample')
    plt.ylabel('dps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path)
        ui.success(f"Plot saved to: {save_path.name}")
    plt.close(fig)


# ======================================================================================================================
# Main Execution
# ======================================================================================================================
def main():
    """
    Resolves IMU ports, starts reading threads, and runs data collection thread.
    :return: None:
    :raises: RuntimeError: if a reading thread stops or cannot be started.
    """
    device_resolution.print_available_serial_ports()

    port_imu1 = device_resolution.resolve_device_port("imu1")
    port_imu2 = device_resolution.resolve_device_port("imu2")

    ui.hr(title="Configure Recording Session")
    session_name = ui.ask("Please enter recording session name [Default: session_<timestamp>]:")
    if not session_name:
        session_name = f"session_{int(time.time())}"

    imu1 = IMUDataInput(port=port_imu1, baudrate=BAUDRATE, name="IMU1")
    imu2 = IMUDataInput(port=port_imu2, baudrate=BAUDRATE, name="IMU2")

    imu1.start()
    imu2.start()

    ui.info("Waiting for sensor initialization (2s)...")
    time.sleep(2.0)

    control_thread = threading.Thread(target=input_thread, args=(imu1, imu2, session_name), daemon=True)
    control_thread.start()

    try:
        while running.is_set():
            if not control_thread.is_alive():
                logger.error("Control thread died. Aborting pipeline.")
                raise RuntimeError("Control thread died.")

            if not imu1.running or not imu2.running:
                logger.error("One of the IMU reading threads stopped running. Aborting pipeline.")
                raise RuntimeError("IMU reading thread stopped.")

            time.sleep(0.1)

    except KeyboardInterrupt:
        running.clear()

    finally:
        imu1.stop()
        imu2.stop()


if __name__ == "__main__":
    main()