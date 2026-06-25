# code/record_data.py
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
from pathlib import Path
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI/threading crashes
import matplotlib.pyplot as plt
from input_data import IMUDataInput
from sync import process_stream
import device_resolution
from data_fusion_project.core.paths import DATA_DIR, GESTURES, get_calibration_file, get_next_recording_file
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
MAX_DEVIATION_OF_TARGET_SAMPLE_RATE = 30  # Max permitted sample count deviation percentage (20%)

PRE_BUFFER_S = 0.05
POST_BUFFER_S = 0.05

# Geste, die kontinuierlich (überlappend) statt sample-weise aufgenommen wird.
NONE_GESTURE_NAME = "none"
# Überlappung aufeinanderfolgender 'none'-Fenster (0 = keine, 0.5 = 50 %).
OVERLAP_RATIO = 0.5

# Pause zwischen einzelnen Samples bei allen Gesten außer 'none'.
PAUSE_DURATION_S = RECORD_DURATION_S

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

    completed_status = ui.progress_bar(duration_s, label="Aufnahme: ", color=Style.SUCCESS, stop_session=stop_session)

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

    with ui.spinner("Verarbeite und synchronisiere Sensordaten..."):
        _merged, valid_windows = process_stream(df1, df2, window_sz=target_samples, max_diff_us=MAX_SYNC_DIFF_US, freq_hz=100, center_gesture=True)

    if not valid_windows:
        logger.error("Synchronization failed (deviation too high).")
        raise RuntimeError("Synchronization failed.")

    # Update global counts for metrics
    global received_counts
    received_counts['IMU1'] += len(snapshot1)
    received_counts['IMU2'] += len(snapshot2)

    # Speichern des ersten validen Fensters
    save_df = valid_windows[0].copy()
    save_df = save_df.drop(columns=['sync_time_us'], errors='ignore')

    if len(save_df) != target_samples:
        logger.error(f"Sample contains invalid row count: {len(save_df)} (expected exactly {target_samples}).")
        raise RuntimeError(f"Invalid row count: {len(save_df)}.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(filename, index=False)
    ui.success(f"Datei erfolgreich gespeichert: {filename.name}")

    # Plot speichern
    if PLOT_EVERY_SAMPLE:
        plot_data(save_df, save_path=filename.with_suffix('.png'))
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
    ui.info("\n=== AUFNAHME PAUSIERT ===")
    if paused_during_recording:
        ui.info("Aktuelles (unvollständiges) Sample wurde verworfen.")
        
    if last_csv and last_csv.exists():
        ui.info(f"Letztes gespeichertes Sample: {last_csv.name}")
        ui.info("Drücke [d] um dieses Sample zu löschen.")
    else:
        ui.info("Kein gespeichertes Sample zum Löschen vorhanden.")
        
    ui.info("Drücke [Leertaste] zum Fortsetzen oder [Enter] zum Beenden.")
    
    ui.flush_input()
    
    while True:
        key = ui.get_key()
        if key == " ":
            ui.success("Setze Aufnahme fort...")
            return "resume"
        elif key in ("\r", "\n"):
            return "stop"
        elif key == "d":
            if last_csv and last_csv.exists():
                try:
                    last_csv.unlink()
                    if last_png and last_png.exists():
                        last_png.unlink()
                    ui.success(f"Gelöscht: {last_csv.name} und zugehöriger Plot.")
                    return "deleted"
                except Exception as e:
                    ui.error(f"Fehler beim Löschen der Datei: {e}")
            else:
                ui.warning("Kein Sample zum Löschen vorhanden.")
        time.sleep(0.05)


def handle_stop(last_csv, last_png):
    """
    Offers the user a 3-second countdown to delete the last saved sample.
    :param: last_csv (Path | None): path of last saved csv.
    :param: last_png (Path | None): path of last saved png.
    :return: None:
    """
    ui.info("\n=== AUFNAHME BEENDET ===")
    if last_csv and last_csv.exists():
        ui.info(f"Letztes gespeichertes Sample: {last_csv.name}")
        ui.warning("Möchtest du dieses letzte Sample löschen?")
        ui.info("Drücke [d] innerhalb von 3 Sekunden zum Löschen, oder eine andere Taste zum Überspringen...")
        
        ui.flush_input()
        start_time = time.time()
        deleted = False
        while time.time() - start_time < 3.0:
            remaining = 3.0 - (time.time() - start_time)
            print(f"\r  Zeit verbleibend: {remaining:.1f}s ... ", end="", flush=True)
            
            key = ui.get_key()
            if key == "d":
                try:
                    last_csv.unlink()
                    if last_png and last_png.exists():
                        last_png.unlink()
                    print()
                    ui.success(f"Gelöscht: {last_csv.name} und zugehöriger Plot.")
                    deleted = True
                    break
                except Exception as e:
                    print()
                    ui.error(f"Fehler beim Löschen: {e}")
                    break
            elif key is not None:
                print()
                ui.info("Löschen übersprungen.")
                break
            time.sleep(0.05)
        if not deleted:
            print()
    else:
        ui.info("Kein gespeichertes Sample zum Löschen vorhanden.")


def input_thread(imu1, imu2, session_name):
    """
    Controls user interaction menu thread for choosing gestures and starting loops.
    :param: imu1 (IMUDataInput): first IMU reader.
    :param: imu2 (IMUDataInput): second IMU reader.
    :param: session_name (str): current session directory name.
    :return: None:
    """
    global current_gesture

    ui.banner("Aufnahme-Controller", subtitle=f"Sitzung: {session_name}")
    _print_dataset_counts()

    choices = list(GESTURES) + ["[Beenden]"]

    while running.is_set():
        selection = ui.ask_choice("\nGeste auswählen:", choices)

        if selection is None or selection == len(choices) - 1:
            logger.info("Beende Aufnahme-Schleife...")
            running.clear()
            break

        current_gesture = selection
        gesture_name = GESTURES[current_gesture]
        ui.hr(title=f"Auswahl: {gesture_name}")

        # Prüfe Kalibrierung
        cal_file = get_calibration_file(gesture_name, session_name)
        if not cal_file.exists():
            ui.warning(f"Keine Kalibrierung für Geste '{gesture_name}' in Sitzung '{session_name}' gefunden.")
            with ui.non_blocking_input():
                ui.wait_for_enter("Bereit? [Enter] startet die 5s Stillstands-Kalibrierung...")
                ui.info("Bitte halte die Sensoren für 5 Sekunden absolut still...")
                success = run_single_recording(imu1, imu2, duration_s=5.0, target_samples=500, filename=cal_file)
                while not success:
                    ui.error("Kalibrierung fehlgeschlagen. Versuche es erneut.")
                    ui.wait_for_enter("Bereit? [Enter] startet die 5s Stillstands-Kalibrierung...")
                    ui.info("Bitte halte die Sensoren für 5 Sekunden absolut still...")
                    success = run_single_recording(imu1, imu2, duration_s=5.0, target_samples=500, filename=cal_file)
            ui.success("Kalibrierung erfolgreich gespeichert!")

        # Bestimme Aufnahmemodus
        if gesture_name == NONE_GESTURE_NAME:
            ui.wait_for_enter("Bereit? [Enter] startet die kontinuierliche Aufnahme...")
            stop_session.clear()
            record_continuous(imu1, imu2, session_name)
        else:
            ui.wait_for_enter("Bereit? [Enter] startet die variable Aufnahmeschleife (Sample/Pause/Sample)...")
            stop_session.clear()
            record_samples_loop(imu1, imu2, session_name)
            
        recording.clear()
        ui.success("Aufnahmesitzung beendet.")


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

    logger.info("KONTINUIERLICHE AUFNAHME '%s' (Fenster %ss, Überlappung %s%%)",
                gesture_name, RECORD_DURATION_S, OVERLAP_RATIO*100)
    ui.info("Drücke [Leertaste] oder [Enter] zum Stoppen...")

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
                rec_file = get_next_recording_file(gesture_name, session_name)
                _save_window(window, rec_file, verbose=False, plot=False)
                saved += 1
                print(f"\r  Überlappende 'none'-Fenster gespeichert: {saved}", end="", flush=True)

            next_start_us += advance_us
            _trim_before(local_buf1, next_start_us)
            _trim_before(local_buf2, next_start_us)

    print()
    global received_counts
    received_counts['IMU1'] += local_cnt1
    received_counts['IMU2'] += local_cnt2
    logger.info("Gespeichert: %d überlappende Fenster für '%s'", saved, gesture_name)
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
    saved = 0
    sample_idx = 0
    
    last_saved_csv = None
    last_saved_png = None
    
    with ui.non_blocking_input():
        while not stop_session.is_set():
            if not imu1.running or not imu2.running:
                logger.error("IMU thread stopped during sample loop. Aborting.")
                raise RuntimeError("IMU thread stopped during sample loop.")

            sample_idx += 1
            rec_file = get_next_recording_file(gesture_name, session_name)

            logger.info("Sample %d ('%s') — Starte Aufnahme...", sample_idx, gesture_name)
            
            # Clear reader queues before starting
            imu1.get_data()
            imu2.get_data()

            recording.set()

            # Pre-buffer recording silently
            time.sleep(PRE_BUFFER_S)

            rec_status = ui.progress_bar(RECORD_DURATION_S, label="Aufnahme: ", color=Style.SUCCESS, stop_session=stop_session)

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

            with ui.spinner("Verarbeite und synchronisiere Sensordaten..."):
                _merged, valid_windows = process_stream(df1, df2, window_sz=TARGET_SAMPLES, max_diff_us=MAX_SYNC_DIFF_US, freq_hz=100, center_gesture=True)

            if not valid_windows:
                logger.error("Synchronization failed for sample (deviation too high). Aborting pipeline.")
                raise RuntimeError("Synchronization failed for sample.")

            # Update global counts for metrics
            global received_counts
            received_counts['IMU1'] += len(snapshot1)
            received_counts['IMU2'] += len(snapshot2)

            save_df = valid_windows[0].copy()
            save_df = save_df.drop(columns=['sync_time_us'], errors='ignore')

            if len(save_df) != TARGET_SAMPLES:
                logger.error(f"Sample contains invalid row count: {len(save_df)} (expected exactly {TARGET_SAMPLES}).")
                raise RuntimeError(f"Invalid row count: {len(save_df)}.")

            rec_file = Path(rec_file)
            rec_file.parent.mkdir(parents=True, exist_ok=True)
            save_df.to_csv(rec_file, index=False)
            ui.success(f"Datei erfolgreich gespeichert: {rec_file.name}")

            last_saved_csv = rec_file
            last_saved_png = rec_file.with_suffix('.png') if PLOT_EVERY_SAMPLE else None
            
            if PLOT_EVERY_SAMPLE:
                plot_data(save_df, save_path=last_saved_png)
                
            saved += 1

            # Check for re-calibration trigger after 50 successfully recorded samples
            if saved > 0 and saved % 50 == 0:
                save_and_plot_energy_distribution(gesture_name, session_name)
                ui.warning(f"\n{saved} Samples aufgenommen. Erneute Kalibrierung erforderlich!")
                cal_file = get_calibration_file(gesture_name, session_name)
                ui.wait_for_enter("Bereit? [Enter] startet die 5s Stillstands-Kalibrierung...")
                ui.info("Bitte halte die Sensoren für 5 Sekunden absolut still...")
                success = run_single_recording(imu1, imu2, duration_s=5.0, target_samples=500, filename=cal_file)
                while not success:
                    ui.error("Kalibrierung fehlgeschlagen. Versuche es erneut.")
                    ui.wait_for_enter("Bereit? [Enter] startet die 5s Stillstands-Kalibrierung...")
                    ui.info("Bitte halte die Sensoren für 5 Sekunden absolut still...")
                    success = run_single_recording(imu1, imu2, duration_s=5.0, target_samples=500, filename=cal_file)
                ui.success("Kalibrierung erfolgreich gespeichert!")

            # Pause Phase
            pause_status = ui.progress_bar(PAUSE_DURATION_S, label="Pause:    ", color=Style.ERROR, stop_session=stop_session)
            
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

    # Berechne und plote Bewegungsenergie-Verteilung beim Beenden der Aufnahme
    logger.info("Gespeichert: %d Samples für '%s'", saved, gesture_name)
    save_and_plot_energy_distribution(gesture_name, session_name)
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
        ui.success(f"Datensatz gespeichert unter: {filename.name}")
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
    logger.info("Empfangene Datenpakete in dieser Session: IMU1 = %d, IMU2 = %d", n1, n2)


def _print_dataset_counts():
    """
    Prints to UI the number of saved samples per gesture.
    :return: None:
    """
    ui.hr(title="Gespeicherte Datensätze pro Geste")
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
    logger.info("Gesamtanzahl valider Datensätze: %d", total)


# ======================================================================================================================
# Analysis & Plotting
# ======================================================================================================================
def save_and_plot_energy_distribution(gesture_name, session_name):
    """
    Computes statistical motion energy, saves it as CSV, and plots PNG.
    :param: gesture_name (str): name of gesture.
    :param: session_name (str): current session directory name.
    :return: None:
    """
    if not PLOT_MOVEMENT_DISTRIBUTION:
        return

    from data_fusion_project.core.paths import get_session_dir
    session_dir = get_session_dir(gesture_name, session_name)
    
    # Find all sample CSVs (excluding calibration and energy_distribution)
    files = [f for f in session_dir.glob("*.csv") if f.name not in ("calibration.csv", "energy_distribution.csv")]
    if not files:
        logger.warning("Keine Samples für Energie-Verteilungs-Analyse in %s gefunden.", session_dir)
        return

    logger.info("Berechne Bewegungsenergie-Verteilung für '%s' (%d Dateien)...", gesture_name, len(files))

    imu1_acc_runs = []
    imu1_gyr_runs = []
    imu2_acc_runs = []
    imu2_gyr_runs = []

    import numpy as np
    import pandas as pd

    for f in files:
        try:
            df = pd.read_csv(f)
            # Standardize length to TARGET_SAMPLES (150)
            if len(df) < TARGET_SAMPLES:
                pad_size = TARGET_SAMPLES - len(df)
                last_row = df.iloc[-1:]
                df = pd.concat([df, pd.concat([last_row] * pad_size, ignore_index=True)], ignore_index=True)
            elif len(df) > TARGET_SAMPLES:
                df = df.iloc[:TARGET_SAMPLES]

            imu1_acc = np.sqrt(df['IMU1_accX']**2 + df['IMU1_accY']**2 + df['IMU1_accZ']**2)
            imu1_gyr = np.sqrt(df['IMU1_gyrX']**2 + df['IMU1_gyrY']**2 + df['IMU1_gyrZ']**2)
            imu2_acc = np.sqrt(df['IMU2_accX']**2 + df['IMU2_accY']**2 + df['IMU2_accZ']**2)
            imu2_gyr = np.sqrt(df['IMU2_gyrX']**2 + df['IMU2_gyrY']**2 + df['IMU2_gyrZ']**2)

            imu1_acc_runs.append(imu1_acc)
            imu1_gyr_runs.append(imu1_gyr)
            imu2_acc_runs.append(imu2_acc)
            imu2_gyr_runs.append(imu2_gyr)
        except Exception as e:
            logger.error("Fehler beim Lesen von %s: %s", f.name, e)

    if not imu1_acc_runs:
        return

    # Calculate statistics
    imu1_acc_runs = np.array(imu1_acc_runs)
    imu1_gyr_runs = np.array(imu1_gyr_runs)
    imu2_acc_runs = np.array(imu2_acc_runs)
    imu2_gyr_runs = np.array(imu2_gyr_runs)

    dist_df = pd.DataFrame({
        'sample_index': range(TARGET_SAMPLES),
        'IMU1_acc_mean': np.mean(imu1_acc_runs, axis=0),
        'IMU1_acc_std': np.std(imu1_acc_runs, axis=0),
        'IMU1_gyr_mean': np.mean(imu1_gyr_runs, axis=0),
        'IMU1_gyr_std': np.std(imu1_gyr_runs, axis=0),
        'IMU2_acc_mean': np.mean(imu2_acc_runs, axis=0),
        'IMU2_acc_std': np.std(imu2_acc_runs, axis=0),
        'IMU2_gyr_mean': np.mean(imu2_gyr_runs, axis=0),
        'IMU2_gyr_std': np.std(imu2_gyr_runs, axis=0),
    })

    dist_csv = session_dir / "energy_distribution.csv"
    dist_df.to_csv(dist_csv, index=False)
    ui.success(f"Bewegungsenergie-Verteilung gespeichert: {dist_csv.name}")

    # Plot creation (non-interactive, Agg)
    plt.close('all')
    t = np.linspace(0, RECORD_DURATION_S, TARGET_SAMPLES)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Bewegungsenergie-Verteilung: '{gesture_name}' ({len(files)} Samples)", fontsize=14, fontweight='bold')

    # IMU1 Acc
    axs[0, 0].plot(t, dist_df['IMU1_acc_mean'], color='#1f77b4', label='Mean')
    axs[0, 0].fill_between(t, dist_df['IMU1_acc_mean'] - dist_df['IMU1_acc_std'],
                           dist_df['IMU1_acc_mean'] + dist_df['IMU1_acc_std'], color='#1f77b4', alpha=0.2)
    axs[0, 0].set_title("IMU1 (Wrist) Accelerometer Magnitude")
    axs[0, 0].set_ylabel("Acceleration (g)")
    axs[0, 0].grid(True, linestyle='--')
    axs[0, 0].legend()

    # IMU1 Gyr
    axs[1, 0].plot(t, dist_df['IMU1_gyr_mean'], color='#ff7f0e', label='Mean')
    axs[1, 0].fill_between(t, dist_df['IMU1_gyr_mean'] - dist_df['IMU1_gyr_std'],
                           dist_df['IMU1_gyr_mean'] + dist_df['IMU1_gyr_std'], color='#ff7f0e', alpha=0.2)
    axs[1, 0].set_title("IMU1 (Wrist) Gyroscope Magnitude")
    axs[1, 0].set_ylabel("Angular Velocity (dps)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].grid(True, linestyle='--')
    axs[1, 0].legend()

    # IMU2 Acc
    axs[0, 1].plot(t, dist_df['IMU2_acc_mean'], color='#2ca02c', label='Mean')
    axs[0, 1].fill_between(t, dist_df['IMU2_acc_mean'] - dist_df['IMU2_acc_std'],
                           dist_df['IMU2_acc_mean'] + dist_df['IMU2_acc_std'], color='#2ca02c', alpha=0.2)
    axs[0, 1].set_title("IMU2 (Finger) Accelerometer Magnitude")
    axs[0, 1].grid(True, linestyle='--')
    axs[0, 1].legend()

    # IMU2 Gyr
    axs[1, 1].plot(t, dist_df['IMU2_gyr_mean'], color='#d62728', label='Mean')
    axs[1, 1].fill_between(t, dist_df['IMU2_gyr_mean'] - dist_df['IMU2_gyr_std'],
                           dist_df['IMU2_gyr_mean'] + dist_df['IMU2_gyr_std'], color='#d62728', alpha=0.2)
    axs[1, 1].set_title("IMU2 (Finger) Gyroscope Magnitude")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].grid(True, linestyle='--')
    axs[1, 1].legend()

    plt.tight_layout()
    dist_png = session_dir / "energy_distribution.png"
    plt.savefig(dist_png)
    plt.close(fig)
    ui.success(f"Plot der Bewegungsenergie-Verteilung gespeichert: {dist_png.name}")


def plot_data(df, save_path=None):
    """
    Plots IMU accelerometer and gyroscope signals to a PNG file.
    :param: df (DataFrame): data to plot.
    :param: save_path (str | Path | None): save destination path.
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
    plt.title('Gyroscope Data')
    plt.xlabel('Sample')
    plt.ylabel('dps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path)
        ui.success(f"Plot gespeichert unter: {save_path.name}")
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

    ui.hr(title="Aufnahmesitzung konfigurieren")
    session_name = ui.ask("Bitte Aufnahmesitzung (Recording Session Name) eingeben [Standard: session_<timestamp>]:")
    if not session_name:
        session_name = f"session_{int(time.time())}"

    imu1 = IMUDataInput(port=port_imu1, baudrate=BAUDRATE, name="IMU1")
    imu2 = IMUDataInput(port=port_imu2, baudrate=BAUDRATE, name="IMU2")

    imu1.start()
    imu2.start()

    ui.info("Warte auf Initialisierung der Sensoren (2s)...")
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