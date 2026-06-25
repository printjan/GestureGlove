import time
import os
import threading
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from input_data import IMUDataInput
from sync import process_stream
import device_resolution

# --- Configuration ---
BAUDRATE = 115200

RECORD_DURATION_S = 1.5
TARGET_SAMPLES = 150  # = RECORD_DURATION_S * 100 Hz

GESTURE_LABELS = {
    0: "none",
    1: "circle_clockwise",
    2: "circle_counterclockwise"
}


# Thread-Steuerung über Events (statt rohe Booleans)
running = threading.Event()
running.set()
recording = threading.Event()

# Puffer + zugehöriger Lock für thread-sicheren Zugriff
_buffer_lock = threading.Lock()
imu1_data_buffer = []
imu2_data_buffer = []
current_gesture = 0


def _show_progress_bar(duration_s, bar_width=40):
    steps = 60
    sleep_time = duration_s / steps
    for i in range(steps + 1):
        filled = int(bar_width * i / steps)
        bar = '█' * filled + '░' * (bar_width - filled)
        elapsed = duration_s * i / steps
        print(f'\r  [{bar}] {elapsed:.1f}s / {duration_s:.1f}s  ', end='', flush=True)
        if i < steps:
            time.sleep(sleep_time)
    print()


def input_thread(imu1, imu2):
    global current_gesture

    print("\n--- Setup Complete ---")
    while running.is_set():
        options = "  ".join(f"{i}: {label}" for i, label in GESTURE_LABELS.items())
        user_input = input(f"\nGeste auswählen [{options}] (oder 'q'): ")

        if user_input.strip().lower() == 'q':
            running.clear()
            break

        try:
            gesture_id = int(user_input.strip())
            if gesture_id not in GESTURE_LABELS:
                print("Ungültige Geste.")
                continue
        except ValueError:
            print("Bitte eine Zahl eingeben.")
            continue

        current_gesture = gesture_id
        print(f"Geste: {GESTURE_LABELS[current_gesture]}")
        input(f"Bereit? [Enter] drücken, dann Geste ausführen ({RECORD_DURATION_S}s)...")

        # Puffer atomar leeren, dann erst recording aktivieren
        with _buffer_lock:
            imu1.get_data()
            imu2.get_data()
            imu1_data_buffer.clear()
            imu2_data_buffer.clear()

        print(f">>> AUFNAHME LÄUFT ({TARGET_SAMPLES} Samples @ 100 Hz) <<<")
        recording.set()
        _show_progress_bar(RECORD_DURATION_S)
        recording.clear()
        print(">>> AUFNAHME BEENDET <<<")

        process_and_save_data()

def process_and_save_data():
    with _buffer_lock:
        snapshot1 = list(imu1_data_buffer)
        snapshot2 = list(imu2_data_buffer)
    print(f"Verarbeite Daten: {len(snapshot1)} Pakete von IMU1, {len(snapshot2)} Pakete von IMU2")
    
    if len(snapshot1) == 0 and len(snapshot2) == 0:
        print("Keine Daten gesammelt. Überspringe Speichern.")
        return

    # In Pandas DataFrames konvertieren
    df1 = pd.DataFrame(snapshot1)
    df2 = pd.DataFrame(snapshot2)

    # Zusammenführen und Synchronisieren über unsere neue Pipeline
    merged_df, valid_windows = process_stream(df1, df2, window_sz=TARGET_SAMPLES, max_diff_us=5000, freq_hz=100)
    print(f"{len(valid_windows)} valide Fenster extrahiert (Abweichung < 5ms).")

    if not valid_windows:
        print("Kein valides Fenster — Aufnahme verworfen.")
        return

    # Erstes valides Fenster speichern (exakt TARGET_SAMPLES Zeilen)
    save_df = valid_windows[0].copy()
    start_time = save_df['sync_time_us'].iloc[0]
    save_df['time_rel_s'] = (save_df['sync_time_us'] - start_time) / 1e6

    gesture_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", GESTURE_LABELS[current_gesture]))
    os.makedirs(gesture_dir, exist_ok=True)
    filename = os.path.join(gesture_dir, f"gesture_record_{int(time.time())}.csv")

    save_df.to_csv(filename, index=False)
    print(f"Datensatz gespeichert unter: {filename}\n")

    # Plot anzeigen
    plot_data(save_df)


def plot_data(df):
    """
    Erzeugt einen rudimentären Plot der zuletzt aufgezeichneten Daten.
    """
    if df.empty or 'time_rel_s' not in df.columns:
        return
        
    plt.figure(figsize=(12, 9))
    
    # --- Accelerometer Plot ---
    plt.subplot(3, 1, 1)
    if 'IMU1_accX' in df.columns:
        plt.plot(df['time_rel_s'], df['IMU1_accX'], label='IMU1 AccX', color='r')
        plt.plot(df['time_rel_s'], df['IMU1_accY'], label='IMU1 AccY', color='g')
        plt.plot(df['time_rel_s'], df['IMU1_accZ'], label='IMU1 AccZ', color='b')
    if 'IMU2_accX' in df.columns:
        plt.plot(df['time_rel_s'], df['IMU2_accX'], label='IMU2 AccX', linestyle='--', color='r')
        plt.plot(df['time_rel_s'], df['IMU2_accY'], label='IMU2 AccY', linestyle='--', color='g')
        plt.plot(df['time_rel_s'], df['IMU2_accZ'], label='IMU2 AccZ', linestyle='--', color='b')
    plt.title('Accelerometer Data')
    plt.xlabel('Time (s)')
    plt.ylabel('g')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # --- Gyroscope Plot ---
    plt.subplot(3, 1, 2)
    if 'IMU1_gyrX' in df.columns:
        plt.plot(df['time_rel_s'], df['IMU1_gyrX'], label='IMU1 GyrX', color='r')
        plt.plot(df['time_rel_s'], df['IMU1_gyrY'], label='IMU1 GyrY', color='g')
        plt.plot(df['time_rel_s'], df['IMU1_gyrZ'], label='IMU1 GyrZ', color='b')
    if 'IMU2_gyrX' in df.columns:
        plt.plot(df['time_rel_s'], df['IMU2_gyrX'], label='IMU2 GyrX', linestyle='--', color='r')
        plt.plot(df['time_rel_s'], df['IMU2_gyrY'], label='IMU2 GyrY', linestyle='--', color='g')
        plt.plot(df['time_rel_s'], df['IMU2_gyrZ'], label='IMU2 GyrZ', linestyle='--', color='b')
    plt.title('Gyroscope Data')
    plt.xlabel('Time (s)')
    plt.ylabel('dps')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # --- Roll & Pitch Plot ---
    plt.subplot(3, 1, 3)
    if 'IMU1_roll_kf' in df.columns:
        plt.plot(df['time_rel_s'], df['IMU1_roll_kf'], label='IMU1 Roll (KF)', color='c')
        plt.plot(df['time_rel_s'], df['IMU1_pitch_kf'], label='IMU1 Pitch (KF)', color='m')
        plt.plot(df['time_rel_s'], df['IMU1_roll'], label='IMU1 Roll', color='y')
        plt.plot(df['time_rel_s'], df['IMU1_pitch'], label='IMU1 Pitch', color='k')
    if 'IMU2_roll_kf' in df.columns:
        plt.plot(df['time_rel_s'], df['IMU2_roll_kf'], label='IMU2 Roll (KF)', linestyle='--', color='c')
        plt.plot(df['time_rel_s'], df['IMU2_pitch_kf'], label='IMU2 Pitch (KF)', linestyle='--', color='m')
        plt.plot(df['time_rel_s'], df['IMU2_roll'], label='IMU2 Roll', linestyle='--', color='y')
        plt.plot(df['time_rel_s'], df['IMU2_pitch'], label='IMU2 Pitch', linestyle='--', color='k')
    plt.title('Orientation (Roll & Pitch - Kalman Filtered)')
    plt.xlabel('Time (s)')
    plt.ylabel('Degrees')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    # block=False bedeutet, dass das Script im Hintergrund weiterläuft,
    # und wir direkt die nächste Aufnahme starten können, während das
    # Fenster offen bleibt.
    plt.show(block=False) 

def main():
    # Sensoren initialisieren
    device_resolution.print_available_serial_ports()

    port_imu1 = device_resolution.resolve_device_port("imu1")
    port_imu2 = device_resolution.resolve_device_port("imu2")

    print(f"Resolved IMU1 -> {port_imu1}")
    print(f"Resolved IMU2 -> {port_imu2}")

    imu1 = IMUDataInput(port=port_imu1, baudrate=BAUDRATE, name="IMU1")
    imu2 = IMUDataInput(port=port_imu2, baudrate=BAUDRATE, name="IMU2")
    
    # Reader-Threads starten
    imu1.start()
    imu2.start()

    # Den Konsolen-Input in einem separaten Thread starten
    control_thread = threading.Thread(target=input_thread, args=(imu1, imu2), daemon=True)
    control_thread.start()

    try:
        # Haupt-Schleife: Pollt die Sensor-Queues
        while running.is_set():
            data1 = imu1.get_data()
            data2 = imu2.get_data()

            # Falls wir gerade aufnehmen, Daten thread-sicher in Puffer schreiben
            if recording.is_set():
                with _buffer_lock:
                    imu1_data_buffer.extend(data1)
                    imu2_data_buffer.extend(data2)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nManuell abgebrochen (STRG+C).")
        running.clear()

    finally:
        imu1.stop()
        imu2.stop()
        print("Programm beendet.")

if __name__ == "__main__":
    main()