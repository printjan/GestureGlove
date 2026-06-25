import time
import os
import random
import logging
import threading
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from input_data import IMUDataInput
from sync import process_stream
import device_resolution

# Die Sync-Pipeline loggt jedes geprüfte Fenster auf INFO. Im kontinuierlichen
# Modus (mehrere Fenster pro Sekunde) flutet das die Konsole und überschreibt
# die Fortschrittsanzeige. Nur noch Warnungen/Fehler anzeigen.
logging.getLogger("IMU_Sync").setLevel(logging.WARNING)

# --- Configuration ---
BAUDRATE = 115200

RECORD_DURATION_S = 1.5
TARGET_SAMPLES = 150  # = RECORD_DURATION_S * 100 Hz

# Geste, die kontinuierlich (überlappend) statt sample-weise aufgenommen wird.
NONE_GESTURE_ID = 0

# Überlappung aufeinanderfolgender 'none'-Fenster (0 = keine, 0.5 = 50 %).
# Bei 50 % wird alle (1 - OVERLAP) * RECORD_DURATION_S ein neues Fenster
# gespeichert, das sich zur Hälfte mit dem vorherigen überschneidet.
OVERLAP_RATIO = 0.5

# Pause zwischen einzelnen Samples bei allen Gesten außer 'none'.
PAUSE_BASE_S = 1.0
# Zufällige Variation der Pause (±), damit die Aufnahme nicht in einen festen
# Rhythmus verfällt — das macht die Pause "variabel".
PAUSE_JITTER_S = 0.4

# Maximal erlaubter Zeitversatz zwischen IMU1 und IMU2 innerhalb eines Fensters.
# Eine volle Abtastperiode (10 ms @ 100 Hz) — ein Restversatz in dieser
# Größenordnung wird beim Resampling auf 100 Hz ohnehin ausgeglichen.
MAX_SYNC_DIFF_US = 10000

GESTURE_LABELS = {
    0: "none",
    1: "circle_clockwise",
    2: "circle_counterclockwise"
}


# Thread-Steuerung über Events (statt rohe Booleans)
running = threading.Event()
running.set()
recording = threading.Event()
# Wird gesetzt, um eine laufende Aufnahme-Session (kontinuierlich oder
# sample-weise) sofort abzubrechen — macht beide Modi unterbrechbar.
stop_session = threading.Event()

# Puffer + zugehöriger Lock für thread-sicheren Zugriff
_buffer_lock = threading.Lock()
imu1_data_buffer = []
imu2_data_buffer = []
# Anzahl der während der laufenden Session empfangenen Pakete je Sensor
# (Dict, damit es ohne 'global' aus mehreren Threads mutiert werden kann).
received_counts = {'IMU1': 0, 'IMU2': 0}
current_gesture = 0


def _show_progress_bar(duration_s, label='', bar_width=40):
    """
    Zeigt einen Fortschrittsbalken über duration_s.

    Wartet die Zeit in kleinen Schritten ab und bricht sofort ab, sobald
    stop_session gesetzt wird. Gibt True zurück, wenn die volle Dauer
    durchlaufen wurde, und False, wenn unterbrochen wurde.
    """
    steps = 60
    sleep_time = duration_s / steps
    for i in range(steps + 1):
        filled = int(bar_width * i / steps)
        bar = '█' * filled + '░' * (bar_width - filled)
        elapsed = duration_s * i / steps
        print(f'\r  {label}[{bar}] {elapsed:.1f}s / {duration_s:.1f}s  ', end='', flush=True)
        if i < steps:
            # stop_session.wait() liefert True, sobald das Event gesetzt ist —
            # damit reagiert der Balken sofort auf einen Abbruch.
            if stop_session.wait(timeout=sleep_time):
                print()
                return False
    print()
    return True


def input_thread(imu1, imu2):
    global current_gesture

    print("\n--- Setup Complete ---")
    _print_dataset_counts()
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
        label = GESTURE_LABELS[current_gesture]
        print(f"Geste: {label}")

        if current_gesture == NONE_GESTURE_ID:
            input("Bereit? [Enter] startet die kontinuierliche Aufnahme...")
            worker_fn = record_continuous
        else:
            input("Bereit? [Enter] startet die Aufnahme-Schleife (Sample / Pause / Sample)...")
            worker_fn = record_samples_loop

        # Aufnahme-Worker in eigenem Thread starten, damit der Input-Thread
        # frei bleibt, um per [Enter] abzubrechen.
        stop_session.clear()
        worker = threading.Thread(target=worker_fn, args=(imu1, imu2), daemon=True)
        worker.start()

        # Zweites [Enter] (oder 'q') beendet die laufende Session.
        stop_input = input(">>> Aufnahme läuft — [Enter] zum Stoppen (oder 'q' zum Beenden) <<<\n")
        stop_session.set()
        worker.join()
        recording.clear()
        print(">>> Aufnahme gestoppt <<<")

        if stop_input.strip().lower() == 'q':
            running.clear()
            break


def _trim_before(buf, cutoff_us):
    """Entfernt alle Pakete vor cutoff_us (Puffer ist zeitlich aufsteigend)."""
    i = 0
    while i < len(buf) and buf[i]['pc_timestamp_us'] < cutoff_us:
        i += 1
    if i:
        del buf[:i]


def record_continuous(imu1, imu2):
    """
    Nimmt die Geste 'none' kontinuierlich mit überlappenden Fenstern auf.

    Es läuft eine durchgehende Aufnahme. Der Fensteranfang (eine gemeinsame
    PC-Zeit für beide IMUs) rückt pro gespeichertem Fenster um
    (1 - OVERLAP_RATIO) * RECORD_DURATION_S vor; alle älteren Pakete werden
    verworfen. Aufeinanderfolgende Fenster überlappen damit zu OVERLAP_RATIO.

    Die Steuerung läuft über pc_timestamp_us (gemeinsame PC-Uhr), nicht über
    Paketanzahlen — das bleibt robust, auch wenn die tatsächliche Abtastrate
    leicht von 100 Hz abweicht.

    Läuft bis stop_session gesetzt wird (über [Enter] im Input-Thread).
    """
    window_us = int(RECORD_DURATION_S * 1e6)
    advance_us = max(1, int(round(RECORD_DURATION_S * (1 - OVERLAP_RATIO) * 1e6)))
    poll_s = 0.05

    # Sauber starten: Queues und Puffer leeren, Paketzähler zurücksetzen,
    # dann Aufnahme aktivieren.
    with _buffer_lock:
        imu1.get_data()
        imu2.get_data()
        imu1_data_buffer.clear()
        imu2_data_buffer.clear()
        received_counts['IMU1'] = 0
        received_counts['IMU2'] = 0
    recording.set()

    print(f">>> KONTINUIERLICHE AUFNAHME '{GESTURE_LABELS[current_gesture]}' "
          f"(Fenster {RECORD_DURATION_S}s, Überlappung {OVERLAP_RATIO*100:.0f}%) <<<")

    saved = 0
    next_start_us = None  # PC-Zeit, ab der das nächste Fenster beginnen soll
    while not stop_session.is_set():
        if stop_session.wait(timeout=poll_s):
            break

        with _buffer_lock:
            snap1 = list(imu1_data_buffer)
            snap2 = list(imu2_data_buffer)
        if not snap1 or not snap2:
            continue

        if next_start_us is None:
            next_start_us = max(snap1[0]['pc_timestamp_us'], snap2[0]['pc_timestamp_us'])

        # Noch nicht genug Daten für ein volles Fenster ab next_start? Weiter sammeln.
        latest_us = min(snap1[-1]['pc_timestamp_us'], snap2[-1]['pc_timestamp_us'])
        if latest_us - next_start_us < window_us:
            continue

        window = _extract_window(snap1, snap2)
        if window is not None:
            _save_window(window, verbose=False)
            saved += 1
            print(f"\r  überlappende 'none'-Fenster gespeichert: {saved}", end='', flush=True)

        # Fensteranfang vorrücken; alles davor verwerfen (hält beide IMUs synchron).
        next_start_us += advance_us
        with _buffer_lock:
            _trim_before(imu1_data_buffer, next_start_us)
            _trim_before(imu2_data_buffer, next_start_us)

    recording.clear()
    print(f"\n>>> {saved} überlappende Fenster für '{GESTURE_LABELS[current_gesture]}' gespeichert <<<")
    _print_received_counts()
    _print_dataset_counts()


def record_samples_loop(imu1, imu2):
    """
    Nimmt für alle Gesten außer 'none' wiederholt einzelne Samples auf:
    Aufnahme -> variable Pause (~1 s) -> nächste Aufnahme -> ...

    Läuft bis stop_session gesetzt wird (über [Enter] im Input-Thread).
    """
    label = GESTURE_LABELS[current_gesture]
    saved = 0
    sample_idx = 0
    # Paketzähler für die gesamte Session zurücksetzen.
    with _buffer_lock:
        received_counts['IMU1'] = 0
        received_counts['IMU2'] = 0
    while not stop_session.is_set():
        sample_idx += 1

        # Puffer atomar leeren, dann erst recording aktivieren.
        with _buffer_lock:
            imu1.get_data()
            imu2.get_data()
            imu1_data_buffer.clear()
            imu2_data_buffer.clear()

        print(f"\n>>> Sample {sample_idx} ('{label}') — AUFNAHME "
              f"({TARGET_SAMPLES} Samples @ 100 Hz) <<<")
        recording.set()
        completed = _show_progress_bar(RECORD_DURATION_S)
        recording.clear()

        if not completed:
            # Während der Aufnahme abgebrochen — unvollständiges Sample verwerfen.
            print(">>> Während der Aufnahme gestoppt — Sample verworfen <<<")
            break

        if process_and_save_data():
            saved += 1

        if stop_session.is_set():
            break

        # Variable Pause um PAUSE_BASE_S herum (unterbrechbar).
        pause_s = max(0.0, PAUSE_BASE_S + random.uniform(-PAUSE_JITTER_S, PAUSE_JITTER_S))
        if not _show_progress_bar(pause_s, label='Pause '):
            break

    print(f"\n>>> {saved} Samples für '{label}' gespeichert <<<")
    _print_received_counts()
    _print_dataset_counts()


def _extract_window(snapshot1, snapshot2):
    """
    Verarbeitet rohe IMU-Pakete über die Sync-Pipeline und gibt das erste
    valide Fenster zurück (oder None, falls keine/zu wenige Daten vorliegen).
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


def _save_window(window_df, verbose=True, plot=False):
    """
    Speichert ein valides Fenster als CSV (nur Sensorwerte, ohne Timestamps).
    Der Dateiname nutzt Millisekunden-Auflösung, damit mehrere Fenster
    innerhalb derselben Sekunde (kontinuierlicher Modus) nicht kollidieren.
    """
    save_df = window_df.copy()

    # Timestamps dienen nur zur Synchronisation und werden nicht gespeichert.
    # Übrig bleiben ausschließlich die reinen Sensorwerte (acc/gyr je IMU).
    save_df = save_df.drop(columns=['sync_time_us'], errors='ignore')

    gesture_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", GESTURE_LABELS[current_gesture]))
    os.makedirs(gesture_dir, exist_ok=True)
    filename = os.path.join(gesture_dir, f"gesture_record_{int(time.time() * 1000)}.csv")

    save_df.to_csv(filename, index=False)
    if verbose:
        print(f"Datensatz gespeichert unter: {filename}\n")
    if plot:
        plot_data(save_df)
    return filename


def _print_received_counts():
    """Gibt die in dieser Session empfangenen Datenpakete je Sensor aus."""
    with _buffer_lock:
        n1 = received_counts['IMU1']
        n2 = received_counts['IMU2']
    print(f"Empfangene Datenpakete: IMU1 = {n1}, IMU2 = {n2}")


def _print_dataset_counts():
    """Gibt die Gesamtzahl gespeicherter Datensätze (CSV) je Geste aus."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    print("Datensätze pro Geste:")
    total = 0
    for label in GESTURE_LABELS.values():
        gesture_dir = os.path.join(base, label)
        n = len([f for f in os.listdir(gesture_dir) if f.endswith('.csv')]) \
            if os.path.isdir(gesture_dir) else 0
        total += n
        print(f"  {label}: {n}")
    print(f"  gesamt: {total}")


def process_and_save_data():
    """
    Snapshot des aktuellen Puffers verarbeiten und das erste valide Fenster
    speichern. Gibt True zurück, wenn ein Fenster gespeichert wurde.
    """
    with _buffer_lock:
        snapshot1 = list(imu1_data_buffer)
        snapshot2 = list(imu2_data_buffer)
    print(f"Verarbeite Daten: {len(snapshot1)} Pakete von IMU1, {len(snapshot2)} Pakete von IMU2")

    window = _extract_window(snapshot1, snapshot2)
    if window is None:
        print("Kein valides Fenster — Aufnahme verworfen.")
        return False

    _save_window(window, verbose=True, plot=True)
    return True


def plot_data(df):
    """
    Erzeugt einen rudimentären Plot der zuletzt aufgezeichneten Daten.
    Da keine Timestamps mehr gespeichert werden, ist die x-Achse der
    Sample-Index (das Fenster ist mit fester Frequenz resampled).
    """
    if df.empty:
        return

    # Vorheriges Fenster schließen, damit die Sample-Schleife nicht dutzende
    # Plot-Fenster öffnet, sondern ein einziges aktualisiert.
    plt.close('all')

    x = range(len(df))
    plt.figure(figsize=(12, 6))

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
    plt.show(block=False)
    plt.pause(0.1)

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
                    received_counts['IMU1'] += len(data1)
                    received_counts['IMU2'] += len(data2)

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