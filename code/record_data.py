import time
import os
import threading
import pandas as pd
import matplotlib.pyplot as plt
from input_data import IMUDataInput

# --- Configuration ---
PORT_IMU1 = 'COM10'
PORT_IMU2 = 'COM11' # Passe dies entsprechend deinem Setup an
BAUDRATE = 115200

GESTURE_LABELS = {
    0: "none",
    1: "circle_clockwise",
    2: "circle_counterclockwise"
}


# Globale Variablen für Threading/Control
running = True
recording = False
current_gesture = 0

imu1_data_buffer = []
imu2_data_buffer = []

def input_thread(imu1, imu2):
    """
    Dieser Thread wartet auf Console-Input vom Nutzer, 
    um die Aufnahme zu starten und zu stoppen.
    """
    global recording, running, imu1_data_buffer, imu2_data_buffer, GESTURE_LABELS
    
    print("\n--- Setup Complete ---")
    while running:
        # Blockiert, bis [Enter] gedrückt wird
        input_str = "Geste auswählen:  "
        for i in GESTURE_LABELS:
            input_str += f"{i}: {GESTURE_LABELS[i]}  "
        user_input = input(f"{input_str} und dann [Enter] um die Aufnahme zu STARTEN (oder tippe 'q' zum Beenden)...\n")
        
        if user_input.strip().lower() == 'q':
            running = False
            break
        
        try:
            gesture_id = int(user_input.strip())
            if gesture_id in GESTURE_LABELS:
                current_gesture = gesture_id
                print(f"Ausgewählte Geste: {GESTURE_LABELS[current_gesture]}")
            else:
                print("Ungültige Eingabe. Bitte eine gültige Geste auswählen.")
                continue
        except ValueError:
            print("Ungültige Eingabe. Bitte eine Zahl eingeben.")
            continue

        print(">>> RECORDING GESTARTET <<<")
        # Puffer leeren, falls Reste existieren
        imu1.get_data()
        imu2.get_data()
        imu1_data_buffer.clear()
        imu2_data_buffer.clear()
        
        recording = True
        
        input(">> Aufzeichnung läuft... Drücke [Enter] um zu STOPPEN...\n")
        print(">>> RECORDING GESTOPPT <<<")
        recording = False
        
        # Sobald gestoppt wurde, verarbeiten und speichern wir die Daten
        process_and_save_data()

def process_and_save_data():
    global imu1_data_buffer, imu2_data_buffer
    print(f"Verarbeite Daten: {len(imu1_data_buffer)} Pakete von IMU1, {len(imu2_data_buffer)} Pakete von IMU2")
    
    if len(imu1_data_buffer) == 0 and len(imu2_data_buffer) == 0:
        print("Keine Daten gesammelt. Überspringe Speichern.")
        return

    # In Pandas DataFrames konvertieren
    df1 = pd.DataFrame(imu1_data_buffer)
    df2 = pd.DataFrame(imu2_data_buffer)
    
    # Prefix hinzufügen, um die Werte beider Sensoren unterscheiden zu können
    if not df1.empty:
        df1 = df1.add_prefix('IMU1_')
        df1.rename(columns={'IMU1_pc_timestamp': 'pc_timestamp'}, inplace=True)
    if not df2.empty:
        df2 = df2.add_prefix('IMU2_')
        df2.rename(columns={'IMU2_pc_timestamp': 'pc_timestamp'}, inplace=True)

    # Zusammenführen beider DataFrames anhand des pc_timestamp 
    if not df1.empty and not df2.empty:
        # Sortieren ist Pflicht für AsOf-Merge
        df1 = df1.sort_values('pc_timestamp')
        df2 = df2.sort_values('pc_timestamp')
        
        # Fügt df2 an df1 an, basierend auf dem zeitlich nächsten pc_timestamp (Toleranz: 100ms)
        try:
            merged_df = pd.merge_asof(df1, df2, on='pc_timestamp', direction='nearest', tolerance=0.1)
        except Exception as e:
            print(f"Fehler beim Zusammenführen der Zeilen: {e}")
            merged_df = df1
    else:
        merged_df = df1 if not df1.empty else df2
        
    # PC-Timestamp relativ auf 0 setzen (macht den Plot lesbarer)
    if not merged_df.empty:
        start_time = merged_df['pc_timestamp'].iloc[0]
        merged_df['time_rel_s'] = merged_df['pc_timestamp'] - start_time
        
    # Ordner für Datensätze erstellen
    gesture_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", GESTURE_LABELS[current_gesture]))
    os.makedirs(gesture_dir, exist_ok=True)
    filename = os.path.join(gesture_dir, f"gesture_record_{int(time.time())}.csv")
    
    merged_df.to_csv(filename, index=False)
    print(f"Datensatz gespeichert unter: {filename}\n")
    
    # Plot anzeigen
    plot_data(merged_df)


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
    global recording, running, imu1_data_buffer, imu2_data_buffer
    
    # Sensoren initialisieren
    imu1 = IMUDataInput(port=PORT_IMU1, baudrate=BAUDRATE, name="IMU1")
    imu2 = IMUDataInput(port=PORT_IMU2, baudrate=BAUDRATE, name="IMU2")
    
    # Reader-Threads starten
    imu1.start()
    imu2.start()
    
    # Den Konsolen-Input in einem separaten Thread starten
    control_thread = threading.Thread(target=input_thread, args=(imu1, imu2), daemon=True)
    control_thread.start()
    
    try:
        # Haupt-Schleife: Pollt die Sensor-Queues
        while running:
            # Puffer regelmäßig leeren, unabhängig ob aufgenommen wird
            # Das verhindert, dass der RAM mit der Zeit vollläuft.
            data1 = imu1.get_data()
            data2 = imu2.get_data()
            
            # Falls wir gerade aufnehmen, hängen wir sie an unsere Speicher-Puffer an
            if recording:
                imu1_data_buffer.extend(data1)
                imu2_data_buffer.extend(data2)
            
            # Kurze Pause um CPU-Last gering zu halten
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\nManuell abgebrochen (STRG+C).")
        running = False
        
    finally:
        # Aufräumen nicht vergessen
        imu1.stop()
        imu2.stop()
        print("Programm beendet.")

if __name__ == "__main__":
    main()