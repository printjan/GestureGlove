import pandas as pd
import numpy as np
import logging

# Logger konfigurieren
logger = logging.getLogger("IMU_Sync")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def align_timestamps(df1, df2):
    '''
    Richtet zwei DataFrames auf eine gemeinsame Zeitachse aus.
    Nutzt den esp_timestamp_us als Basis, aber gleicht den Offset
    beider Sensoren relativ zum PC-Timestamp ab.
    '''
    if df1.empty or df2.empty:
        return df1, df2
        
    pc_start1 = df1['pc_timestamp_us'].iloc[0]
    esp_start1 = df1['esp_timestamp_us'].iloc[0]
    
    pc_start2 = df2['pc_timestamp_us'].iloc[0]
    esp_start2 = df2['esp_timestamp_us'].iloc[0]
    
    anchor_pc = min(pc_start1, pc_start2)
    
    # Berechne die absolute Zeit auf Basis der esp_timestamps plus Offset zum pc_start
    # Dadurch bleibt die Stabilität des ESP erhalten, aber sie sind untereinander snychronisiert.
    t1 = (df1['esp_timestamp_us'] - esp_start1) + (pc_start1 - anchor_pc)
    t2 = (df2['esp_timestamp_us'] - esp_start2) + (pc_start2 - anchor_pc)
    
    df1_aligned = df1.copy()
    df2_aligned = df2.copy()
    df1_aligned['sync_time_us'] = t1
    df2_aligned['sync_time_us'] = t2
    
    return df1_aligned, df2_aligned

def interpolate_and_merge(df1, df2, freq_hz=100):
    '''
    Interpoliert die Daten beider Sensoren linear auf eine gemeinsame
    Zeitbasis mit der angegebenen Frequenz.
    '''
    if df1.empty or df2.empty:
        return pd.DataFrame()
        
    start_time = max(df1['sync_time_us'].iloc[0], df2['sync_time_us'].iloc[0])
    end_time = min(df1['sync_time_us'].iloc[-1], df2['sync_time_us'].iloc[-1])
    
    # Gemeinsame Zeitachse erstellen
    period_us = int(1e6 / freq_hz)
    common_time = np.arange(start_time, end_time, period_us)
    common_df = pd.DataFrame({'sync_time_us': common_time})
    
    # Hilfsfunktion zur Interpolation
    def interp_sensor(df, suffix):
        # Numeric columns only
        num_cols = df.select_dtypes(include=[np.number]).columns
        # drop non-sensor cols if needed, but we can just interpolate all num_cols
        df_num = df[['sync_time_us'] + [c for c in num_cols if c != 'sync_time_us']]
        
        # Merge-basis
        merged = pd.merge(common_df, df_num, on='sync_time_us', how='outer').sort_values('sync_time_us')
        merged = merged.set_index('sync_time_us')
        # Lineare Interpolation
        merged = merged.interpolate(method='index')
        # Zurück auf common_time filtern
        merged = merged.loc[common_time].reset_index()
        # Prefix anbringen
        merged = merged.add_prefix(suffix)
        merged = merged.rename(columns={f'{suffix}sync_time_us': 'sync_time_us'})
        return merged
        
    df1_interp = interp_sensor(df1, 'IMU1_')
    df2_interp = interp_sensor(df2, 'IMU2_')
    
    # Zusammenführen
    merged_df = pd.merge(df1_interp, df2_interp, on='sync_time_us', how='inner')
    
    return merged_df

def window_data(merged_df, original_df1, original_df2, window_size_samples, max_time_diff_us=5000):
    '''
    Unterteilt den Datensatz in Fenster.
    Prüft die tatsächliche zeitliche Abweichung der Originaldaten innerhalb des Fensters.
    Falls Abweichung > max_time_diff_us, wird das Fenster markiert/verworfen.
    '''
    windows = []
    
    num_samples = len(merged_df)
    for start_idx in range(0, num_samples, window_size_samples):
        end_idx = start_idx + window_size_samples
        if end_idx > num_samples:
            break
            
        window = merged_df.iloc[start_idx:end_idx]
        t_start = window['sync_time_us'].iloc[0]
        t_end = window['sync_time_us'].iloc[-1]
        
        # Originaldaten in diesem Zeitraum suchen
        orig1_mask = (original_df1['sync_time_us'] >= t_start) & (original_df1['sync_time_us'] <= t_end)
        orig2_mask = (original_df2['sync_time_us'] >= t_start) & (original_df2['sync_time_us'] <= t_end)
        
        t1 = original_df1.loc[orig1_mask, 'sync_time_us'].values
        t2 = original_df2.loc[orig2_mask, 'sync_time_us'].values
        
        # Maximale Abweichung der nächsten Samples schätzen (vereinfacht)
        # Besser: wir berechnen im Voraus die Nearest-Neighbor-Distanz
        # und nehmen den maximalen Abstand im Fenster.
        
        window_valid = True
        max_diff = 0
        if len(t1) == 0 or len(t2) == 0:
            window_valid = False
        else:
            # Für jedes t1 das nächste t2 finden
            # (In performanterem Code würde man pd.merge_asof nutzen und vorher berechnen)
            diffs = [np.min(np.abs(t2 - t)) for t in t1]
            max_diff = np.max(diffs) if diffs else 0
            
            if max_diff > max_time_diff_us:
                window_valid = False
                
        logger.info(f"Fenster {len(windows)}: Diff {max_diff:.1f} us -> Valid: {window_valid}")
        
        if window_valid:
            windows.append(window)
            
    return windows

def process_stream(df1, df2, window_sz=50, max_diff_us=5000, freq_hz=100):
    '''
    Kombinierte Pipeline für Echtzeit / Offline:
    1. Align
    2. Interpolate
    3. Window & Validate
    '''
    df1_a, df2_a = align_timestamps(df1, df2)
    merged = interpolate_and_merge(df1_a, df2_a, freq_hz)
    windows = window_data(merged, df1_a, df2_a, window_sz, max_diff_us)
    return merged, windows
