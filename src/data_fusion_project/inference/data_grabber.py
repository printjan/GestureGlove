# src/data_fusion_project/inference/data_grabber.py
"""
Asynchronous Data Grabber for real-time sensor processing and inference decoupling.
"""

import threading
import time
from typing import Callable, Optional, Tuple, Any
import numpy as np
import pandas as pd

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.recording.input_data import IMUDataInput
from data_fusion_project.recording.sync import process_stream
from data_fusion_project.processing import process_window, PipelineConfig, CalibrationProfile

logger = get_logger("DataGrabber")


class AsynchronousDataGrabber:
    """
    Asynchronous Data Grabber that runs a background thread to fetch raw IMU data,
    aligns the streams, pre-processes the sliding windows, and stores the latest frame
    for the inference system.
    """
    def __init__(
        self,
        imu1: IMUDataInput,
        imu2: IMUDataInput,
        pipeline_config: PipelineConfig,
        calibration_profile: CalibrationProfile,
        window_size_samples: int = 150,
        advance_samples: int = 10,
        freq_hz: float = 100.0,
        max_diff_us: int = 10000,
        transform_fn: Optional[Callable[[np.ndarray, list[str]], Any]] = None,
        poll_interval_s: float = 0.01,
        enable_zupt: bool = True,
    ) -> None:
        self.imu1 = imu1
        self.imu2 = imu2
        self.pipeline_config = pipeline_config
        self.calibration_profile = calibration_profile
        self.window_size_samples = window_size_samples
        self.advance_samples = advance_samples
        self.freq_hz = freq_hz
        self.max_diff_us = max_diff_us
        self.transform_fn = transform_fn
        self.poll_interval_s = poll_interval_s
        self.enable_zupt = enable_zupt

        self.window_us = int((self.window_size_samples / self.freq_hz) * 1e6)
        self.advance_us = int((self.advance_samples / self.freq_hz) * 1e6)
        self._last_zupt_log_time = 0.0

        # Thread-safe slot for the latest preprocessed frame/data
        self._latest_frame: Optional[Any] = None
        self._frame_lock = threading.Lock()
        self._new_frame_event = threading.Event()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the data grabber background thread."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="DataGrabberThread")
        self._thread.start()
        logger.info("Asynchronous Data Grabber thread started.")

    def stop(self) -> None:
        """Stops the data grabber background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        logger.info("Asynchronous Data Grabber thread stopped.")

    def check_health(self) -> bool:
        """Checks if the background thread and both sensors are running."""
        if not self._running or self._thread is None or not self._thread.is_alive():
            return False
        if not self.imu1.running or not self.imu2.running:
            return False
        return True

    def _trim_before(self, buf: list, cutoff_us: int) -> None:
        i = 0
        while i < len(buf) and buf[i]['pc_timestamp_us'] < cutoff_us:
            i += 1
        if i:
            del buf[:i]

    def _run_loop(self) -> None:
        local_buf1 = []
        local_buf2 = []
        next_start_us = None

        # Clean/drain inputs before starting
        self.imu1.get_data()
        self.imu2.get_data()

        while self._running:
            time.sleep(self.poll_interval_s)

            # Check sensor health to fail fast
            if not self.imu1.running or not self.imu2.running:
                logger.error("One or both sensors stopped running. Exiting data grabber thread.")
                break

            # Fetch new packets
            local_buf1.extend(self.imu1.get_data())
            local_buf2.extend(self.imu2.get_data())

            if not local_buf1 or not local_buf2:
                continue

            if next_start_us is None:
                next_start_us = max(local_buf1[0]['pc_timestamp_us'], local_buf2[0]['pc_timestamp_us'])

            latest_us = min(local_buf1[-1]['pc_timestamp_us'], local_buf2[-1]['pc_timestamp_us'])
            
            # Process all ready windows to catch up
            while next_start_us is not None and latest_us - next_start_us >= self.window_us:
                df1 = pd.DataFrame(local_buf1)
                df2 = pd.DataFrame(local_buf2)
                merged_win, valid_windows = process_stream(
                    df1, df2, window_sz=self.window_size_samples, max_diff_us=self.max_diff_us, freq_hz=self.freq_hz
                )

                if valid_windows:
                    window_df = valid_windows[0]
                    
                    # --- Zero-Velocity Updates (ZUPT) background calibration ---
                    if self.enable_zupt:
                        beta = 0.1  # EMA smoothing factor
                        gyro_std_threshold = 3.0   # dps
                        acc_std_threshold = 0.025  # g
                        
                        for imu in ["IMU1", "IMU2"]:
                            acc_cols = [f"{imu}_accX", f"{imu}_accY", f"{imu}_accZ"]
                            gyr_cols = [f"{imu}_gyrX", f"{imu}_gyrY", f"{imu}_gyrZ"]
                            if not all(c in window_df.columns for c in acc_cols + gyr_cols):
                                continue
                            
                            acc_raw = window_df[acc_cols].to_numpy(dtype=float)
                            gyr_raw = window_df[gyr_cols].to_numpy(dtype=float)
                            
                            acc_std = np.std(acc_raw, axis=0)
                            gyr_std = np.std(gyr_raw, axis=0)
                            
                            if np.all(acc_std < acc_std_threshold) and np.all(gyr_std < gyro_std_threshold):
                                # Measured raw gyro bias
                                measured_gyro_bias = np.mean(gyr_raw, axis=0)
                                
                                from data_fusion_project.processing.calibration import ImuCalibration
                                calib = self.calibration_profile.per_imu.get(imu)
                                if calib is None:
                                    calib = ImuCalibration()
                                    self.calibration_profile.per_imu[imu] = calib
                                    
                                old_bias = calib.gyro_bias.copy()
                                calib.gyro_bias = (1 - beta) * old_bias + beta * measured_gyro_bias
                                
                                now = time.time()
                                if now - self._last_zupt_log_time > 1.0:
                                    logger.info("[%s ZUPT] Stillness detected. Updated gyro bias from %s to %s",
                                                imu, np.round(old_bias, 3), np.round(calib.gyro_bias, 3))
                                    import sys
                                    sys.stdout.write(f"\n\033[94m[ZUPT] Stillness detected. Recalibrated {imu} gyro bias to {np.round(calib.gyro_bias, 3)}\033[0m\n")
                                    sys.stdout.flush()
                                    self._last_zupt_log_time = now
                    
                    try:
                        channels, channel_names, _, _ = process_window(
                            window_df, self.calibration_profile, self.pipeline_config
                        )
                        frame = (channels, channel_names)
                        if self.transform_fn is not None:
                            frame = self.transform_fn(channels, channel_names)

                        with self._frame_lock:
                            self._latest_frame = frame
                            self._new_frame_event.set()
                    except Exception as e:
                        logger.error(f"Error processing window: {e}")

                next_start_us += self.advance_us
                self._trim_before(local_buf1, next_start_us)
                self._trim_before(local_buf2, next_start_us)

    def get_newest_frame(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Retrieves the newest preprocessed data frame.
        If block is True, it waits up to timeout seconds for a new frame.
        Once retrieved, the new frame event is cleared, meaning subsequent calls
        without new data will block or return None.
        """
        if not self.check_health():
            raise RuntimeError("Asynchronous Data Grabber or IMU sensors are not running.")

        if block:
            signaled = self._new_frame_event.wait(timeout=timeout)
            if not signaled:
                return None

        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()
            return frame
