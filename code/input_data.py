# code/input_data.py
"""
IMU input module that manages threaded serial connection reading.

Input:
Serial connection from ESP32 streaming IMU packets.
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================
import serial
import time
import threading
import queue
from queue import Queue
from data_fusion_project.core.logger_setup import get_logger

logger = get_logger("IMU_Input")


# ======================================================================================================================
# IMUDataInput Class
# ======================================================================================================================
class IMUDataInput:
    """
    Manages threaded serial communication to receive IMU packets.
    """
    def __init__(self, port, baudrate=115200, name="IMU"):
        self.port = port
        self.baudrate = baudrate
        self.name = name
        self.ser = None
        self.data_queue = Queue()
        self.running = False
        self.thread = None

    def connect(self) -> bool:
        """
        Establishes connection to the serial port.
        :return: status (bool): True if connection succeeded, False otherwise.
        """
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            logger.info("[%s] Connected to %s at %s baud.", self.name, self.port, self.baudrate)
            time.sleep(2)  # Wait for the serial connection to initialize
            return True
        except serial.SerialException as e:
            logger.error("[%s] Connection error: %s", self.name, e)
            return False

    def start(self) -> None:
        """
        Starts the background reading thread.
        :return: None:
        """
        if not self.ser:
            if not self.connect():
                return
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("[%s] Reading thread started.", self.name)

    def stop(self) -> None:
        """
        Stops the background reading thread and closes the serial connection.
        :return: None:
        """
        self.running = False
        if self.thread:
            self.thread.join()
        if self.ser:
            self.ser.close()
        logger.info("[%s] Stopped and disconnected.", self.name)

    def _read_loop(self) -> None:
        """
        Continuous loop running in a background thread to read from serial.
        Sets self.running to False and breaks on connection errors to fail fast.
        :return: None:
        """
        # Clear buffer before starting
        self.ser.reset_input_buffer()
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                pc_timestamp = time.time()
                
                # Expected format: timestamp_us, accX, accY, accZ, gyrX, gyrY, gyrZ
                data = line.split(',')
                if len(data) == 7:
                    try:
                        esp_us = int(data[0])
                        ax, ay, az, gx, gy, gz = map(float, data[1:])

                        packed_data = {
                            'sensor_id': self.name,
                            'pc_timestamp_us': int(pc_timestamp * 1e6),
                            'esp_timestamp_us': esp_us,
                            'accX': ax, 'accY': ay, 'accZ': az,
                            'gyrX': gx, 'gyrY': gy, 'gyrZ': gz,
                        }
                        self.data_queue.put(packed_data)
                    except ValueError:
                        pass # Ignore parse errors (e.g. malformed serial line)
            except Exception as e:
                logger.error("[%s] Error reading from serial: %s", self.name, e)
                self.running = False
                break

    def get_data(self) -> list:
        """
        Returns all currently buffered data from the queue.
        :return: data_list (list): list of collected IMU packets.
        """
        data_list = []
        while True:
            try:
                data_list.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data_list
