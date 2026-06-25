import serial
import time
import threading
import queue
from queue import Queue

class IMUDataInput:
    def __init__(self, port, baudrate=115200, name="IMU"):
        self.port = port
        self.baudrate = baudrate
        self.name = name
        self.ser = None
        self.data_queue = Queue()
        self.running = False
        self.thread = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"[{self.name}] Connected to {self.port} at {self.baudrate} baud.")
            time.sleep(2)  # Wait for the serial connection to initialize
            return True
        except serial.SerialException as e:
            print(f"[{self.name}] Connection error: {e}")
            return False

    def start(self):
        if not self.ser:
            if not self.connect():
                return
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(f"[{self.name}] Reading thread started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.ser:
            self.ser.close()
        print(f"[{self.name}] Stopped and disconnected.")

    def _read_loop(self):
        # Clear buffer before starting
        self.ser.reset_input_buffer()
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                pc_timestamp = time.time()
                
                # Expected format now: timestamp_us, accX, accY, accZ, gyrX, gyrY, gyrZ
                data = line.split(',')
                if len(data) == 7:
                    try:
                        esp_us = int(data[0])
                        ax, ay, az, gx, gy, gz = map(float, data[1:])

                        # Nur rohe Sensorwerte speichern. Die Timestamps werden
                        # ausschließlich zur Synchronisation beider IMUs genutzt.
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
                print(f"[{self.name}] Error reading from serial: {e}")
                time.sleep(0.1)

    def get_data(self):
        """Returns all currently buffered data from the queue."""
        data_list = []
        while True:
            try:
                data_list.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data_list

if __name__ == "__main__":
    # Test script - just to confirm the class works standalone
    imu = IMUDataInput(port='COM11')
    imu.start()
    try:
        while True:
            data = imu.get_data()
            if data:
                #print(f"Received {len(data)} packets from queue. Latest timestamp: {data[-1]['esp_ms']} ms")
                print(data[-1])  # Print the latest packet
            time.sleep(0.1)
    except KeyboardInterrupt:
        imu.stop()
    
