import serial
import time
import threading
import math
import queue
from queue import Queue

class SimpleKalmanFilter:
    def __init__(self, err_measure, err_estimate, q):
        self.err_measure = err_measure
        self.err_estimate = err_estimate
        self.q = q
        self.current_estimate = 0.0
        self.last_estimate = 0.0
        self.kalman_gain = 0.0

    def update_estimate(self, mea):
        self.kalman_gain = self.err_estimate / (self.err_estimate + self.err_measure)
        self.current_estimate = self.last_estimate + self.kalman_gain * (mea - self.last_estimate)
        self.err_estimate = (1.0 - self.kalman_gain) * self.err_estimate + abs(self.last_estimate - self.current_estimate) * self.q
        self.last_estimate = self.current_estimate
        return self.current_estimate

class IMUDataInput:
    def __init__(self, port, baudrate=115200, name="IMU"):
        self.port = port
        self.baudrate = baudrate
        self.name = name
        self.ser = None
        self.data_queue = Queue()
        self.running = False
        self.thread = None
        
        # Kalman Filter für Roll und Pitch
        self.kf_roll = SimpleKalmanFilter(0.1, 0.1, 0.05)
        self.kf_pitch = SimpleKalmanFilter(0.1, 0.1, 0.05)
        
    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
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
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    pc_timestamp = time.time()
                    
                    # Expected format now: timestamp_us, accX, accY, accZ, gyrX, gyrY, gyrZ
                    data = line.split(',')
                    if len(data) == 7:
                        try:
                            esp_us = int(data[0])
                            ax, ay, az, gx, gy, gz = map(float, data[1:])
                            
                            # Roll und Pitch berechnen
                            roll = math.atan2(ay, az) * 180.0 / math.pi
                            pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az)) * 180.0 / math.pi
                            
                            # Mit Kalman Filter glätten
                            roll_kf = self.kf_roll.update_estimate(roll)
                            pitch_kf = self.kf_pitch.update_estimate(pitch)
                            
                            packed_data = {
                                'sensor_id': self.name,
                                'pc_timestamp_us': int(pc_timestamp * 1e6),
                                'esp_timestamp_us': esp_us,
                                'accX': ax, 'accY': ay, 'accZ': az,
                                'gyrX': gx, 'gyrY': gy, 'gyrZ': gz,
                                'roll': roll, 'pitch': pitch,
                                'roll_kf': roll_kf, 'pitch_kf': pitch_kf
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
    
