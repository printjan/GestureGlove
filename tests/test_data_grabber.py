#!/usr/bin/env python
# tests/test_data_grabber.py
"""
Unit tests for the AsynchronousDataGrabber class.
Exercises alignment, preprocessing, thread lifecycle, and health checking.
"""

import sys
import time
import threading
import numpy as np
import pandas as pd
from pathlib import Path

# Add project src/ directory to the python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_fusion_project.processing import PipelineConfig
from data_fusion_project.processing.calibration import identity_profile
from data_fusion_project.inference.data_grabber import AsynchronousDataGrabber


class MockIMUInput:
    def __init__(self, name):
        self.name = name
        self.running = False
        self._data = []
        self._lock = threading.Lock()

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def get_data(self):
        with self._lock:
            data = list(self._data)
            self._data.clear()
            return data

    def feed_packets(self, packets):
        with self._lock:
            self._data.extend(packets)


def make_imu_packet(name, timestamp_us, acc_z=1.0):
    return {
        'sensor_id': name,
        'pc_timestamp_us': timestamp_us,
        'esp_timestamp_us': timestamp_us,
        'imu_timestamp_ms': int(timestamp_us / 1000),
        'accX': 0.0, 'accY': 0.0, 'accZ': acc_z,
        'gyrX': 0.0, 'gyrY': 0.0, 'gyrZ': 0.0,
    }


def test_grabber_lifecycle():
    print("Testing grabber thread start/stop lifecycle...")
    imu1 = MockIMUInput("IMU1")
    imu2 = MockIMUInput("IMU2")
    imu1.start()
    imu2.start()

    config = PipelineConfig()
    profile = identity_profile()

    grabber = AsynchronousDataGrabber(
        imu1=imu1,
        imu2=imu2,
        pipeline_config=config,
        calibration_profile=profile,
        window_size_samples=150,
        advance_samples=10,
    )

    assert not grabber.check_health(), "Grabber should not be healthy before start"
    
    grabber.start()
    time.sleep(0.05)
    
    assert grabber.check_health(), "Grabber should be healthy after start"
    
    grabber.stop()
    time.sleep(0.05)
    
    assert not grabber.check_health(), "Grabber should not be healthy after stop"
    print("Lifecycle test PASSED.")


def test_grabber_alignment_and_processing():
    print("Testing stream alignment and callback processing...")
    imu1 = MockIMUInput("IMU1")
    imu2 = MockIMUInput("IMU2")
    imu1.start()
    imu2.start()

    config = PipelineConfig()
    profile = identity_profile()

    transformed_frames = []

    def dummy_transform(channels, channel_names):
        transformed_frames.append((channels, channel_names))
        return "transformed_dummy_frame"

    grabber = AsynchronousDataGrabber(
        imu1=imu1,
        imu2=imu2,
        pipeline_config=config,
        calibration_profile=profile,
        window_size_samples=100, # smaller window size to speed up test
        advance_samples=10,
        transform_fn=dummy_transform,
        poll_interval_s=0.005,
    )

    grabber.start()

    # Feed packets (100 Hz). Need 100 packets to fill the first window
    start_time_us = int(time.time() * 1e6)
    packets1 = [make_imu_packet("IMU1", start_time_us + i * 10000) for i in range(110)]
    packets2 = [make_imu_packet("IMU2", start_time_us + i * 10000) for i in range(110)]

    imu1.feed_packets(packets1)
    imu2.feed_packets(packets2)

    # Allow grabber to process
    time.sleep(0.2)

    frame = grabber.get_newest_frame(block=False)
    assert frame == "transformed_dummy_frame", "Failed to retrieve preprocessed frame"
    assert len(transformed_frames) > 0, "Transform callback was not executed"
    
    channels, channel_names = transformed_frames[0]
    assert channels.shape[0] == 100, f"Expected 100 samples in window, got {channels.shape[0]}"
    assert "IMU1_accX" in channel_names, "Channel names list is missing expected columns"

    grabber.stop()
    print("Alignment and processing test PASSED.")


def test_grabber_health_monitoring():
    print("Testing sensor health monitoring...")
    imu1 = MockIMUInput("IMU1")
    imu2 = MockIMUInput("IMU2")
    imu1.start()
    imu2.start()

    config = PipelineConfig()
    profile = identity_profile()

    grabber = AsynchronousDataGrabber(
        imu1=imu1,
        imu2=imu2,
        pipeline_config=config,
        calibration_profile=profile,
    )

    grabber.start()
    time.sleep(0.05)
    assert grabber.check_health()

    # Simulate sensor disconnection
    imu2.stop()
    time.sleep(0.05)

    assert not grabber.check_health(), "Grabber should detect sensor disconnection"

    # Verifying get_newest_frame raises RuntimeError on unhealthy grabber
    try:
        grabber.get_newest_frame(block=False)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

    grabber.stop()
    print("Health monitoring test PASSED.")


def test_grabber_blocking_behavior():
    print("Testing grabber blocking/timeout behavior...")
    imu1 = MockIMUInput("IMU1")
    imu2 = MockIMUInput("IMU2")
    imu1.start()
    imu2.start()

    config = PipelineConfig()
    profile = identity_profile()

    grabber = AsynchronousDataGrabber(
        imu1=imu1,
        imu2=imu2,
        pipeline_config=config,
        calibration_profile=profile,
        window_size_samples=50,
        advance_samples=5,
        poll_interval_s=0.005,
    )

    grabber.start()

    # Call get_newest_frame on empty buffer with timeout, should return None
    frame = grabber.get_newest_frame(block=True, timeout=0.05)
    assert frame is None, "Expected timeout to return None"

    # Now feed packets asynchronously in another thread to unblock
    def feed_after_delay():
        time.sleep(0.05)
        start_time_us = int(time.time() * 1e6)
        packets1 = [make_imu_packet("IMU1", start_time_us + i * 10000) for i in range(60)]
        packets2 = [make_imu_packet("IMU2", start_time_us + i * 10000) for i in range(60)]
        imu1.feed_packets(packets1)
        imu2.feed_packets(packets2)

    feeder = threading.Thread(target=feed_after_delay)
    feeder.start()

    # Block and wait for frame
    t_start = time.time()
    frame = grabber.get_newest_frame(block=True, timeout=0.5)
    t_duration = time.time() - t_start

    assert frame is not None, "Failed to wait and grab frame"
    assert 0.04 < t_duration < 0.3, f"Expected to wait briefly, took {t_duration}s"

    grabber.stop()
    feeder.join()
    print("Blocking behavior test PASSED.")


def run_all_tests():
    test_grabber_lifecycle()
    test_grabber_alignment_and_processing()
    test_grabber_health_monitoring()
    test_grabber_blocking_behavior()
    print("All tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()
