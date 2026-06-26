# src/data_fusion_project/recording/__init__.py
"""
Data recording subpackage: serial capture and synchronization for the dual-IMU rig.

This package contains the importable building blocks used by the recording entry point
``scripts/record_data.py``:

- :class:`IMUDataInput`       threaded serial receiver for a single ESP32 IMU stream
- :func:`process_stream`      align, resample and window two raw IMU streams onto a grid
- device resolution helpers   map configured boards to concrete serial ports

Quick start
-----------
>>> from data_fusion_project.recording import IMUDataInput, process_stream, resolve_device_port
>>> port = resolve_device_port("imu1")
"""

from data_fusion_project.recording.input_data import IMUDataInput
from data_fusion_project.recording.sync import (
    SyncDiagnostics,
    align_timestamps,
    interpolate_and_merge,
    window_data,
    process_stream,
)
from data_fusion_project.recording.device_resolution import (
    load_device_config,
    available_serial_ports,
    print_available_serial_ports,
    resolve_device_port,
)

__all__ = [
    # serial input
    "IMUDataInput",
    # synchronization
    "SyncDiagnostics",
    "align_timestamps",
    "interpolate_and_merge",
    "window_data",
    "process_stream",
    # device resolution
    "load_device_config",
    "available_serial_ports",
    "print_available_serial_ports",
    "resolve_device_port",
]
