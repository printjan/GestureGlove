# src/data_fusion_project/processing/filters.py
"""
Digital signal filters for IMU pre-processing.

Implements the filters listed in the project README:
- Low-pass filtering   (noise reduction)
- High-pass filtering   (drift / offset removal)
- Band-pass filtering    (combination of the two)
- Gravity removal         (split accelerometer into gravity + linear acceleration)

All filters are zero-phase (forward-backward, ``scipy.signal.sosfiltfilt``) so that they
introduce no time shift, which matters for downstream windowing and orientation fusion.
Filters operate column-wise on ``(T, C)`` arrays along the time axis.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from data_fusion_project.core.logger_setup import get_logger
from data_fusion_project.processing.config import FilterType

logger = get_logger(__name__)


# ======================================================================================================================
# Core Butterworth helpers
# ======================================================================================================================
def _design_sos(filter_type: FilterType, cutoff_hz, fs: float, order: int) -> np.ndarray | None:
    """
    Designs a Butterworth filter as second-order sections (SOS) for numerical stability.
    :param: filter_type (FilterType): low-/high-/band-pass selector.
    :param: cutoff_hz (float | tuple): cutoff frequency in Hz (tuple for band-pass).
    :param: fs (float): sampling frequency in Hz.
    :param: order (int): filter order.
    :return: sos (np.ndarray | None): SOS coefficients, or None if no filtering applies.
    """
    nyquist = 0.5 * fs

    if filter_type == FilterType.LOWPASS:
        wn = float(cutoff_hz) / nyquist
        btype = "lowpass"
    elif filter_type == FilterType.HIGHPASS:
        wn = float(cutoff_hz) / nyquist
        btype = "highpass"
    elif filter_type == FilterType.BANDPASS:
        low, high = cutoff_hz
        wn = [float(low) / nyquist, float(high) / nyquist]
        btype = "bandpass"
    else:
        return None

    # Clip normalized cutoffs into the open interval (0, 1) to keep the design valid.
    wn = np.clip(wn, 1e-6, 1.0 - 1e-6)
    return signal.butter(order, wn, btype=btype, output="sos")


def _sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Applies zero-phase filtering along the time axis, guarding against short signals.
    :param: sos (np.ndarray): second-order-section coefficients.
    :param: x (np.ndarray): input array of shape (T,) or (T, C).
    :return: filtered (np.ndarray): filtered array with the same shape as the input.
    """
    n = x.shape[0]
    # sosfiltfilt requires the signal to be longer than the internal padding length.
    default_padlen = 3 * (2 * sos.shape[0] + 1)
    if n <= default_padlen:
        padlen = max(0, n - 1)
        logger.debug("Signal length %d <= default padlen %d; reducing padlen to %d.", n, default_padlen, padlen)
    else:
        padlen = default_padlen
    return signal.sosfiltfilt(sos, x, axis=0, padlen=padlen)


# ======================================================================================================================
# Public filter API
# ======================================================================================================================
def apply_filter(x: np.ndarray, filter_type: FilterType, cutoff_hz, fs: float, order: int = 2) -> np.ndarray:
    """
    Filters a signal block with the given Butterworth filter.
    :param: x (np.ndarray): input of shape (T,) or (T, C); filtered along axis 0.
    :param: filter_type (FilterType): filter selector; FilterType.NONE returns the input unchanged.
    :param: cutoff_hz (float | tuple): cutoff frequency/frequencies in Hz.
    :param: fs (float): sampling frequency in Hz.
    :param: order (int): Butterworth filter order.
    :return: filtered (np.ndarray): filtered copy of the input (or the input if no filtering applies).
    """
    if filter_type == FilterType.NONE:
        return np.asarray(x, dtype=float)

    sos = _design_sos(filter_type, cutoff_hz, fs, order)
    if sos is None:
        return np.asarray(x, dtype=float)

    return _sosfiltfilt(sos, np.asarray(x, dtype=float))


def lowpass(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """
    Low-pass filters a signal for noise reduction.
    :param: x (np.ndarray): input of shape (T,) or (T, C).
    :param: cutoff_hz (float): cutoff frequency in Hz.
    :param: fs (float): sampling frequency in Hz.
    :param: order (int): filter order.
    :return: filtered (np.ndarray): low-pass filtered signal.
    """
    return apply_filter(x, FilterType.LOWPASS, cutoff_hz, fs, order)


def highpass(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """
    High-pass filters a signal to remove slow drift and constant offsets.
    :param: x (np.ndarray): input of shape (T,) or (T, C).
    :param: cutoff_hz (float): cutoff frequency in Hz.
    :param: fs (float): sampling frequency in Hz.
    :param: order (int): filter order.
    :return: filtered (np.ndarray): high-pass filtered signal.
    """
    return apply_filter(x, FilterType.HIGHPASS, cutoff_hz, fs, order)


def bandpass(x: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 2) -> np.ndarray:
    """
    Band-pass filters a signal between two cutoff frequencies.
    :param: x (np.ndarray): input of shape (T,) or (T, C).
    :param: low_hz (float): lower cutoff frequency in Hz.
    :param: high_hz (float): upper cutoff frequency in Hz.
    :param: fs (float): sampling frequency in Hz.
    :param: order (int): filter order.
    :return: filtered (np.ndarray): band-pass filtered signal.
    """
    return apply_filter(x, FilterType.BANDPASS, (low_hz, high_hz), fs, order)


def estimate_gravity(acc: np.ndarray, fs: float, cutoff_hz: float = 0.5, order: int = 2) -> np.ndarray:
    """
    Estimates the slowly-varying gravity component of an accelerometer signal via low-pass.
    :param: acc (np.ndarray): accelerometer block of shape (T, 3).
    :param: fs (float): sampling frequency in Hz.
    :param: cutoff_hz (float): low-pass cutoff isolating the gravity component (Hz).
    :param: order (int): filter order.
    :return: gravity (np.ndarray): estimated gravity component, shape (T, 3).
    """
    return lowpass(acc, cutoff_hz, fs, order)


def remove_gravity(acc: np.ndarray, fs: float, cutoff_hz: float = 0.5, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits an accelerometer signal into gravity and gravity-free linear acceleration.
    :param: acc (np.ndarray): accelerometer block of shape (T, 3).
    :param: fs (float): sampling frequency in Hz.
    :param: cutoff_hz (float): low-pass cutoff isolating gravity (Hz).
    :param: order (int): filter order.
    :return: result (tuple): (linear_acc, gravity), each of shape (T, 3).
    """
    gravity = estimate_gravity(acc, fs, cutoff_hz, order)
    linear = np.asarray(acc, dtype=float) - gravity
    return linear, gravity
