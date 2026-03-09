"""Utility functions for applying a band‑pass filter to one‑dimensional signals.

The functions in this module leverage SciPy's digital filter design
and filtering capabilities to isolate the frequency band associated
with the human pulse. A typical heart rate falls between roughly
0.75 Hz (45 beats per minute) and 4 Hz (240 beats per minute). When
processing remote photoplethysmography (rPPG) signals it is common
to apply a band‑pass filter to remove low‑frequency drift and
high‑frequency noise.

Example usage:

    from utils.bandpass_filter import bandpass_filter
    filtered = bandpass_filter(signal, fs=30, low=0.75, high=4.0)

"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = 0.75,
    high: float = 4.0,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth band‑pass filter to the input signal.

    Parameters
    ----------
    signal : np.ndarray
        One‑dimensional array containing the raw rPPG signal.
    fs : float
        Sampling frequency of the signal in Hertz. This should
        correspond to the frame rate of the video from which the
        signal was extracted.
    low : float, optional
        Lower cutoff frequency in Hertz. Defaults to 0.75 Hz.
    high : float, optional
        Upper cutoff frequency in Hertz. Defaults to 4.0 Hz.
    order : int, optional
        Order of the Butterworth filter. Higher orders have
        steeper roll‑off but may introduce more phase distortion.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as the input.
    """
    if signal.size == 0:
        return signal
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    # Ensure the cutoff frequencies are within (0, 1)
    low_norm = max(low_norm, 1e-5)
    high_norm = min(high_norm, 0.999)
    b, a = butter(order, [low_norm, high_norm], btype="bandpass")
    min_len = 3 * max(len(a), len(b))
    if signal.size <= min_len:
        # Avoid filtfilt pad-length errors on short startup buffers.
        return signal.copy()
    # Use filtfilt to achieve zero‑phase filtering (no group delay)
    filtered = filtfilt(b, a, signal)
    return filtered
