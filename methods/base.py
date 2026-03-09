"""Base classes and common utilities for remote photoplethysmography methods.

All methods follow the same four-stage processing model:

1) ``signal_extraction``: derive one scalar value from the ROI frame
2) ``normalization``: remove slow trends / scale effects
3) ``filtering``: isolate the heart-rate band
4) ``hr_estimation``: estimate BPM from the dominant spectral peak
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rppg_core.utils.bandpass_filter import bandpass_filter


class RPPGMethod:
    """Abstract base class for rPPG methods.

    Parameters
    ----------
    fs : float
        Sampling frequency (frame rate) of the video in Hertz.
    buffer_size : int
        Number of samples to retain for heart rate estimation. The
        buffer should be long enough to capture several cardiac
        cycles; a typical buffer corresponds to 8–12 seconds of
        video.
    """

    def __init__(self, fs: float = 30.0, buffer_size: int = 300) -> None:
        self.fs = fs
        self.buffer_size = buffer_size
        self.signal_buffer: list[float] = []
        self.times: list[float] = []
        self.last_hr: Optional[float] = None
        self.last_raw_hr: Optional[float] = None
        self.last_peak_freq_hz: Optional[float] = None
        self.last_hr_confidence: Optional[float] = None
        self.welch_window_seconds = 8.0
        self.welch_overlap_ratio = 0.5
        self.min_hr_confidence = 1.05
        self.hr_smoothing_alpha = 0.30
        self.max_hr_jump_bpm_per_s = 12.0
        self.latency_seconds = 0.0

    def reset(self) -> None:
        """Clear the internal signal buffer and reset heart rate."""
        self.signal_buffer.clear()
        self.times.clear()
        self.last_hr = None
        self.last_raw_hr = None
        self.last_peak_freq_hz = None
        self.last_hr_confidence = None

    def update(self, roi_frame: np.ndarray) -> None:
        """Process a new ROI frame and update the internal buffer.

        Subclasses must override this method to extract the per‑frame
        photoplethysmographic value (e.g. mean green intensity or a
        chrominance combination) from the ROI.

        Parameters
        ----------
        roi_frame : np.ndarray
            The cropped BGR image corresponding to the subject's
            forehead.
        """
        raise NotImplementedError

    def update_from_value(self, value: float) -> None:
        """Run the common pipeline for one extracted method value."""
        self._append_value(value)
        raw_hr = self.estimate_hr_bpm()
        self.last_raw_hr = raw_hr
        if raw_hr is None:
            self.last_hr = None
            return

        if self.last_hr is None:
            self.last_hr = float(raw_hr)
            return

        max_step = self.max_hr_jump_bpm_per_s / max(self.fs, 1e-6)
        clamped = self.last_hr + float(np.clip(raw_hr - self.last_hr, -max_step, max_step))
        alpha = float(np.clip(self.hr_smoothing_alpha, 0.0, 1.0))
        self.last_hr = float(alpha * clamped + (1.0 - alpha) * self.last_hr)

    def _append_value(self, value: float) -> None:
        """Append a raw value to the signal buffer and trim if necessary."""
        self.signal_buffer.append(value)
        # Maintain buffer length
        if len(self.signal_buffer) > self.buffer_size:
            # Remove oldest values
            excess = len(self.signal_buffer) - self.buffer_size
            self.signal_buffer = self.signal_buffer[excess:]

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalization stage: remove DC component."""
        if signal.size == 0:
            return signal
        return signal - np.mean(signal)

    def filter_signal(self, normalized_signal: np.ndarray) -> np.ndarray:
        """Filtering stage: apply standard rPPG heart-rate band-pass."""
        return bandpass_filter(normalized_signal, fs=self.fs, low=0.75, high=4.0)

    def get_filtered_signal(self) -> np.ndarray:
        """Return the current normalized and filtered signal."""
        signal = np.array(self.signal_buffer, dtype=np.float64)
        if signal.size == 0:
            return signal
        normalized = self.normalize_signal(signal)
        return self.filter_signal(normalized)

    def compute_psd(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one-sided FFT magnitude spectrum for the filtered signal."""
        filtered = self.get_filtered_signal()
        if filtered.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        fft_mag = np.abs(np.fft.rfft(filtered))
        return freqs, fft_mag

    def estimate_hr_bpm(self) -> Optional[float]:
        """Heart-rate estimation stage from Welch-averaged in-band spectrum."""
        signal = np.array(self.signal_buffer, dtype=np.float64)
        if signal.size < int(self.fs * 4):
            self.last_hr_confidence = None
            return None
        filtered = self.get_filtered_signal()
        if filtered.size == 0:
            self.last_hr_confidence = None
            return None
        n = filtered.size
        seg_len = int(min(n, max(int(round(self.fs * 4.0)), int(round(self.fs * self.welch_window_seconds)))))
        if seg_len < int(self.fs * 4):
            self.last_hr_confidence = None
            return None
        hop = max(1, int(round(seg_len * (1.0 - self.welch_overlap_ratio))))

        starts = list(range(0, n - seg_len + 1, hop))
        if not starts:
            starts = [0]

        psd_accum = None
        freqs = None
        for start in starts:
            seg = filtered[start : start + seg_len]
            if seg.size != seg_len:
                continue
            window = np.hanning(seg_len)
            seg = seg - np.mean(seg)
            fft = np.fft.rfft(seg * window)
            power = (np.abs(fft) ** 2) / max(seg_len, 1)
            f = np.fft.rfftfreq(seg_len, d=1.0 / self.fs)
            if psd_accum is None:
                psd_accum = power
                freqs = f
            else:
                psd_accum += power

        if psd_accum is None or freqs is None:
            self.last_hr_confidence = None
            return None

        psd = psd_accum / float(len(starts))
        if freqs.size == 0:
            self.last_hr_confidence = None
            return None
        mask = (freqs >= 0.75) & (freqs <= 3.0)
        if not np.any(mask):
            self.last_hr_confidence = None
            return None
        in_band_freqs = freqs[mask]
        in_band_power = psd[mask]
        harmonic_power = np.interp(2.0 * in_band_freqs, freqs, psd, left=0.0, right=0.0)
        # Combine fundamental and second harmonic evidence to reduce sub-harmonic lock.
        score = in_band_power + 0.5 * harmonic_power
        peak_idx = int(np.argmax(score))
        peak_power = float(score[peak_idx])
        baseline = float(np.median(score) + 1e-10)
        self.last_hr_confidence = peak_power / baseline
        if self.last_hr_confidence < self.min_hr_confidence:
            return None
        peak_freq = in_band_freqs[peak_idx]
        self.last_peak_freq_hz = float(peak_freq)
        return float(peak_freq * 60.0)

    def get_hr(self) -> Optional[float]:
        """Return the most recent heart rate estimation.

        This method should be called after ``update`` has been
        invoked on successive frames. The result is cached to avoid
        recomputing the FFT on every frame.
        """
        return self.last_hr

    def get_ppg_signal(self) -> np.ndarray:
        """Return the current filtered PPG signal for plotting."""
        return self.get_filtered_signal()

    def get_confidence(self) -> Optional[float]:
        """Return spectral-confidence proxy for last HR estimate."""
        return self.last_hr_confidence
