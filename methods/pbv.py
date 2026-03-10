"""Blood Volume Pulse Signature (PBV) style rPPG method.

Reference:
    de Haan, G., & van Leest, A. (2014).
    Improved motion robustness of remote-PPG by using the blood volume pulse signature.
    Physiological Measurement, 35(9), 1913-1926.
"""

from __future__ import annotations

import numpy as np

from rppg_core.methods.base import RPPGMethod
from rppg_core.utils.color_signal import robust_mean_bgr


class PBVMethod(RPPGMethod):
    """PBV-inspired linear projection in normalized RGB space."""

    def __init__(self, fs: float = 30.0, buffer_size: int = 300, window_seconds: float = 1.6) -> None:
        super().__init__(fs=fs, buffer_size=buffer_size)
        self.welch_window_seconds = 8.0
        self.min_hr_confidence = 1.08
        self.hr_smoothing_alpha = 0.35
        self.max_hr_jump_bpm_per_s = 14.0

        self.window_size = max(24, int(round(window_seconds * fs)))
        self.analysis_size = max(self.window_size, int(round(4.0 * fs)))
        self.latency_seconds = (self.analysis_size / self.fs) * 0.5
        self._tail_mean_samples = max(1, int(round(0.15 * fs)))
        self._signature_smoothing = 0.35
        self._peak_continuity_hz = 0.35

        self.r_buffer: list[float] = []
        self.g_buffer: list[float] = []
        self.b_buffer: list[float] = []

        # Canonical blood-volume pulse signature direction (RGB domain).
        self.pbv_signature = np.array([0.77, 0.51, 0.38], dtype=np.float64)
        self._previous_signature: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
        self._previous_signature = None

    def update(self, roi_frame: np.ndarray) -> None:
        if roi_frame is None or roi_frame.size == 0:
            return

        b, g, r = robust_mean_bgr(roi_frame)
        self.b_buffer.append(b)
        self.g_buffer.append(g)
        self.r_buffer.append(r)

        if len(self.r_buffer) > self.buffer_size:
            excess = len(self.r_buffer) - self.buffer_size
            self.r_buffer = self.r_buffer[excess:]
            self.g_buffer = self.g_buffer[excess:]
            self.b_buffer = self.b_buffer[excess:]

        if len(self.r_buffer) < self.analysis_size:
            return

        rgb = np.stack(
            [
                np.array(self.r_buffer[-self.analysis_size :], dtype=np.float64),
                np.array(self.g_buffer[-self.analysis_size :], dtype=np.float64),
                np.array(self.b_buffer[-self.analysis_size :], dtype=np.float64),
            ],
            axis=1,
        )
        rgb = (rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-8)) - 1.0
        rgb = rgb - np.mean(rgb, axis=0, keepdims=True)

        cov = np.cov(rgb, rowvar=False)
        cov += np.eye(3, dtype=np.float64) * 1e-5
        inv_cov = np.linalg.pinv(cov)

        dynamic_signature = np.std(rgb, axis=0)
        dynamic_norm = float(np.linalg.norm(dynamic_signature))
        if dynamic_norm <= 1e-8:
            dynamic_signature = self.pbv_signature.copy()
            dynamic_norm = float(np.linalg.norm(dynamic_signature))
        dynamic_signature = dynamic_signature / max(dynamic_norm, 1e-12)

        canonical_signature = self.pbv_signature / (np.linalg.norm(self.pbv_signature) + 1e-12)
        if self._previous_signature is None:
            smoothed_signature = dynamic_signature
        else:
            alpha = float(np.clip(self._signature_smoothing, 0.0, 1.0))
            smoothed_signature = (1.0 - alpha) * self._previous_signature + alpha * dynamic_signature
            smoothed_signature /= np.linalg.norm(smoothed_signature) + 1e-12

        signatures = [
            dynamic_signature,
            smoothed_signature,
            0.5 * (smoothed_signature + canonical_signature),
        ]
        green_reference = rgb[:, 1] - np.mean(rgb[:, 1])

        best_signature = None
        best_filtered = None
        best_key = None
        for signature in signatures:
            signature = signature / (np.linalg.norm(signature) + 1e-12)
            w = inv_cov @ signature
            denom = float(signature.T @ inv_cov @ signature)
            if abs(denom) < 1e-10:
                continue
            w /= denom

            signal = self._standardize_signal(rgb @ w)
            if float(np.dot(signal, green_reference)) < 0.0:
                signal = -signal
            spec = self._compute_candidate_spectrum(signal)
            confidence = float(spec["confidence"] or 0.0)
            score = float(spec["score"] or float("-inf"))
            peak_freq_hz = spec["peak_freq_hz"]
            continuity = 1.0
            if self.last_peak_freq_hz is not None and peak_freq_hz is not None:
                continuity = 1.0 / (1.0 + abs(float(peak_freq_hz) - self.last_peak_freq_hz) / self._peak_continuity_hz)
            key = (confidence * continuity, confidence, score)
            if best_key is None or key > best_key:
                best_key = key
                best_signature = signature.copy()
                best_filtered = self._standardize_signal(np.array(spec["filtered"], dtype=np.float64))

        if best_signature is None or best_filtered is None:
            return
        self._previous_signature = best_signature
        tail = best_filtered[-self._tail_mean_samples :]
        self.update_from_value(float(np.mean(tail)))
