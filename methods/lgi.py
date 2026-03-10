"""Local Group Invariance (LGI) style rPPG method.

Reference:
    Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018).
    Local Group Invariance for Heart Rate Estimation From Face Videos in the Wild.
    CVPR Workshops, 1254-1262.
"""

from __future__ import annotations

import numpy as np

from rppg_core.methods.base import RPPGMethod
from rppg_core.utils.color_signal import robust_mean_bgr


class LGIMethod(RPPGMethod):
    """LGI-inspired subspace projection in normalized RGB traces."""

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
        self._peak_continuity_hz = 0.35

        self.r_buffer: list[float] = []
        self.g_buffer: list[float] = []
        self.b_buffer: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()

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

        # LGI-style projection: remove dominant group component, retain invariant residual channels.
        _, _, vh = np.linalg.svd(rgb, full_matrices=False)
        principal = vh[0]
        projection = np.eye(3, dtype=np.float64) - np.outer(principal, principal)
        residual = rgb @ projection

        green_reference = rgb[:, 1] - np.mean(rgb[:, 1])
        candidates: list[np.ndarray] = []
        for axis_idx in range(residual.shape[1]):
            candidates.append(residual[:, axis_idx])

        cov_res = np.cov(residual, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_res)
        order = np.argsort(eigvals)[::-1]
        for axis_idx in order:
            candidates.append(residual @ eigvecs[:, axis_idx])

        best_signal = None
        best_filtered = None
        best_key = None
        for candidate in candidates:
            s = self._standardize_signal(candidate)
            if float(np.dot(s, green_reference)) < 0.0:
                s = -s
            spec = self._compute_candidate_spectrum(s)
            confidence = float(spec["confidence"] or 0.0)
            score = float(spec["score"] or float("-inf"))
            peak_freq_hz = spec["peak_freq_hz"]
            continuity = 1.0
            if self.last_peak_freq_hz is not None and peak_freq_hz is not None:
                continuity = 1.0 / (1.0 + abs(float(peak_freq_hz) - self.last_peak_freq_hz) / self._peak_continuity_hz)
            key = (confidence * continuity, confidence, score)
            if best_key is None or key > best_key:
                best_key = key
                best_signal = s
                best_filtered = self._standardize_signal(np.array(spec["filtered"], dtype=np.float64))

        if best_signal is None or best_filtered is None:
            return

        tail = best_filtered[-self._tail_mean_samples :]
        self.update_from_value(float(np.mean(tail)))
