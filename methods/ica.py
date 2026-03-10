"""Independent Component Analysis (ICA) style rPPG method.

Reference:
    Poh, M.-Z., McDuff, D. J., & Picard, R. W. (2010).
    Non-contact, automated cardiac pulse measurements using video imaging
    and blind source separation. Optics Express, 18(10), 10762-10774.
"""

from __future__ import annotations

import numpy as np

from rppg_core.methods.base import RPPGMethod
from rppg_core.utils.color_signal import robust_mean_bgr


class ICAMethod(RPPGMethod):
    """FastICA-inspired extraction over normalized RGB traces."""

    def __init__(self, fs: float = 30.0, buffer_size: int = 300, window_seconds: float = 1.6) -> None:
        super().__init__(fs=fs, buffer_size=buffer_size)
        self.welch_window_seconds = 8.0
        self.min_hr_confidence = 1.06
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

    @staticmethod
    def _fastica_components(x: np.ndarray, n_components: int = 3, max_iter: int = 160, tol: float = 1e-6) -> np.ndarray:
        """Estimate multiple ICA components with symmetric decorrelation."""
        n_features = x.shape[1]
        n_components = min(max(1, n_components), n_features)
        init = np.eye(n_features, dtype=np.float64)[:n_components].copy()
        if init.shape[0] < n_components:
            pad = np.ones((n_components - init.shape[0], n_features), dtype=np.float64)
            init = np.vstack([init, pad])
        w = init
        for row in range(w.shape[0]):
            w[row] /= np.linalg.norm(w[row]) + 1e-12

        for _ in range(max_iter):
            wx = x @ w.T
            g = np.tanh(wx)
            g_prime = 1.0 - g * g
            w_new = (g.T @ x) / x.shape[0] - np.diag(np.mean(g_prime, axis=0)) @ w

            cov = w_new @ w_new.T
            eigvals, eigvecs = np.linalg.eigh(cov)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.clip(eigvals, 1e-8, None))) @ eigvecs.T
            w_new = inv_sqrt @ w_new

            deltas = np.abs(np.sum(w_new * w, axis=1))
            if np.all(deltas > (1.0 - tol)):
                w = w_new
                break
            w = w_new

        return x @ w.T

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
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        xw = rgb @ whitener

        components = self._fastica_components(xw, n_components=3)
        green_reference = rgb[:, 1] - np.mean(rgb[:, 1])
        best_component = None
        best_filtered = None
        best_key = None

        for idx in range(components.shape[1]):
            comp = self._standardize_signal(components[:, idx])
            if float(np.dot(comp, green_reference)) < 0.0:
                comp = -comp
            spec = self._compute_candidate_spectrum(comp)
            confidence = float(spec["confidence"] or 0.0)
            score = float(spec["score"] or float("-inf"))
            peak_freq_hz = spec["peak_freq_hz"]
            continuity = 1.0
            if self.last_peak_freq_hz is not None and peak_freq_hz is not None:
                continuity = 1.0 / (1.0 + abs(float(peak_freq_hz) - self.last_peak_freq_hz) / self._peak_continuity_hz)
            key = (confidence * continuity, confidence, score)
            if best_key is None or key > best_key:
                best_key = key
                best_component = comp
                best_filtered = self._standardize_signal(np.array(spec["filtered"], dtype=np.float64))

        if best_component is None or best_filtered is None:
            return

        tail = best_filtered[-self._tail_mean_samples :]
        self.update_from_value(float(np.mean(tail)))
