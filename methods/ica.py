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
        self.latency_seconds = (self.window_size / self.fs) * 0.5

        self.r_buffer: list[float] = []
        self.g_buffer: list[float] = []
        self.b_buffer: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()

    @staticmethod
    def _fastica_one_component(x: np.ndarray, max_iter: int = 120, tol: float = 1e-6) -> np.ndarray:
        # x shape: (n_samples, n_features), whitened and zero mean.
        n_features = x.shape[1]
        w = np.ones(n_features, dtype=np.float64)
        w /= np.linalg.norm(w) + 1e-12

        for _ in range(max_iter):
            wx = x @ w
            g = np.tanh(wx)
            g_prime = 1.0 - g * g
            w_new = (x.T @ g) / x.shape[0] - np.mean(g_prime) * w
            w_new /= np.linalg.norm(w_new) + 1e-12
            if abs(float(np.dot(w_new, w))) > (1.0 - tol):
                w = w_new
                break
            w = w_new

        return x @ w

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

        if len(self.r_buffer) < self.window_size:
            return

        rgb = np.stack(
            [
                np.array(self.r_buffer[-self.window_size :], dtype=np.float64),
                np.array(self.g_buffer[-self.window_size :], dtype=np.float64),
                np.array(self.b_buffer[-self.window_size :], dtype=np.float64),
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

        comp = self._fastica_one_component(xw)
        comp = comp - np.mean(comp)
        std = float(np.std(comp))
        if std > 1e-8:
            comp = comp / std

        self.update_from_value(float(comp[-1]))
