"""Implementation of the Plane-Orthogonal-to-Skin (POS) rPPG method.

Reference:
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
    Algorithmic Principles of Remote PPG.
    IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
"""

from __future__ import annotations

import numpy as np

from .base import RPPGMethod
from rppg_core.utils.color_signal import robust_mean_bgr


class POSMethod(RPPGMethod):
    """POS implementation with sliding temporal window projection."""

    def __init__(
        self,
        fs: float = 30.0,
        buffer_size: int = 300,
        window_seconds: float = 1.6,
    ) -> None:
        super().__init__(fs=fs, buffer_size=buffer_size)
        self.welch_window_seconds = 8.0
        self.min_hr_confidence = 1.08
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
        mean_rgb = np.mean(rgb, axis=0, keepdims=True)
        cn = (rgb / (mean_rgb + 1e-8)) - 1.0

        s1 = cn[:, 1] - cn[:, 2]
        s2 = -2.0 * cn[:, 0] + cn[:, 1] + cn[:, 2]
        std_s2 = float(np.std(s2))
        alpha = 0.0 if std_s2 <= 1e-10 else float(np.std(s1) / (std_s2 + 1e-10))
        h = s1 + alpha * s2
        h = h - np.mean(h)
        std_h = float(np.std(h))
        if std_h > 1e-8:
            h = h / std_h

        self.update_from_value(float(h[-1]))
