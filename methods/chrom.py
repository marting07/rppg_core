"""Implementation of the chrominance-based rPPG method (CHROM)."""

from __future__ import annotations

import numpy as np

from .base import RPPGMethod
from rppg_core.utils.color_signal import robust_mean_bgr


class ChromMethod(RPPGMethod):
    """CHROM implementation using temporal color normalization and alpha balancing."""

    def __init__(self, fs: float = 30.0, buffer_size: int = 300) -> None:
        super().__init__(fs=fs, buffer_size=buffer_size)
        self.welch_window_seconds = 8.0
        self.min_hr_confidence = 1.1
        self.hr_smoothing_alpha = 0.4
        self.max_hr_jump_bpm_per_s = 18.0
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

        self.r_buffer.append(r)
        self.g_buffer.append(g)
        self.b_buffer.append(b)
        if len(self.r_buffer) > self.buffer_size:
            excess = len(self.r_buffer) - self.buffer_size
            self.r_buffer = self.r_buffer[excess:]
            self.g_buffer = self.g_buffer[excess:]
            self.b_buffer = self.b_buffer[excess:]
        if len(self.r_buffer) < 2:
            return

        r_arr = np.array(self.r_buffer, dtype=np.float64)
        g_arr = np.array(self.g_buffer, dtype=np.float64)
        b_arr = np.array(self.b_buffer, dtype=np.float64)
        r_n = (r_arr / (np.mean(r_arr) + 1e-8)) - 1.0
        g_n = (g_arr / (np.mean(g_arr) + 1e-8)) - 1.0
        b_n = (b_arr / (np.mean(b_arr) + 1e-8)) - 1.0

        x_arr = 3.0 * r_n - 2.0 * g_n
        y_arr = 1.5 * r_n + g_n - 1.5 * b_n

        std_x = np.std(x_arr) + 1e-8
        std_y = np.std(y_arr)
        alpha = 0.0 if std_y <= 1e-12 else float(std_x / std_y)
        s_arr = x_arr - alpha * y_arr
        s = float(s_arr[-1])
        self.update_from_value(s)
