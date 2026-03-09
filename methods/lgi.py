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
        rgb = (rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-8)) - 1.0
        rgb = rgb - np.mean(rgb, axis=0, keepdims=True)

        # LGI-like projection: remove dominant group component, retain invariant residual.
        _, _, vh = np.linalg.svd(rgb, full_matrices=False)
        principal = vh[0]
        projection = np.eye(3, dtype=np.float64) - np.outer(principal, principal)
        y = rgb @ projection

        # Use dominant residual axis as pulse proxy.
        cov_res = np.cov(y, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_res)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        s = y @ axis
        s = s - np.mean(s)
        std = float(np.std(s))
        if std > 1e-8:
            s = s / std

        self.update_from_value(float(s[-1]))
