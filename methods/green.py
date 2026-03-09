"""Implementation of the basic green channel rPPG method.

Signal extraction equation:
    s_t = mean(G_t)
where G_t is the green channel in the forehead ROI at frame t.

Reference:
    Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008).
    Remote plethysmographic imaging using ambient light.
    Optics Express, 16(26), 21434-21445.
"""

from __future__ import annotations

import numpy as np

from .base import RPPGMethod
from rppg_core.utils.color_signal import robust_mean_bgr


class GreenMethod(RPPGMethod):
    """Simple rPPG method using mean green channel intensity."""

    def __init__(self, fs: float = 30.0, buffer_size: int = 300) -> None:
        super().__init__(fs=fs, buffer_size=buffer_size)
        self.welch_window_seconds = 8.0
        self.min_hr_confidence = 1.1
        self.hr_smoothing_alpha = 0.2
        self.max_hr_jump_bpm_per_s = 10.0

    def update(self, roi_frame: np.ndarray) -> None:
        """Run method stages:
        1) signal_extraction: mean green value
        2) normalization/filtering/hr_estimation: shared base pipeline
        """
        if roi_frame is None or roi_frame.size == 0:
            return
        _, mean_green, _ = robust_mean_bgr(roi_frame)
        self.update_from_value(mean_green)
