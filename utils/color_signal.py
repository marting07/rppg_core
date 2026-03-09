"""Robust color-statistics extraction for rPPG ROI frames."""

from __future__ import annotations

import cv2  # type: ignore
import numpy as np


def robust_mean_bgr(roi_frame: np.ndarray) -> tuple[float, float, float]:
    """Return robust per-channel means from a facial ROI.

    Steps:
    1) reject clipped/saturated pixels,
    2) prefer a broad skin-color mask in YCrCb,
    3) trimmed mean per channel to reduce outliers/specular spots.
    """
    if roi_frame is None or roi_frame.size == 0:
        return 0.0, 0.0, 0.0

    roi_u8 = roi_frame.astype(np.uint8)
    pixels = roi_u8.reshape(-1, 3)
    if pixels.size == 0:
        return 0.0, 0.0, 0.0

    clipped = np.any((pixels <= 5) | (pixels >= 250), axis=1)
    valid = ~clipped

    ycrcb = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    cr = ycrcb[:, 1]
    cb = ycrcb[:, 2]
    skin = (cr >= 120) & (cr <= 180) & (cb >= 80) & (cb <= 140)

    keep = valid & skin
    if np.sum(keep) < 80:
        keep = valid
    if np.sum(keep) < 20:
        keep = np.ones(pixels.shape[0], dtype=bool)

    sel = pixels[keep].astype(np.float64)

    means: list[float] = []
    for c in range(3):
        v = sel[:, c]
        if v.size == 0:
            means.append(0.0)
            continue
        lo, hi = np.percentile(v, [10, 90])
        m = (v >= lo) & (v <= hi)
        means.append(float(np.mean(v[m])) if np.any(m) else float(np.mean(v)))

    return means[0], means[1], means[2]
