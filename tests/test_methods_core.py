from __future__ import annotations

import unittest

import numpy as np

from rppg_core import ChromMethod, GreenMethod, POSMethod, SSRMethod


def roi_from_bgr(b: float, g: float, r: float, size: int = 12) -> np.ndarray:
    roi = np.zeros((size, size, 3), dtype=np.uint8)
    roi[:, :, 0] = np.clip(round(b), 0, 255)
    roi[:, :, 1] = np.clip(round(g), 0, 255)
    roi[:, :, 2] = np.clip(round(r), 0, 255)
    return roi


def textured_roi_from_bgr(
    b: float,
    g: float,
    r: float,
    size: int = 24,
) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    pattern = 1.0 + 0.12 * np.sin(2.0 * np.pi * xx / max(size, 1)) + 0.10 * np.cos(2.0 * np.pi * yy / max(size, 1))
    roi = np.zeros((size, size, 3), dtype=np.float64)
    roi[:, :, 0] = np.clip(b * pattern, 0.0, 255.0)
    roi[:, :, 1] = np.clip(g * pattern, 0.0, 255.0)
    roi[:, :, 2] = np.clip(r * pattern, 0.0, 255.0)
    return roi.astype(np.uint8)


class MethodBehaviorTests(unittest.TestCase):
    def test_empty_roi_is_ignored(self) -> None:
        methods = [GreenMethod(), ChromMethod(), POSMethod(), SSRMethod()]
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        for method in methods:
            method.update(empty)
            self.assertEqual(len(method.signal_buffer), 0)

    def test_green_recovers_known_hr_from_synthetic_signal(self) -> None:
        fs = 30.0
        bpm_target = 72.0
        freq = bpm_target / 60.0
        method = GreenMethod(fs=fs, buffer_size=600)
        duration_s = 12.0
        n = int(duration_s * fs)
        for idx in range(n):
            t = idx / fs
            g = 128.0 + 20.0 * np.sin(2.0 * np.pi * freq * t)
            method.update(roi_from_bgr(b=90.0, g=g, r=100.0))
        hr = method.get_hr()
        self.assertIsNotNone(hr)
        assert hr is not None
        self.assertLess(abs(hr - bpm_target), 5.0)

    def test_chrom_recovers_known_hr_from_synthetic_signal(self) -> None:
        fs = 30.0
        bpm_target = 72.0
        freq = bpm_target / 60.0
        method = ChromMethod(fs=fs, buffer_size=600)
        duration_s = 12.0
        n = int(duration_s * fs)
        for idx in range(n):
            t = idx / fs
            pulse_r = np.sin(2.0 * np.pi * freq * t)
            pulse_g = np.sin(2.0 * np.pi * freq * t + 0.3)
            pulse_b = np.sin(2.0 * np.pi * freq * t - 0.2)
            r = 130.0 + 14.0 * pulse_r
            g = 110.0 + 8.0 * pulse_g
            b = 95.0 + 4.0 * pulse_b
            method.update(roi_from_bgr(b=b, g=g, r=r))
        hr = method.get_hr()
        self.assertIsNotNone(hr)
        assert hr is not None
        self.assertLess(abs(hr - bpm_target), 7.0)

    def test_pos_recovers_known_hr_from_synthetic_signal(self) -> None:
        fs = 30.0
        bpm_target = 78.0
        freq = bpm_target / 60.0
        method = POSMethod(fs=fs, buffer_size=600)
        n = int(12.0 * fs)
        for idx in range(n):
            t = idx / fs
            pulse = np.sin(2.0 * np.pi * freq * t)
            motion = 0.8 * np.sin(2.0 * np.pi * 0.35 * t)
            r = 130.0 + 8.0 * pulse + 5.0 * motion
            g = 115.0 + 11.0 * pulse - 3.5 * motion
            b = 95.0 + 5.0 * pulse + 2.0 * motion
            method.update(roi_from_bgr(b=b, g=g, r=r))
        hr = method.get_hr()
        self.assertIsNotNone(hr)
        assert hr is not None
        self.assertLess(abs(hr - bpm_target), 8.0)

    def test_ssr_is_deterministic_for_same_input(self) -> None:
        fs = 30.0
        bpm_target = 70.0
        freq = bpm_target / 60.0
        method_a = SSRMethod(fs=fs, buffer_size=600)
        method_b = SSRMethod(fs=fs, buffer_size=600)
        n = int(14.0 * fs)
        rois: list[np.ndarray] = []
        for idx in range(n):
            t = idx / fs
            pulse = np.sin(2.0 * np.pi * freq * t)
            illum = 0.6 * np.sin(2.0 * np.pi * 0.22 * t)
            r = 125.0 + 8.0 * pulse + 4.5 * illum
            g = 110.0 + 7.0 * pulse - 3.0 * illum
            b = 95.0 + 4.0 * pulse + 2.0 * illum
            rois.append(textured_roi_from_bgr(b=b, g=g, r=r))
        for roi in rois:
            method_a.update(roi)
            method_b.update(roi)
        hr_a = method_a.get_hr()
        hr_b = method_b.get_hr()
        self.assertIsNotNone(hr_a)
        self.assertIsNotNone(hr_b)
        assert hr_a is not None and hr_b is not None
        self.assertAlmostEqual(hr_a, hr_b, places=6)

    def test_get_ppg_signal_handles_short_buffers(self) -> None:
        method = GreenMethod(fs=30.0, buffer_size=300)
        for _ in range(5):
            method.update(roi_from_bgr(90.0, 120.0, 100.0))
        signal = method.get_ppg_signal()
        self.assertEqual(signal.size, 5)


if __name__ == "__main__":
    unittest.main()
