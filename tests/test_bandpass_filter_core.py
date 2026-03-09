from __future__ import annotations

import unittest

import numpy as np

from rppg_core.utils.bandpass_filter import bandpass_filter


class BandpassFilterTests(unittest.TestCase):
    def test_short_signal_does_not_crash(self) -> None:
        signal = np.array([0.1, 0.2, 0.3, 0.1], dtype=np.float64)
        filtered = bandpass_filter(signal, fs=30.0, low=0.75, high=4.0)
        self.assertEqual(filtered.shape, signal.shape)


if __name__ == "__main__":
    unittest.main()
