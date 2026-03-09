"""Implementation of an SSR/2SR-style subspace-rotation rPPG method.

Reference:
    Wang, W., Stuijk, S., & de Haan, G. (2015).
    A Novel Algorithm for Remote Photoplethysmography: Spatial Subspace Rotation.
    IEEE Transactions on Biomedical Engineering, 63(9), 1974-1984.
"""

from __future__ import annotations

import cv2  # type: ignore
import numpy as np

from .base import RPPGMethod


class SSRMethod(RPPGMethod):
    """Windowed SSR/2SR-style extraction with temporal stabilization."""

    def __init__(
        self,
        fs: float = 30.0,
        buffer_size: int = 300,
        smooth_seconds: float = 0.3,
    ) -> None:
        super().__init__(fs=fs, buffer_size=buffer_size)
        self.welch_window_seconds = 8.0
        self.min_hr_confidence = 1.06
        self.hr_smoothing_alpha = 0.30
        self.max_hr_jump_bpm_per_s = 14.0

        self._smooth_len = max(1, int(round(smooth_seconds * fs)))
        self._rot_buffer: list[float] = []
        self._prev_basis: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._rot_buffer.clear()
        self._prev_basis = None

    def update(self, roi_frame: np.ndarray) -> None:
        if roi_frame is None or roi_frame.size == 0:
            return

        basis = self._compute_skin_subspace_basis(roi_frame)
        if basis is None:
            return

        if self._prev_basis is None:
            self._prev_basis = basis
            return

        basis = self._align_basis_with_previous(basis, self._prev_basis)

        # Signed in-plane rotation proxy between consecutive 2D subspaces.
        a11 = float(np.dot(self._prev_basis[:, 0], basis[:, 0]))
        a12 = float(np.dot(self._prev_basis[:, 0], basis[:, 1]))
        a21 = float(np.dot(self._prev_basis[:, 1], basis[:, 0]))
        a22 = float(np.dot(self._prev_basis[:, 1], basis[:, 1]))
        rot_signal = np.arctan2(a12 - a21, a11 + a22)

        self._rot_buffer.append(float(rot_signal))
        if len(self._rot_buffer) > self.buffer_size:
            excess = len(self._rot_buffer) - self.buffer_size
            self._rot_buffer = self._rot_buffer[excess:]

        if len(self._rot_buffer) >= self._smooth_len:
            value = float(np.mean(self._rot_buffer[-self._smooth_len :]))
        else:
            value = float(self._rot_buffer[-1])

        self.update_from_value(value)
        self._prev_basis = basis

    def _compute_skin_subspace_basis(self, roi_frame: np.ndarray) -> np.ndarray | None:
        roi_u8 = roi_frame.astype(np.uint8)
        ycrcb = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 120) & (cr <= 180) & (cb >= 80) & (cb <= 140)

        pixels = roi_u8.reshape(-1, 3).astype(np.float64)
        if np.any(skin_mask):
            skin_pixels = roi_u8[skin_mask].reshape(-1, 3).astype(np.float64)
            if skin_pixels.shape[0] >= 32:
                pixels = skin_pixels

        if pixels.shape[0] < 16:
            return None

        # Work in RGB order and normalize by channel means.
        rgb = pixels[:, ::-1]
        rgb = (rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-8)) - 1.0
        rgb = rgb - np.mean(rgb, axis=0, keepdims=True)

        cov = (rgb.T @ rgb) / max(rgb.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        basis = eigvecs[:, :2]
        if basis.shape != (3, 2):
            return None

        # Deterministic sign convention to avoid arbitrary flips.
        for j in range(2):
            dominant = int(np.argmax(np.abs(basis[:, j])))
            if basis[dominant, j] < 0:
                basis[:, j] *= -1.0
        return basis

    @staticmethod
    def _align_basis_with_previous(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        # Resolve 2D basis permutation ambiguity by maximal overlap.
        d00 = abs(float(np.dot(previous[:, 0], current[:, 0])))
        d11 = abs(float(np.dot(previous[:, 1], current[:, 1])))
        d01 = abs(float(np.dot(previous[:, 0], current[:, 1])))
        d10 = abs(float(np.dot(previous[:, 1], current[:, 0])))
        aligned = current.copy()
        if (d01 + d10) > (d00 + d11):
            aligned = aligned[:, [1, 0]]
        # Resolve sign ambiguity with respect to previous basis.
        for j in range(2):
            if float(np.dot(previous[:, j], aligned[:, j])) < 0.0:
                aligned[:, j] *= -1.0
        return aligned
