"""Utilities for detecting faces and extracting the forehead region of interest (ROI).

This module uses OpenCV's Haar cascade classifier to locate faces
within a video frame. It then computes a rectangular region
corresponding to the forehead, which is commonly used for
remote photoplethysmography (rPPG) because it tends to be less
affected by facial expressions or motion than other parts of the face.

The functions defined here encapsulate face detection and ROI
extraction so that the rPPG methods can focus solely on signal
processing.
"""

from __future__ import annotations

import cv2  # type: ignore
import numpy as np


class FaceDetector:
    """Wrapper around OpenCV's Haar cascade face detector.

    Parameters
    ----------
    cascade_path : str | None
        Path to the Haar cascade XML file. If ``None``, the default
        cascade provided by OpenCV will be used. On most systems
        cv2 includes the necessary file in ``cv2.data.haarcascades``.
    """

    def __init__(self, cascade_path: str | None = None) -> None:
        if cascade_path is None:
            # Use OpenCV's default frontal face classifier
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError(f"Failed to load face cascade from {cascade_path}")

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in a grayscale frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image in which to detect faces.

        Returns
        -------
        list of tuple[int, int, int, int]
            A list of bounding boxes ``(x, y, w, h)`` for each detected face.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def extract_forehead_roi(
    frame: np.ndarray, face_bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract a forehead region from the input frame using the face bounding box.

    The forehead ROI is defined as the top 20 % of the face bounding box.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame.
    face_bbox : tuple[int, int, int, int]
        The bounding box of the face as returned by ``detect_faces``.

    Returns
    -------
    roi : np.ndarray
        The extracted forehead region in BGR format.
    roi_box : tuple[int, int, int, int]
        The bounding box of the ROI within the input frame (x, y, w, h).
    """
    x, y, w, h = face_bbox
    # Use a centered forehead patch to avoid hairline and eyebrow edges.
    roi_height = int(h * 0.22)
    roi_y = y + int(h * 0.14)
    roi_w = int(w * 0.60)
    roi_x = x + int((w - roi_w) * 0.5)
    roi_h = roi_height
    # Ensure ROI is within image bounds
    roi_y = max(0, roi_y)
    roi_y_end = min(frame.shape[0], roi_y + roi_h)
    roi_x_end = min(frame.shape[1], roi_x + roi_w)
    roi = frame[roi_y:roi_y_end, roi_x:roi_x_end].copy()
    return roi, (roi_x, roi_y, roi_w, roi_h)


def _extract_box_roi(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(frame.shape[1], x0 + max(1, int(w)))
    y1 = min(frame.shape[0], y0 + max(1, int(h)))
    roi = frame[y0:y1, x0:x1].copy()
    return roi, (x0, y0, int(x1 - x0), int(y1 - y0))


def extract_left_cheek_roi(
    frame: np.ndarray, face_bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = face_bbox
    roi_w = int(w * 0.24)
    roi_h = int(h * 0.22)
    roi_x = x + int(w * 0.16)
    roi_y = y + int(h * 0.52)
    return _extract_box_roi(frame, roi_x, roi_y, roi_w, roi_h)


def extract_right_cheek_roi(
    frame: np.ndarray, face_bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = face_bbox
    roi_w = int(w * 0.24)
    roi_h = int(h * 0.22)
    roi_x = x + int(w * 0.60)
    roi_y = y + int(h * 0.52)
    return _extract_box_roi(frame, roi_x, roi_y, roi_w, roi_h)


def extract_named_face_rois(
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    include_cheeks: bool = True,
) -> dict[str, tuple[np.ndarray, tuple[int, int, int, int]]]:
    rois = {"forehead": extract_forehead_roi(frame, face_bbox)}
    if include_cheeks:
        rois["left_cheek"] = extract_left_cheek_roi(frame, face_bbox)
        rois["right_cheek"] = extract_right_cheek_roi(frame, face_bbox)
    return rois
