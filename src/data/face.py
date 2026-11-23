"""Face detection, cropping and normalization shared by training and webcam.

The same `preprocess_face` must run on GENKI photos and on live webcam frames,
otherwise the model sees a different input distribution at game time.

Steps: BGR/gray -> grayscale -> largest frontal face (OpenCV Haar cascade)
-> square crop with a small margin -> resize to IMG_SIZE -> histogram
equalization. FER-2013 images are already tight 48x48 face crops, so they skip
detection (`assume_cropped=True`) and only get resized and equalized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import FACE_MARGIN, IMG_SIZE

BBox = tuple[int, int, int, int]  # x, y, w, h in source-image pixels

_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def _cascade(name: str) -> cv2.CascadeClassifier:
    if name not in _CASCADES:
        path = cv2.data.haarcascades + name
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            raise RuntimeError(
                f"Could not load {path}. Use opencv-python(-headless) 4.x; "
                "the 5.x wheels no longer ship cv2.data.haarcascades.")
        _CASCADES[name] = clf
    return _CASCADES[name]


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@dataclass
class FaceResult:
    crop: Optional[np.ndarray]   # IMG_SIZE x IMG_SIZE uint8, or None
    bbox: Optional[BBox]         # crop box in source pixels, or None
    n_faces: int                 # raw detections from the primary cascade
    method: str                  # "haar_default", "haar_alt2", "center", "precropped", "none"


def detect_faces(gray: np.ndarray, cascade: str = "haarcascade_frontalface_default.xml",
                 min_frac: float = 0.15) -> list[BBox]:
    h, w = gray.shape[:2]
    m = max(24, int(min(h, w) * min_frac))
    eq = cv2.equalizeHist(gray)
    faces = _cascade(cascade).detectMultiScale(
        eq, scaleFactor=1.1, minNeighbors=5, minSize=(m, m))
    return [tuple(int(v) for v in f) for f in faces]


def square_box(box: BBox, img_shape, margin: float = FACE_MARGIN) -> BBox:
    """Grow a detection box by `margin` and make it square, clipped to the image."""
    x, y, w, h = box
    H, W = img_shape[:2]
    side = int(round(max(w, h) * (1 + 2 * margin)))
    side = min(side, H, W)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = min(max(0, x0), W - side)
    y0 = min(max(0, y0), H - side)
    return x0, y0, side, side


def normalize(gray_crop: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    interp = cv2.INTER_AREA if gray_crop.shape[0] > size else cv2.INTER_LINEAR
    out = cv2.resize(gray_crop, (size, size), interpolation=interp)
    return cv2.equalizeHist(out)


def preprocess_face(img: np.ndarray, assume_cropped: bool = False,
                    center_fallback: bool = False) -> FaceResult:
    """Return a normalized face crop from a BGR or grayscale image.

    assume_cropped:  skip detection (FER-2013).
    center_fallback: if no face is found, use the central square instead of
                     returning None. Used only for GENKI training images, which
                     were already selected by a face detector; never for webcam.
    """
    gray = to_gray(img)
    H, W = gray.shape[:2]
    if assume_cropped:
        return FaceResult(normalize(gray), (0, 0, W, H), 1, "precropped")

    faces = detect_faces(gray)
    method = "haar_default"
    n = len(faces)
    if not faces:
        faces = detect_faces(gray, "haarcascade_frontalface_alt2.xml")
        method = "haar_alt2"
    if faces:
        box = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = square_box(box, gray.shape)
        return FaceResult(normalize(gray[y:y + h, x:x + w]), (x, y, w, h), n, method)
    if center_fallback:
        side = min(H, W)
        x, y = (W - side) // 2, (H - side) // 2
        return FaceResult(normalize(gray[y:y + side, x:x + side]), (x, y, side, side), 0, "center")
    return FaceResult(None, None, 0, "none")
