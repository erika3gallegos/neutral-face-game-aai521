"""Swap-in facial-expression detectors sharing one Prediction interface."""
from __future__ import annotations

import cv2
from scoring import Prediction
from pretrained_model import MediaPipeSmileDetector


class KeyboardDetector:
    """Deterministic demo: hold S=smile, hold X=face missing."""
    label = "KEYBOARD DEMO"

    def predict(self, frame_bgr, timestamp: float,
                smile_pressed: bool = False,
                missing_pressed: bool = False) -> Prediction:
        height, width = frame_bgr.shape[:2]
        return Prediction(not missing_pressed, float(smile_pressed), timestamp,
                          None if missing_pressed else
                          (int(width * 0.36), int(height * 0.13),
                           int(width * 0.28), int(height * 0.64)),
                          "keyboard_demo")


class HaarSmileDetector:
    """Existing, untrained-for-this-project OpenCV cascade placeholder.

    It detects a face and a rough smile-like pattern, not emotion or a
    calibrated smile probability. It is intentionally replaceable by Person 2.
    """
    label = "OPENCV DEMO"

    def __init__(self):
        data_dir = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(data_dir + "haarcascade_frontalface_default.xml")
        self.smile = cv2.CascadeClassifier(data_dir + "haarcascade_smile.xml")
        if self.face.empty() or self.smile.empty():
            raise RuntimeError("OpenCV face/smile cascade XML files were not found")

    def predict(self, frame_bgr, timestamp: float,
                smile_pressed: bool = False,
                missing_pressed: bool = False) -> Prediction:
        scale = 0.55
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face.detectMultiScale(gray, scaleFactor=1.12,
                                          minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            return Prediction(False, 0.0, timestamp, source="opencv_haar_demo")
        x, y, w, h = map(int, max(faces, key=lambda r: int(r[2]) * int(r[3])))
        # Limit to lower half of face to avoid eye/eyebrow false positives.
        mouth_y = y + int(h * 0.48)
        mouth_roi = gray[mouth_y:y + h, x:x + w]
        smiles = self.smile.detectMultiScale(mouth_roi, scaleFactor=1.6,
                                             minNeighbors=22,
                                             minSize=(int(w * 0.18), 12))
        inv = 1.0 / scale
        bbox = (round(x * inv), round(y * inv), round(w * inv), round(h * inv))
        return Prediction(True, 1.0 if len(smiles) else 0.0, timestamp,
                          bbox, "opencv_haar_demo")


class Person2Detector:
    """Adapter for Person 2's trained detector, activated with READY=True."""
    label = "PERSON 2 MODEL"

    def __init__(self):
        from person2_adapter import READY, predict_expression
        if not READY:
            raise RuntimeError("Person 2's model is not connected yet")
        self.predict_expression = predict_expression

    def predict(self, frame_bgr, timestamp: float,
                smile_pressed: bool = False,
                missing_pressed: bool = False) -> Prediction:
        raw = self.predict_expression(frame_bgr, timestamp)
        if isinstance(raw, Prediction):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("Person 2 must return a Prediction or dict")
        return Prediction(
            face_detected=bool(raw["face_detected"]),
            smile_score=float(raw["smile_score"]),
            timestamp=float(raw.get("timestamp", timestamp)),
            bbox=tuple(raw["bbox"]) if raw.get("bbox") is not None else None,
            source="person2_model")


def available_modes() -> list[str]:
    modes = ["MEDIAPIPE PRETRAINED", "OPENCV DEMO", "KEYBOARD DEMO"]
    try:
        from person2_adapter import READY
        if READY:
            modes.append("PERSON 2 MODEL")
    except ImportError:
        pass
    return modes


def build_detector(mode: str):
    if mode == "MEDIAPIPE PRETRAINED":
        return MediaPipeSmileDetector()
    if mode == "OPENCV DEMO":
        return HaarSmileDetector()
    if mode == "KEYBOARD DEMO":
        return KeyboardDetector()
    if mode == "PERSON 2 MODEL":
        return Person2Detector()
    raise ValueError(f"Unknown detector mode: {mode}")
