"""Thread-safe, frame-only interface between browser WebRTC and the game.

The WebRTC callback executes outside Streamlit's script thread. It may NOT
read or modify st.session_state, call st.*, or update the GameEngine. Only
latest predictions and error strings pass through this thread-safe bridge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

import cv2

from scoring import Prediction


STALE_AFTER_SECONDS = 1.0


@dataclass
class LatestPrediction:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _prediction: Prediction | None = None
    _error: str | None = None

    def publish(self, prediction: Prediction) -> None:
        with self._lock:
            self._prediction = prediction
            self._error = None

    def publish_error(self, message: str) -> None:
        with self._lock:
            self._prediction = None
            self._error = message[:200]

    def clear(self) -> None:
        with self._lock:
            self._prediction = None
            self._error = None

    def read(self, now: float, max_age: float = STALE_AFTER_SECONDS) -> tuple[Prediction, str | None]:
        with self._lock:
            p, error = self._prediction, self._error
        if p is None or p.timestamp > now + 0.2 or now - p.timestamp > max_age:
            return Prediction(False, 0.0, now, source="camera_unavailable"), error
        return p, error


def simulated_prediction(*, face_visible: bool, smile: bool, now: float) -> Prediction:
    """Same contract as the CV model; no browser camera is used."""
    return Prediction(face_visible, float(smile) if face_visible else 0.0,
                      now, source="simulation")


class WebcamProcessor:
    """On-frame WebRTC processing using either a placeholder or Person 2 model."""

    def __init__(self, detector, latest: LatestPrediction):
        self.detector = detector
        self.latest = latest

    def on_frame(self, frame):
        """AV VideoFrame -> AV VideoFrame; no Streamlit operations here."""
        import av  # Loaded only when app is used with the webcam.
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        try:
            prediction = self.detector.predict(image, now)
            self.latest.publish(prediction)
            rendered = image.copy()
            if prediction.face_detected and prediction.bbox:
                x, y, w, h = (int(v) for v in prediction.bbox)
                cv2.rectangle(rendered, (x, y), (x + w, y + h), (35, 190, 135), 2)
            status = (f"SMILE SIGNAL: {prediction.smile_score:.2f}"
                      if prediction.face_detected else "NO FACE DETECTED")
            cv2.rectangle(rendered, (0, 0), (min(360, rendered.shape[1]), 38),
                          (22, 40, 54), -1)
            cv2.putText(rendered, status, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as exc:
            self.latest.publish_error(f"Detector error: {type(exc).__name__}: {exc}")
            rendered = image  # Keep webcam feed usable while reporting error.
        return av.VideoFrame.from_ndarray(rendered, format="bgr24")
