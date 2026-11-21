"""MediaPipe's pretrained Face Landmarker as a drop-in webcam detector.

No facial-expression training is performed by this module. The upstream task
bundle provides 52 blendshape coefficients. We combine mouthSmileLeft and
mouthSmileRight into a *heuristic smile signal*, NOT an emotion probability.

The game, its score rules, and the Person 2 adapter do not depend on MediaPipe.
"""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
import threading
from urllib.request import Request, urlopen

import cv2
import numpy as np

from scoring import Prediction

# Versioned, official model URL used by Google's MediaPipe examples.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
_DOWNLOAD_LOCK = threading.Lock()
MIN_MODEL_SIZE_BYTES = 1_000_000


def ensure_model_file() -> Path:
    """Use an already downloaded model, or download once to app/models/.

    Set NEUTRAL_FACE_MODEL_PATH to point to a pre-downloaded .task file if
    the deployment environment has no outbound network connection.
    """
    path = Path(os.environ.get("NEUTRAL_FACE_MODEL_PATH", str(MODEL_PATH))).expanduser()
    with _DOWNLOAD_LOCK:
        if path.is_file() and path.stat().st_size >= MIN_MODEL_SIZE_BYTES:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=".face_landmarker_",
                suffix=".tmp", delete=False
            ) as temp:
                tmp_name = temp.name
                request = Request(MODEL_URL, headers={"User-Agent": "NeutralFaceGame/1.0"})
                with urlopen(request, timeout=35) as response:
                    shutil.copyfileobj(response, temp)
            if os.stat(tmp_name).st_size < MIN_MODEL_SIZE_BYTES:
                raise RuntimeError("Received an incomplete MediaPipe model file")
            os.replace(tmp_name, path)
        except Exception as exc:
            raise RuntimeError(
                "Could not download Google's pretrained face_landmarker.task. "
                "Check your network connection, or download the file from "
                f"{MODEL_URL} and place it at {path}. Original error: {exc}"
            ) from exc
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
        return path


def smile_score_from_blendshapes(categories) -> float:
    """Mean of two ARKit-style mouth smile coefficients, in [0, 1].

    Coefficients describe mouth geometry; the value is not the likelihood
    that the person is happy or a calibrated smile-class probability.
    """
    scores = {
        c.category_name: float(c.score)
        for c in categories
        if hasattr(c, "category_name") and hasattr(c, "score")
    }
    left = scores.get("mouthSmileLeft", 0.0)
    right = scores.get("mouthSmileRight", 0.0)
    return float(np.clip((left + right) / 2.0, 0.0, 1.0))


def bounding_box_from_landmarks(landmarks, height: int, width: int):
    """Compute (x, y, w, h) around the returned landmarks in source pixels."""
    if not landmarks:
        return None
    xs = np.clip([pt.x for pt in landmarks], 0.0, 1.0)
    ys = np.clip([pt.y for pt in landmarks], 0.0, 1.0)
    left = max(0, int(np.floor(min(xs) * width)))
    top = max(0, int(np.floor(min(ys) * height)))
    right = min(width, int(np.ceil(max(xs) * width)))
    bottom = min(height, int(np.ceil(max(ys) * height)))
    return (left, top, right - left, bottom - top)


def prediction_from_result(result, frame_shape, timestamp: float) -> Prediction:
    """Pure result conversion; intentionally independently unit-testable."""
    if not result.face_landmarks:
        return Prediction(False, 0.0, timestamp, source="mediapipe_pretrained")
    height, width = frame_shape[:2]
    bbox = bounding_box_from_landmarks(result.face_landmarks[0], height, width)
    scores = result.face_blendshapes[0] if result.face_blendshapes else []
    return Prediction(
        face_detected=True,
        smile_score=smile_score_from_blendshapes(scores),
        timestamp=timestamp,
        bbox=bbox,
        source="mediapipe_pretrained",
    )


class MediaPipeSmileDetector:
    """Pretrained MediaPipe face and blendshape inference for webcam frames.

    Uses IMAGE mode with a lock because Streamlit/WebRTC can deliver concurrent
    frame callbacks. Inference is independent for each video frame.
    """
    label = "MEDIAPIPE PRETRAINED"

    def __init__(self):
        import mediapipe as mp

        self._mp = mp
        model_file = ensure_model_file()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._inference_lock = threading.Lock()

    def predict(self, frame_bgr, timestamp: float, **_ignored) -> Prediction:
        if frame_bgr is None or frame_bgr.size == 0:
            return Prediction(False, 0.0, timestamp, source="mediapipe_pretrained")
        # Preserve the ORIGINAL pixel dimensions for the returned box.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )
        with self._inference_lock:
            result = self._landmarker.detect(image)
        return prediction_from_result(result, frame_bgr.shape, timestamp)

    def close(self) -> None:
        with self._inference_lock:
            self._landmarker.close()
