"""Webcam-frame wrapper around the frozen HOG baseline.

Two entry points, because the repo currently has two contracts:

  predict(frame)                         README "Detector output contract"
  predict_expression(frame, timestamp)   app/person2_adapter.py contract (what the
                                         Streamlit game actually consumes)

To try the baseline in the game, paste into app/person2_adapter.py:

    import sys; sys.path.insert(0, "..")          # repo root on the path
    from src.models.baseline_detector import predict_expression
    READY = True
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from src.data.config import MODELS
from src.data.face import preprocess_face
from src.models.baseline import HOGBaseline

MODEL_NAME = "baseline_genki_smile"


@lru_cache(maxsize=1)
def _model() -> HOGBaseline:
    return HOGBaseline.load(MODELS / MODEL_NAME)


def _score(frame_bgr: np.ndarray):
    face = preprocess_face(frame_bgr)            # no center fallback on live frames
    if face.crop is None:
        return face, None
    return face, float(_model().score(face.crop[None])[0])


def predict(frame_bgr: np.ndarray) -> dict:
    face, s = _score(frame_bgr)
    m = _model()
    usable = s is not None
    return {
        "timestamp": time.time(),
        "face_usable": usable,
        "is_neutral": bool(usable and s < m.cfg.threshold),
        "scores": {"neutral": 1 - s, "smile": s} if usable else {},
        "model": MODEL_NAME,
        "bbox": face.bbox,
    }


def predict_expression(frame_bgr: np.ndarray, timestamp: float) -> dict:
    face, s = _score(frame_bgr)
    return {
        "face_detected": s is not None,
        "smile_score": float(np.clip(s, 0.0, 1.0)) if s is not None else 0.0,
        "timestamp": timestamp,
        "bbox": face.bbox,
    }
