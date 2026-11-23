"""Baseline expression detector: HOG features + logistic regression.

Input:  64x64 grayscale face crops from src.data.face.preprocess_face
Output: P(expression), used as the game's smile_score (0..1).

Logistic regression is used instead of an SVM so the score is a probability-like
value the game can threshold, which is what the app's Prediction.smile_score
expects. It is not a calibrated probability of an emotion.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class HOGConfig:
    orientations: int = 9
    pixels_per_cell: int = 8
    cells_per_block: int = 2

    def features(self, X: np.ndarray) -> np.ndarray:
        """X: (N, H, W) uint8 -> (N, D) float32 HOG descriptors."""
        ppc, cpb = self.pixels_per_cell, self.cells_per_block
        return np.stack([
            hog(x, orientations=self.orientations, pixels_per_cell=(ppc, ppc),
                cells_per_block=(cpb, cpb), block_norm="L2-Hys", feature_vector=True)
            for x in X]).astype(np.float32)


@dataclass
class BaselineConfig:
    hog: HOGConfig = field(default_factory=HOGConfig)
    C: float = 0.01
    flip_augment: bool = True
    threshold: float = 0.5
    task: str = "genki_smile"


class HOGBaseline:
    def __init__(self, cfg: BaselineConfig | None = None):
        self.cfg = cfg or BaselineConfig()
        self.clf: Pipeline | None = None

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, y: np.ndarray, feats: np.ndarray | None = None):
        if self.cfg.flip_augment:          # training-only augmentation
            X = np.concatenate([X, X[:, :, ::-1]])
            y = np.concatenate([y, y])
            feats = None
        F = feats if feats is not None else self.cfg.hog.features(X)
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.cfg.C, class_weight="balanced", max_iter=3000))
        self.clf.fit(F, y)
        return self

    # ------------------------------------------------------------------ inference
    def score(self, X: np.ndarray, feats: np.ndarray | None = None) -> np.ndarray:
        """P(expression) for each 64x64 crop."""
        F = feats if feats is not None else self.cfg.hog.features(X)
        return self.clf.predict_proba(F)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) >= self.cfg.threshold).astype(int)

    # ------------------------------------------------------------------ io
    def save(self, path: Path, extra: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cfg": asdict(self.cfg), "clf": self.clf}, path.with_suffix(".joblib"))
        meta = {"config": asdict(self.cfg), **(extra or {})}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, path: Path) -> "HOGBaseline":
        blob = joblib.load(Path(path).with_suffix(".joblib"))
        c = blob["cfg"]
        cfg = BaselineConfig(hog=HOGConfig(**c["hog"]), C=c["C"], flip_augment=c["flip_augment"],
                             threshold=c["threshold"], task=c["task"])
        m = cls(cfg)
        m.clf = blob["clf"]
        return m
