"""Shared paths and constants for the data pipeline and baseline.

Everything downstream (baseline, improved detector, webcam game) should import
IMG_SIZE and the preprocessing functions from src.data so training images and
webcam frames go through the same steps.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

GENKI_RAW = RAW / "genki4k"
FER_RAW = RAW / "fer2013"

SEED = 42
IMG_SIZE = 64            # every face crop is IMG_SIZE x IMG_SIZE grayscale uint8
FACE_MARGIN = 0.10       # extra border around the Haar box, as a fraction of box size
GENKI_SPLIT = (0.70, 0.15, 0.15)   # train / val / test, stratified by smile label

FER_CLASSES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

# Game label: 0 = neutral, 1 = expression (anything the game should penalize)
GAME_LABEL = {
    "genki": {0: 0, 1: 1},
    "fer": {c: (0 if c == "neutral" else 1) for c in FER_CLASSES},
}
