"""Data loading, preprocessing and label mapping (Person 1)."""
from .config import IMG_SIZE
from .face import preprocess_face

__all__ = ["IMG_SIZE", "preprocess_face"]
