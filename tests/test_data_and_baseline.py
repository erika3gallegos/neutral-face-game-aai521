"""Labeled tests for the data pipeline and the frozen baseline.

Run from the repo root:  python -m pytest tests -q
Tests that need the downloaded data or trained model skip when those are absent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.config import GAME_LABEL, IMG_SIZE, MODELS, PROCESSED, SPLITS
from src.data.face import preprocess_face, square_box

HAS_SPLITS = (SPLITS / "genki4k.csv").exists() and (SPLITS / "fer2013.csv").exists()
HAS_MODEL = (MODELS / "baseline_genki_smile.joblib").exists()
HAS_ARRAYS = (PROCESSED / "genki4k_64.npz").exists()


# ----------------------------------------------------------------------- label mapping
def test_label_mapping_neutral_is_zero():
    assert GAME_LABEL["genki"] == {0: 0, 1: 1}
    assert GAME_LABEL["fer"]["neutral"] == 0
    assert all(v == 1 for k, v in GAME_LABEL["fer"].items() if k != "neutral")
    assert len(GAME_LABEL["fer"]) == 7


# ----------------------------------------------------------------------- preprocessing
def test_square_box_stays_inside_image():
    for box in [(0, 0, 50, 60), (90, 90, 40, 40), (10, 5, 200, 150)]:
        x, y, w, h = square_box(box, (120, 160))
        assert w == h and x >= 0 and y >= 0 and x + w <= 160 and y + h <= 120


def test_blank_frame_has_no_face():
    r = preprocess_face(np.zeros((480, 640, 3), np.uint8))
    assert r.crop is None and r.bbox is None and r.method == "none"


def test_center_fallback_and_precropped_shapes():
    img = np.random.default_rng(0).integers(0, 255, (200, 180, 3), dtype=np.uint8)
    r = preprocess_face(img, center_fallback=True)
    assert r.crop.shape == (IMG_SIZE, IMG_SIZE) and r.crop.dtype == np.uint8
    r = preprocess_face(img[:48, :48, 0], assume_cropped=True)
    assert r.crop.shape == (IMG_SIZE, IMG_SIZE) and r.method == "precropped"


# ----------------------------------------------------------------------- Kaggle layouts
def _touch_jpg(path: Path):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((48, 48), 128, np.uint8))


def test_fer_kaggle_layout_recovers_official_split(tmp_path):
    from src.data.datasets import index_fer, split_fer
    root = tmp_path / "fer2013" / "archive"          # Kaggle zips may add a folder level
    _touch_jpg(root / "train" / "happy" / "Training_101.jpg")
    _touch_jpg(root / "train" / "neutral" / "Training_102.jpg")
    _touch_jpg(root / "test" / "happy" / "PublicTest_201.jpg")
    _touch_jpg(root / "test" / "neutral" / "PrivateTest_301.jpg")
    df = index_fer(tmp_path / "fer2013")
    assert df.attrs["layout"] == "kaggle_train_test" and df.attrs["official_split"]
    split = dict(zip(df["path"].map(lambda s: Path(s).name), split_fer(df)))
    assert split == {"Training_101.jpg": "train", "Training_102.jpg": "train",
                     "PublicTest_201.jpg": "val", "PrivateTest_301.jpg": "test"}
    assert df.set_index(df["path"].map(lambda s: Path(s).name))["game_label"].to_dict() == {
        "Training_101.jpg": 1, "Training_102.jpg": 0, "PublicTest_201.jpg": 1, "PrivateTest_301.jpg": 0}


def test_genki_labels_txt_and_folder_layouts_agree(tmp_path):
    from src.data.datasets import index_genki
    a = tmp_path / "a" / "GENKI-4K"
    (a).mkdir(parents=True)
    (a / "labels.txt").write_text("1 0.1 0 0\n0 -0.2 0 0\n")
    _touch_jpg(a / "files" / "file0001.jpg"); _touch_jpg(a / "files" / "file0002.jpg")
    b = tmp_path / "b"
    _touch_jpg(b / "smile" / "file0001.jpg"); _touch_jpg(b / "non_smile" / "file0002.jpg")
    ga, gb = index_genki(tmp_path / "a"), index_genki(b)
    assert ga.attrs["layout"] == "labels_txt" and gb.attrs["layout"] == "label_folders"
    assert ga.set_index("image_id")["game_label"].to_dict() == gb.set_index("image_id")["game_label"].to_dict() \
        == {"genki_0001": 1, "genki_0002": 0}
    assert ga["yaw"].notna().all() and gb["yaw"].isna().all()


# ----------------------------------------------------------------------- splits
@pytest.mark.skipif(not HAS_SPLITS, reason="run python -m src.data.prepare first")
@pytest.mark.parametrize("tag,n", [("genki4k", 4000), ("fer2013", 35887)])
def test_split_files_complete_and_disjoint(tag, n):
    df = pd.read_csv(SPLITS / f"{tag}.csv")
    assert len(df) == n and df["image_id"].is_unique
    assert set(df["split"]) == {"train", "val", "test"}
    assert (df.loc[~df["keep"], "drop_reason"].fillna("") != "").all()


@pytest.mark.skipif(not HAS_SPLITS, reason="run python -m src.data.prepare first")
def test_genki_labels_and_stratification():
    df = pd.read_csv(SPLITS / "genki4k.csv")
    assert (df["raw_label"] == "smile").sum() == 2162          # published GENKI-4K count
    rate = df[df["keep"]].groupby("split")["game_label"].mean()
    assert rate.max() - rate.min() < 0.02                       # stratified


@pytest.mark.skipif(not HAS_SPLITS, reason="run python -m src.data.prepare first")
def test_fer_split_sizes():
    df = pd.read_csv(SPLITS / "fer2013.csv")
    counts = df.groupby("split").size()
    assert counts.sum() == 35887
    if {"PublicTest", "PrivateTest"} <= set(df["origin"]):     # Kaggle msambare keeps usage in names
        assert counts["train"] == 28709 and counts["val"] == 3589 and counts["test"] == 3589
    else:                                                      # no usage prefix: val carved from train
        assert counts["val"] == 3589 and counts["test"] == 7178


# ----------------------------------------------------------------------- baseline
@pytest.mark.skipif(not (HAS_MODEL and HAS_ARRAYS and HAS_SPLITS), reason="needs data + trained model")
def test_frozen_baseline_beats_chance_on_val():
    from sklearn.metrics import roc_auc_score
    from src.models.baseline import HOGBaseline
    df = pd.read_csv(SPLITS / "genki4k.csv")
    X = np.load(PROCESSED / "genki4k_64.npz", allow_pickle=True)["X"]
    m = (df["split"] == "val").values & df["keep"].values
    s = HOGBaseline.load(MODELS / "baseline_genki_smile").score(X[m])
    assert ((s >= 0) & (s <= 1)).all()
    assert roc_auc_score(df.loc[m, "game_label"], s) > 0.9


@pytest.mark.skipif(not HAS_MODEL, reason="needs trained model")
def test_detector_contracts_on_empty_frame():
    from src.models.baseline_detector import predict, predict_expression
    frame = np.zeros((480, 640, 3), np.uint8)
    out = predict(frame)
    assert out["face_usable"] is False and out["is_neutral"] is False
    app = predict_expression(frame, 12.5)
    assert app == {"face_detected": False, "smile_score": 0.0, "timestamp": 12.5, "bbox": None}
