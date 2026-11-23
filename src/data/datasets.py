"""Index GENKI-4K and FER-2013 into tidy tables with game labels and splits.

Both loaders tolerate the different layouts people end up with:

GENKI-4K  any folder tree that contains labels.txt and fileNNNN.jpg images
          (Kaggle talhasar/genki4k, the original UCSD zip, or Smile/not_Smile
          sub-folders). Labels always come from labels.txt, line i = file{i:04d}.
FER-2013  (a) Training/ PublicTest/ PrivateTest/ <Class>/ *.jpg  (original Usage)
          (b) train/ test/ <class>/ *.jpg   (Kaggle msambare/fer2013)
          (c) fer2013.csv with emotion, pixels, Usage columns
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FER_CLASSES, GAME_LABEL, GENKI_SPLIT, SEED

_GENKI_RE = re.compile(r"file(\d{4})\.jpe?g$", re.IGNORECASE)
_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _walk(root: Path):
    """Recursive file listing that also follows symlinked folders."""
    import os
    for d, _, files in os.walk(root, followlinks=True):
        for f in files:
            yield Path(d) / f


# --------------------------------------------------------------------------- GENKI
_SMILE_DIRS = {"smile": 1, "smiling": 1, "1": 1,
               "not_smile": 0, "non_smile": 0, "nonsmile": 0, "no_smile": 0,
               "not_smiling": 0, "non-smile": 0, "0": 0}


def index_genki(root: Path) -> pd.DataFrame:
    """Index GENKI-4K. df.attrs["layout"] records which layout was found.

    Preferred: labels.txt + fileNNNN.jpg (UCSD release, Kaggle talhasar/genki4k).
    Fallback: images inside smile / non_smile folders with no labels.txt; head
    pose is then unavailable (NaN) and the pose analyses are skipped.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run: kaggle datasets download -d talhasar/genki4k -p {root} --unzip")
    files = list(_walk(root))
    labels = sorted(p for p in files if p.name.lower() == "labels.txt")
    imgs = [p for p in files if p.suffix.lower() in _IMG_EXT]

    if labels:
        lab = pd.read_csv(labels[0], sep=r"\s+", header=None,
                          names=["smile", "yaw", "pitch", "roll"])
        lab["num"] = np.arange(1, len(lab) + 1)
        paths = {}
        for p in imgs:
            m = _GENKI_RE.search(p.name)
            if m:
                paths.setdefault(int(m.group(1)), p)
        if not paths:
            raise FileNotFoundError(f"labels.txt found but no fileNNNN.jpg images under {root}.")
        lab = lab[lab["num"].isin(paths)].copy()
        df = pd.DataFrame({
            "dataset": "genki",
            "image_id": [f"genki_{n:04d}" for n in lab["num"]],
            "path": [str(paths[n]) for n in lab["num"]],
            "game_label": lab["smile"].astype(int).values,
            "yaw": lab["yaw"].values, "pitch": lab["pitch"].values, "roll": lab["roll"].values,
        })
        layout = "labels_txt"
    else:
        rows = []
        for p in imgs:
            y = _SMILE_DIRS.get(p.parent.name.lower())
            if y is not None:
                m = re.search(r"(\d+)", p.stem)
                rows.append((p, y, int(m.group(1)) if m else None))
        if not rows:
            raise FileNotFoundError(
                f"No labels.txt and no smile / non_smile image folders under {root}.")
        df = pd.DataFrame(rows, columns=["path", "game_label", "num"]).sort_values("path")
        df["image_id"] = [f"genki_{n:04d}" if n is not None and not pd.isna(n) else f"genki_{Path(pth).stem}"
                          for n, pth in zip(df["num"], df["path"])]
        df["path"] = df["path"].astype(str)
        df["dataset"] = "genki"
        for c in ("yaw", "pitch", "roll"):
            df[c] = np.nan
        layout = "label_folders"

    df["raw_label"] = np.where(df["game_label"] == 1, "smile", "non_smile")
    df["game_label"] = df["game_label"].map(GAME_LABEL["genki"]).astype(int)
    df["origin"] = "genki"
    df = df[["dataset", "image_id", "path", "raw_label", "game_label",
             "yaw", "pitch", "roll", "origin"]].reset_index(drop=True)
    if not df["image_id"].is_unique:
        raise ValueError("GENKI image ids are not unique; check for duplicate copies of the dataset.")

    # If labels.txt was used and images also sit in label-named folders, cross-check them.
    if layout == "labels_txt":
        folder = df["path"].map(lambda s: Path(s).parent.name.lower())
        named = folder.isin(set(_SMILE_DIRS))
        if named.any():
            f_lab = folder[named].map(_SMILE_DIRS)
            bad = int((f_lab != df.loc[named, "game_label"]).sum())
            if bad:
                raise ValueError(f"{bad} GENKI images sit in a folder that contradicts labels.txt")
    df.attrs["layout"] = layout
    return df


def split_genki(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Stratified 70/15/15 split, deterministic given the seed and image ids."""
    tr, va, te = GENKI_SPLIT
    ids = df["image_id"].values
    y = df["game_label"].values
    id_tr, id_rest, _, y_rest = train_test_split(ids, y, test_size=va + te,
                                                 stratify=y, random_state=seed)
    id_va, id_te = train_test_split(id_rest, test_size=te / (va + te),
                                    stratify=y_rest, random_state=seed)
    split = pd.Series("train", index=df.index)
    split[df["image_id"].isin(id_va)] = "val"
    split[df["image_id"].isin(id_te)] = "test"
    return split


# --------------------------------------------------------------------------- FER
def _fer_class(name: str) -> str | None:
    n = name.lower()
    return n if n in FER_CLASSES else None


_USAGE = {"training": "Training", "publictest": "PublicTest", "privatetest": "PrivateTest"}


def _find_fer_root(root: Path, depth: int = 3) -> Path | None:
    """Kaggle zips sometimes unpack into an extra folder (archive/, fer2013/).
    Return the first folder at most `depth` levels down that holds the split folders."""
    frontier = [root]
    for _ in range(depth + 1):
        nxt = []
        for d in frontier:
            if not d.is_dir():
                continue
            names = {p.name.lower() for p in d.iterdir() if p.is_dir()}
            if {"train", "test"} <= names or set(_USAGE) <= names:
                return d
            nxt += [p for p in d.iterdir() if p.is_dir()]
        frontier = nxt
    return None


def _usage_from_name(name: str) -> str | None:
    """Kaggle msambare/fer2013 names files <Usage>_<id>.jpg, e.g. PrivateTest_91810341.jpg."""
    return _USAGE.get(name.split("_")[0].lower())


def index_fer(root: Path) -> pd.DataFrame:
    """Index FER-2013. df.attrs["layout"] records which layout was found."""
    root = Path(root)
    rows = []
    base = _find_fer_root(root) if root.exists() else None
    sub = {p.name.lower(): p for p in base.iterdir() if p.is_dir()} if base else {}

    if set(_USAGE) <= set(sub):                                        # layout (a)
        layout = "usage_folders"
        for key, origin in _USAGE.items():
            for cdir in sub[key].iterdir():
                cls = _fer_class(cdir.name)
                if cls is None:
                    continue
                for p in cdir.iterdir():
                    if p.suffix.lower() in _IMG_EXT:
                        rows.append((p, cls, origin))
    elif {"train", "test"} <= set(sub):                                # layout (b): Kaggle msambare
        layout = "kaggle_train_test"
        for key in ("train", "test"):
            for cdir in sub[key].iterdir():
                cls = _fer_class(cdir.name)
                if cls is None:
                    continue
                for p in cdir.iterdir():
                    if p.suffix.lower() not in _IMG_EXT:
                        continue
                    origin = _usage_from_name(p.name) or ("Training" if key == "train" else "test")
                    rows.append((p, cls, origin))
    else:                                                              # layout (c)
        layout = "csv"
        csvs = sorted(p for p in _walk(root) if p.name == "fer2013.csv")
        if not csvs:
            raise FileNotFoundError(
                f"No FER-2013 train/test folders or fer2013.csv under {root}. "
                "Run: kaggle datasets download -d msambare/fer2013 -p data/raw/fer2013 --unzip")
        raw = pd.read_csv(csvs[0])
        for i, r in raw.iterrows():
            rows.append((f"csv:{csvs[0]}:{i}", FER_CLASSES[int(r["emotion"])], r["Usage"]))

    df = pd.DataFrame(rows, columns=["path", "raw_label", "origin"])
    df["path"] = df["path"].astype(str)
    df = df.sort_values("path").reset_index(drop=True)
    stem = df["path"].map(lambda s: Path(s).stem if not s.startswith("csv:") else s.rsplit(":", 1)[1])
    df.insert(0, "image_id", "fer_" + df["origin"].str.lower() + "_" + df["raw_label"] + "_" + stem)
    df.insert(0, "dataset", "fer")
    df["game_label"] = df["raw_label"].map(GAME_LABEL["fer"]).astype(int)
    for c in ("yaw", "pitch", "roll"):
        df[c] = np.nan
    out = df[["dataset", "image_id", "path", "raw_label", "game_label",
              "yaw", "pitch", "roll", "origin"]].copy()
    out.attrs["layout"] = layout
    out.attrs["official_split"] = bool((out["origin"] == "PublicTest").any()
                                       and (out["origin"] == "PrivateTest").any())
    return out


def split_fer(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Official FER-2013 usage: Training->train, PublicTest->val, PrivateTest->test.

    Kaggle msambare/fer2013 keeps the usage in each file name (PublicTest_*.jpg,
    PrivateTest_*.jpg), so the official split is recovered from it. Only if the
    names carry no usage prefix does val become a stratified 3,589-image slice
    of train (the size of PublicTest) and test the whole test/ folder.
    """
    split = df["origin"].map({"Training": "train", "PublicTest": "val",
                              "PrivateTest": "test", "test": "test"})
    if (split == "val").sum() == 0:
        tr_idx = df.index[split == "train"]
        _, va_idx = train_test_split(tr_idx, test_size=3589, random_state=seed,
                                     stratify=df.loc[tr_idx, "raw_label"])
        split[va_idx] = "val"
    return split


_CSV_CACHE: dict[str, pd.Series] = {}


def load_fer_image(path: str):
    """Read one FER image as 48x48 uint8 from disk or from a csv:<file>:<row> key."""
    import cv2
    if path.startswith("csv:"):
        f, row = path[4:].rsplit(":", 1)
        if f not in _CSV_CACHE:
            _CSV_CACHE[f] = pd.read_csv(f, usecols=["pixels"])["pixels"]
        px = _CSV_CACHE[f].iloc[int(row)]
        return np.array(px.split(), dtype=np.uint8).reshape(48, 48)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
