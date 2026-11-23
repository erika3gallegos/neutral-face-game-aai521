"""Build processed arrays, quality flags and train/val/test splits.

    python -m src.data.prepare            # both datasets, writes splits + summary
    python -m src.data.prepare --no-figures

Outputs
  data/splits/genki4k.csv, data/splits/fer2013.csv   (committed: one row per image)
  data/processed/genki4k_64.npz, fer2013_64.npz      (git-ignored: X uint8 N x 64 x 64)
  results/data/summary.json + figures                (committed)

Rows are never deleted from the split files. Unusable images carry
keep=False and a drop_reason, so every number in the report can be traced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import FER_RAW, GENKI_RAW, IMG_SIZE, PROCESSED, RESULTS, SPLITS
from .datasets import index_fer, index_genki, load_fer_image, split_fer, split_genki
from .face import preprocess_face, to_gray

BLANK_STD = 8.0          # raw grayscale std below this = blank / near-uniform image
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}   # who keeps a cross-split duplicate


def dhash(img: np.ndarray, size: int = 16) -> str:
    """256-bit difference hash of a normalized crop; equal hashes = near-identical images."""
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def _process(df: pd.DataFrame, root: Path, is_fer: bool) -> tuple[pd.DataFrame, np.ndarray]:
    X = np.zeros((len(df), IMG_SIZE, IMG_SIZE), np.uint8)
    meta = {k: [] for k in ("orig_h", "orig_w", "raw_std", "face_method", "n_faces",
                            "bbox", "hash", "readable")}
    for i, p in enumerate(df["path"]):
        img = load_fer_image(p) if is_fer else cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            for k, v in zip(meta, (0, 0, 0.0, "unreadable", 0, "", "", False)):
                meta[k].append(v)
            continue
        g = to_gray(img)
        res = preprocess_face(img, assume_cropped=is_fer, center_fallback=not is_fer)
        X[i] = res.crop
        meta["orig_h"].append(g.shape[0]); meta["orig_w"].append(g.shape[1])
        meta["raw_std"].append(float(g.std()))
        meta["face_method"].append(res.method); meta["n_faces"].append(res.n_faces)
        meta["bbox"].append(" ".join(map(str, res.bbox)))
        meta["hash"].append(dhash(res.crop)); meta["readable"].append(True)
    out = df.copy()
    for k, v in meta.items():
        out[k] = v
    out["rel_path"] = [str(Path(p).relative_to(root)) if not p.startswith("csv:") else p
                       for p in out["path"]]
    return out, X


def _flag(df: pd.DataFrame) -> pd.DataFrame:
    """Set keep / drop_reason. Order matters: first reason wins."""
    df = df.copy()
    df["keep"] = True
    df["drop_reason"] = ""

    def drop(mask, reason):
        m = mask & df["keep"]
        df.loc[m, "keep"] = False
        df.loc[m, "drop_reason"] = reason

    drop(~df["readable"], "unreadable")
    drop(df["raw_std"] < BLANK_STD, "blank_or_uniform")

    live = df[df["keep"]]
    # 1) near-identical images that carry different labels: label noise, drop every copy
    n_lab = live.groupby("hash")["raw_label"].transform("nunique")
    drop(df.index.isin(n_lab.index[n_lab > 1]), "duplicate_conflicting_label")

    # 2) the same image in two splits: keep the copy in the most protected split
    live = df[df["keep"]].copy()
    live["prio"] = live["split"].map(SPLIT_PRIORITY)
    best = live.groupby("hash")["prio"].transform("min")
    drop(df.index.isin(live.index[live["prio"] > best]), "duplicate_of_heldout_split")

    # 3) repeated copies within one split: keep the first
    live = df[df["keep"]]
    dup = live.duplicated(["hash", "split"], keep="first")
    drop(df.index.isin(live.index[dup]), "duplicate_within_split")
    return df


def _summary(df: pd.DataFrame) -> dict:
    kept = df[df["keep"]]
    s = {
        "n_images": int(len(df)),
        "n_kept": int(len(kept)),
        "drop_reasons": df.loc[~df["keep"], "drop_reason"].value_counts().to_dict(),
        "split_sizes": kept["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "raw_label_counts_all": df["raw_label"].value_counts().to_dict(),
        "raw_label_by_split": {sp: kept.loc[kept["split"] == sp, "raw_label"].value_counts().to_dict()
                               for sp in ("train", "val", "test")},
        "game_label_by_split": {sp: {"neutral": int((g == 0).sum()), "expression": int((g == 1).sum())}
                                for sp, g in kept.groupby("split")["game_label"]},
        "face_method": df["face_method"].value_counts().to_dict(),
        "orig_size": {"h_min": int(df["orig_h"].min()), "h_median": float(df["orig_h"].median()),
                      "h_max": int(df["orig_h"].max()), "w_min": int(df["orig_w"].min()),
                      "w_median": float(df["orig_w"].median()), "w_max": int(df["orig_w"].max())},
    }
    if df["yaw"].notna().any():
        deg = np.degrees(df[["yaw", "pitch", "roll"]].abs())
        s["abs_pose_deg"] = {c: {"median": round(float(deg[c].median()), 1),
                                 "p95": round(float(deg[c].quantile(0.95)), 1),
                                 "max": round(float(deg[c].max()), 1)} for c in deg}
        s["n_faces_gt1"] = int((df["n_faces"] > 1).sum())
    return s


SPLIT_COLS = ["image_id", "rel_path", "raw_label", "game_label", "split", "keep", "drop_reason",
              "origin", "face_method", "n_faces", "bbox", "orig_h", "orig_w", "raw_std",
              "yaw", "pitch", "roll"]


def build(name: str, write_figures: bool = True) -> dict:
    if name == "genki":
        root, df = GENKI_RAW, index_genki(GENKI_RAW)
        source = {"layout": df.attrs.get("layout"), "head_pose": bool(df["yaw"].notna().any())}
        df["split"] = split_genki(df).values
        tag, is_fer = "genki4k", False
    else:
        root, df = FER_RAW, index_fer(FER_RAW)
        source = {"layout": df.attrs.get("layout"), "official_split": df.attrs.get("official_split")}
        if not source["official_split"]:
            print("[prepare] WARNING: FER file names carry no PublicTest/PrivateTest prefix; "
                  "val is carved from train and test is the whole test/ folder.", flush=True)
        df["split"] = split_fer(df).values
        tag, is_fer = "fer2013", True

    df, X = _process(df, root, is_fer)
    df = _flag(df)

    SPLITS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df[SPLIT_COLS].to_csv(SPLITS / f"{tag}.csv", index=False, float_format="%.6f")
    np.savez_compressed(PROCESSED / f"{tag}_{IMG_SIZE}.npz", X=X, image_id=df["image_id"].values)

    s = {"source": source, **_summary(df)}
    if write_figures:
        from .eda import dataset_figures
        dataset_figures(tag, df, X, RESULTS / "data")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["genki", "fer"], choices=["genki", "fer"])
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    (RESULTS / "data").mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "data" / "summary.json"
    summary = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in a.datasets:
        print(f"[prepare] {name} ...", flush=True)
        summary["genki4k" if name == "genki" else "fer2013"] = build(name, not a.no_figures)
    out_path.write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
