"""Report figures for the dataset and preprocessing sections."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#2b2b2b"
C_NEU = "#4c78a8"   # neutral / non-smile
C_EXP = "#e45756"   # expression / smile
SPLIT_ORDER = ["train", "val", "test"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.dpi": 150, "savefig.bbox": "tight"})


def _grid(ax_row, imgs, title=None):
    for ax, im in zip(ax_row, imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if title:
        ax_row[0].set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9)


def class_balance(tag, df, out: Path):
    kept = df[df["keep"]]
    tab = kept.pivot_table(index="raw_label", columns="split", values="image_id",
                           aggfunc="count").reindex(columns=SPLIT_ORDER).fillna(0)
    tab = tab.loc[tab.sum(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6, 2.6))
    left = np.zeros(len(tab))
    shades = {"train": "#4c78a8", "val": "#9ecae9", "test": "#f58518"}
    for sp in SPLIT_ORDER:
        ax.barh(tab.index, tab[sp], left=left, color=shades[sp], label=sp, height=0.65)
        left += tab[sp].values
    for y, tot in enumerate(left):
        ax.text(tot, y, f" {int(tot):,}", va="center", fontsize=8, color=INK)
    ax.invert_yaxis(); ax.set_xlabel("images kept")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    ax.set_title(f"{tag}: label counts by split", loc="left", fontsize=10)
    fig.savefig(out / f"{tag}_class_balance.png"); plt.close(fig)


def samples(tag, df, X, out: Path, per=8, seed=0):
    rng = np.random.default_rng(seed)
    kept = df[df["keep"]]
    labels = list(kept["raw_label"].value_counts().index)
    fig, axes = plt.subplots(len(labels), per, figsize=(per * 0.9, len(labels) * 0.95))
    axes = np.atleast_2d(axes)
    for r, lab in enumerate(labels):
        idx = kept.index[kept["raw_label"] == lab].values
        pick = rng.choice(idx, size=min(per, len(idx)), replace=False)
        _grid(axes[r], [X[i] for i in pick], lab)
    fig.suptitle(f"{tag}: preprocessed 64x64 crops (random, kept images)", x=0.02, ha="left", fontsize=10)
    fig.savefig(out / f"{tag}_samples.png"); plt.close(fig)


def dropped(tag, df, X, out: Path, per=8):
    d = df[~df["keep"]]
    if d.empty:
        return
    reasons = list(d["drop_reason"].value_counts().index)
    fig, axes = plt.subplots(len(reasons), per, figsize=(per * 0.9, len(reasons) * 0.95), squeeze=False)
    for r, rs in enumerate(reasons):
        idx = d.index[d["drop_reason"] == rs].values[:per]
        _grid(axes[r], [X[i] for i in idx] + [np.full_like(X[0], 255)] * (per - len(idx)),
              f"{rs}\n(n={int((d['drop_reason'] == rs).sum())})")
    fig.suptitle(f"{tag}: examples of removed images", x=0.02, ha="left", fontsize=10)
    fig.savefig(out / f"{tag}_dropped.png"); plt.close(fig)


def genki_pose(df, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.2), sharey=True)
    for ax, c in zip(axes, ["yaw", "pitch", "roll"]):
        bins = np.linspace(-45, 45, 31)
        for lab, col, name in ((0, C_NEU, "non-smile"), (1, C_EXP, "smile")):
            v = np.degrees(df.loc[df["game_label"] == lab, c])
            ax.hist(v.clip(-45, 45), bins=bins, color=col, alpha=0.55, label=name)
        ax.set_xlabel(f"{c} (deg)")
    axes[0].set_ylabel("images"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("GENKI-4K head pose by label", x=0.02, ha="left", fontsize=10)
    fig.savefig(out / "genki4k_pose.png"); plt.close(fig)


def genki_pipeline(df, X, out: Path, n=5, seed=3):
    """Raw photo with detected crop box next to the normalized 64x64 input."""
    rng = np.random.default_rng(seed)
    kept = df[df["keep"] & df["face_method"].str.startswith("haar")]
    pick = list(rng.choice(kept.index.values, n, replace=False))
    fb = df.index[df["face_method"] == "center"].values[:1]
    pick += list(fb)
    fig, axes = plt.subplots(2, len(pick), figsize=(len(pick) * 1.4, 3.0))
    for j, i in enumerate(pick):
        img = cv2.cvtColor(cv2.imread(df.at[i, "path"]), cv2.COLOR_BGR2RGB)
        x, y, w, h = map(int, df.at[i, "bbox"].split())
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 60, 60), 2)
        axes[0, j].imshow(img); axes[1, j].imshow(X[i], cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(df.at[i, "face_method"].replace("haar_", "haar "), fontsize=7)
        for a in axes[:, j]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, 0].set_ylabel("raw + box"); axes[1, 0].set_ylabel("64x64 input")
    fig.suptitle("GENKI-4K: face crop and normalization", x=0.02, ha="left", fontsize=10)
    fig.savefig(out / "genki4k_pipeline.png"); plt.close(fig)


def dataset_figures(tag, df, X, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    class_balance(tag, df, out)
    samples(tag, df, X, out)
    dropped(tag, df, X, out)
    if tag == "genki4k":
        if df["yaw"].notna().any():
            genki_pose(df, out)
        genki_pipeline(df, X, out)
