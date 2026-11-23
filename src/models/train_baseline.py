"""Train, tune, freeze and evaluate the HOG + logistic-regression baseline.

    python -m src.models.train_baseline                 # both tasks, val tuning, then test once
    python -m src.models.train_baseline --skip-test     # stop after freezing on val

Tasks
  genki_smile     GENKI-4K smile (1) vs non-smile (0)          -> the game's detector
  fer_expression  FER-2013 any expression (1) vs neutral (0)   -> multi-expression variant

Protocol: hyperparameters are chosen on val ROC-AUC, the decision threshold on
val balanced accuracy (Youden J), then the model + threshold are written to models/ and only then scored
on the test split. Test numbers never feed back into a choice.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src.data.config import MODELS, PROCESSED, RESULTS, SEED, SPLITS
from src.models.baseline import BaselineConfig, HOGBaseline, HOGConfig

OUT = RESULTS / "baseline"
TASKS = {
    "genki_smile": {"tag": "genki4k", "grid": {"ppc": [6, 8, 10], "C": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1],
                                               "flip": [False, True]},
                    "pos": "smile", "neg": "non-smile"},
    "fer_expression": {"tag": "fer2013", "grid": {"ppc": [8], "C": [1e-4, 1e-3, 1e-2],
                                                  "flip": [False, True]},
                       "pos": "expression", "neg": "neutral"},
}


# ----------------------------------------------------------------------------- data
def load(tag: str):
    df = pd.read_csv(SPLITS / f"{tag}.csv")
    z = np.load(PROCESSED / f"{tag}_64.npz", allow_pickle=True)
    assert (z["image_id"] == df["image_id"].values).all(), "processed arrays out of sync; rerun prepare"
    keep = df["keep"].values
    return df[keep].reset_index(drop=True), z["X"][keep]


def split_hash(tag: str) -> str:
    return hashlib.md5((SPLITS / f"{tag}.csv").read_bytes()).hexdigest()[:12]


# ----------------------------------------------------------------------------- metrics
def metrics(y, s, thr) -> dict:
    p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "threshold": float(thr),
            "accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p),
            "f1": f1_score(y, p), "false_alarm_rate": fp / max(1, fp + tn),
            "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def youden_threshold(y, s) -> float:
    """Threshold maximizing recall - false_alarm_rate (= balanced accuracy).

    Max-F1 was tried first and is degenerate on FER-2013, where 83% of images
    are positive: it flags nearly every neutral face. Youden's J weighs the two
    classes equally regardless of prevalence, so one rule serves both tasks.
    """
    fpr, tpr, th = roc_curve(y, s)
    return float(th[int(np.argmax(tpr - fpr))])


def fpr_threshold(y, s, max_fpr=0.05) -> float:
    """Lowest threshold whose false-alarm rate on neutral faces is <= max_fpr."""
    neg = np.sort(s[y == 0])
    k = int(np.floor(max_fpr * len(neg)))
    return float(np.nextafter(neg[len(neg) - k - 1], 1)) if k < len(neg) else 0.0


def rounded(d):
    if isinstance(d, dict):
        return {k: rounded(v) for k, v in d.items()}
    if isinstance(d, (float, np.floating)):
        return round(float(d), 4)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


# ----------------------------------------------------------------------------- tuning
def tune(task: str, df, X):
    spec = TASKS[task]
    tr, va = (df["split"] == "train").values, (df["split"] == "val").values
    ytr, yva = df.loc[tr, "game_label"].values, df.loc[va, "game_label"].values
    rows, cache = [], {}
    g = spec["grid"]
    for ppc, flip, C in itertools.product(g["ppc"], g["flip"], g["C"]):
        hc = HOGConfig(pixels_per_cell=ppc)
        if (ppc, flip) not in cache:
            Xtr = np.concatenate([X[tr], X[tr][:, :, ::-1]]) if flip else X[tr]
            cache[(ppc, flip)] = (hc.features(Xtr), hc.features(X[va]))
        Ftr, Fva = cache[(ppc, flip)]
        y_fit = np.concatenate([ytr, ytr]) if flip else ytr
        m = HOGBaseline(BaselineConfig(hog=hc, C=C, flip_augment=False, task=task))
        t0 = time.time(); m.fit(None, y_fit, feats=Ftr); fit_s = time.time() - t0
        s = m.score(None, feats=Fva)
        rows.append({"pixels_per_cell": ppc, "flip_augment": flip, "C": C, "n_features": Ftr.shape[1],
                     "val_roc_auc": roc_auc_score(yva, s), "val_pr_auc": average_precision_score(yva, s),
                     "val_f1@0.5": f1_score(yva, (s >= 0.5).astype(int)), "fit_seconds": fit_s})
        print(f"  {task} ppc={ppc} flip={flip} C={C:g}: val AUC={rows[-1]['val_roc_auc']:.4f}", flush=True)
    grid = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False)
    return grid, grid.iloc[0]


# ----------------------------------------------------------------------------- figures
def plot_curves(task, curves: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, lw=1.6, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
        pr, rc, _ = precision_recall_curve(y, s)
        axes[1].plot(rc, pr, lw=1.6, label=f"{name} (AP {average_precision_score(y, s):.3f})")
    axes[0].plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    axes[0].set(xlabel="false alarm rate (neutral flagged)", ylabel="recall (expression caught)", title="ROC")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    for a in axes:
        a.legend(frameon=False, fontsize=7, loc="lower right" if a is axes[0] else "lower left")
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"HOG baseline, {task}", x=0.02, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(task, cm: dict, labels, path):
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(3, 2.7))
    ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]}\n{M[i, j] / M[i].sum():.1%}", ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black", fontsize=9)
    ax.set_xticks([0, 1], [f"pred {l}" for l in labels]); ax.set_yticks([0, 1], [f"true {l}" for l in labels])
    ax.set_title(f"{task}: test confusion", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_errors(task, df_te, X_te, s, thr, path, n=8):
    y = df_te["game_label"].values
    fp = np.where((y == 0) & (s >= thr))[0]; fp = fp[np.argsort(-s[fp])][:n]
    fn = np.where((y == 1) & (s < thr))[0]; fn = fn[np.argsort(s[fn])][:n]
    fig, axes = plt.subplots(2, n, figsize=(n * 0.95, 2.5))
    for r, (idx, name) in enumerate(((fp, "false alarm\n(neutral flagged)"), (fn, "missed\n(expression)"))):
        for j in range(n):
            a = axes[r, j]; a.set_xticks([]); a.set_yticks([])
            [sp.set_visible(False) for sp in a.spines.values()]
            if j < len(idx):
                a.imshow(X_te[idx[j]], cmap="gray", vmin=0, vmax=255)
                a.set_title(f"{s[idx[j]]:.2f}", fontsize=7)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"{task}: most confident test errors (score shown)", x=0.02, ha="left", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_hog(X, model, path):
    from skimage.feature import hog
    fig, axes = plt.subplots(2, 4, figsize=(6, 3.2))
    for j, i in enumerate([0, 1, 2, 3]):
        h = model.cfg.hog
        _, vis = hog(X[i], orientations=h.orientations, pixels_per_cell=(h.pixels_per_cell,) * 2,
                     cells_per_block=(h.cells_per_block,) * 2, visualize=True)
        axes[0, j].imshow(X[i], cmap="gray"); axes[1, j].imshow(vis, cmap="magma")
        [a.axis("off") for a in axes[:, j]]
    fig.suptitle(f"64x64 input and its HOG ({model.cfg.hog.pixels_per_cell}px cells)", x=0.02, ha="left", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- main flow
def run(task: str, skip_test: bool):
    spec = TASKS[task]
    df, X = load(spec["tag"])
    print(f"[baseline] tuning {task} on {len(df)} kept images", flush=True)
    grid, best = tune(task, df, X)
    OUT.mkdir(parents=True, exist_ok=True)
    grid.round(4).to_csv(OUT / f"{task}_grid.csv", index=False)

    cfg = BaselineConfig(hog=HOGConfig(pixels_per_cell=int(best["pixels_per_cell"])), C=float(best["C"]),
                         flip_augment=bool(best["flip_augment"]), task=task)
    tr, va = (df["split"] == "train").values, (df["split"] == "val").values
    model = HOGBaseline(cfg).fit(X[tr], df.loc[tr, "game_label"].values)
    yva = df.loc[va, "game_label"].values
    sva = model.score(X[va])
    model.cfg.threshold = youden_threshold(yva, sva)
    thr_strict = fpr_threshold(yva, sva, 0.05)
    val = {"default_threshold": metrics(yva, sva, model.cfg.threshold),
           "strict_threshold": metrics(yva, sva, thr_strict)}

    frozen = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": SEED,
              "split_file_md5": split_hash(spec["tag"]),
              "train_size": int(tr.sum()), "val_size": int(va.sum()),
              "alt_threshold_fpr5": thr_strict, "val_metrics": rounded(val)}
    model.save(MODELS / f"baseline_{task}", frozen)
    if task == "genki_smile":
        plot_hog(X[va], model, OUT / "hog_visualization.png")
    result = {"task": task, "config": asdict(model.cfg), "val": rounded(val)}
    if skip_test:
        return result, model

    te = (df["split"] == "test").values
    df_te, X_te, yte = df[te].reset_index(drop=True), X[te], df.loc[te, "game_label"].values
    t0 = time.time(); ste = model.score(X_te); ms = (time.time() - t0) / len(X_te) * 1000
    test = {"default_threshold": metrics(yte, ste, model.cfg.threshold),
            "strict_threshold": metrics(yte, ste, thr_strict),
            "ms_per_crop": ms}
    if task == "genki_smile":
        test["by_face_method"] = {
            k: metrics(yte[m], ste[m], model.cfg.threshold)
            for k, m in (("haar_detected", df_te["face_method"].str.startswith("haar").values),
                         ("center_fallback", (df_te["face_method"] == "center").values)) if m.sum() > 1}
        yaw = np.degrees(df_te["yaw"].abs().values)
        if not np.isnan(yaw).all():          # pose labels exist only with labels.txt
            test["by_abs_yaw_deg"] = {k: metrics(yte[m], ste[m], model.cfg.threshold)
                                      for k, m in (("<10", yaw < 10), ("10-20", (yaw >= 10) & (yaw < 20)),
                                                   (">=20", yaw >= 20))}
    else:
        test["by_fer_class_flag_rate"] = {
            c: float((ste[(df_te["raw_label"] == c).values] >= model.cfg.threshold).mean())
            for c in sorted(df_te["raw_label"].unique())}
    result["test"] = rounded(test)
    plot_curves(task, {"val": (yva, sva), "test": (yte, ste)}, OUT / f"{task}_roc_pr.png")
    plot_confusion(task, test["default_threshold"]["confusion"], ["neu", "exp"] if task != "genki_smile" else ["non", "smile"],
                   OUT / f"{task}_confusion.png")
    plot_errors(task, df_te, X_te, ste, model.cfg.threshold, OUT / f"{task}_errors.png")
    return result, model


def cross_domain(models: dict) -> dict:
    """Score each frozen model on the other dataset's test split (no retuning)."""
    out = {}
    if "genki_smile" in models:
        df, X = load("fer2013")
        m = (df["split"] == "test").values & df["raw_label"].isin(["happy", "neutral"]).values
        y = (df.loc[m, "raw_label"] == "happy").astype(int).values
        g = models["genki_smile"]
        out["genki_model_on_fer_test_happy_vs_neutral"] = metrics(y, g.score(X[m]), g.cfg.threshold)
    if "fer_expression" in models:
        df, X = load("genki4k")
        m = (df["split"] == "test").values
        y = df.loc[m, "game_label"].values
        f = models["fer_expression"]
        out["fer_model_on_genki_test"] = metrics(y, f.score(X[m]), f.cfg.threshold)
    return rounded(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args(argv)
    summary, models = {}, {}
    for t in a.tasks:
        summary[t], models[t] = run(t, a.skip_test)
    if not a.skip_test:
        summary["cross_domain"] = cross_domain(models)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "baseline_summary.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(summary)
    path.write_text(json.dumps(prev, indent=2, default=float))
    print(json.dumps(rounded(summary), indent=2, default=float))


if __name__ == "__main__":
    main()
