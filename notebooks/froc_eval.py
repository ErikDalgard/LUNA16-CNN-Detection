"""LUNA16-style evaluation for the multi-scale FP-reduction model.

Scores every candidate in a held-out subset (e.g. masked_scans3) once, with no
augmentation and no negative subsampling, then computes the FROC curve and the
CPM (mean sensitivity at 0.125, 0.25, 0.5, 1, 2, 4, 8 false positives per scan).

Preprocessing (cut, clip, scale, optional mean-centering, layout) is taken from
cub_cutter, so it MUST be called with the same arch / layout / means used in
training, or the inputs will not match what the model learned.

Example:
    from froc_eval import evaluate, plot_froc

    res = evaluate(archi_1_2D, Path(PROJECT_DIR) / "masked_scans3",
                   arch="archi1", layout="2d", means=None)
    print("CPM:", res["cpm"])
    for fp, s in zip(res["fppi"], res["sensitivity"]):
        print(f"  {fp:>6.3f} FP/scan : sensitivity {s:.4f}")
    plot_froc(res)                       # optional, needs matplotlib

Candidate-level caveat: candidates_index.csv carries a per-candidate class label
but no nodule grouping, so sensitivity here is over positive *candidates*, not
over distinct nodules. The official noduleCADEvaluationLUNA16.py matches
detections to annotations geometrically and bootstraps confidence intervals;
bootstrap_cpm below reproduces the scan-level resampling part of that.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from cub_cutter import SIZES, cut_cube, normalize

LUNA_FP_POINTS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _layout_cube(cube, name, layout, means):
    """Mirror cub_cutter._finalize so eval preprocessing equals training."""
    cube = normalize(cube)
    if means is not None:
        cube = cube - np.float32(means[name])
    if layout == "3d":
        cube = cube[..., None]                  # (Z, Y, X, 1)
    else:
        cube = np.transpose(cube, (1, 2, 0))    # (Y, X, Z)
    return np.ascontiguousarray(cube, dtype=np.float32)


def predict_candidates(model, subset_dir, arch="archi1", layout="2d",
                       means=None, batch_size=128):
    """Run the model over every candidate in subset_dir exactly once (no aug).

    Returns (probs, labels, n_scans, index_df) with probs and labels aligned to
    index_df row order. Candidates are read in seriesuid order so the single
    volume cache stays warm.
    """
    subset_dir = Path(subset_dir)
    if arch not in SIZES:
        raise ValueError(f"arch must be one of {list(SIZES)}, got {arch!r}")
    if layout not in ("2d", "3d"):
        raise ValueError(f"layout must be '2d' or '3d', got {layout!r}")

    idx = (pd.read_csv(subset_dir / "candidates_index.csv")
             .sort_values("seriesuid", kind="mergesort")
             .reset_index(drop=True))

    size = SIZES[arch]
    dz, dy, dx = size
    feat_shape = (dz, dy, dx, 1) if layout == "3d" else (dy, dx, dz)

    cache = {"uid": None, "vol": None}

    def get_vol(uid):
        if uid != cache["uid"]:
            cache["vol"] = np.load(subset_dir / f"{uid}.npz")["volume"]
            cache["uid"] = uid
        return cache["vol"]

    def gen():
        for row in idx.itertuples(index=False):
            vol = get_vol(row.seriesuid)
            cube = cut_cube(vol, int(row.voxel_z), int(row.voxel_y),
                            int(row.voxel_x), size)
            yield _layout_cube(cube, arch, layout, means)

    ds = (tf.data.Dataset
          .from_generator(gen, output_signature=tf.TensorSpec(shape=feat_shape,
                                                              dtype=tf.float32))
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))

    probs = np.asarray(model.predict(ds, verbose=0)).reshape(-1)
    labels = idx["class"].to_numpy().astype(np.int32)
    n_scans = int(idx["seriesuid"].nunique())

    if len(probs) != len(labels):
        raise RuntimeError(f"prediction count {len(probs)} != candidate count {len(labels)}")
    return probs, labels, n_scans, idx


def compute_froc(probs, labels, n_scans, fp_points=LUNA_FP_POINTS):
    """FROC curve and CPM from per-candidate scores.

    sensitivity = detected positive candidates / total positive candidates
    fp_per_scan = false positive candidates / number of scans
    Both increase as the decision threshold drops, so sensitivity is read off the
    curve at each target FP/scan by interpolation.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    total_pos = int((labels == 1).sum())
    if total_pos == 0:
        raise ValueError("no positive candidates in the evaluation set")

    # threshold >= p: sort by score descending, accumulate hits
    order = np.argsort(-probs, kind="mergesort")
    y = labels[order]
    p = probs[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    sens = tp / total_pos
    fppi = fp / n_scans

    # collapse tied scores to the point after the whole tie group (">= t" semantics)
    last_of_tie = np.empty(len(p), dtype=bool)
    last_of_tie[-1] = True
    last_of_tie[:-1] = p[1:] != p[:-1]
    fppi, sens = fppi[last_of_tie], sens[last_of_tie]

    # for equal FP/scan keep the highest sensitivity, prepend the origin, make monotone
    uniq_fp, inv = np.unique(fppi, return_inverse=True)
    best_sens = np.zeros_like(uniq_fp)
    np.maximum.at(best_sens, inv, sens)
    curve_fppi = np.concatenate([[0.0], uniq_fp])
    curve_sens = np.maximum.accumulate(np.concatenate([[0.0], best_sens]))

    interp = np.interp(fp_points, curve_fppi, curve_sens, right=curve_sens[-1])
    return {
        "fppi": np.asarray(fp_points, dtype=float),
        "sensitivity": interp,
        "cpm": float(np.mean(interp)),
        "curve_fppi": curve_fppi,
        "curve_sens": curve_sens,
        "total_positives": total_pos,
        "total_negatives": int((labels == 0).sum()),
        "n_scans": int(n_scans),
    }


def evaluate(model, subset_dir, arch="archi1", layout="2d", means=None,
             batch_size=128, fp_points=LUNA_FP_POINTS):
    """predict_candidates + compute_froc, plus raw scores for reuse/CI."""
    probs, labels, n_scans, idx = predict_candidates(
        model, subset_dir, arch=arch, layout=layout, means=means, batch_size=batch_size)
    res = compute_froc(probs, labels, n_scans, fp_points=fp_points)
    res["probs"] = probs
    res["labels"] = labels
    res["seriesuids"] = idx["seriesuid"].to_numpy()
    return res


def bootstrap_cpm(probs, labels, seriesuids, n_bootstrap=1000, seed=0,
                  fp_points=LUNA_FP_POINTS):
    """LUNA16-style 95% CI on CPM by resampling scans with replacement."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    seriesuids = np.asarray(seriesuids)

    uids = np.unique(seriesuids)
    by_uid = {u: np.where(seriesuids == u)[0] for u in uids}
    rng = np.random.default_rng(seed)

    scores = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        picked = rng.choice(uids, size=len(uids), replace=True)
        rows = np.concatenate([by_uid[u] for u in picked])
        scores[b] = compute_froc(probs[rows], labels[rows], len(uids),
                                 fp_points=fp_points)["cpm"]
    return {
        "cpm_mean": float(scores.mean()),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
        "samples": scores,
    }


def plot_froc(res, ax=None, label=None):
    """Semilog FROC curve with the 7 operating points marked."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.semilogx(res["curve_fppi"][1:], res["curve_sens"][1:],
                label=label or f"CPM = {res['cpm']:.4f}")
    ax.scatter(res["fppi"], res["sensitivity"], zorder=3)
    ax.set_xlabel("false positives per scan")
    ax.set_ylabel("sensitivity")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.1, 10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax
