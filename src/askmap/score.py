"""Scores: F1 of answer masks over the test tiles, and bootstrap intervals over queries."""

from __future__ import annotations

import numpy as np


def f1(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> float:
    """F1 of two boolean masks over ``valid`` pixels; 1.0 if both are empty."""
    if valid is not None:
        pred, truth = pred[valid], truth[valid]
    tp = np.sum(pred & truth)
    fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth)
    if tp + fp + fn == 0:
        return 1.0
    return float(2 * tp / (2 * tp + fp + fn))


def pooled_counts(pred: dict, truth: dict, valid: dict) -> dict:
    """Precision, recall and F1 with pixels pooled over every tile of ``pred`` (masks per tile).

    Precision is None when nothing is predicted; recall is None when nothing is true."""
    p = np.concatenate([pred[t][valid[t]] for t in pred])
    g = np.concatenate([truth[t][valid[t]] for t in pred])
    tp, fp, fn = int(np.sum(p & g)), int(np.sum(p & ~g)), int(np.sum(~p & g))
    return {
        "predicted": tp + fp,
        "true": tp + fn,
        "both": tp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": f1(p, g),
    }


def pooled_f1(pred: dict, truth: dict, valid: dict, tiles: list[str]) -> float:
    """F1 with pixels pooled over ``tiles``."""
    p = np.concatenate([pred[t][valid[t]] for t in tiles])
    g = np.concatenate([truth[t][valid[t]] for t in tiles])
    return f1(p, g)


def median_ci(values: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """Median and 95% interval from resampling the values (queries) with replacement."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    boots = np.median(v[rng.integers(0, v.size, size=(n, v.size))], axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.median(v)), float(lo), float(hi)


def share_ci(flags: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """Share of True and its 95% bootstrap interval."""
    v = np.asarray(flags, dtype=float)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


def paired_median_gap(
    a: np.ndarray, b: np.ndarray, n: int, seed: int, groups: np.ndarray | None = None
) -> tuple[float, float, float]:
    """median(a) - median(b) for paired rows, and its 95% interval from resampling the rows with replacement
    (the same rows for both). With ``groups``, whole groups are resampled (a group brings all of its rows)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("a and b must be paired, non-empty 1-d arrays")
    point = float(np.median(a) - np.median(b))
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, a.size, size=(n, a.size))
        gaps = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    else:
        groups = np.asarray(groups)
        if groups.shape != a.shape:
            raise ValueError("one group label per row is needed")
        uniq = np.unique(groups)
        if len(uniq) < 2:
            raise ValueError("need at least two groups to resample")
        rows = [np.flatnonzero(groups == g) for g in uniq]
        gaps = np.empty(n)
        for i in range(n):
            pick = np.concatenate([rows[k] for k in rng.integers(0, len(uniq), len(uniq))])
            gaps[i] = np.median(a[pick]) - np.median(b[pick])
    lo, hi = np.quantile(gaps, [0.025, 0.975])
    return point, float(lo), float(hi)
