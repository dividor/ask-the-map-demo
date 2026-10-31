"""General land maps from AlphaEarth for every class at once, without class-specific rules.

- one-vs-rest probabilities (the notebook 12 classifiers) and their prior correction: each intercept shifted
  by log-odds(true share) - log-odds(training share) (Saerens, Latinne & Decaestecker 2002, Neural
  Computation 14(1));
- one multinomial (softmax) logistic regression fitted on pixels in their natural proportions, used as argmax;
- per-class scoring against an official map, macro-F1, and a block bootstrap of the macro-F1 difference.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from askmap.landctx import context_features, scaled_onehot


def log_odds(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("shares must be strictly between 0 and 1")
    return np.log(p / (1 - p))


def ovr_training_share(class_counts: np.ndarray, per_class: int) -> np.ndarray:
    """Positive share each notebook 12 one-vs-rest classifier was fitted on: positives = min(count, per_class)
    from the class-stratified sample; negatives = min(rest of that sample, per_class)."""
    pos = np.minimum(np.asarray(class_counts), per_class)
    neg = np.minimum(pos.sum() - pos, per_class)
    return pos / (pos + neg)


def prior_shift(intercept: np.ndarray, train_share: np.ndarray, true_share: np.ndarray) -> np.ndarray:
    """Intercepts of one-vs-rest logistic classifiers moved from their training class shares to the true
    shares."""
    return np.asarray(intercept, dtype=np.float64) + log_odds(true_share) - log_odds(train_share)


def ovr_probs(coef: np.ndarray, intercept: np.ndarray, ae: np.ndarray) -> np.ndarray:
    """(classes, rows, cols) probabilities of one-vs-rest classifiers on AlphaEarth ``ae`` (64, rows,
    cols)."""
    lin = np.einsum("lc,chw->lhw", coef, np.nan_to_num(ae)) + np.asarray(intercept)[:, None, None]
    return 1 / (1 + np.exp(-lin))


def natural_sample(
    embeddings: list[np.ndarray], labels: list[np.ndarray], keep: list[np.ndarray], n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` pixels drawn uniformly (without replacement) from the kept, finite pixels of all tiles: classes
    keep their natural proportions."""
    xs, ys = [], []
    for a, lab, k in zip(embeddings, labels, keep, strict=True):
        x = a.reshape(a.shape[0], -1).T
        ok = k.ravel() & np.isfinite(x).all(axis=1)
        xs.append(x[ok].astype(np.float32))
        ys.append(lab.ravel()[ok])
    X, y = np.concatenate(xs), np.concatenate(ys)
    if n > len(X):
        raise ValueError(f"asked for {n} pixels but only {len(X)} are available")
    idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def fit_multinomial(X: np.ndarray, y: np.ndarray, n_classes: int, max_iter: int = 2000):
    """One softmax logistic regression over three or more classes; returns (coef (classes, features),
    intercept (classes)). Every class must occur in ``y``."""
    if n_classes < 3:
        raise ValueError("the multinomial model needs at least three classes")
    missing = sorted(set(range(n_classes)) - set(np.unique(y).tolist()))
    if missing:
        raise ValueError(f"classes {missing} have no pixels in the sample")
    model = LogisticRegression(max_iter=max_iter).fit(X, y)
    order = np.argsort(model.classes_)
    return model.coef_[order].astype(np.float32), model.intercept_[order].astype(np.float32)


def argmax_onehot(coef: np.ndarray, intercept: np.ndarray, ae: np.ndarray) -> np.ndarray:
    """(classes, rows, cols) one-hot of the multinomial model's most likely class per pixel."""
    lin = np.einsum("lc,chw->lhw", coef, np.nan_to_num(ae)) + np.asarray(intercept)[:, None, None]
    top = lin.argmax(axis=0)
    return (np.arange(len(coef))[:, None, None] == top[None]).astype(np.float32)


def block_counts(masks: np.ndarray, truth: np.ndarray, valid: np.ndarray, block: int) -> np.ndarray:
    """(blocks, classes, 3) true positives, false positives and false negatives per square block of
    ``block`` x ``block`` cells. ``masks`` is (classes, rows, cols) bool; ``truth`` is the official class
    index per cell."""
    n_cls, rows, cols = masks.shape
    by, bx = np.arange(rows) // block, np.arange(cols) // block
    bid = (by[:, None] * (bx.max() + 1) + bx[None, :])[valid]
    nb = int((by.max() + 1) * (bx.max() + 1))
    out = np.zeros((nb, n_cls, 3), np.int64)
    t = truth[valid]
    for c in range(n_cls):
        p, g = masks[c][valid], t == c
        out[:, c, 0] = np.bincount(bid, weights=p & g, minlength=nb)
        out[:, c, 1] = np.bincount(bid, weights=p & ~g, minlength=nb)
        out[:, c, 2] = np.bincount(bid, weights=~p & g, minlength=nb)
    return out


def class_scores(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, F1 per class from (classes, 3) pooled counts; NaN where undefined."""
    tp, fp, fn = (counts[:, i].astype(float) for i in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "true": tp + fn, "predicted": tp + fp}


def macro_f1(counts: np.ndarray, classes: np.ndarray) -> float:
    """Mean F1 over ``classes`` (indices) from (classes, 3) pooled counts. A class with no true and no
    predicted cells (possible in a bootstrap resample) has no F1 and is left out of that mean."""
    return float(np.nanmean(class_scores(counts)["f1"][classes]))


def bootstrap_macro_diff(
    blocks_new: np.ndarray, blocks_old: np.ndarray, classes: np.ndarray, n: int, seed: int
) -> tuple[float, float, float]:
    """Macro-F1 difference (new - old) on all blocks, and its 95% interval from resampling blocks."""
    if blocks_new.shape != blocks_old.shape:
        raise ValueError("block counts of the two methods differ in shape")
    point = macro_f1(blocks_new.sum(0), classes) - macro_f1(blocks_old.sum(0), classes)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(blocks_new), len(blocks_new))
        diffs.append(macro_f1(blocks_new[i].sum(0), classes) - macro_f1(blocks_old[i].sum(0), classes))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


LAND_METHODS = ("one_vs_rest", "multinomial", "context_more")


def land_probs(method: str, params: dict, ae: np.ndarray, stride: int = 1) -> np.ndarray:
    """(classes, rows, cols) land maps for the demo on every ``stride``-th pixel of the full-resolution tile
    ``ae`` (64, rows, cols): ``one_vs_rest`` probabilities (notebook 12), the ``multinomial`` model's one-hot
    argmax (notebook 20), or ``context_more``, the model on neighbourhood features fitted with the extra
    training tiles (notebook 23; features are computed on the full tile before thinning). ``params`` holds the
    arrays of ``land_methods.npz`` (and, for ``context_more``, those of ``land_methods_a7.npz``)."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if method == "context_more":
        sizes = tuple(int(s) for s in np.atleast_1d(params["mean_sizes"]))
        feats = context_features(ae, sizes, int(params["std_size"]))[:, ::stride, ::stride]
        model = {k: params[f"NM_{k}"] for k in ("coef", "intercept", "mean", "scale")}
        return scaled_onehot(model, feats)
    thin = ae[:, ::stride, ::stride]
    if method == "one_vs_rest":
        return ovr_probs(params["ovr_coef"], params["ovr_intercept"], thin)
    if method == "multinomial":
        return argmax_onehot(params["mn_coef"], params["mn_intercept"], thin)
    raise ValueError(f"unknown land method {method!r}; choose one of {LAND_METHODS}")
