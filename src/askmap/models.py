"""Land maps predicted from AlphaEarth embeddings, and September's one-step "average example" search."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def pixels(ae: np.ndarray) -> np.ndarray:
    """(bands, rows, cols) -> (pixels, bands) as float32."""
    return ae.reshape(ae.shape[0], -1).T.astype(np.float32)


def training_sample(
    embeddings: list[np.ndarray],
    labels: list[np.ndarray],
    keep: list[np.ndarray],
    n_classes: int,
    per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``per_class`` random pixels of each class from the training tiles (pixels where ``keep``)."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([pixels(a) for a in embeddings])
    y = np.concatenate([lab.ravel() for lab in labels])
    k = np.concatenate([m.ravel() for m in keep]) & np.isfinite(X).all(axis=1)
    xs, ys = [], []
    for c in range(n_classes):
        idx = np.flatnonzero(k & (y == c))
        if len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        xs.append(X[idx])
        ys.append(y[idx])
    return np.concatenate(xs), np.concatenate(ys)


def fit_classifier(
    X: np.ndarray, y: np.ndarray, trees: int, min_samples_leaf: int, seed: int
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=seed
    )
    return rf.fit(X, y)


def predict_map(model, ae: np.ndarray, fill: int) -> np.ndarray:
    """Predicted leaf map; pixels without an embedding get ``fill``."""
    X = pixels(ae)
    ok = np.isfinite(X).all(axis=1)
    out = np.full(X.shape[0], fill, dtype=np.int16)
    if ok.any():
        out[ok] = model.predict(X[ok])
    return out.reshape(ae.shape[1:])


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def average_example(train_pos: np.ndarray, train_rate: float, test_unit: np.ndarray) -> np.ndarray:
    """September's baseline: mark the share ``train_rate`` of test pixels most similar (cosine) to the mean
    embedding of the training pixels that answer the query. Empty answer if there are no training examples."""
    if len(train_pos) == 0 or train_rate == 0:
        return np.zeros(len(test_unit), dtype=bool)
    sim = test_unit @ unit(train_pos.mean(axis=0))
    return sim >= np.quantile(sim, 1 - train_rate)


def like_point_scores(
    ae: np.ndarray, clim: np.ndarray, ex_ae: np.ndarray, ex_clim: np.ndarray, climate_weight: float
) -> np.ndarray:
    """ "Places like these points": cosine similarity to the examples' mean AlphaEarth embedding minus
    ``climate_weight`` times the distance to their mean standardised climate summary. Higher = more alike."""
    land = unit(ae) @ unit(ex_ae.mean(axis=0))
    dist = np.linalg.norm(clim - ex_clim.mean(axis=0), axis=1)
    return land - climate_weight * dist
