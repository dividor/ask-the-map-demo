"""Check W: ways of finding one class (wetland) from AlphaEarth vectors, from one example to a classifier.

Every method returns a score per pixel (higher = more like the class); scoring is average precision and the
best F1 over thresholds.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier

from askmap.models import unit


def prototype_scores(x_unit: np.ndarray, protos: np.ndarray) -> np.ndarray:
    """Best cosine similarity of each (unit) pixel vector to any of the prototype vectors (one per row)."""
    protos = unit(np.atleast_2d(protos))
    return (x_unit @ protos.T).max(axis=1)


def direction_scores(x_unit: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Projection on the class direction: mean of the class examples minus mean of the other examples."""
    return x_unit @ (pos.mean(axis=0) - neg.mean(axis=0))


def kmeans_centres(pos: np.ndarray, k: int, seed: int) -> np.ndarray:
    """``k`` k-means centres of the class examples."""
    if len(pos) < k:
        raise ValueError(f"need at least {k} examples, got {len(pos)}")
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit(pos).cluster_centers_


def knn_scores(pos: np.ndarray, neg: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """Share of the class among the ``k`` nearest examples (cosine distance)."""
    X = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    model = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1).fit(X, y)
    return model.predict_proba(x)[:, 1]


def linear_scores(pos: np.ndarray, neg: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Probability of the class from a logistic regression fitted on the examples."""
    X = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return LogisticRegression(max_iter=2000).fit(X, y).predict_proba(x)[:, 1]


def best_f1(truth: np.ndarray, score: np.ndarray) -> float:
    """Highest F1 over all thresholds on ``score``."""
    if not truth.any():
        raise ValueError("no positive pixels")
    p, r, _ = precision_recall_curve(truth, score)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(f.max())


def west_half(shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of the western (left) half of a tile's columns."""
    m = np.zeros(shape, dtype=bool)
    m[:, : shape[1] // 2] = True
    return m
