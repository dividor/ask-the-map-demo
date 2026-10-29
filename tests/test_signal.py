import numpy as np
import pytest

from askmap.models import unit
from askmap.signal import (
    best_f1,
    direction_scores,
    kmeans_centres,
    knn_scores,
    linear_scores,
    prototype_scores,
    west_half,
)


def two_groups(seed: int = 0, n: int = 300):
    """Class pixels around two separate centres (two kinds of wetland), others around a third.

    The centres are fixed; ``seed`` only changes the noise (new pixels of the same classes).
    """
    c = unit(np.random.default_rng(99).normal(size=(3, 16)))
    rng = np.random.default_rng(seed)
    pos = np.concatenate([c[0] + 0.1 * rng.normal(size=(n, 16)), c[1] + 0.1 * rng.normal(size=(n, 16))])
    neg = c[2] + 0.1 * rng.normal(size=(2 * n, 16))
    return unit(pos), unit(neg)


def test_prototype_scores_take_best_match():
    x = unit(np.array([[1.0, 0.0], [0.0, 1.0]]))
    s = prototype_scores(x, np.array([[1.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(s, [1.0, 1.0])
    np.testing.assert_allclose(prototype_scores(x, np.array([1.0, 0.0])), [1.0, 0.0])


def test_several_centres_find_both_kinds_and_direction_separates():
    pos, neg = two_groups()
    x = np.concatenate([pos, neg])
    truth = np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]
    centres = kmeans_centres(pos, 2, seed=0)
    assert best_f1(truth, prototype_scores(x, centres)) > 0.99
    assert best_f1(truth, direction_scores(x, pos, neg)) > 0.99
    with pytest.raises(ValueError):
        kmeans_centres(pos[:1], 2, seed=0)


def test_knn_and_linear_scores_separate_classes():
    pos, neg = two_groups(1)
    tp, tn = two_groups(2)
    x = np.concatenate([tp, tn])
    truth = np.r_[np.ones(len(tp), bool), np.zeros(len(tn), bool)]
    assert best_f1(truth, knn_scores(pos, neg, x, k=5)) > 0.99
    assert best_f1(truth, linear_scores(pos, neg, x)) > 0.99


def test_best_f1_and_west_half():
    truth = np.array([True, False, True, False])
    assert best_f1(truth, np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx(1.0)
    assert best_f1(truth, np.array([0.1, 0.9, 0.2, 0.8])) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        best_f1(np.zeros(3, bool), np.zeros(3))
    m = west_half((2, 5))
    assert m[:, :2].all() and not m[:, 2:].any()
