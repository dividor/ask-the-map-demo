import numpy as np
import pytest

from askmap.landctx import context_features, scaled_onehot
from askmap.landmaps import (
    argmax_onehot,
    block_counts,
    bootstrap_macro_diff,
    class_scores,
    fit_multinomial,
    land_probs,
    log_odds,
    macro_f1,
    natural_sample,
    ovr_probs,
    ovr_training_share,
    prior_shift,
)


def test_training_share_follows_notebook_12_sampling():
    # three classes with 50, 30_000 and 5 pixels, 20_000 per class at most
    share = ovr_training_share(np.array([50, 30_000, 5]), 20_000)
    np.testing.assert_allclose(share, [50 / 20_050, 20_000 / 20_055, 5 / 20_005])


def test_prior_shift_moves_probability_to_the_true_share():
    # a classifier that says 0.5 on a 50/50 training sample should say the true share after the shift
    icpt = prior_shift(np.array([0.0]), np.array([0.5]), np.array([0.01]))
    p = 1 / (1 + np.exp(-icpt))
    np.testing.assert_allclose(p, [0.01])
    with pytest.raises(ValueError):
        log_odds(np.array([0.0]))


def test_ovr_probs_and_argmax_onehot():
    ae = np.zeros((2, 1, 2))
    ae[0, 0, 1] = 1.0
    coef = np.array([[0.0, 0.0], [10.0, 0.0]])
    icpt = np.array([0.0, -5.0])
    p = ovr_probs(coef, icpt, ae)
    assert p.shape == (2, 1, 2) and p[0, 0, 0] == pytest.approx(0.5) and p[1, 0, 1] > 0.99
    oh = argmax_onehot(coef, icpt, ae)
    np.testing.assert_array_equal(oh[:, 0, :], [[1, 0], [0, 1]])


def test_natural_sample_keeps_proportions_and_fails_when_too_large():
    a = np.random.default_rng(0).normal(size=(3, 10, 10)).astype(np.float32)
    lab = np.zeros((10, 10), int)
    lab[:, :3] = 1  # 30% class 1
    keep = np.ones((10, 10), bool)
    keep[0, 0] = False
    a[:, 9, 9] = np.nan  # not finite: never sampled
    X, y = natural_sample([a], [lab], [keep], 98, seed=0)
    assert X.shape == (98, 3) and np.isfinite(X).all()
    assert (y == 1).sum() == 29  # every usable pixel: 30 class-1 pixels minus the one not kept
    with pytest.raises(ValueError):
        natural_sample([a], [lab], [keep], 99, seed=0)


def test_fit_multinomial_learns_separable_classes_and_needs_every_class():
    rng = np.random.default_rng(0)
    centres = np.array([[-3, 0], [3, 0], [0, 3]])
    X = np.concatenate([rng.normal(c, 0.3, (50, 2)) for c in centres])
    y = np.repeat(np.arange(3), 50)
    coef, icpt = fit_multinomial(X, y, 3)
    oh = argmax_onehot(coef, icpt, X.T[:, None, :])
    assert (oh[:, 0].argmax(axis=0) == y).all()
    with pytest.raises(ValueError):
        fit_multinomial(X, y, 4)  # class 3 has no pixels
    with pytest.raises(ValueError):
        fit_multinomial(X[y < 2], y[y < 2], 2)


def test_block_counts_scores_and_bootstrap():
    truth = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    masks = np.zeros((2, 2, 4), bool)
    masks[0, :, :2] = True  # class 0 exactly right
    masks[1, :, 1:] = True  # class 1: two false positives
    valid = np.ones((2, 4), bool)
    b = block_counts(masks, truth, valid, block=2)
    assert b.shape == (2, 2, 3)
    pooled = b.sum(0)
    np.testing.assert_array_equal(pooled, [[4, 0, 0], [4, 2, 0]])
    s = class_scores(pooled)
    assert s["f1"][0] == 1.0 and s["precision"][1] == pytest.approx(4 / 6)
    assert macro_f1(pooled, np.array([0, 1])) == pytest.approx((1 + 0.8) / 2)
    point, lo, hi = bootstrap_macro_diff(b, b, np.array([0, 1]), n=50, seed=0)
    assert point == 0 and lo == 0 and hi == 0
    with pytest.raises(ValueError):
        bootstrap_macro_diff(b, b[:1], np.array([0, 1]), n=5, seed=0)


def test_land_probs_picks_the_configured_method():
    ae = np.zeros((2, 1, 2))
    ae[0, 0, 1] = 1.0
    params = {
        "ovr_coef": np.array([[0.0, 0.0], [10.0, 0.0]]),
        "ovr_intercept": np.array([0.0, -5.0]),
        "mn_coef": np.array([[0.0, 0.0], [10.0, 0.0]]),
        "mn_intercept": np.array([0.0, -5.0]),
    }
    expected = ovr_probs(params["ovr_coef"], params["ovr_intercept"], ae)
    np.testing.assert_allclose(land_probs("one_vs_rest", params, ae), expected)
    np.testing.assert_array_equal(land_probs("multinomial", params, ae)[:, 0, :], [[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        land_probs("per_class_rules", params, ae)
    with pytest.raises(ValueError):
        land_probs("multinomial", params, ae, stride=0)


def test_land_probs_context_more_uses_full_tile_features_then_thins():
    rng = np.random.default_rng(0)
    ae = rng.normal(size=(2, 6, 6)).astype(np.float32)
    feats = context_features(ae, (3,), 3)  # 2 bands x (own, mean 3, std 3) = 6 features
    model = {
        "coef": rng.normal(size=(3, 6)).astype(np.float32),
        "intercept": np.zeros(3, np.float32),
        "mean": feats.reshape(6, -1).mean(1),
        "scale": feats.reshape(6, -1).std(1),
    }
    params = {f"NM_{k}": v for k, v in model.items()} | {"mean_sizes": np.array([3]), "std_size": np.array(3)}
    got = land_probs("context_more", params, ae, stride=2)
    np.testing.assert_array_equal(got, scaled_onehot(model, feats[:, ::2, ::2]))
    assert got.shape == (3, 3, 3) and (got.sum(axis=0) == 1).all()
    thin = land_probs("multinomial", {"mn_coef": np.eye(2), "mn_intercept": np.zeros(2)}, ae, stride=3)
    assert thin.shape == (2, 2, 2)
