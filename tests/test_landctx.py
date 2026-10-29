import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import Point, box

from askmap.landctx import (
    context_features,
    fit_scaled_multinomial,
    natural_sample_tiles,
    pick_new_tiles,
    scaled_onehot,
    window_mean_std,
)


def test_window_mean_std_uses_only_valid_pixels_inside_the_tile():
    ae = np.arange(25, dtype=np.float32).reshape(1, 5, 5)
    ae[0, 2, 3] = np.nan
    mean, std = window_mean_std(ae, 3)
    # centre pixel (2, 2): the 3 x 3 block without the missing (2, 3)
    block = np.array([6, 7, 8, 11, 12, 16, 17, 18], dtype=float)
    assert mean[0, 2, 2] == pytest.approx(block.mean())
    assert std[0, 2, 2] == pytest.approx(block.std(), rel=1e-5)
    # corner (0, 0): only the 2 x 2 pixels inside the tile
    assert mean[0, 0, 0] == pytest.approx(np.mean([0, 1, 5, 6]))
    with pytest.raises(ValueError):
        window_mean_std(ae, 4)


def test_context_features_stack_and_keep_missing_pixels_missing():
    rng = np.random.default_rng(0)
    ae = rng.normal(size=(2, 6, 6)).astype(np.float32)
    ae[:, 0, 0] = np.nan
    f = context_features(ae, (3, 5), 5)
    assert f.shape == (2 * 4, 6, 6)
    np.testing.assert_allclose(f[:2, 3, 3], ae[:, 3, 3])
    assert np.isnan(f[:, 0, 0]).all() and np.isfinite(f[:, 1:, 1:]).all()


def test_natural_sample_tiles_draws_usable_pixels_in_natural_proportions():
    tiles = {
        "a": (np.zeros((1, 10, 10), np.float32), np.zeros((10, 10), int), np.ones((10, 10), bool)),
        "b": (np.ones((1, 10, 10), np.float32), np.ones((10, 10), int), np.ones((10, 10), bool)),
    }
    tiles["b"][2][:, 5:] = False  # half of b is not usable
    X, y = natural_sample_tiles(["a", "b"], lambda n: tiles[n], lambda ae: ae * 2, 150, seed=0)
    assert X.shape == (150, 1) and len(y) == 150
    np.testing.assert_array_equal(X[:, 0], 2.0 * y)  # features follow their labels
    assert (y == 0).sum() == 100 and (y == 1).sum() == 50
    with pytest.raises(ValueError):
        natural_sample_tiles(["a", "b"], lambda n: tiles[n], lambda ae: ae, 151, seed=0)


def test_scaled_multinomial_separates_classes_and_fails_loudly():
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
    y = np.repeat(np.arange(3), 50)
    X = centres[y] + rng.normal(size=(150, 2)).astype(np.float32)
    model = fit_scaled_multinomial(X, y, 3)
    feats = np.moveaxis(centres, 1, 0)[:, None, :]  # (features, 1 row, 3 cols)
    onehot = scaled_onehot(model, feats)
    np.testing.assert_array_equal(onehot[:, 0, :], np.eye(3))
    with pytest.raises(ValueError):
        fit_scaled_multinomial(np.c_[X, np.ones(150)], y, 3)  # a constant feature
    with pytest.raises(ValueError):
        fit_scaled_multinomial(X[:100], y[:100], 3)  # class 2 missing
    feats_nan = feats.copy()
    feats_nan[0, 0, 0] = np.nan
    assert scaled_onehot(model, feats_nan)[:, 0, 0].sum() == 0


def test_pick_new_tiles_follows_the_rule():
    to_ll = Transformer.from_crs("EPSG:5070", 4326, always_xy=True)

    # two 400 km square "states" in equal-area metres, given in lon/lat
    def ll_box(x0, y0):
        square = [(x0, y0), (x0 + 4e5, y0), (x0 + 4e5, y0 + 4e5), (x0, y0 + 4e5)]
        corners = [to_ll.transform(x, y) for x, y in square]
        from shapely.geometry import Polygon

        return Polygon(corners)

    states = gpd.GeoDataFrame(
        {"STUSPS": ["AA", "BB", "CC"]},
        geometry=[ll_box(0, 1.5e6), ll_box(6e5, 1.5e6), ll_box(2e6, 1.5e6)],
        crs=4326,
    )
    farm_xy = [(2e5, 1.7e6), (2.05e5, 1.7e6), (8e5, 1.7e6), (2.2e6, 1.7e6)]  # one too close, one in state CC
    farms = gpd.GeoDataFrame(
        {"p_state": ["AA", "AA", "BB", "CC"]},
        geometry=[box(*to_ll.transform(x, y), *to_ll.transform(x + 500, y + 500)) for x, y in farm_xy],
        crs=4326,
    )
    picked = pick_new_tiles(
        states, ["AA", "BB"], farms, n_random=3, n_farms=2, existing_lonlat=[], min_km=30, seed=0
    )
    assert [p[0] for p in picked] == ["R1", "R2", "R3", "F1", "F2"]
    to_eq = Transformer.from_crs(4326, "EPSG:5070", always_xy=True)
    xy = np.array([to_eq.transform(lon, lat) for _, lon, lat in picked])
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    assert d[np.triu_indices(len(xy), 1)].min() >= 30_000
    union = states[states.STUSPS.isin(["AA", "BB"])].to_crs("EPSG:5070").geometry.union_all()
    assert all(union.contains(Point(*p)) for p in xy)
    with pytest.raises(RuntimeError):  # only two farms in AA/BB are at least 30 km apart
        pick_new_tiles(
            states, ["AA", "BB"], farms, n_random=0, n_farms=3, existing_lonlat=[], min_km=30, seed=0
        )
