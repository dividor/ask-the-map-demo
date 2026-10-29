import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from askmap.geo import make_tile
from askmap.models import average_example, fit_classifier, predict_map, training_sample, unit
from askmap.score import f1, median_ci, paired_median_gap, pooled_f1, share_ci
from askmap.tiles import pixel_lonlat, solar_farm_ids, solar_raster


def test_classifier_learns_separable_classes():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 3, size=(20, 20))
    ae = np.stack([lab + 0.05 * rng.normal(size=lab.shape) for _ in range(4)]).astype(np.float16)
    ae[:, 0, 0] = np.nan
    X, y = training_sample([ae], [lab], [np.ones_like(lab, bool)], 3, per_class=50, seed=0)
    assert np.isfinite(X).all() and np.bincount(y).max() <= 50
    pred = predict_map(fit_classifier(X, y, 20, 1, 0), ae, fill=9)
    assert pred[0, 0] == 9
    assert (pred[1:, 1:] == lab[1:, 1:]).mean() > 0.95


def test_average_example_picks_most_similar_share():
    test = unit(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]))
    out = average_example(np.array([[1.0, 0.0]]), 0.5, test)
    assert out.tolist() == [True, True, False, False]
    assert not average_example(np.empty((0, 2)), 0.5, test).any()


def test_scores():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert f1(a, b) == pytest.approx(0.5)
    assert f1(np.zeros(3, bool), np.zeros(3, bool)) == 1.0
    assert f1(a, b, valid=np.array([True, False, False, True])) == 1.0
    assert pooled_f1({"x": a}, {"x": b}, {"x": np.ones(4, bool)}, ["x"]) == pytest.approx(0.5)
    m, lo, hi = median_ci(np.arange(11.0), 500, 0)
    assert m == 5.0 and lo <= m <= hi
    s, lo, hi = share_ci(np.array([True, False, True, True]), 500, 0)
    assert s == 0.75 and lo <= s <= hi


def test_paired_median_gap():
    a = np.arange(20.0) + 1.0
    b = np.arange(20.0)
    g, lo, hi = paired_median_gap(a, b, 500, 0)
    assert g == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)  # every pair +1
    g, lo, hi = paired_median_gap(b, b, 500, 0)
    assert g == lo == hi == 0.0
    words = np.repeat(np.arange(5), 2)
    noisy = np.random.default_rng(1).normal(size=10)
    g, lo, hi = paired_median_gap(noisy + 2.0, noisy, 500, 0, groups=words)
    assert g == pytest.approx(2.0) and lo <= g <= hi
    with pytest.raises(ValueError):
        paired_median_gap(a, b[:3], 10, 0)
    with pytest.raises(ValueError):
        paired_median_gap(a, b, 10, 0, groups=np.zeros(20))


def test_solar_raster_and_ids():
    tile = make_tile("t", -78.0, 35.45, 3000, 30)
    xmin, ymin, xmax, ymax = tile.bounds
    gdf = gpd.GeoDataFrame(
        {"p_year": [2015, 2024, None], "case_id": [1, 2, 3]},
        geometry=[
            box(xmin, ymin, xmin + 300, ymin + 300),
            box(xmin + 600, ymin, xmin + 900, ymin + 300),
            box(xmax + 5000, ymax + 5000, xmax + 6000, ymax + 6000),
        ],
        crs="EPSG:5070",
    )
    r = solar_raster(gdf, tile, 2023)
    assert (r == 1).sum() == 100 and (r == 2).sum() == 100
    assert solar_farm_ids(gdf, tile, 2023) == [1]
    lon, lat = pixel_lonlat(tile)
    assert lon.shape == (100, 100) and abs(lat.mean() - 35.45) < 0.01


def test_fetch_file_downloads_once(tmp_path, monkeypatch):
    import requests

    from askmap.data import fetch_file, read_uspvdb
    from askmap.manifest import Manifest

    calls = []
    gdf = gpd.GeoDataFrame({"p_year": [2020], "case_id": [7]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    src = tmp_path / "src.geojson"
    gdf.to_file(src, driver="GeoJSON")
    import zipfile

    with zipfile.ZipFile(tmp_path / "src.zip", "w") as z:
        z.write(src, "u.geojson")
    payload = (tmp_path / "src.zip").read_bytes()

    class Resp:
        def __init__(self, body, announced):
            self.body, self.headers = body, {"Content-Length": str(announced)}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            for i in range(0, len(self.body), 7):  # small chunks, to exercise the streaming loop
                yield self.body[i : i + 7]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_get(url, timeout, stream):
        assert stream
        calls.append(url)
        return Resp(payload, len(payload))

    monkeypatch.setattr(requests, "get", fake_get)
    out = tmp_path / "dl" / "u.zip"
    fetch_file("https://x/u.zip", out, Manifest(tmp_path / "dl"))
    fetch_file("https://x/u.zip", out, Manifest(tmp_path / "dl"))
    assert len(calls) == 1
    assert out.read_bytes() == payload
    assert Manifest(tmp_path / "dl").entries["u.zip"]["source"] == "https://x/u.zip"
    assert read_uspvdb(out, "u.geojson").case_id.tolist() == [7]

    monkeypatch.setattr(requests, "get", lambda url, timeout, stream: Resp(payload[:10], len(payload)))
    with pytest.raises(OSError, match="expected"):
        fetch_file("https://x/short.zip", tmp_path / "dl" / "short.zip", Manifest(tmp_path / "dl"))
    assert not (tmp_path / "dl" / "short.zip").exists()


def test_like_point_scores_prefers_same_land_and_climate():
    from askmap.models import like_point_scores

    ae = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    clim = np.array([[0.0], [3.0], [0.0]])
    s = like_point_scores(ae, clim, np.array([[1.0, 0.1]]), np.array([[0.0]]), 0.1)
    assert s.argmax() == 0 and s[1] > s[2]
