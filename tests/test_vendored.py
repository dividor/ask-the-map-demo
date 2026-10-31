"""Tests for code adapted from the author's earlier projects (cropint, powerwm), so this folder runs on its
own."""

import numpy as np
import pytest

from askmap.earth_engine import band_groups, grid_spec, is_rate_limited, resolve_project, with_backoff
from askmap.geo import make_tile, tile_bbox_4326, within
from askmap.layers import annual_crop_codes
from askmap.manifest import Manifest, sha256_of
from askmap.nwi import NWI_WATER, NWI_WETLAND, nwi_raster, read_nwi_tile
from askmap.power import nearest_index


def test_make_tile_snapped_and_sized():
    t = make_tile("x", -99.0, 46.95, 20_000, 30)
    assert t.size_px == 667
    assert t.x0 % 30 == 0 and t.y1 % 30 == 0
    w, s, e, n = tile_bbox_4326(t)
    assert w < -99.0 < e and s < 46.95 < n
    x0, y0, x1, y1 = t.bounds
    assert x1 - x0 == y1 - y0 == 667 * 30


def test_within():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    assert within(m, 30, 30).sum() == 5  # centre + 4 neighbours at exactly 30 m
    assert within(m, 45, 30).sum() == 9  # diagonals at ~42 m
    assert not within(np.zeros((3, 3), bool), 100, 30).any()


def test_nwi_raster_and_precedence():
    import geopandas as gpd
    from shapely.geometry import box

    t = make_tile("x", -99.0, 46.95, 3_000, 30)
    xmin, ymin, xmax, ymax = tile_bbox_4326(t)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    gdf = gpd.GeoDataFrame(
        {"WETLAND_TYPE": ["Lake", "Freshwater Emergent Wetland", "Riverine"]},
        geometry=[
            box(cx - 0.005, cy - 0.005, cx + 0.005, cy + 0.005),
            box(cx - 0.002, cy - 0.002, cx + 0.002, cy + 0.002),
            box(xmin, ymin, xmin + 0.001, ymin + 0.001),
        ],
        crs=4326,
    )
    r = nwi_raster(gdf, t)
    mid = t.size_px // 2
    assert r[mid, mid] == NWI_WETLAND  # wetland drawn over water
    assert (r == NWI_WATER).sum() > (r == NWI_WETLAND).sum() > 0
    assert set(np.unique(r)) == {0, NWI_WETLAND, NWI_WATER}  # riverine is neither
    assert nwi_raster(gdf.iloc[:0], t).sum() == 0


def test_read_nwi_tile_filters_to_tile(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    t = make_tile("x", -99.0, 46.95, 3_000, 30)
    xmin, ymin, xmax, ymax = t.bounds
    inside = box(xmin + 100, ymin + 100, xmin + 400, ymin + 400)
    far = box(xmax + 50_000, ymax + 50_000, xmax + 50_300, ymax + 50_300)
    gdf = gpd.GeoDataFrame({"WETLAND_TYPE": ["Lake", "Lake"]}, geometry=[inside, far], crs="EPSG:5070")
    path = tmp_path / "nwi.gpkg"
    gdf.to_file(path, layer="ND_Wetlands", driver="GPKG")
    out = read_nwi_tile(str(path), "ND_Wetlands", t)
    assert len(out) == 1 and out.crs.to_epsg() == 5070
    gpd.GeoDataFrame({"x": [1]}, geometry=[inside], crs="EPSG:5070").to_file(path, layer="bad", driver="GPKG")
    with pytest.raises(KeyError, match="WETLAND_TYPE"):
        read_nwi_tile(str(path), "bad", t)


def test_band_groups_and_grid_spec():
    bands = [f"A{i:02d}" for i in range(64)]
    groups = band_groups(bands, 4, 667 * 667, max_bytes=30_000_000)
    assert [b for g in groups for b in g] == bands
    assert all(len(g) * 4 * 667 * 667 <= 30_000_000 for g in groups)
    with pytest.raises(ValueError, match="exceeds"):
        band_groups(bands, 4, 10_000_000, max_bytes=1_000)
    g = grid_spec(make_tile("x", -99.0, 46.95, 20_000, 30))
    assert g["dimensions"] == {"width": 667, "height": 667}
    assert g["affineTransform"]["scaleY"] == -30 and g["crsCode"] == "EPSG:5070"


def test_resolve_project(monkeypatch):
    monkeypatch.setenv("EE_PROJECT", "from-env")
    assert resolve_project("from-arg") == "from-arg"
    assert resolve_project() == "from-env"
    monkeypatch.delenv("EE_PROJECT")
    with pytest.raises(RuntimeError, match="MANUAL STEPS"):
        resolve_project()


def test_with_backoff_retries_rate_limits_only():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("HttpError 429 Too Many Requests: Exceeded Earth Engine concurrency limit")
        return "ok"

    assert with_backoff(flaky, retries=5, first_wait_s=0, log=lambda *a: None) == "ok"
    assert calls["n"] == 3
    assert is_rate_limited(RuntimeError("concurrency limit")) and not is_rate_limited(ValueError("bad band"))

    def broken():
        raise ValueError("bad band name")

    with pytest.raises(ValueError, match="bad band"):
        with_backoff(broken, retries=5, first_wait_s=0, log=lambda *a: None)

    def always_limited():
        raise RuntimeError("429")

    with pytest.raises(RuntimeError):
        with_backoff(always_limited, retries=2, first_wait_s=0, log=lambda *a: None)


def test_annual_crop_codes_ranges():
    assert annual_crop_codes([[1, 3], [38, 39], [2, 4]]) == [1, 2, 3, 4, 38, 39]
    with pytest.raises(ValueError):
        annual_crop_codes([])


def test_manifest_resume(tmp_path):
    m = Manifest(tmp_path)
    f = tmp_path / "x.nc"
    f.write_bytes(b"abc")
    assert not m.is_current(f)
    m.record(f, source="s3://x")
    again = Manifest(tmp_path)
    assert again.is_current(f)
    assert again.entries["x.nc"]["sha256"] == sha256_of(f) and again.entries["x.nc"]["source"] == "s3://x"
    f.write_bytes(b"changed")
    assert not Manifest(tmp_path).is_current(f)


def test_nearest_index():
    got = nearest_index(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.74, 2.0]))
    np.testing.assert_array_equal(got, [0, 1, 2])
