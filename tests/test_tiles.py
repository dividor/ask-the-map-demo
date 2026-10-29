import geopandas as gpd
import numpy as np
from shapely.geometry import box

import askmap.earth_engine as ee_mod
from askmap.geo import make_tile, tile_bbox_4326
from askmap.layers import SOLAR_EXISTING, SOLAR_LATER
from askmap.nwi import NWI_WETLAND
from askmap.tiles import build_ee_tile, build_nwi_tile, ee_land_layers


def fake_earth_engine(monkeypatch, n):
    """Stand-ins for the Earth Engine image builders and pixel download (no network)."""
    monkeypatch.setattr(ee_mod, "alphaearth_image", lambda year: ("ae", year))
    monkeypatch.setattr(ee_mod, "cdl_image", lambda years: ("cdl", tuple(years)))

    def fetch(image, bands, tile, log=print):
        assert tile.size_px == n
        if image[0] == "ae":
            return np.stack([np.full((n, n), i / 100, np.float32) for i in range(len(bands))])
        return np.full((1, n, n), 5.0, np.float32)

    monkeypatch.setattr(ee_mod, "fetch_tile", fetch)


def test_ee_land_layers_types(monkeypatch):
    t = make_tile("x", -99.0, 46.95, 3_000, 30)
    fake_earth_engine(monkeypatch, t.size_px)
    out = ee_land_layers(t, 2024, log=lambda *a: None)
    assert out["ae"].shape == (64, t.size_px, t.size_px) and out["ae"].dtype == np.float16
    assert out["cdl"].dtype == np.int16 and (out["cdl"] == 5).all()
    assert float(out["ae"][10, 0, 0]) == np.float16(0.10)


def test_build_nwi_tile_adds_wetlands(monkeypatch, tmp_path):
    t = make_tile("x", -99.0, 46.95, 3_000, 30)
    fake_earth_engine(monkeypatch, t.size_px)
    xmin, ymin = t.bounds[:2]
    gdf = gpd.GeoDataFrame(
        {"WETLAND_TYPE": ["Freshwater Emergent Wetland"]},
        geometry=[box(xmin, ymin, xmin + 600, ymin + 600)],
        crs="EPSG:5070",
    )
    path = tmp_path / "nwi.gpkg"
    gdf.to_file(path, layer="ND_Wetlands", driver="GPKG")
    out = build_nwi_tile(t, 2024, str(path), "ND_Wetlands", log=lambda *a: None)
    assert set(out) == {"ae", "cdl", "nwi"}
    assert (out["nwi"] == NWI_WETLAND).sum() == 20 * 20  # a 600 m square on the 30 m grid


def test_build_ee_tile_adds_solar(monkeypatch):
    t = make_tile("x", -78.0, 35.45, 3_000, 30)
    fake_earth_engine(monkeypatch, t.size_px)
    w, s, e, n = tile_bbox_4326(t)
    cx, cy = (w + e) / 2, (s + n) / 2
    pv = gpd.GeoDataFrame(
        {"p_year": [2020, 2025], "case_id": [1, 2]},
        geometry=[box(cx - 0.003, cy - 0.003, cx, cy), box(cx, cy, cx + 0.003, cy + 0.003)],
        crs=4326,
    )
    out = build_ee_tile(t, 2024, pv, 2023, log=lambda *a: None)
    assert set(out) == {"ae", "cdl", "solar"}
    assert (out["solar"] == SOLAR_EXISTING).any() and (out["solar"] == SOLAR_LATER).any()
