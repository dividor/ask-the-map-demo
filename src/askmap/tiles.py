"""Study-area tiles: AlphaEarth embeddings, CDL, NWI (North Dakota) and USPVDB solar farms (other areas).

Tiles are 20 km squares on a 30 m EPSG:5070 grid (``askmap.geo.make_tile``). Each built tile is saved
as one ``.npz`` (embeddings as float16) and recorded in a manifest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from askmap.geo import EQUAL_AREA, Tile, make_tile
from askmap.layers import SOLAR_EXISTING, SOLAR_LATER
from askmap.manifest import Manifest
from askmap.nwi import nwi_raster, read_nwi_tile


def area_tiles(area_cfg: dict, grid: dict) -> dict[str, tuple[Tile, str]]:
    """Tile objects of an area with their role ("train" or "test")."""
    out = {}
    for role in ("train", "test"):
        for name, (lon, lat) in area_cfg.get(f"{role}_tiles", {}).items():
            out[name] = (make_tile(name, lon, lat, grid["tile_size_m"], grid["res_m"]), role)
    return out


def pixel_lonlat(tile: Tile) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude of every pixel centre of the tile, (rows, cols) each."""
    from pyproj import Transformer

    n, r = tile.size_px, tile.res_m
    xs = tile.x0 + (np.arange(n) + 0.5) * r
    ys = tile.y1 - (np.arange(n) + 0.5) * r
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(EQUAL_AREA, 4326, always_xy=True).transform(gx, gy)
    return np.asarray(lon), np.asarray(lat)


def solar_raster(gdf, tile: Tile, last_year: int) -> np.ndarray:
    """USPVDB polygons on the tile: 1 = built by ``last_year``, 2 = built later or year unknown, 0 = none."""
    from rasterio.features import rasterize

    g = gdf.to_crs(EQUAL_AREA)
    xmin, ymin, xmax, ymax = tile.bounds
    g = g.cx[slice(xmin, xmax), slice(ymin, ymax)]
    if len(g) == 0:
        return np.zeros((tile.size_px, tile.size_px), dtype=np.uint8)
    year = g["p_year"].astype("float")
    kind = np.where(year.notna() & (year <= last_year), SOLAR_EXISTING, SOLAR_LATER)
    # later farms first, then existing farms on top
    shapes = sorted(
        ((geom, int(k)) for geom, k in zip(g.geometry, kind, strict=True) if geom is not None),
        key=lambda s: -s[1],
    )
    return rasterize(
        shapes, out_shape=(tile.size_px, tile.size_px), transform=tile.transform, fill=0, dtype="uint8"
    )


def solar_farm_ids(gdf, tile: Tile, last_year: int) -> list:
    """USPVDB case ids of farms built by ``last_year`` whose polygon intersects the tile."""
    g = gdf.to_crs(EQUAL_AREA)
    xmin, ymin, xmax, ymax = tile.bounds
    g = g.cx[slice(xmin, xmax), slice(ymin, ymax)]
    year = g["p_year"].astype("float")
    return sorted(g.loc[year.notna() & (year <= last_year), "case_id"].tolist())


def ee_land_layers(tile: Tile, land_year: int, log=print) -> dict:
    """AlphaEarth embeddings (64 bands, float16) and the CDL class map (int16) of ``land_year``, from Earth
    Engine."""
    from askmap.earth_engine import alphaearth_image, cdl_image, fetch_tile

    ae = fetch_tile(alphaearth_image(land_year), [f"A{i:02d}" for i in range(64)], tile, log=log)
    cdl = fetch_tile(cdl_image([land_year]), [f"cdl_{land_year}"], tile, log=log)[0]
    return {"ae": ae.astype(np.float16), "cdl": cdl.astype(np.int16)}


def build_nwi_tile(tile: Tile, land_year: int, nwi_path: str, nwi_layer: str, log=print) -> dict:
    """A North Dakota tile: AlphaEarth and CDL from Earth Engine plus the National Wetlands Inventory raster
    (1 = wetland, 2 = water) from the state geodatabase at ``nwi_path``."""
    nwi = nwi_raster(read_nwi_tile(nwi_path, nwi_layer, tile), tile)
    return {**ee_land_layers(tile, land_year, log), "nwi": nwi}


def build_ee_tile(tile: Tile, land_year: int, solar_gdf, last_year: int, log=print) -> dict:
    """A tile fetched from Earth Engine (AlphaEarth and CDL for ``land_year``) plus USPVDB solar farms."""
    return {**ee_land_layers(tile, land_year, log), "solar": solar_raster(solar_gdf, tile, last_year)}


def save_tile(path: Path, arrays: dict, manifest: Manifest, **info) -> None:
    tmp = path.with_suffix(".part.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    manifest.record(path, **info)


def check_tile(arrays: dict, max_missing: float) -> dict:
    """Share of pixels with missing embeddings (any band NaN); fails above ``max_missing``."""
    missing = float(np.isnan(arrays["ae"]).any(axis=0).mean())
    if missing > max_missing:
        raise ValueError(f"{missing:.1%} of pixels have no embedding")
    return {"missing_embedding_share": missing}
