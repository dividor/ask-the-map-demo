"""National Wetlands Inventory (NWI) polygons for a tile, rasterised to wetland / water.

Adapted from Matt Harris's earlier wetland-query code (``cropint.wetlands`` in the
climate-embeddings-crop-integrity project).
"""

from __future__ import annotations

import numpy as np

from askmap.geo import EQUAL_AREA, Tile

NWI_WETLAND_TYPES = {"Freshwater Emergent Wetland", "Freshwater Forested/Shrub Wetland"}
NWI_WATER_TYPES = {"Freshwater Pond", "Lake"}
# raster values
NWI_WETLAND, NWI_WATER = 1, 2


def read_nwi_tile(path: str, layer: str, tile: Tile):
    """NWI polygons intersecting ``tile`` from a state geodatabase, in EPSG:5070.

    ``path`` may point inside a zip (``zip://...zip!X.gdb``)."""
    import geopandas as gpd

    xmin, ymin, xmax, ymax = tile.bounds
    gdf = gpd.read_file(path, layer=layer, bbox=(xmin, ymin, xmax, ymax))
    if "WETLAND_TYPE" not in gdf:
        raise KeyError(f"WETLAND_TYPE missing from {layer}; columns {list(gdf.columns)}")
    return gdf.to_crs(EQUAL_AREA)


def nwi_raster(gdf, tile: Tile) -> np.ndarray:
    """Rasterise NWI polygons (with ``WETLAND_TYPE``) onto the tile: 1 = wetland, 2 = water, 0 = none."""
    from rasterio.features import rasterize

    if len(gdf) == 0:
        return np.zeros((tile.size_px, tile.size_px), dtype=np.uint8)
    gdf = gdf.to_crs(EQUAL_AREA)
    kind = gdf["WETLAND_TYPE"].map(
        lambda t: NWI_WETLAND if t in NWI_WETLAND_TYPES else (NWI_WATER if t in NWI_WATER_TYPES else 0)
    )
    shapes = [
        (geom, int(k)) for geom, k in zip(gdf.geometry, kind, strict=True) if k > 0 and geom is not None
    ]
    if not shapes:
        return np.zeros((tile.size_px, tile.size_px), dtype=np.uint8)
    # water first, then wetland on top, so wetland polygons take precedence where they overlap
    shapes.sort(key=lambda s: -s[1])
    return rasterize(
        shapes, out_shape=(tile.size_px, tile.size_px), transform=tile.transform, fill=0, dtype="uint8"
    )
