"""Tiles on a fixed EPSG:5070 grid and the "within X metres of" operator.

Adapted from the author's earlier wetland-query code (``cropint.wetlands`` in the
climate-embeddings-crop-integrity project), so this folder needs nothing outside itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EQUAL_AREA = "EPSG:5070"


@dataclass(frozen=True)
class Tile:
    name: str
    x0: float  # EPSG:5070 left edge (m)
    y1: float  # EPSG:5070 top edge (m)
    size_px: int
    res_m: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        span = self.size_px * self.res_m
        return self.x0, self.y1 - span, self.x0 + span, self.y1

    @property
    def transform(self):
        from affine import Affine

        return Affine(self.res_m, 0.0, self.x0, 0.0, -self.res_m, self.y1)


def make_tile(name: str, lon: float, lat: float, size_m: float, res_m: float) -> Tile:
    """Square tile centred on ``lon, lat``, snapped to a ``res_m`` grid in EPSG:5070."""
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, EQUAL_AREA, always_xy=True).transform(lon, lat)
    size_px = round(size_m / res_m)
    half = size_px * res_m / 2
    x0 = np.floor((x - half) / res_m) * res_m
    y1 = np.floor((y + half) / res_m) * res_m
    return Tile(name, float(x0), float(y1), size_px, res_m)


def tile_bbox_4326(tile: Tile) -> tuple[float, float, float, float]:
    """Lon/lat bounding box (west, south, east, north) covering the tile."""
    from pyproj import Transformer

    xmin, ymin, xmax, ymax = tile.bounds
    t = Transformer.from_crs(EQUAL_AREA, 4326, always_xy=True)
    xs, ys = t.transform([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
    return min(xs), min(ys), max(xs), max(ys)


def within(mask: np.ndarray, distance_m: float, res_m: float) -> np.ndarray:
    """Pixels whose centre is within ``distance_m`` of any True pixel in ``mask`` (includes ``mask``). A
    distance below one cell (``res_m``) means the neighbouring cell: cells that touch are 0 m apart, but their
    centres are ``res_m`` apart, so without this a 30 m rule on a 90 m grid could match nothing."""
    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    dist = distance_transform_edt(~mask) * res_m
    return dist <= max(distance_m, res_m)
