"""The climate "universe": every land NASA POWER cell in the configured boxes, as one flat list of cells.

A cell is identified by its box and its row/column in that box's daily dataset. Land = root-zone soil wetness
is present on every day (sea cells have none). Cells whose centre lies in a held-out state are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class Universe:
    cells: pd.DataFrame  # one row per cell: box, iy, ix, lat, lon, state, heldout
    daily: dict[str, xr.Dataset]  # box -> daily dataset (with derived variables)

    def index(self, box: str, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """Universe row of each (box, iy, ix); fails if a cell is not a land cell of the universe."""
        ds = self.daily[box]
        lut = np.full((ds.sizes["lat"], ds.sizes["lon"]), -1, dtype=np.int64)
        g = self.cells[self.cells.box == box]
        lut[g.iy.to_numpy(), g.ix.to_numpy()] = g.index.to_numpy()
        out = lut[np.asarray(iy), np.asarray(ix)]
        if (out < 0).any():
            raise ValueError(f"some pixels fall in cells of {box} that are not land cells of the universe")
        return out


def land_cells(ds: xr.Dataset, box: str) -> pd.DataFrame:
    """Cells of one box with root-zone wetness on every day."""
    ok = np.isfinite(ds["GWETROOT"].values).all(axis=0)
    iy, ix = np.nonzero(ok)
    return pd.DataFrame({"box": box, "iy": iy, "ix": ix, "lat": ds.lat.values[iy], "lon": ds.lon.values[ix]})


def assign_states(cells: pd.DataFrame, states, heldout: list[str]) -> pd.DataFrame:
    """Add the state (USPS code, "" if none) of each cell centre and a held-out flag."""
    import geopandas as gpd

    pts = gpd.GeoDataFrame(cells.copy(), geometry=gpd.points_from_xy(cells.lon, cells.lat), crs=4326)
    joined = gpd.sjoin(pts, states[["STUSPS", "geometry"]].to_crs(4326), how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    out = cells.copy()
    out["state"] = joined["STUSPS"].fillna("").to_numpy()
    out["heldout"] = out["state"].isin(heldout)
    return out


def box_of(lon: float, lat: float, boxes: dict) -> str:
    """Name of the box containing (lon, lat); fails if none does."""
    for name, b in boxes.items():
        if b["lon_min"] <= lon <= b["lon_max"] and b["lat_min"] <= lat <= b["lat_max"]:
            return name
    raise ValueError(f"({lon}, {lat}) is in no climate box")
