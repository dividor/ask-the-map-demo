"""NASA POWER daily data for a study area, derived daily variables, and climate conditions per grid cell.

All daily variables are put on the MERRA-2 grid (0.5 x 0.625 degrees). Sunlight comes from CERES SYN1deg
(1 degree); each MERRA-2 cell takes the value of the nearest SYN1deg cell. Sunlight is converted from the
store's W/m2 (daily mean) to kWh/m2/day, the unit POWER users see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from askmap.manifest import Manifest

W_M2_TO_KWH_DAY = 24 / 1000
BASE_VARS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "GWETROOT", "WS50M", "RH2M", "ALLSKY_SFC_SW_DWN"]
DERIVED_VARS = ["GDD10", "HDD18", "CDD18", "LOW_WIND_SUN"]
STATS = ("mean", "total", "days_above", "days_below")


def open_store(url: str) -> xr.Dataset:
    """A NASA POWER Zarr store on AWS Open Data (anonymous access)."""
    return xr.open_zarr(url, storage_options={"anon": True}, consolidated=True)


def nearest_index(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Index of the nearest ``source`` coordinate for each ``target`` coordinate.

    ``open_store`` and ``nearest_index`` are adapted from Matt Harris's earlier POWER downloader
    (``powerwm.data.power``)."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    return np.abs(target[:, None] - source[None, :]).argmin(axis=1)


def box_for(tiles_lonlat: list[tuple[float, float, float, float]], margin: float) -> dict:
    """Lon/lat box covering tile bounding boxes (west, south, east, north) plus ``margin`` degrees."""
    a = np.array(tiles_lonlat)
    return {
        "lon_min": float(a[:, 0].min() - margin),
        "lat_min": float(a[:, 1].min() - margin),
        "lon_max": float(a[:, 2].max() + margin),
        "lat_max": float(a[:, 3].max() + margin),
    }


def fetch_daily(cfg: dict, box: dict, path: Path, log=print) -> xr.Dataset:
    """Daily MERRA-2 variables and SYN1deg sunlight over ``box``, on the MERRA-2 grid; cached at ``path``.

    Missing values are kept (sea cells have no soil wetness); ``condition_cells`` fails if a study-area cell
    is missing."""
    manifest = Manifest(path.parent)
    if not manifest.is_current(path):
        sel = {
            "time": slice(cfg["start"], cfg["end"]),
            "lat": slice(box["lat_min"], box["lat_max"]),
            "lon": slice(box["lon_min"], box["lon_max"]),
        }
        m = open_store(cfg["merra2_daily"])
        ds = xr.Dataset({v: m[v].sel(**sel).load().astype("float32") for v in cfg["merra2_vars"]})
        s = open_store(cfg["syn1deg_daily"])
        sw = (
            s["ALLSKY_SFC_SW_DWN"]
            .sel(
                time=sel["time"],
                lat=slice(box["lat_min"] - 1, box["lat_max"] + 1),
                lon=slice(box["lon_min"] - 1, box["lon_max"] + 1),
            )
            .load()
        )
        iy = nearest_index(sw.lat.values, ds.lat.values)
        ix = nearest_index(sw.lon.values, ds.lon.values)
        if not np.array_equal(sw.time.values, ds.time.values):
            raise ValueError("SYN1deg and MERRA-2 days differ")
        ds["ALLSKY_SFC_SW_DWN"] = (
            ("time", "lat", "lon"),
            (sw.values[:, iy][:, :, ix] * W_M2_TO_KWH_DAY).astype("float32"),
        )
        # cells over the sea have no soil wetness; they are kept as missing and checked where used
        n_missing = {v: int(ds[v].isnull().sum()) for v in ds.data_vars}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".nc.part")
        ds.to_netcdf(tmp, engine="h5netcdf")
        tmp.replace(path)
        manifest.record(
            path,
            sources=[cfg["merra2_daily"], cfg["syn1deg_daily"]],
            box=box,
            time=[cfg["start"], cfg["end"]],
            missing_values=n_missing,
        )
        log(f"POWER daily: {dict(ds.sizes)}; missing values {n_missing}")
    return xr.open_dataset(path, engine="h5netcdf").load()


def add_derived(ds: xr.Dataset, low_pct: float) -> xr.Dataset:
    """Growing degree days (base 10 C, max capped at 30 C), heating/cooling degree days (base 18.3 C), and
    LOW_WIND_SUN (1 on days when 50 m wind and sunlight are both below the cell's own ``low_pct`` percentile
    over the whole record)."""
    tmax = ds["T2M_MAX"].clip(max=30.0)
    tmin = ds["T2M_MIN"].clip(min=10.0)
    out = ds.copy()
    out["GDD10"] = ((tmax + tmin) / 2 - 10.0).clip(min=0.0)
    out["HDD18"] = (18.3 - ds["T2M"]).clip(min=0.0)
    out["CDD18"] = (ds["T2M"] - 18.3).clip(min=0.0)
    w_lo = ds["WS50M"].quantile(low_pct / 100, dim="time")
    s_lo = ds["ALLSKY_SFC_SW_DWN"].quantile(low_pct / 100, dim="time")
    out["LOW_WIND_SUN"] = ((ds["WS50M"] < w_lo) & (ds["ALLSKY_SFC_SW_DWN"] < s_lo)).astype("float32")
    return out.drop_vars("quantile", errors="ignore")


def _threshold(day_threshold: float | None) -> float:
    if day_threshold is None:
        raise ValueError("days_above and days_below need a day_threshold")
    return day_threshold


def yearly_values(
    da: xr.DataArray, years: list[int], months: list[int], stat: str, day_threshold: float | None = None
) -> xr.DataArray:
    """One value per year and cell: ``stat`` over the days of ``months`` in that year.

    ``mean`` and ``total`` use the daily values; ``days_above``/``days_below`` count days strictly above/below
    ``day_threshold``. Fails if a year has no days in the record.
    """
    if stat not in STATS:
        raise ValueError(f"unknown stat {stat!r}")
    if stat.startswith("days_") and day_threshold is None:
        raise ValueError(f"{stat} needs a day_threshold")
    if not months or any(m < 1 or m > 12 for m in months):
        raise ValueError(f"bad months {months}")
    t = da.time.dt
    vals = []
    for y in years:
        d = da.sel(time=(t.year == y) & t.month.isin(months))
        if d.sizes["time"] == 0:
            raise ValueError(f"no days for {y} months {months}")
        if stat == "mean":
            v = d.mean("time")
        elif stat == "total":
            v = d.sum("time")
        elif stat == "days_above":
            v = (d > _threshold(day_threshold)).sum("time")
        else:
            v = (d < _threshold(day_threshold)).sum("time")
        # a cell with any missing day gets no value (sums and counts would otherwise skip the gaps silently)
        vals.append(v.astype("float64").where(~d.isnull().any("time")))
    return xr.concat(vals, dim=xr.DataArray(years, dims="year", name="year"))


def condition_values(ds: xr.Dataset, cond: dict, area_cells: np.ndarray) -> dict:
    """The numbers behind a climate condition (see the query language in PLAN.md), per grid cell (lat, lon).

    Returns ``value`` (the period value: the mean over ``years`` of the yearly values), ``threshold`` (same
    shape) and ``op`` (">=" or "<="); a cell meets the condition when ``value op threshold``. Tests:
    ``absolute`` (op >= or <= value); ``history_rank`` (in the highest/lowest ``pct``% of yearly values over
    the ``history`` years, per cell); ``area_rank`` (in the highest/lowest ``pct``% of the area's cells).
    """
    var = cond["variable"]
    if var not in BASE_VARS + DERIVED_VARS:
        raise ValueError(f"unknown variable {var!r}")
    y0, y1 = cond["years"]
    period = (
        yearly_values(
            ds[var], list(range(y0, y1 + 1)), cond["months"], cond["stat"], cond.get("day_threshold")
        )
        .mean("year")
        .values
    )
    if np.isnan(period[area_cells]).any():
        raise ValueError(f"{var} is missing in a study-area cell")
    test = cond["test"]
    kind = test["kind"]
    if kind == "absolute":
        if test["op"] not in (">=", "<="):
            raise ValueError(f"bad op {test['op']!r}")
        return {"value": period, "threshold": np.full(period.shape, float(test["value"])), "op": test["op"]}
    side, pct = test["side"], float(test["pct"])
    if side not in ("high", "low") or not 0 < pct < 100:
        raise ValueError(f"bad rank test {test}")
    op = ">=" if side == "high" else "<="
    q = 1 - pct / 100 if side == "high" else pct / 100
    if kind == "history_rank":
        h0, h1 = test["history"]
        hist = yearly_values(
            ds[var], list(range(h0, h1 + 1)), cond["months"], cond["stat"], cond.get("day_threshold")
        ).values
        return {"value": period, "threshold": np.quantile(hist, q, axis=0), "op": op}
    if kind == "area_rank":
        ref = period[area_cells]
        if ref.size == 0:
            raise ValueError("study area has no cells")
        return {"value": period, "threshold": np.full(period.shape, np.quantile(ref, q)), "op": op}
    raise ValueError(f"unknown test kind {kind!r}")


def condition_cells(ds: xr.Dataset, cond: dict, area_cells: np.ndarray) -> np.ndarray:
    """Boolean (lat, lon): cells meeting a climate condition (``condition_values`` gives the numbers)."""
    v = condition_values(ds, cond, area_cells)
    if v["op"] == ">=":
        return v["value"] >= v["threshold"]
    return v["value"] <= v["threshold"]


def pixel_cells(
    grid_lat: np.ndarray, grid_lon: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest grid cell for each pixel; fails if a pixel is > half a cell outside the grid."""
    step_la, step_lo = np.diff(grid_lat).max(), np.diff(grid_lon).max()
    if (
        lat.min() < grid_lat[0] - step_la / 2
        or lat.max() > grid_lat[-1] + step_la / 2
        or lon.min() < grid_lon[0] - step_lo / 2
        or lon.max() > grid_lon[-1] + step_lo / 2
    ):
        raise ValueError("pixels fall outside the POWER grid")
    return nearest_index(grid_lat, lat.ravel()).reshape(lat.shape), nearest_index(
        grid_lon, lon.ravel()
    ).reshape(lon.shape)
