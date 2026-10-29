import numpy as np
import pandas as pd
import pytest
import xarray as xr

from askmap.power import (
    add_derived,
    box_for,
    condition_cells,
    condition_values,
    pixel_cells,
    yearly_values,
)


def daily(values_by_cell, start="2001-01-01", end="2004-12-31", name="PRECTOTCORR"):
    t = pd.date_range(start, end, freq="D")
    v = np.stack([f(t) for f in values_by_cell], axis=-1)[:, None, :].astype("float32")
    return xr.Dataset(
        {name: (("time", "lat", "lon"), v)},
        coords={"time": t, "lat": [47.0], "lon": np.arange(v.shape[2]) * 0.625},
    )


def test_yearly_values_stats():
    ds = daily([lambda t: np.where(t.year == 2002, 2.0, 1.0)])
    da = ds["PRECTOTCORR"]
    tot = yearly_values(da, [2001, 2002], [6], "total")
    np.testing.assert_allclose(tot.values[:, 0, 0], [30.0, 60.0])
    np.testing.assert_allclose(yearly_values(da, [2002], [6, 7], "mean").values[0, 0, 0], 2.0)
    above = yearly_values(da, [2001, 2002], [6], "days_above", 1.5)
    np.testing.assert_allclose(above.values[:, 0, 0], [0, 30])
    with pytest.raises(ValueError):
        yearly_values(da, [2001], [6], "days_above")
    with pytest.raises(ValueError):
        yearly_values(da, [1999], [6], "total")


def test_condition_tests():
    # cell 0: 2003 is its wettest June; cell 1: always 1 mm/day; cell 2: always 3 mm/day
    ds = daily(
        [
            lambda t: np.where(t.year == 2003, 5.0, 1.0),
            lambda t: np.ones(len(t)),
            lambda t: 3 * np.ones(len(t)),
        ]
    )
    base = {
        "variable": "PRECTOTCORR",
        "years": [2003, 2003],
        "months": [6],
        "stat": "total",
        "day_threshold": None,
    }
    cells = np.ones((1, 3), bool)
    hist = dict(base, test={"kind": "history_rank", "side": "high", "pct": 25, "history": [2001, 2004]})
    assert condition_cells(ds, hist, cells).tolist() == [[True, True, True]]  # ties at the top count
    hist["years"] = [2002, 2002]
    assert condition_cells(ds, hist, cells).tolist() == [[False, True, True]]
    area = dict(base, test={"kind": "area_rank", "side": "low", "pct": 34})
    assert condition_cells(ds, area, cells).tolist() == [[False, True, False]]
    ab = dict(base, test={"kind": "absolute", "op": ">=", "value": 90})
    assert condition_cells(ds, ab, cells).tolist() == [[True, False, True]]
    with pytest.raises(ValueError):
        condition_cells(ds, dict(base, variable="SNOW", test=ab["test"]), cells)
    with pytest.raises(ValueError):
        condition_cells(ds, dict(base, test={"kind": "absolute", "op": ">", "value": 1}), cells)
    # the numbers behind each test
    v = condition_values(ds, ab, cells)
    np.testing.assert_allclose(v["value"][0], [150.0, 30.0, 90.0])
    assert v["op"] == ">=" and (v["threshold"] == 90).all()
    a = condition_values(ds, area, cells)
    assert a["op"] == "<=" and np.allclose(a["threshold"], np.quantile([150.0, 30.0, 90.0], 0.34))
    h = condition_values(ds, hist, cells)
    assert h["op"] == ">=" and h["threshold"].shape == (1, 3) and np.isclose(h["threshold"][0, 1], 30.0)
    ds["PRECTOTCORR"][:, 0, 2] = np.nan
    assert condition_cells(ds, ab, np.array([[True, True, False]])).tolist()[0][:2] == [True, False]
    with pytest.raises(ValueError):
        condition_cells(ds, ab, cells)


def test_add_derived_degree_days_and_low_days():
    t = pd.date_range("2001-01-01", periods=100, freq="D")
    shape = (len(t), 1, 1)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "T2M": (("time", "lat", "lon"), np.full(shape, 20.3, "float32")),
            "T2M_MAX": (("time", "lat", "lon"), np.full(shape, 35.0, "float32")),
            "T2M_MIN": (("time", "lat", "lon"), np.full(shape, 5.0, "float32")),
            "WS50M": (("time", "lat", "lon"), rng.random(shape).astype("float32")),
            "ALLSKY_SFC_SW_DWN": (("time", "lat", "lon"), rng.random(shape).astype("float32")),
        },
        coords={"time": t, "lat": [0.0], "lon": [0.0]},
    )
    out = add_derived(ds, 10)
    np.testing.assert_allclose(out["GDD10"].values, 10.0)  # (30 + 10) / 2 - 10
    np.testing.assert_allclose(out["CDD18"].values, 2.0, atol=1e-5)
    np.testing.assert_allclose(out["HDD18"].values, 0.0)
    low = out["LOW_WIND_SUN"].values.ravel().astype(bool)
    w, s = ds["WS50M"].values.ravel(), ds["ALLSKY_SFC_SW_DWN"].values.ravel()
    assert (low == ((w < np.quantile(w, 0.1)) & (s < np.quantile(s, 0.1)))).all()


def test_pixel_cells_and_box():
    lat, lon = np.array([47.0, 47.5]), np.array([-99.375, -98.75, -98.125])
    iy, ix = pixel_cells(lat, lon, np.array([[47.2, 47.3]]), np.array([[-99.2, -98.2]]))
    assert iy.tolist() == [[0, 1]] and ix.tolist() == [[0, 2]]
    with pytest.raises(ValueError):
        pixel_cells(lat, lon, np.array([[49.0]]), np.array([[-99.0]]))
    b = box_for([(-100, 46, -99, 47), (-98, 47, -97, 48)], 1.0)
    assert b == {"lon_min": -101, "lat_min": 45, "lon_max": -96, "lat_max": 49}
