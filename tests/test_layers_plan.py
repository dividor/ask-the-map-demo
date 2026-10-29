import numpy as np
import pandas as pd
import pytest
import xarray as xr

from askmap.catalogue import area_classes
from askmap.config import PROJECT_ROOT, load_config
from askmap.layers import class_mask, class_names, official_leaf_map
from askmap.plan import AreaContext, execute, min_patch, validate

CFG = load_config(PROJECT_ROOT / "configs/feasibility.yaml")
CLS = CFG["classes"]
ND = CLS["leaves"]
NC = CLS["leaves"]


def test_official_leaf_map_precedence():
    cdl = np.array([[1, 5, 24, 176], [111, 190, 121, 0]])
    nwi = np.array([[0, 1, 0, 0], [0, 0, 2, 0]])
    m = official_leaf_map(cdl, ND, CLS, nwi=nwi)
    names = [[ND[i] for i in row] for row in m]
    assert names == [["corn", "wetland", "other_crop", "grassland"], ["water", "other", "water", "other"]]
    solar = np.array([[0, 0, 0, 1], [0, 0, 2, 0]])
    m2 = official_leaf_map(cdl, NC, CLS, solar=solar)
    names2 = [[NC[i] for i in row] for row in m2]
    # NC: CDL wetland classes count as wetland; solar farms built by the cut-off override; later ones do not
    assert names2 == [
        ["corn", "soybeans", "other_crop", "solar_farm"],
        ["water", "wetland", "developed", "other"],
    ]


def test_class_mask_groups_and_unknown():
    m = official_leaf_map(np.array([[1, 5, 24, 176]]), ND, CLS)
    assert class_mask(m, "cropland", ND, CLS["groups"]).tolist() == [[True, True, True, False]]
    with pytest.raises(ValueError):
        class_mask(m, "other", ND, CLS["groups"])
    assert "cropland" in class_names(ND, CLS["groups"]) and "other" not in class_names(ND, CLS["groups"])


def test_min_patch_keeps_large_patches():
    m = np.zeros((6, 6), bool)
    m[0:2, 0:2] = True  # 4 px
    m[4, 4] = True  # 1 px
    out = min_patch(m, 3)
    assert out.sum() == 4 and not out[4, 4]


def ctx_for(leaves, power, area_cells, pixel_cell):
    return AreaContext(
        leaves=leaves,
        groups=CLS["groups"],
        res_m=30.0,
        power=power,
        area_cells=area_cells,
        pixel_cell=pixel_cell,
    )


def toy_power():
    t = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    rain = np.zeros((len(t), 1, 2), dtype="float32")
    rain[:, 0, 1] = 1.0  # cell 1 rains 1 mm/day, cell 0 is dry
    return xr.Dataset(
        {"PRECTOTCORR": (("time", "lat", "lon"), rain)},
        coords={"time": t, "lat": [47.0], "lon": [-99.0, -98.375]},
    )


def test_execute_combines_land_near_and_climate():
    leaf = np.full((5, 6), ND.index("grassland"), dtype=np.int16)
    leaf[:, 0] = ND.index("wetland")
    leaf[:, 3] = ND.index("corn")
    leaf[:, 5] = ND.index("corn")
    iy = np.zeros((5, 6), int)
    ix = np.zeros((5, 6), int)
    ix[:, 4:] = 1  # right part of the tile is in the rainy cell
    ctx = ctx_for(ND, toy_power(), np.ones((1, 2), bool), {"t": (iy, ix)})
    plan = {
        "area": "north_dakota",
        "land": {"classes": ["cropland"], "min_patch_ha": None},
        "near": [],
        "climate": [],
        "output": "map",
    }
    assert execute(plan, {"t": leaf}, ctx)["t"].sum() == 10
    plan["near"] = [{"classes": ["wetland"], "distance_m": 100, "negate": True}]
    out = execute(plan, {"t": leaf}, ctx)["t"]
    assert out[:, 5].all() and not out[:, 3].any()  # column 5 is 150 m from the wetland, column 3 is 90 m
    plan["near"] = [{"classes": ["wetland"], "distance_m": 100, "negate": False}]
    assert execute(plan, {"t": leaf}, ctx)["t"].sum() == 5  # column 3 is 90 m away
    plan["near"] = []
    plan["climate"] = [
        {
            "variable": "PRECTOTCORR",
            "years": [2002, 2002],
            "months": [6],
            "stat": "total",
            "day_threshold": None,
            "test": {"kind": "absolute", "op": ">=", "value": 10},
        }
    ]
    out = execute(plan, {"t": leaf}, ctx)["t"]
    assert out[:, 5].all() and not out[:, 3].any()
    v = ctx.climate_values(plan["climate"][0])
    assert v["op"] == ">=" and (v["threshold"] == 10).all()
    np.testing.assert_array_equal(v["value"] >= v["threshold"], ctx.climate(plan["climate"][0]))
    assert ctx.climate_values(plan["climate"][0]) is v  # cached


def good_plan():
    return {
        "area": "north_dakota",
        "land": {"classes": ["cropland"], "min_patch_ha": None},
        "near": [{"classes": ["wetland"], "distance_m": 100, "negate": False}],
        "climate": [
            {
                "variable": "GWETROOT",
                "years": [2021, 2021],
                "months": [5, 6],
                "stat": "mean",
                "day_threshold": None,
                "test": {"kind": "history_rank", "side": "low", "pct": 10, "history": [2001, 2025]},
            }
        ],
        "output": "map",
    }


def test_validate_accepts_good_and_rejects_bad():
    cat = area_classes(CFG)
    validate(good_plan(), cat, (2001, 2025))
    bad = [
        lambda p: p.update(area="mars"),
        lambda p: p["land"].update(classes=["parking"]),
        lambda p: p.update(extra=1),
        lambda p: p["near"][0].update(distance_m=-5),
        lambda p: p["climate"][0].update(variable="SNOW"),
        lambda p: p["climate"][0].update(stat="days_above"),  # needs a threshold
        lambda p: p["climate"][0].update(years=[1990, 2021]),
        lambda p: p["climate"][0]["test"].update(pct=150),
        lambda p: p["climate"][0]["test"].pop("history"),
        lambda p: p.update(output="table"),
    ]
    for change in bad:
        p = good_plan()
        change(p)
        with pytest.raises(ValueError):
            validate(p, cat, (2001, 2025))


def test_reference_plans_are_valid():
    q = load_config(PROJECT_ROOT / "configs/queries.yaml")
    cat = area_classes(CFG)
    ids = [x["id"] for x in q["queries"]]
    assert len(ids) == len(set(ids))
    for x in q["queries"]:
        validate(x["plan"], cat, (2001, 2025))
        assert x["persona"] in q["personas"] and x["source"] in q["sources"]


def test_class_indices_leaves_and_groups():
    from askmap.layers import class_indices

    leaves = ["corn", "soybeans", "wetland", "other"]
    groups = {"cropland": ["corn", "soybeans", "cotton"]}
    assert class_indices(["cropland"], leaves, groups) == [0, 1]
    assert class_indices(["wetland", "corn"], leaves, groups) == [0, 2]
    with pytest.raises(ValueError):
        class_indices(["forest"], leaves, groups)
    with pytest.raises(ValueError):
        class_indices(["other"], leaves, groups)
