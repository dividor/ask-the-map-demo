import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from shapely.geometry import box

from askmap.captions import CLIMATE, Location, Vocab, make_caption, n_extra_conditions, render_plan
from askmap.climate_facts import condition_fact, fact_table, history_flags, summary_features, tier_flags
from askmap.config import PROJECT_ROOT, load_config
from askmap.local_facts import FAR, distances, leaf_patch_ha, patch_area_ha, sample_pixels
from askmap.power import add_derived
from askmap.space import Items, Space, multi_positive_loss, search, specs_from, truth
from askmap.universe import Universe, assign_states, box_of, land_cells

CFG = load_config(PROJECT_ROOT / "configs/feasibility.yaml")
CAP = load_config(PROJECT_ROOT / "configs/captions.yaml")
LEAVES = CFG["classes"]["leaves"]
GROUPS = CFG["classes"]["groups"]
YEARS = list(range(2001, 2026))


def toy_universe(n_lon=3):
    t = pd.date_range("2001-01-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(0)
    shape = (len(t), 2, n_lon)

    def v(scale, base):
        return (base + scale * rng.random(shape)).astype("float32")

    ds = xr.Dataset(
        {
            "T2M": (("time", "lat", "lon"), v(10, 10)),
            "T2M_MAX": (("time", "lat", "lon"), v(15, 20)),
            "T2M_MIN": (("time", "lat", "lon"), v(10, 5)),
            "PRECTOTCORR": (("time", "lat", "lon"), v(4, 0)),
            "GWETROOT": (("time", "lat", "lon"), v(0.5, 0.2)),
            "WS50M": (("time", "lat", "lon"), v(5, 2)),
            "RH2M": (("time", "lat", "lon"), v(50, 30)),
            "ALLSKY_SFC_SW_DWN": (("time", "lat", "lon"), v(4, 2)),
        },
        coords={"time": t, "lat": [40.0, 40.5], "lon": -100 + 0.625 * np.arange(n_lon)},
    )
    ds["GWETROOT"][:, 1, 0] = np.nan  # a sea cell
    ds = add_derived(ds, 10)
    cells = land_cells(ds, "plains")
    cells["state"] = "XX"
    cells["heldout"] = False
    return Universe(cells=cells, daily={"plains": ds})


def test_universe_land_cells_and_index():
    u = toy_universe()
    assert len(u.cells) == 5
    np.testing.assert_array_equal(u.index("plains", np.array([0, 1]), np.array([2, 1])), [2, 3])
    with pytest.raises(ValueError):
        u.index("plains", np.array([1]), np.array([0]))


def test_assign_states():
    cells = pd.DataFrame({"box": "b", "iy": [0, 0], "ix": [0, 1], "lat": [0.5, 5.0], "lon": [0.5, 5.0]})
    states = gpd.GeoDataFrame({"STUSPS": ["MN"]}, geometry=[box(0, 0, 1, 1)], crs=4326)
    out = assign_states(cells, states, ["MN"])
    assert out.state.tolist() == ["MN", ""] and out.heldout.tolist() == [True, False]


def test_history_and_tier_flags():
    vals = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [10.0, 5.0]])
    assert history_flags(vals, "high", 25)[:, 0].tolist() == [False, False, False, True]
    assert history_flags(vals, "low", 25)[:, 0].tolist() == [True, False, False, False]
    assert tier_flags(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "high", 50).tolist() == [
        False,
        True,
        True,
    ]


def test_fact_table_and_summary_features():
    u = toy_universe()
    ids, table = fact_table(u, CAP, YEARS)
    assert table.shape == (5, len(ids)) and len(set(ids)) == len(ids)
    k = ids.index("hist|PRECTOTCORR|AMJ|2022|high|10")
    assert 0 <= table[:, k].sum() <= 5
    # every cell is in its own wettest 10% in exactly 3 of 25 years (ties aside)
    ks = [ids.index(f"hist|PRECTOTCORR|AMJ|{y}|high|10") for y in YEARS]
    assert (table[:, ks].sum(axis=1) == 3).all()
    f = summary_features(u, CAP, YEARS)
    assert f.shape[0] == 5 and np.isfinite(f).all()


def test_condition_fact_maps_reference_plans():
    q = load_config(PROJECT_ROOT / "configs/queries.yaml")
    u = toy_universe()
    ids, _ = fact_table(u, CAP, YEARS)
    for x in q["queries"]:
        for c in x["plan"]["climate"]:
            assert condition_fact(c, CAP) in ids
        text = render_plan(x["plan"], CAP, lambda c: condition_fact(c, CAP))
        assert text.startswith("Show ") and "study area" not in text
    assert sum(n_extra_conditions(x["plan"]) >= 2 for x in q["queries"]) >= 10


def test_local_facts():
    leaf = np.full((10, 10), LEAVES.index("grassland"))
    leaf[:, :2] = LEAVES.index("wetland")
    leaf[0, 9] = LEAVES.index("corn")
    d = distances(leaf, ["wetland", "solar_farm", "cropland"], LEAVES, GROUPS, 30.0)
    assert d[0, 5, 0] == pytest.approx(120.0) and d[0, 0, 1] == FAR and d[0, 9, 2] == 0
    p = leaf_patch_ha(leaf, len(LEAVES), 30.0)
    assert p[5, 5] == pytest.approx(79 * 0.09) and p[0, 9] == pytest.approx(0.09)
    assert patch_area_ha(np.zeros((3, 3), bool), 30).sum() == 0
    idx = sample_pixels(leaf, np.ones_like(leaf, bool), 30, len(LEAVES), 0)
    assert len(np.unique(idx)) == 30 and LEAVES.index("corn") in leaf.ravel()[idx]


def test_captions_are_true_for_their_location():
    u = toy_universe()
    ids, table = fact_table(u, CAP, YEARS)
    vocab = Vocab(CAP, LEAVES, GROUPS, ids)
    rng = np.random.default_rng(0)
    fams = {
        f: np.array([k for k, i in enumerate(ids) if i.split("|")[0] in (f, "sun" if f == "long" else f)])
        for f in ("hist", "heat", "rain", "gdd", "long")
    }
    n_items = 40
    leaf = rng.choice([i for i, n in enumerate(LEAVES) if n != "other"], size=n_items)
    dist = rng.uniform(0, 3000, size=(n_items, len(vocab.near_classes))).astype("float32")
    pl, pc = rng.uniform(0, 50, n_items), rng.uniform(0, 50, n_items)
    cell = rng.integers(0, 5, n_items)
    caps, item_of = [], []
    for j in range(n_items):
        true = {f: fams[f][table[cell[j], fams[f]]] for f in fams}
        loc = Location(int(leaf[j]), dist[j], float(pl[j]), float(pc[j]), true, "north_dakota")
        for _ in range(10):
            caps.append(make_caption(loc, vocab, rng))
            item_of.append(j)
    assert any(c["spec"]["type"] == CLIMATE for c in caps) and all(c["full"] for c in caps)
    items = Items(
        torch.as_tensor(leaf),
        torch.as_tensor(dist),
        torch.as_tensor(pl, dtype=torch.float32),
        torch.as_tensor(pc, dtype=torch.float32),
        torch.as_tensor(cell),
    )
    specs = specs_from([c["spec"] for c in caps], "cpu")
    ok = truth(specs, items, torch.arange(n_items), torch.as_tensor(table))
    assert ok[torch.arange(len(caps)), torch.as_tensor(item_of)].all()
    with pytest.raises(ValueError):
        make_caption(Location(LEAVES.index("other"), dist[0], 0, 0, {}, "north_dakota"), vocab, rng)


def test_loss_and_search():
    sim = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    good = multi_positive_loss(sim, torch.eye(2, dtype=torch.bool), 0.1)
    bad = multi_positive_loss(sim, ~torch.eye(2, dtype=torch.bool), 0.1)
    assert good < bad
    m = Space(4, 3, 2, 8, 16, 0.0)
    s = search(m, np.ones((2, 4), "float32"), np.ones((5, 3), "float32"), np.ones((5, 2), "float32"), "cpu")
    assert s.shape == (2, 5) and np.allclose(s[:, 0], s[:, 1])
    assert search(
        Space(4, 3, None, 8, 16, 0.0), np.ones((1, 4), "float32"), np.ones((5, 3), "float32"), None, "cpu"
    ).shape == (1, 5)


def test_box_of():
    boxes = {"a": {"lon_min": 0, "lon_max": 1, "lat_min": 0, "lat_max": 1}}
    assert box_of(0.5, 0.5, boxes) == "a"
    with pytest.raises(ValueError):
        box_of(5, 5, boxes)
