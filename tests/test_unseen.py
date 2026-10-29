import numpy as np
import pytest
from pyproj import Transformer

from askmap.geo import EQUAL_AREA, make_tile
from askmap.unseen import (
    ap_and_chance,
    block_shuffle,
    bootstrap_median_by_word,
    bootstrap_median_difference,
    element_shape,
    overpass_query_selectors,
    rule_mask,
    terms_in_captions,
)

TILE = make_tile("t", -98.5, 46.5, 3000, 30)  # 100 x 100 pixels of 30 m
TO_LONLAT = Transformer.from_crs(EQUAL_AREA, 4326, always_xy=True)


def lonlat(x, y):
    lon, lat = TO_LONLAT.transform(x, y)
    return {"lon": lon, "lat": lat}


def centre_xy(row, col):
    return TILE.x0 + (col + 0.5) * 30, TILE.y1 - (row + 0.5) * 30


def test_query_lists_each_selector_for_each_type_and_rejects_unknown_types():
    q = overpass_query_selectors(
        (-99, 46, -98, 47),
        [
            {"filter": '["railway"="rail"]', "types": ["way"]},
            {"filter": '["power"="generator"]', "types": ["node", "way"]},
        ],
        60,
    )
    assert q.startswith("[out:json][timeout:60];") and q.endswith("out geom;")
    assert 'way["railway"="rail"](46.000000,-99.000000,47.000000,-98.000000);' in q
    assert q.count('["power"="generator"]') == 2 and 'node["power"="generator"]' in q
    with pytest.raises(ValueError):
        overpass_query_selectors((-99, 46, -98, 47), [{"filter": '["x"]', "types": ["area"]}], 60)


def test_element_shape_by_kind():
    node = {"type": "node", "lon": -98.5, "lat": 46.5}
    assert element_shape(node, ["point"])[0] == "point" and element_shape(node, ["line"]) is None
    sq = [
        {"lon": 0, "lat": 0},
        {"lon": 1, "lat": 0},
        {"lon": 1, "lat": 1},
        {"lon": 0, "lat": 1},
        {"lon": 0, "lat": 0},
    ]
    way = {"type": "way", "geometry": sq}
    assert element_shape(way, ["area", "line"])[0] == "area"
    assert element_shape(way, ["line"])[0] == "line"  # a closed way that may only be a line
    assert element_shape({"type": "way", "geometry": sq[:2]}, ["area"]) is None
    assert element_shape({"type": "way", "geometry": sq[:1]}, ["line"]) is None


def test_rule_mask_buffers_lines_and_points_then_takes_every_stride_pixel():
    row = 30
    y = centre_xy(row, 0)[1]
    line = {
        "type": "way",
        "tags": {"railway": "rail"},
        "geometry": [lonlat(TILE.x0, y), lonlat(TILE.x0 + 3000, y)],
    }
    rule = {"match": [{"railway": ["rail"]}], "kinds": ["line"]}
    m = rule_mask([line], TILE, rule, line_buffer_m=45, point_buffer_m=60, stride=1)
    rows = np.flatnonzero(m.any(axis=1))
    assert set(rows) == {row - 1, row, row + 1}  # centres 30 m away are inside 45 m; 60 m away are not
    assert rule_mask([line], TILE, rule, 30, 60, stride=3).shape == (34, 34)
    x, y = centre_xy(50, 50)
    pt = {
        "type": "node",
        "lon": lonlat(x, y)["lon"],
        "lat": lonlat(x, y)["lat"],
        "tags": {"power": "generator", "generator:source": "wind"},
    }
    rule_pt = {"match": [{"power": ["generator"], "generator:source": ["wind"]}], "kinds": ["point"]}
    mp = rule_mask([pt], TILE, rule_pt, 30, 75, stride=1)
    assert mp[50, 50] and mp[50, 52] and not mp[50, 53]  # 75 m reaches the centre 60 m away, not 90 m
    other = dict(pt, tags={"power": "generator", "generator:source": "solar"})
    assert not rule_mask([other], TILE, rule_pt, 30, 75, stride=1).any()


def test_terms_in_captions_matches_whole_words_any_case():
    caps = ["Corn near a Wetland", "a river channel", "a damp field"]
    assert terms_in_captions(caps, ["wetland", "dam", "river channel", "airport"]) == {
        "wetland",
        "river channel",
    }


def test_ap_and_chance():
    truth = np.array([1, 0, 1, 0, 0], bool)
    ap, chance = ap_and_chance(truth, np.array([0.9, 0.1, 0.8, 0.2, 0.3]))
    assert ap == pytest.approx(1.0) and chance == pytest.approx(0.4)
    with pytest.raises(ValueError):
        ap_and_chance(np.zeros(3, bool), np.zeros(3))


def test_block_shuffle_moves_whole_blocks_and_keeps_counts():
    m = np.zeros((10, 9), bool)
    m[0:4, 0:4] = True
    out = block_shuffle(m, 4, np.random.default_rng(1))
    assert out.sum() == m.sum() and out.shape == m.shape
    blocks = [out[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4] for i in range(2) for j in range(2)]
    assert sum(b.all() for b in blocks) == 1 and sum(b.any() for b in blocks) == 1
    with pytest.raises(ValueError):
        block_shuffle(np.zeros((3, 3), bool), 4, np.random.default_rng(0))


def test_bootstrap_median_by_word():
    words = np.array(["a", "a", "b", "c"])
    vals = np.array([1.0, 3.0, 2.0, 10.0])
    med, lo, hi = bootstrap_median_by_word(words, vals, 500, 0)
    assert med == pytest.approx(2.5) and lo <= med <= hi
    with pytest.raises(ValueError):
        bootstrap_median_by_word(np.array(["a", "a"]), np.array([1.0, 2.0]), 10, 0)


def test_bootstrap_median_difference():
    wa, va = np.array(["a", "b", "c"]), np.array([5.0, 6.0, 7.0])
    wb, vb = np.array(["x", "y", "z"]), np.array([1.0, 2.0, 3.0])
    d, lo, hi = bootstrap_median_difference(wa, va, wb, vb, 500, 0)
    assert d == pytest.approx(4.0) and lo <= d <= hi and lo > 0
    with pytest.raises(ValueError):
        bootstrap_median_difference(np.array(["a"]), np.array([1.0]), wb, vb, 10, 0)
