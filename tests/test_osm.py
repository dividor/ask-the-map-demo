import json

import pytest
from pyproj import Transformer

from askmap.geo import EQUAL_AREA, make_tile
from askmap.osm import (
    element_polygon,
    fetch_overpass,
    matches,
    new_tag_entries,
    overpass_query,
    tag_layers,
    tag_pixel_counts,
)


def square(lon0, lat0, d):
    return [
        {"lon": lon0, "lat": lat0},
        {"lon": lon0 + d, "lat": lat0},
        {"lon": lon0 + d, "lat": lat0 + d},
        {"lon": lon0, "lat": lat0 + d},
        {"lon": lon0, "lat": lat0},
    ]


def test_overpass_query_lists_every_key_for_ways_and_relations():
    q = overpass_query((-99.0, 46.0, -98.0, 47.0), ["landuse", "natural"], 120)
    assert q.startswith("[out:json][timeout:120];")
    assert 'way["landuse"](46.000000,-99.000000,47.000000,-98.000000);' in q
    assert q.count("relation[") == 2 and q.endswith("out geom;")


def test_closed_way_and_multipolygon_with_split_outer_and_hole():
    way = {"type": "way", "geometry": square(0, 0, 1), "tags": {"landuse": "farmland"}}
    assert element_polygon(way).area == pytest.approx(1.0)
    open_way = {"type": "way", "geometry": square(0, 0, 1)[:-1]}
    assert element_polygon(open_way) is None
    ring = square(0, 0, 2)
    rel = {
        "type": "relation",
        "tags": {"type": "multipolygon", "natural": "wetland"},
        "members": [
            {"type": "way", "role": "outer", "geometry": ring[:3]},
            {"type": "way", "role": "outer", "geometry": ring[2:]},
            {"type": "way", "role": "inner", "geometry": square(0.5, 0.5, 1)},
        ],
    }
    assert element_polygon(rel).area == pytest.approx(3.0)
    assert element_polygon({"type": "relation", "tags": {"type": "route"}, "members": []}) is None


def test_matches_any_condition_all_keys():
    cond = [{"power": ["plant", "generator"], "plant:source": ["solar"]}, {"landuse": ["quarry"]}]
    assert matches({"power": "plant", "plant:source": "solar"}, cond)
    assert not matches({"power": "plant", "plant:source": "gas"}, cond)
    assert matches({"landuse": "quarry", "name": "x"}, cond)


def test_tag_layers_rasterise_matching_polygons_onto_the_tile():
    tile = make_tile("t", -98.0, 46.0, 3000, 30)
    to_ll = Transformer.from_crs(EQUAL_AREA, 4326, always_xy=True).transform
    x0, y0, x1, y1 = tile.bounds
    # a square covering the tile's left half (in EPSG:5070), sent as lon/lat
    xm = (x0 + x1) / 2
    corners = [to_ll(x, y) for x, y in [(x0, y0), (xm, y0), (xm, y1), (x0, y1), (x0, y0)]]
    geom = [{"lon": a, "lat": b} for a, b in corners]
    el = {"type": "way", "geometry": geom, "tags": {"natural": "wetland"}}
    vocab = {"wet": {"match": [{"natural": ["wetland"]}]}, "farm": {"match": [{"landuse": ["farmland"]}]}}
    layers = tag_layers([el], tile, vocab)
    assert layers["wet"].shape == (100, 100)
    assert layers["wet"][:, :45].all() and not layers["wet"][:, 55:].any()
    assert not layers["farm"].any()


def test_tag_pixel_counts_and_new_tag_entries():
    tile = make_tile("t", -98.0, 46.0, 3000, 30)
    to_ll = Transformer.from_crs(EQUAL_AREA, 4326, always_xy=True).transform
    x0, y0, x1, y1 = tile.bounds
    xm = (x0 + x1) / 2
    corners = [to_ll(x, y) for x, y in [(x0, y0), (xm, y0), (xm, y1), (x0, y1), (x0, y0)]]
    left = [{"lon": a, "lat": b} for a, b in corners]
    els = [{"type": "way", "geometry": left, "tags": {"leisure": "resort", "name": "x"}}]
    counts = tag_pixel_counts(els, tile, ["leisure", "landuse"])
    assert set(counts) == {"leisure=resort"} and 4500 <= counts["leisure=resort"] <= 5500
    counts = {
        "leisure=resort": 2000,
        "landuse=forest": 9000,
        "power=plant": 3000,
        "power=generator": 1500,
        "man_made=clearcut": 2000,
        "leisure=track": 500,
    }
    vocab = {
        "forest": {"match": [{"landuse": ["forest"]}]},
        "solar": {"match": [{"power": ["plant", "generator"], "plant:source": ["solar"]}]},
    }
    excluded = [{"power": ["generator"], "generator:source": ["wind"]}]
    got = new_tag_entries(counts, vocab, excluded, 1000)
    assert got == ["power=plant", "leisure=resort", "man_made=clearcut"]


def test_fetch_overpass_reuses_a_saved_reply(tmp_path):
    path = tmp_path / "reply.json"
    path.write_text(json.dumps({"elements": [1, 2]}))
    got = fetch_overpass(
        "q", "http://unused.invalid", path, retries=0, wait_s=0, user_agent="ua", log=lambda *_: None
    )
    assert got == {"elements": [1, 2]}
    with pytest.raises(RuntimeError):
        fetch_overpass(
            "q",
            "http://unused.invalid",
            tmp_path / "new.json",
            retries=0,
            wait_s=0,
            user_agent="ua",
            log=lambda *_: None,
        )


def test_fetch_overpass_sends_the_user_agent_and_saves(tmp_path, monkeypatch):
    import askmap.osm as osm

    seen = {}

    class Reply:
        status_code = 200

        def json(self):
            return {"elements": []}

    def fake_post(url, data, headers, timeout):
        seen.update(headers)
        return Reply()

    monkeypatch.setattr(osm.requests, "post", fake_post)
    got = fetch_overpass("q", "http://x", tmp_path / "a.json", retries=0, wait_s=0, user_agent="my-ua")
    assert got == {"elements": []} and seen["User-Agent"] == "my-ua" and (tmp_path / "a.json").exists()
