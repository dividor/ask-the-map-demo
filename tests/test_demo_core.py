import io

import numpy as np
import pytest
from PIL import Image

from askmap.demo.core import (
    cell_rects,
    class_shares,
    class_words,
    climate_words,
    describe_plan,
    display_index,
    overlay_rgba,
    period_words,
    png_bytes,
    shown_bounds,
    top_mask,
)
from askmap.geo import make_tile


def test_display_index_covers_the_tile_and_marks_outside():
    tile = make_tile("t", -98.0, 46.0, 3000, 30)
    idx, (w, s, e, n) = display_index(tile, stride=3, n_out=40)
    assert idx.shape == (40, 40) and w < -98.0 < e and s < 46.0 < n
    side = int(np.ceil(tile.size_px / 3))
    assert idx.max() < side * side and (idx >= 0).mean() > 0.9
    # north-west display cell maps to the first row, south-east to the last
    assert idx[1, 1] // side <= 1 and idx[-2, -2] // side >= side - 2


def test_top_mask_and_overlay():
    scores = np.arange(10, dtype=float).reshape(2, 5)
    valid = np.ones((2, 5), bool)
    valid[0, 0] = False
    m = top_mask(scores, valid, 0.25)
    assert m.sum() == 3 and m[1, 4] and not m[0, 0]
    with pytest.raises(ValueError):
        top_mask(scores, valid, 0)
    index = np.array([[0, 9], [-1, 5]])
    rgba = overlay_rgba(m, index)
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 1, 3] == 255 and rgba[0, 0, 3] == 0 and rgba[1, 0, 3] == 0
    shown = rgba[rgba[..., 3] > 0]
    # every match is drawn the same solid bright red
    assert (shown[:, 3] == 255).all() and (shown[:, :3] == (255, 23, 23)).all()
    img = Image.open(io.BytesIO(png_bytes(rgba)))
    assert img.size == (2, 2) and img.mode == "RGBA"


def test_class_shares_orders_by_count():
    got = class_shares(np.array([1, 1, 2, 1, 0]), ["a", "b", "c"], top=2)
    assert got == [{"class": "b", "share": 0.6}, {"class": "a", "share": 0.2}]
    assert class_shares(np.array([], int), ["a"]) == []


def test_plain_english_wording():
    assert class_words(["corn"]) == "corn" and class_words(["corn", "soybeans", "solar_farm"]) == (
        "corn, soybeans or solar farm"
    )
    assert period_words([7], [2021, 2021]) == "July 2021"
    assert period_words([4, 5, 6], [2022, 2022]) == "April to June 2022"
    assert period_words([6, 7, 8], [2021, 2023]) == "June to August, 2021 to 2023 (average)"
    assert period_words(list(range(1, 13)), [2021, 2021]) == "2021, all year"
    heat = {
        "variable": "T2M_MAX",
        "years": [2021, 2021],
        "months": [7],
        "stat": "days_above",
        "day_threshold": 35,
        "test": {"kind": "absolute", "op": ">=", "value": 5},
    }
    assert climate_words(heat) == "July 2021 had at least 5 days with daily high temperature above 35 °C"
    wet = {
        "variable": "PRECTOTCORR",
        "years": [2022, 2022],
        "months": [4, 5, 6],
        "stat": "total",
        "day_threshold": None,
        "test": {"kind": "history_rank", "side": "high", "pct": 10, "history": [2001, 2025]},
    }
    assert climate_words(wet) == "April to June 2022: total rainfall in the highest 10% of years 2001 to 2025"
    sun = {
        "variable": "ALLSKY_SFC_SW_DWN",
        "years": [2001, 2025],
        "months": [6, 7, 8],
        "stat": "mean",
        "day_threshold": None,
        "test": {"kind": "area_rank", "side": "high", "pct": 25},
    }
    assert climate_words(sun).endswith("mean sunlight in the highest 25% of this area's places")
    warm = {
        "variable": "T2M",
        "years": [2022, 2022],
        "months": [7],
        "stat": "mean",
        "day_threshold": None,
        "test": {"kind": "history_rank", "side": "high", "pct": 10, "history": [2001, 2025]},
    }
    assert climate_words(warm) == "July 2022: mean temperature in the highest 10% of years 2001 to 2025"
    plan = {
        "area": "x",
        "land": {"classes": ["grassland"], "min_patch_ha": 40},
        "near": [{"classes": ["wetland", "water"], "distance_m": 500, "negate": True}],
        "climate": [wet],
        "output": "map",
    }
    assert describe_plan(plan) == (
        "Grassland in patches of at least 40 ha, more than 500 m from any wetland or water, where April to "
        "June 2022: total rainfall in the highest 10% of years 2001 to 2025 (NASA POWER)."
    )
    corn = dict(plan, near=[], climate=[], land={"classes": ["corn"], "min_patch_ha": None})
    assert describe_plan(corn) == "Corn."


def test_shown_bounds_and_cell_rects():
    show = np.zeros((4, 4), bool)
    show[1, 2] = True
    assert shown_bounds(show, (0.0, 0.0, 4.0, 4.0)) == [2.0, 2.0, 3.0, 3.0]
    assert shown_bounds(np.zeros((2, 2), bool), (0, 0, 1, 1)) is None
    cells = np.array([[True, False], [False, True]])
    r = cell_rects(np.array([46.0, 46.5]), np.array([-98.0, -97.375]), cells)
    assert [(x["iy"], x["ix"]) for x in r] == [(0, 0), (1, 1)]
    assert r[0]["bounds"] == [-98.3125, 45.75, -97.6875, 46.25]
