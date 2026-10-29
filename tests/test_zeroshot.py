import numpy as np
import pytest

from askmap.captions import render_plan
from askmap.climate_facts import condition_fact
from askmap.config import PROJECT_ROOT, load_config
from askmap.demo.core import climate_words
from askmap.zeroshot import climate_text, distance_ok, leaf_grid, leaf_phrases, plan_text, zero_shot_leaf

CAP = load_config(PROJECT_ROOT / "configs" / "captions.yaml")
LEAVES = ["corn", "soybeans", "wetland", "other"]


class Ctx:
    def __init__(self):
        self.res_m, self.leaves, self.groups = 30.0, LEAVES, {"cropland": ["corn", "soybeans"]}


def test_leaf_phrases_are_wording_0_of_every_leaf_but_other():
    idx, texts = leaf_phrases(CAP, LEAVES)
    assert idx == [0, 1, 2] and texts == ["corn fields", "soybean fields", "wetlands"]
    with pytest.raises(ValueError, match="no land phrase"):
        leaf_phrases(CAP, ["corn", "moon_rock"])


def test_zero_shot_leaf_takes_the_most_similar_candidate():
    sim = np.array([[0.1, 0.9, 0.2], [0.5, 0.1, 0.1], [0.0, 0.0, 0.3]])
    assert zero_shot_leaf(sim, [0, 1, 2]).tolist() == [1, 0, 2]
    with pytest.raises(ValueError):
        zero_shot_leaf(sim, [0, 1])


def test_leaf_grid_fills_valid_pixels_in_order():
    valid = np.array([[True, False], [True, True]])
    g = leaf_grid(valid, np.array([2, 0, 1]))
    assert g.tolist() == [[2, -1], [0, 1]]
    with pytest.raises(ValueError):
        leaf_grid(valid, np.array([1]))


def test_distance_ok_uses_geometry_on_the_map():
    m = np.zeros(
        (1, 6), np.int16
    )  # corn everywhere but a wetland pixel at the right end and one missing pixel
    m[0, 5], m[0, 0] = 2, -1
    near = {"classes": ["wetland"], "distance_m": 60, "negate": False}
    plan = {"near": [near]}
    assert distance_ok(plan, m, Ctx()).tolist() == [[False, False, False, True, True, True]]
    far = {"near": [dict(near, negate=True)]}
    assert distance_ok(far, m, Ctx()).tolist() == [[True, True, True, False, False, False]]
    both = {"near": [near, {"classes": ["cropland"], "distance_m": 0, "negate": False}]}
    assert distance_ok(both, m, Ctx()).tolist() == [[False, False, False, True, True, False]]
    assert distance_ok({"near": []}, m, Ctx()).all()


def test_plan_text_matches_render_plan_on_the_reference_plans():
    for q in load_config(PROJECT_ROOT / "configs" / "queries.yaml")["queries"]:
        plan = q["plan"]
        full = render_plan(plan, CAP, lambda c: condition_fact(c, CAP))
        assert plan_text(plan, CAP, distances=True, climate=True) == full
        land = render_plan(dict(plan, near=[], climate=[]), CAP, None)
        assert plan_text(plan, CAP, distances=False, climate=False) == land
        no_dist = plan_text(plan, CAP, distances=False, climate=True)
        assert no_dist == render_plan(dict(plan, near=[]), CAP, lambda c: condition_fact(c, CAP))


def test_a_condition_without_a_caption_fact_is_written_in_plain_english():
    cond = {
        "variable": "LOW_WIND_SUN",
        "years": [2001, 2025],
        "months": list(range(1, 13)),
        "stat": "total",
        "day_threshold": None,
        "test": {"kind": "area_rank", "side": "low", "pct": 33},
    }
    with pytest.raises(ValueError, match="no long-term caption tier"):
        condition_fact(cond, CAP)
    assert climate_text(cond, CAP) == climate_words(cond)
    third = dict(cond, test=dict(cond["test"], pct=33.3))
    assert (
        climate_text(third, CAP) == "in the fewest third of places by low-wind, low-sun days over 2001-2025"
    )
