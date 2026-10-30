import base64
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from askmap.config import PROJECT_ROOT, load_config
from askmap.demo.app import create_app
from askmap.demo.engine import Engine, TileData, Unavailable, Unsupported, UserError

CAP = load_config(PROJECT_ROOT / "configs" / "captions.yaml")
LEAVES = ["corn", "wetland", "other"]


class Power:
    """Stands in for the POWER dataset's coordinates (``ds.lat.values``, ``ds.lon.values``)."""

    lat = SimpleNamespace(values=np.array([46.0]))
    lon = SimpleNamespace(values=np.array([-98.0, -97.375]))


class Ctx:
    """Two POWER cells side by side; pixel columns alternate between them. Cell values 7 and 2 days."""

    res_m = 90.0
    power = Power()

    def __init__(self, values=(7.0, 2.0)):
        self.leaves, self.groups = LEAVES, {"cropland": ["corn"]}
        self.pixel_cell = {"t": (np.zeros((4, 4), int), np.tile([0, 1, 0, 1], (4, 1)))}
        self.area_cells = np.ones((1, 2), bool)
        self.values = np.array([values])

    def climate_values(self, cond):
        thr = np.full((1, 2), float(cond["test"]["value"]))
        return {"value": self.values, "threshold": thr, "op": cond["test"]["op"]}

    def climate(self, cond):
        v = self.climate_values(cond)
        return v["value"] >= v["threshold"] if v["op"] == ">=" else v["value"] <= v["threshold"]


def tile():
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    z = np.zeros((15, 2), np.float16)
    z[:, 0] = np.linspace(-1, 1, 15)
    z[:, 1] = 1.0
    probs = np.zeros((3, 4, 4), np.float16)
    probs[0, :, :2] = 0.9
    leaf = np.zeros((4, 4), np.uint8)
    leaf[:, 2:] = 1
    index = np.arange(16).reshape(4, 4)
    return TileData("t", (-98.0, 46.0, -97.9, 46.1), index, valid, z, probs, leaf)


HEAT = {
    "variable": "T2M_MAX",
    "years": [2021, 2021],
    "months": [7],
    "stat": "days_above",
    "day_threshold": 35,
    "test": {"kind": "absolute", "op": ">=", "value": 5},
}
PLAN = {
    "area": "a",
    "land": {"classes": ["cropland"], "min_patch_ha": None},
    "near": [],
    "climate": [HEAT],
    "output": "map",
}


def engine(best="aligned", planner=lambda q, a: PLAN, ctx=None):
    return Engine(
        {"a": [tile()]},
        LEAVES,
        lambda texts: np.array([[1.0, 0.0]] * len(texts)),
        planner,
        {"a": ctx or Ctx()},
        best,
        CAP,
        top_share=0.25,
    )


def test_aligned_search_shows_the_top_share():
    out = engine(planner=lambda q, a: dict(PLAN, climate=[])).search("corn", "a", "aligned")
    assert out["used"] == "aligned" and out["plan"]["climate"] == []
    assert out["facts"]["shown_pixels"] == 4 and out["power_cells"] == []
    assert out["facts"]["text"] == "Show cropland" and out["facts"]["distance_gate"] is False
    assert base64.b64decode(out["tiles"][0]["png"])[:4] == b"\x89PNG"


def test_aligned_shows_power_cells_for_comparison_without_gating_on_them():
    out = engine().search("corn where it was hot", "a", "aligned")
    # the top 4 of 15 valid cells are the bottom row: columns 0 and 2 lie in the hot cell, 1 and 3 do not
    assert out["facts"]["shown_pixels"] == 4 and out["tiles"][0]["count_meeting_climate"] == 2
    assert out["plan"] == PLAN and out["facts"]["comparison_only"] is True
    assert out["facts"]["climate_gate"] is False
    # Method 1 keeps the climate as caption wording in its text
    want = "Show cropland where July 2021 had at least 5 days with a daily maximum above 35 C"
    assert out["facts"]["text"] == want
    assert [c["pass"] for c in out["power_cells"]] == [True, False]
    assert [c["value"] for c in out["facts"]["climate"][0]["cells"]] == [7.0, 2.0]
    assert 0.0 <= out["facts"]["share_meeting_climate_conditions"] <= 1.0
    tile_out = out["tiles"][0]
    in_green = round(out["facts"]["share_meeting_climate_conditions"] * out["facts"]["shown_pixels"])
    assert tile_out["count_meeting_climate"] == in_green <= tile_out["count"]


def test_methods_1_and_2_need_the_plan_for_their_distance_rules():
    def failing(q, a):
        raise UserError("no usable plan")

    for method in ("aligned", "aligned_land_power_climate"):
        with pytest.raises(UserError, match="no usable plan"):
            engine(planner=failing).search("corn", "a", method)
        with pytest.raises(Unavailable, match="language model is not configured"):
            engine(planner=None).search("corn", "a", method)


def test_classifier_and_best_use_the_plan_and_climate():
    out = engine().search("corn where it was hot", "a", "classifier")
    assert out["plan"] == PLAN and out["facts"]["share_meeting_climate_conditions"] == 1.0
    assert out["tiles"][0]["count_meeting_climate"] == out["tiles"][0]["count"]
    assert {x["class"] for x in out["facts"]["land_of_shown_pixels"]} == {"corn"}


def test_plan_answers_show_only_places_meeting_every_step():
    out = engine().search("corn where July 2021 was hot", "a", "classifier")
    f = out["facts"]
    # corn (probability 0.9): columns 0-1 minus the invalid corner = 7 places; column 0 lies in the hot cell
    assert [st["places"] for st in f["steps"]] == [7, 3]
    assert [st["kind"] for st in f["steps"]] == ["land", "climate"]
    assert f["shown_pixels"] == 3 and out["tiles"][0]["count"] == 3 and f["empty_reason"] is None
    assert f["plan_words"].startswith("Cropland, where July 2021 had at least 5 days with daily high")
    cells = f["climate"][0]["cells"]
    assert [c["value"] for c in cells] == [7.0, 2.0] and [c["pass"] for c in cells] == [True, False]
    assert [b["pass"] for b in out["power_cells"]] == [True, False]
    w, s_, e, n = out["tiles"][0]["match_bounds"]
    W, S, E, N = out["tiles"][0]["bounds"]
    assert W <= w < e <= E and S <= s_ < n <= N and (e - w) < (E - W)  # only part of the tile matches


def test_empty_answer_names_the_step_that_removed_everything():
    out = engine(ctx=Ctx(values=(2.0, 1.0))).search("corn where July 2021 was hot", "a", "classifier")
    f = out["facts"]
    assert f["shown_pixels"] == 0 and out["tiles"][0]["match_bounds"] is None
    assert "No NASA POWER cell in this area meets" in f["empty_reason"]
    assert "the closest cell had 2 against a threshold of 5" in f["empty_reason"]
    no_corn = dict(PLAN, land={"classes": ["wetland"], "min_patch_ha": None}, climate=[])
    f = engine(planner=lambda q, a: no_corn).search("wetland", "a", "classifier")["facts"]
    assert f["empty_reason"] == "The land classifier found no wetland in this area."
    far = dict(PLAN, climate=[], near=[{"classes": ["wetland"], "distance_m": 100, "negate": False}])
    f = engine(planner=lambda q, a: far).search("corn near wetland", "a", "classifier")["facts"]
    assert f["empty_reason"] == "None of the cropland places is within 100 m of wetland."
    out = engine(best="aligned_land_power_climate").search("corn where it was hot", "a", "best")
    assert out["used"] == "aligned_land_power_climate"


def test_errors():
    with pytest.raises(UserError):
        engine().search("x", "nowhere", "aligned")
    with pytest.raises(UserError):
        engine().search("x", "a", "magic")
    with pytest.raises(Unavailable):
        engine(planner=None).search("x", "a", "classifier")
    with pytest.raises(ValueError):
        engine(best="unknown")


def test_app_endpoints():
    meta = {"areas": {"a": {"label": "A", "tiles": []}}, "examples": ["corn"]}
    client = TestClient(create_app(engine(planner=lambda q, a: dict(PLAN, climate=[])), meta))
    page = client.get("/")
    assert "Ask the map" in page.text and page.headers["cache-control"] == "no-store"
    assert client.get("/api/meta").json() == meta
    r = client.post("/api/search", json={"question": "corn", "area": "a", "method": "aligned"})
    assert r.status_code == 200 and r.json()["facts"]["shown_pixels"] == 4
    assert (
        client.post("/api/search", json={"question": " ", "area": "a", "method": "aligned"}).status_code
        == 400
    )
    assert (
        client.post("/api/search", json={"question": "x", "area": "b", "method": "aligned"}).status_code
        == 400
    )
    no_model = TestClient(create_app(engine(planner=None), meta))
    for method in ("aligned", "classifier"):
        r = no_model.post("/api/search", json={"question": "x", "area": "a", "method": method})
        assert (
            r.status_code == 503
            and r.json()["detail"] == "Not available yet: The language model is not configured."
        )


def test_not_ready_methods_are_refused_with_their_reason_not_replaced():
    e = Engine(
        {"a": [tile()]},
        LEAVES,
        lambda texts: np.array([[1.0, 0.0]] * len(texts)),
        lambda q, a: PLAN,
        {"a": Ctx()},
        None,
        CAP,
        top_share=0.25,
        not_ready={"classifier": "Stage 3 classifiers missing."},
    )
    with pytest.raises(Unavailable, match="stage 3 has not finished"):
        e.search("corn", "a", "best")
    with pytest.raises(Unavailable, match="Stage 3 classifiers missing"):
        e.search("corn", "a", "classifier")
    assert e.search("corn", "a", "aligned")["used"] == "aligned"
    e2 = Engine(
        {"a": [tile()]},
        LEAVES,
        lambda texts: np.array([[1.0, 0.0]] * len(texts)),
        lambda q, a: PLAN,
        {"a": Ctx()},
        "classifier",
        CAP,
        not_ready={"classifier": "missing"},
    )
    with pytest.raises(Unavailable, match="uses 'classifier'"):
        e2.search("corn", "a", "best")


def test_answer_masks_match_the_shown_places():
    e = engine()
    masks = e.answer_masks(PLAN, "a")
    out = e.search("corn where July 2021 was hot", "a", "classifier")
    assert masks["t"].sum() == out["facts"]["shown_pixels"] == 3
    assert masks["t"][:, 0].sum() == 3 and not masks["t"][:, 1:].any()  # the hot cell's corn column only


def test_answer_masks_of_methods_1_and_2_match_the_shown_places():
    e = geometry_engine(dict(PLAN, near=[NEAR]))
    for how in ("aligned", "aligned_land_power_climate"):
        masks = e.answer_masks(e.planner("q", "a"), "a", how)
        out = e.search("corn fields near a wetland", "a", how)
        assert masks["t"].sum() == out["facts"]["shown_pixels"] == out["tiles"][0]["count"] == 2
        assert masks["t"][2:, 2].all() and masks["t"].sum() == 2  # rows 2 and 3 of column 2


def test_each_tile_carries_its_own_details():
    out = engine().search("corn where it was hot", "a", "classifier")
    tf = out["tiles"][0]["facts"]
    # one tile, so the tile's details equal the area's
    assert [x["places"] for x in tf["steps"]] == [x["places"] for x in out["facts"]["steps"]]
    assert tf["land_of_shown_pixels"] == out["facts"]["land_of_shown_pixels"]
    assert tf["cells"] == 15  # land cells in the tile (one of 16 pixels is invalid)
    aligned = engine().search("corn", "a", "aligned")
    assert aligned["tiles"][0]["facts"]["land_of_shown_pixels"] == aligned["facts"]["land_of_shown_pixels"]


def test_method_2_shows_only_places_in_power_cells_that_meet_the_climate():
    out = engine().search("corn where it was hot", "a", "aligned_land_power_climate")
    assert out["used"] == "aligned_land_power_climate" and out["plan"] == PLAN
    t = out["tiles"][0]
    # cell 0 (7 days) passes, cell 1 (2 days) fails: every shown place lies in cell 0
    assert t["count"] > 0 and t["count_meeting_climate"] == t["count"]
    assert [c["pass"] for c in out["power_cells"]] == [True, False]
    assert out["facts"]["climate_gate"] is True and "comparison_only" not in out["facts"]
    assert [c["value"] for c in out["facts"]["climate"][0]["cells"]] == [7.0, 2.0]


def test_method_2_says_why_when_no_power_cell_meets_the_climate():
    dry = engine(ctx=Ctx(values=(2.0, 2.0)))
    out = dry.search("corn where it was hot", "a", "aligned_land_power_climate")
    assert out["facts"]["shown_pixels"] == 0
    assert out["facts"]["empty_reason"].startswith("No NASA POWER cell in this area meets")
    assert "closest cell had 2" in out["facts"]["empty_reason"]


def geometry_engine(plan, corn_like=None):
    """A 4 x 4 tile whose right-hand column is wetland on the aligned space's own land map (its cells point
    along the wetland phrase), the rest corn; similarity to the land text grows down the columns, or is
    ``corn_like[column]`` for columns 0-2 when given."""
    z = np.zeros((16, 2), np.float16)
    col = np.tile(np.arange(4), 4)
    row = np.repeat(np.arange(4), 4)
    sim = 0.2 + 0.2 * row if corn_like is None else np.asarray([*corn_like, 0.0])[col]
    z[:, 0] = np.where(col == 3, 0.0, sim)
    z[:, 1] = np.where(col == 3, 1.0, 0.0)
    t = TileData(
        "t",
        (-98.0, 46.0, -97.9, 46.1),
        np.arange(16).reshape(4, 4),
        np.ones((4, 4), bool),
        z,
        np.zeros((3, 4, 4), np.float16),
        np.zeros((4, 4), np.uint8),
    )

    def encode(texts):
        return np.array([[0.0, 1.0] if "wetland" in x else [1.0, 0.0] for x in texts])

    ctx = {"a": Ctx()}
    return Engine({"a": [t]}, LEAVES, encode, lambda q, a: plan, ctx, "classifier", CAP, top_share=0.25)


NEAR = {"classes": ["wetland"], "distance_m": 90, "negate": False}


def test_zero_shot_map_takes_the_most_similar_land_phrase():
    e = geometry_engine(dict(PLAN, near=[NEAR]))
    m = e.zero_shot_map(e.areas["a"][0])
    assert (m[:, 3] == LEAVES.index("wetland")).all() and (m[:, :3] == LEAVES.index("corn")).all()


def test_methods_1_and_2_measure_distances_with_geometry_not_wording():
    e = geometry_engine(dict(PLAN, near=[NEAR]))
    # candidates: the most similar 25% of all 16 cells (the cut falls on the tie at 0.6), rows 2 and 3 of
    # columns 0-2; within 90 m of the wetland column (90 m cells) are columns 2 and 3, so only rows 2 and 3
    # of column 2 are shown
    m1 = e.search("corn fields near a wetland", "a", "aligned")
    assert m1["facts"]["distance_gate"] is True and "within" not in m1["facts"]["text"]
    assert m1["tiles"][0]["count"] == 2 and m1["tiles"][0]["count_meeting_climate"] == 2
    # Method 2 also needs the hot NASA POWER cell (columns 0 and 2): the same two cells
    m2 = e.search("corn fields near a wetland", "a", "aligned_land_power_climate")
    assert m2["facts"]["text"] == "Show cropland" and m2["facts"]["climate_gate"] is True
    assert m2["tiles"][0]["count"] == m2["tiles"][0]["count_meeting_climate"] == 2


def test_methods_1_and_2_pick_candidates_before_the_distance_rule_filters_them():
    # the land text is most similar to column 0, far from the wetland column; the cells near the wetland
    # (columns 2 and 3) are poor matches, so nothing is both a candidate and within 90 m of the wetland
    e = geometry_engine(dict(PLAN, climate=[], near=[NEAR]), corn_like=[1.0, 0.5, 0.1])
    for method in ("aligned", "aligned_land_power_climate"):
        out = e.search("corn fields near a wetland", "a", method)
        assert out["facts"]["shown_pixels"] == 0
        assert out["facts"]["empty_reason"] == (
            "None of the 4 places most similar to “Show cropland” is within 90 m of wetland "
            "on the aligned space's land map."
        )


def test_far_rules_are_negated_and_an_impossible_rule_says_why():
    far = dict(PLAN, climate=[], near=[dict(NEAR, negate=True)])
    out = geometry_engine(far).search("corn far from wetlands", "a", "aligned")
    sel = out["tiles"][0]
    assert sel["count"] == 4  # of the candidates (rows 2 and 3 of columns 0-2), columns 0 and 1 are far away
    nowhere = dict(PLAN, climate=[], near=[{"classes": ["cropland"], "distance_m": 90, "negate": True}])
    empty = geometry_engine(nowhere).search("x", "a", "aligned")
    assert empty["facts"]["shown_pixels"] == 0
    assert empty["facts"]["empty_reason"].startswith("No place in this area is more than 90 m from any")


def test_unsupported_questions_refuse_the_classifier_but_free_text_search_the_aligned_space():
    def planner(q, a):
        raise Unsupported("barns are not a land class")

    e = engine(best="classifier", planner=planner)
    with pytest.raises(Unsupported, match="cannot answer that: barns"):
        e.search("barns", "a", "best")
    with pytest.raises(Unsupported):
        e.search("barns", "a", "classifier")
    for method in ("aligned", "aligned_land_power_climate"):
        out = e.search("barns", "a", method)
        f = out["facts"]
        assert out["plan"] is None and f["free_text"] and f["text"] == "barns" and f["climate"] == []
        assert f["shown_pixels"] > 0 and f["rule"] == "top 25%" and "barns" in f["plan_words"]
        assert "barns are not a land class" in f["plan_words"] and out["power_cells"] == []


def test_methods_1_and_2_embed_the_questions_own_land_words():
    seen = []

    def encode(texts):
        seen.extend(texts)
        return np.array([[1.0, 0.0]] * len(texts))

    plan = dict(PLAN, land=dict(PLAN["land"], words="maize"))
    e = Engine(
        {"a": [tile()]}, LEAVES, encode, lambda q, a: plan, {"a": Ctx()}, "aligned", CAP, top_share=0.25
    )
    out = e.search("maize", "a", "aligned")
    # the question's words for the land, then Method 1's climate wording as before
    assert out["facts"]["text"].startswith("maize where") and any(x.startswith("maize") for x in seen)
    # without words the plan's caption wording is embedded, as before
    e2 = Engine(
        {"a": [tile()]}, LEAVES, encode, lambda q, a: PLAN, {"a": Ctx()}, "aligned", CAP, top_share=0.25
    )
    assert e2.search("corn", "a", "aligned")["facts"]["text"].startswith("Show ")


def test_method_3_ranks_by_the_words_when_no_class_fits_and_filters_by_geometry():
    plan = {
        "area": "a",
        "land": {"classes": [], "min_patch_ha": None, "words": "barns"},
        "near": [],
        "climate": [],
        "output": "map",
    }
    e = engine(best="classifier", planner=lambda q, a: plan)
    out = e.search("barns", "a", "best")
    f = out["facts"]
    assert f["word_land"] == {"words": "barns", "share": 0.25} and f["shown_pixels"] > 0
    assert f["steps"][0]["label"].startswith("places most like \u201cbarns\u201d")
    assert e.answer_masks(plan, "a")["t"].sum() == f["shown_pixels"]
    # a smaller share shows fewer places
    assert e.search("barns", "a", "best", 0.1)["facts"]["shown_pixels"] <= f["shown_pixels"]
    with pytest.raises(UserError, match="between 0 and 20%"):
        e.search("barns", "a", "best", 0.5)
