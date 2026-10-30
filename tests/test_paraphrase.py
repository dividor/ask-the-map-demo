import json

import numpy as np
import pytest

from askmap.captions import Location, Vocab, make_caption
from askmap.config import PROJECT_ROOT, load_config
from askmap.paraphrase import ask, extended_cap, keep_valid, parallel_lists, placeholders

CAP = load_config(PROJECT_ROOT / "configs" / "captions.yaml")
LEAVES = ["corn", "soybeans", "grassland", "wetland", "other"]
GROUPS = {"cropland": ["corn", "soybeans"]}
FACTS = ["heat|JUL|35|2021|5", "long|T2M|high|third"]


def test_parallel_lists_cover_every_indexed_wording_list():
    lists = parallel_lists(CAP)
    assert all(len(v) == 3 for v in lists.values())
    for key in (
        "land.corn",
        "near.classes.wetland",
        "near.far_templates",
        "climate_year.seasons.JUL.words",
        "climate_long.variables.T2M.words",
        "gdd.templates_at_least",
        "climate_long.sunlight_templates",
    ):
        assert key in lists
    bad = extended_cap(CAP, {"land.corn": ["maize"]})
    with pytest.raises(ValueError, match="differ in length"):
        parallel_lists(bad)


def test_placeholders():
    assert placeholders("within {d} of {c}") == ["c", "d"]
    assert placeholders("a forest") == []


def test_keep_valid_checks_placeholders_repeats_and_banned_terms():
    orig = ["within {d} of {c}", "less than {d} from {c}"]
    cands = [
        "near {c}",
        "Within {d} of {c}",
        "close to {c} ({d})",
        "by a railway {d} from {c}",
        "{d} from {c}",
        "inside {d} of {c}",
        "x {e} {d} {c}",
    ]
    assert keep_valid(orig, cands, ["railway"], 3) == [
        "close to {c} ({d})",
        "{d} from {c}",
        "inside {d} of {c}",
    ]
    with pytest.raises(ValueError, match="only 3 usable"):
        keep_valid(orig, cands, ["railway"], 4)
    assert keep_valid(["Show ", ""], ["Where are the", " Map  ", "show"], [], 2, prefix=True) == [
        "Where are the ",
        "Map ",
    ]
    assert keep_valid(["near {c}"], ["by {c", "by {c}"], [], 1) == [
        "by {c}"
    ]  # an unbalanced brace is skipped
    with pytest.raises(ValueError, match="different placeholders"):
        keep_valid(["{a}", "{b}"], ["{a}"], [], 1)


def test_extended_cap_appends_without_changing_the_original():
    ext = extended_cap(CAP, {"land.corn": ["maize", "cornfields"], "prefixes": ["Map "]})
    assert ext["land"]["corn"][3:] == ["maize", "cornfields"] and ext["prefixes"][-1] == "Map "
    assert len(CAP["land"]["corn"]) == 3
    with pytest.raises(ValueError, match="not a wording list"):
        extended_cap(CAP, {"near.union_prob": ["x"]})


def test_captions_draw_from_every_wording_when_extended():
    extra = {k: [f"{v[0]} qqq{i}" for i in range(5)] for k, v in parallel_lists(CAP).items()}
    ext = extended_cap(CAP, extra)
    vocab = Vocab(ext, LEAVES, GROUPS, FACTS, wordings=8)
    loc = Location(
        LEAVES.index("corn"),
        np.array([50.0, 400.0] + [1000.0] * (len(CAP["near"]["classes"]) - 2)),
        30.0,
        30.0,
        {"heat": np.array([0]), "long": np.array([1])},
        "north_dakota",
    )
    rng = np.random.default_rng(0)
    texts = [make_caption(loc, vocab, rng)["full"] for _ in range(400)]
    assert any("qqq4" in t for t in texts) and not any("{" in t for t in texts)
    # the default (three wordings) never uses the added ones
    base = Vocab(ext, LEAVES, GROUPS, FACTS)
    rng = np.random.default_rng(0)
    assert not any("qqq" in make_caption(loc, base, rng)["full"] for _ in range(400))


def test_ask_reuses_the_saved_reply_and_fails_on_missing_keys(tmp_path):
    log = tmp_path / "reply.json"
    log.write_text(json.dumps({"content": json.dumps({"land.corn": ["maize", "cornfields"]})}))
    got = ask(None, "m", {"land.corn": ["corn fields"]}, 2, [], "", log, {}, 100)
    assert got == {"land.corn": ["maize", "cornfields"]}
    with pytest.raises(ValueError, match="no candidates for"):
        ask(None, "m", {"land.corn": ["corn fields"], "land.hay": ["hay land"]}, 2, [], "", log, {}, 100)
