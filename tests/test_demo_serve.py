import numpy as np
import pytest

from askmap.config import PROJECT_ROOT, load_config
from askmap.demo.engine import Unsupported, UserError
from askmap.demo.serve import demo_leaves, load_tiles, make_planner, page_examples, server_host

CFG = load_config(PROJECT_ROOT / "configs" / "feasibility.yaml")


def test_page_examples_keep_question_and_area_and_fail_on_bad_entries():
    areas = {"north_dakota": {}, "georgia": {}}
    got = page_examples([{"question": " corn ", "area": "georgia"}], areas)
    assert got == [{"question": "corn", "area": "georgia"}]
    with pytest.raises(ValueError):
        page_examples([{"question": "corn", "area": "mars"}], areas)
    with pytest.raises(ValueError):
        page_examples([{"question": "  ", "area": "georgia"}], areas)


def test_configured_page_examples_name_demo_areas():
    au = load_config(PROJECT_ROOT / "configs" / "aether_us.yaml")
    got = page_examples(au["demo"]["page_examples"], {a: {} for a in au["demo"]["areas"]})
    assert len(got) == 12
    assert len({x["question"] for x in got}) == len(got)  # no question is listed twice


def test_load_tiles_reads_notebook_13_files(tmp_path):
    np.savez_compressed(
        tmp_path / "A1.npz",
        index=np.zeros((2, 2), int),
        valid=np.ones((2, 2), bool),
        z=np.zeros((4, 3), np.float16),
        probs=np.zeros((2, 2, 2), np.float16),
        leaf=np.zeros((2, 2), np.uint8),
        iy=np.zeros((2, 2), int),
        ix=np.zeros((2, 2), int),
    )
    meta = {"areas": {"a": {"tiles": [{"name": "A1", "bounds": [0, 1, 2, 3]}]}}}
    got = load_tiles(tmp_path, meta)
    assert got["a"][0].name == "A1" and got["a"][0].bounds == (0, 1, 2, 3) and got["a"][0].z.shape == (4, 3)


def test_planner_names_the_area_validates_and_caches_by_question(tmp_path):
    seen = []
    plan = {
        "area": "north_dakota",
        "land": {"classes": ["corn"], "min_patch_ha": None},
        "near": [],
        "climate": [],
        "output": "map",
    }

    def translate(text, path):
        seen.append((text, path.name))
        return {"plan": plan}

    planner = make_planner(CFG, translate, tmp_path)
    got = planner("corn fields", "minnesota")
    assert got["area"] == "minnesota" and got["land"]["classes"] == ["corn"]
    assert seen[0][0].startswith("In the Minnesota study area: corn fields")
    planner("corn fields in the Minnesota study area", "minnesota")
    assert seen[1][0] == "corn fields in the Minnesota study area" and seen[0][1] != seen[1][1]

    unsupported = make_planner(
        CFG, lambda t, p: {"plan": {"unsupported": "the data has no record of fire"}}, tmp_path
    )
    with pytest.raises(Unsupported, match="cannot answer that: the data has no record of fire") as info:
        unsupported("corn fields that burned", "minnesota")
    assert info.value.reason == "the data has no record of fire"
    silent = make_planner(CFG, lambda t, p: {"plan": {"unsupported": ""}}, tmp_path)
    with pytest.raises(UserError, match="gave no reason"):
        silent("corn fields that burned", "minnesota")
    bad = make_planner(CFG, lambda t, p: {"plan": None}, tmp_path)
    with pytest.raises(UserError):
        bad("x", "minnesota")
    invalid = make_planner(
        CFG, lambda t, p: {"plan": dict(plan, land={"classes": ["rice"], "min_patch_ha": None})}, tmp_path
    )
    with pytest.raises(UserError):
        invalid("rice", "minnesota")


def test_server_listens_on_localhost_unless_told_otherwise():
    assert server_host({}) == "127.0.0.1"
    assert server_host({"ASKMAP_HOST": ""}) == "127.0.0.1"
    assert server_host({"ASKMAP_HOST": "0.0.0.0"}) == "0.0.0.0"


def test_demo_leaves_come_from_the_data_and_are_required():
    old = ["corn", "developed", "other"]
    assert demo_leaves({"leaves": old}) == old
    with pytest.raises(KeyError, match="notebook 13"):
        demo_leaves({"stride": 3})


def test_planner_validates_against_the_data_classes_not_the_config(tmp_path):
    plan = {
        "area": "x",
        "land": {"classes": ["town"], "min_patch_ha": None},
        "near": [],
        "climate": [],
        "output": "map",
    }
    old_data = make_planner(CFG, lambda t, p: {"plan": plan}, tmp_path, ["corn", "developed", "other"])
    with pytest.raises(UserError, match="unknown classes"):
        old_data("towns", "minnesota")
    assert make_planner(CFG, lambda t, p: {"plan": plan}, tmp_path)("towns", "minnesota")["land"][
        "classes"
    ] == ["town"]
