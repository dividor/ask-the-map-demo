"""Start the demo: ``uv run python -m askmap.demo.serve`` (after notebook 13 has written the demo data).

Loads the precomputed tiles, the text encoder of the stage 2 model, the open language model (Hugging Face
router, token from the local login) and NASA POWER contexts, then serves the app on localhost.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from askmap.catalogue import area_classes, catalogue_text
from askmap.config import PROJECT_ROOT, data_root, load_config, load_env
from askmap.demo.engine import Engine, TileData, UserError
from askmap.plan import AreaContext, validate


def load_tiles(demo_dir: Path, meta: dict) -> dict[str, list[TileData]]:
    """TileData per area from notebook 13's files."""
    out: dict[str, list[TileData]] = {}
    for area, a in meta["areas"].items():
        out[area] = []
        for t in a["tiles"]:
            z = np.load(demo_dir / f"{t['name']}.npz")
            out[area].append(
                TileData(t["name"], tuple(t["bounds"]), z["index"], z["valid"], z["z"], z["probs"], z["leaf"])
            )
    return out


def make_contexts(cfg: dict, meta: dict, demo_dir: Path) -> dict[str, AreaContext]:
    """POWER contexts on the demo grid (cell indices saved by notebook 13; resolution = stride x 30 m)."""
    from askmap.power import add_derived, fetch_daily

    P = cfg["power"]
    root = data_root()
    daily = {
        box: add_derived(
            fetch_daily(P, b, root / "power" / f"universe_{box}_daily.nc"), P["low_wind_sun_pct"]
        )
        for box, b in P["boxes"].items()
    }
    out = {}
    for area, a in meta["areas"].items():
        ds = daily[a["box"]]
        cells = np.zeros((ds.sizes["lat"], ds.sizes["lon"]), bool)
        pc = {}
        for t in a["tiles"]:
            z = np.load(demo_dir / f"{t['name']}.npz")
            pc[t["name"]] = (z["iy"], z["ix"])
            cells[z["iy"], z["ix"]] = True
        out[area] = AreaContext(
            leaves=cfg["classes"]["leaves"],
            groups=cfg["classes"]["groups"],
            res_m=cfg["grid"]["res_m"] * meta["stride"],
            power=ds,
            area_cells=cells,
            pixel_cell=pc,
        )
    return out


def page_examples(examples: list[dict], areas: dict) -> list[dict]:
    """The page's clickable examples, ``{"question", "area"}``; fails on an empty question or unknown area."""
    out = []
    for e in examples:
        question, area = str(e["question"]).strip(), e["area"]
        if not question:
            raise ValueError("an example question is empty")
        if area not in areas:
            raise ValueError(f"example area {area!r} is not a demo area")
        out.append({"question": question, "area": area})
    return out


def make_planner(cfg: dict, translate, log_dir: Path):
    """``planner(question, area)``: the main open model's plan for the question in the chosen area."""
    labels = cfg["question_areas"]
    catalogue = area_classes(cfg)

    def planner(question: str, area: str) -> dict:
        text = (
            question
            if labels[area].split(" (")[0].lower() in question.lower()
            else f"In the {labels[area].split(' (')[0]}: {question}"
        )
        key = hashlib.sha256(f"{area}|{text}".encode()).hexdigest()[:16]
        rec = translate(text, log_dir / f"{key}.json")
        plan = rec.get("plan")
        if plan is None:
            raise UserError("The language model did not return a usable plan; try rewording the question.")
        plan = dict(plan, area=area)
        try:
            validate(plan, catalogue, (int(cfg["power"]["start"][:4]), int(cfg["power"]["end"][:4])))
        except (ValueError, KeyError, TypeError) as e:
            raise UserError(
                f"The language model's plan was not valid ({e}); try rewording the question."
            ) from e
        return plan

    return planner


def main() -> None:
    import uvicorn

    from askmap.aether_us import TextEncoder
    from askmap.demo.app import create_app
    from askmap.llm import make_client, system_prompt, translate

    load_env()
    cfg = load_config(PROJECT_ROOT / "configs" / "feasibility.yaml")
    au = load_config(PROJECT_ROOT / "configs" / "aether_us.yaml")
    cap = load_config(PROJECT_ROOT / "configs" / "captions.yaml")
    demo_dir = data_root() / "demo"
    meta = json.loads((demo_dir / "meta.json").read_text())
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cache = os.environ["HF_HOME"]
    encode = TextEncoder(meta["encoder"], meta["encoder_kind"], device, cache)

    llm = cfg["llm"]
    client = make_client(llm["base_url"])
    system = system_prompt((PROJECT_ROOT / "configs" / "language.md").read_text(), catalogue_text(cfg))
    planner = make_planner(
        cfg,
        lambda q, path: translate(client, llm["models"]["main"], system, q, path, llm),
        PROJECT_ROOT / "outputs" / "llm" / "demo",
    )
    engine = Engine(
        load_tiles(demo_dir, meta),
        cfg["classes"]["leaves"],
        encode,
        planner,
        make_contexts(cfg, meta, demo_dir),
        meta["best"],
        cap,
        top_share=meta["top_share"],
        not_ready=meta["not_ready"],
    )
    examples = page_examples(au["demo"]["page_examples"], meta["areas"])
    uvicorn.run(
        create_app(
            engine,
            {
                "areas": meta["areas"],
                "best": meta["best"],
                "land_method": meta["land_method"],
                "examples": examples,
                "not_ready": engine.not_ready,
            },
        ),
        host="127.0.0.1",
        port=au["demo"]["port"],
    )


if __name__ == "__main__":
    main()
