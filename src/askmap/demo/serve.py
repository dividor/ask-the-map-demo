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
from askmap.demo.engine import Engine, TileData, Unsupported, UserError
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
            leaves=demo_leaves(meta),
            groups=cfg["classes"]["groups"],
            res_m=cfg["grid"]["res_m"] * meta["stride"],
            power=ds,
            area_cells=cells,
            pixel_cell=pc,
        )
    return out


def server_host(env: dict | None = None) -> str:
    """The address the server listens on: ``ASKMAP_HOST`` if set, else localhost only. A container sets it to
    ``0.0.0.0`` so the platform can reach the server; nothing else should."""
    value = (os.environ if env is None else env).get("ASKMAP_HOST", "").strip()
    return value or "127.0.0.1"


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


def demo_leaves(meta: dict) -> list[str]:
    """The class list the demo data was built with (notebook 13 records it). The server, the planner's
    catalogue and the engine all use this list, so a class list changed in the config after the data was built
    cannot shift the data's columns."""
    if "leaves" not in meta:
        raise KeyError("demo meta.json has no 'leaves': rebuild the demo data with notebook 13")
    return list(meta["leaves"])


def plan_key(area: str, text: str, prompt: str = "") -> str:
    """The saved-reply file name for a question: the area, the text sent to the model and the prompt it was
    sent with, so a changed prompt or catalogue never reuses a reply made under the old one."""
    salt = hashlib.sha256(prompt.encode()).hexdigest()[:8] if prompt else ""
    return hashlib.sha256(f"{area}|{text}|{salt}".encode()).hexdigest()[:16]


def tidy_words(plan: dict) -> None:
    """The question's own words in a validated plan, lower-cased and single-spaced: the model copies them with
    or without a capital, and the same words must give the same plan and the same embedding."""
    for part in [plan["land"], *plan["near"]]:
        if part.get("words"):
            part["words"] = " ".join(part["words"].split()).lower()


def make_planner(cfg: dict, translate, log_dir: Path, leaves: list[str] | None = None, prompt: str = ""):
    """``planner(question, area)``: the main open model's plan for the question in the chosen area.
    ``leaves`` is the class list the plan may use (default: the configured one); ``prompt`` is the system
    prompt the replies are asked with (part of the saved-reply key)."""
    labels = cfg["question_areas"]
    catalogue = area_classes(cfg, leaves)

    def planner(question: str, area: str) -> dict:
        text = (
            question
            if labels[area].split(" (")[0].lower() in question.lower()
            else f"In the {labels[area].split(' (')[0]}: {question}"
        )
        rec = translate(text, log_dir / f"{plan_key(area, text, prompt)}.json")
        plan = rec.get("plan")
        if plan is None:
            raise UserError("The language model did not return a usable plan; try rewording the question.")
        if "unsupported" in plan:
            # the model says a part of the question cannot be expressed with the data (configs/language.md);
            # the engine decides what that means for each method
            why = plan["unsupported"]
            if not isinstance(why, str) or not why.strip():
                raise UserError("The language model said the question is unsupported but gave no reason.")
            raise Unsupported(why.strip())
        plan = dict(plan, area=area)
        try:
            validate(plan, catalogue, (int(cfg["power"]["start"][:4]), int(cfg["power"]["end"][:4])))
            tidy_words(plan)
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
    leaves = demo_leaves(meta)
    if leaves != cfg["classes"]["leaves"]:
        print(
            "note: the demo data was built with a different class list from configs/feasibility.yaml; "
            f"serving the data's classes {leaves} until notebook 13 rebuilds it"
        )
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cache = os.environ["HF_HOME"]
    encode = TextEncoder(meta["encoder"], meta["encoder_kind"], device, cache)

    llm = cfg["llm"]
    client = make_client(llm["base_url"])
    years = (int(cfg["power"]["start"][:4]), int(cfg["power"]["end"][:4]))
    system = system_prompt(
        (PROJECT_ROOT / "configs" / "language.md").read_text(), catalogue_text(cfg, leaves), data_years=years
    )
    planner = make_planner(
        cfg,
        lambda q, path: translate(client, llm["models"]["main"], system, q, path, llm),
        PROJECT_ROOT / "outputs" / "llm" / "demo",
        leaves,
        system,
    )
    engine = Engine(
        load_tiles(demo_dir, meta),
        leaves,
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
        host=server_host(),
        port=au["demo"]["port"],
    )


if __name__ == "__main__":
    main()
