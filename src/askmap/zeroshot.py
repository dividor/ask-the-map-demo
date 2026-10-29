"""Amendment A8: land maps read straight from the aligned space, and distance rules measured on them.

- zero-shot land map: each pixel's projected vector against the wording-0 phrase of every land class except
  "other"; the pixel takes the class of the most similar phrase (no per-class thresholds or weights);
- a distance rule is ``within`` on the zero-shot map of its classes (negated for "more than");
- the text the aligned space ranks by is the plan in caption wording with its distance rules taken out (patch
  size stays as wording; climate is kept for Method 1 and left to NASA POWER for Method 2).
"""

from __future__ import annotations

import numpy as np

from askmap.captions import fact_text, land_text, near_text, patch_text
from askmap.climate_facts import condition_fact
from askmap.demo.core import climate_words
from askmap.layers import class_indices
from askmap.plan import within


def leaf_phrases(cap: dict, leaves: list[str]) -> tuple[list[int], list[str]]:
    """Leaf indices and wording-0 land phrases of every leaf except "other" (the zero-shot candidates)."""
    idx = [i for i, n in enumerate(leaves) if n != "other"]
    missing = [leaves[i] for i in idx if leaves[i] not in cap["land"]]
    if missing:
        raise ValueError(f"no land phrase for {missing}")
    return idx, [cap["land"][leaves[i]][0] for i in idx]


def zero_shot_leaf(sim: np.ndarray, candidates: list[int]) -> np.ndarray:
    """Leaf index per pixel from (candidates, pixels) similarities: the most similar candidate's leaf."""
    sim = np.asarray(sim)
    if sim.ndim != 2 or sim.shape[0] != len(candidates):
        raise ValueError("similarities must be (candidates, pixels)")
    return np.asarray(candidates, dtype=np.int16)[sim.argmax(axis=0)]


def leaf_grid(valid: np.ndarray, leaf_of_valid: np.ndarray) -> np.ndarray:
    """(rows, cols) leaf map with the valid pixels' leaves (row-major order) and -1 elsewhere."""
    if int(valid.sum()) != len(leaf_of_valid):
        raise ValueError("one leaf per valid pixel is needed")
    out = np.full(valid.shape, -1, dtype=np.int16)
    out[valid] = leaf_of_valid
    return out


def distance_ok(plan: dict, leaf_map: np.ndarray, ctx) -> np.ndarray:
    """(rows, cols) bool: pixels that pass every distance rule of the plan, measured on ``leaf_map``
    (-1 = no class) at ``ctx.res_m``."""
    ok = np.ones(leaf_map.shape, dtype=bool)
    for n in plan["near"]:
        target = np.isin(leaf_map, class_indices(n["classes"], ctx.leaves, ctx.groups))
        near = within(target, n["distance_m"], ctx.res_m)
        ok &= ~near if n["negate"] else near
    return ok


def climate_text(cond: dict, cap: dict) -> str:
    """A climate condition in caption wording, or in the demo's plain English when no caption fact matches."""
    try:
        fid = condition_fact(cond, cap)
    except ValueError:
        return climate_words(cond)
    return fact_text(fid, cap, 0)


def plan_text(plan: dict, cap: dict, *, distances: bool, climate: bool) -> str:
    """The plan in caption wording (first wording of every phrase, no area name): land, patch size, the
    distance rules if ``distances``, the climate conditions if ``climate``."""
    extras = []
    if plan["land"]["min_patch_ha"] is not None:
        extras.append(patch_text(plan["land"]["min_patch_ha"], cap, 0))
    if distances:
        extras += [near_text(n["classes"], n["distance_m"], n["negate"], cap, 0) for n in plan["near"]]
    if climate:
        extras += [climate_text(c, cap) for c in plan["climate"]]
    land = land_text(plan["land"]["classes"], cap, 0)
    return cap["prefixes"][0] + land + ("" if not extras else " " + " and ".join(extras))
