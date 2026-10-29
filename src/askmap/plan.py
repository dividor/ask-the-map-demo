"""The query language (a JSON "plan") and its executor.

A plan names a study area, the land to return (classes, optional minimum patch size), distance rules to other
classes, NASA POWER climate conditions, and the output (a map, or its area). Everything is combined with AND.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy.ndimage import label

from askmap.geo import within as _within
from askmap.layers import class_mask
from askmap.power import BASE_VARS, DERIVED_VARS, STATS, condition_cells, condition_values

PLAN_KEYS = {"area", "land", "near", "climate", "output"}
LAND_KEYS = {"classes", "min_patch_ha"}
NEAR_KEYS = {"classes", "distance_m", "negate"}
COND_KEYS = {"variable", "years", "months", "stat", "day_threshold", "test"}
OUTPUTS = ("map", "area")
EIGHT = np.ones((3, 3), dtype=int)


def _keys(obj: dict, allowed: set, required: set, what: str) -> None:
    if not isinstance(obj, dict):
        raise ValueError(f"{what} must be an object")
    extra, missing = set(obj) - allowed, required - set(obj)
    if extra or missing:
        raise ValueError(f"{what}: unexpected keys {sorted(extra)}, missing keys {sorted(missing)}")


def _classes(names, known: set[str], what: str) -> None:
    if not isinstance(names, list) or not names:
        raise ValueError(f"{what}: classes must be a non-empty list")
    bad = [n for n in names if n not in known]
    if bad:
        raise ValueError(f"{what}: unknown classes {bad}")


def validate(plan: dict, catalogue: dict[str, list[str]], year_range: tuple[int, int]) -> None:
    """Raise ValueError if ``plan`` is not a valid plan. ``catalogue`` maps area -> class names."""
    _keys(plan, PLAN_KEYS, PLAN_KEYS, "plan")
    if plan["area"] not in catalogue:
        raise ValueError(f"unknown area {plan['area']!r}")
    known = set(catalogue[plan["area"]])
    _validate_land(plan["land"], known)
    if not isinstance(plan["near"], list) or not isinstance(plan["climate"], list):
        raise ValueError("near and climate must be lists")
    for i, n in enumerate(plan["near"]):
        _validate_near(n, known, f"near[{i}]")
    for i, c in enumerate(plan["climate"]):
        _validate_condition(c, year_range, f"climate[{i}]")
    if plan["output"] not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}")


def _validate_land(land: dict, known: set[str]) -> None:
    _keys(land, LAND_KEYS, LAND_KEYS, "land")
    _classes(land["classes"], known, "land")
    mp = land["min_patch_ha"]
    if mp is not None and (not isinstance(mp, int | float) or mp <= 0):
        raise ValueError("land: min_patch_ha must be null or > 0")


def _validate_near(n: dict, known: set[str], what: str) -> None:
    _keys(n, NEAR_KEYS, NEAR_KEYS, what)
    _classes(n["classes"], known, what)
    if not isinstance(n["distance_m"], int | float) or n["distance_m"] <= 0:
        raise ValueError(f"{what}: distance_m must be > 0")
    if not isinstance(n["negate"], bool):
        raise ValueError(f"{what}: negate must be true or false")


def _validate_condition(c: dict, year_range: tuple[int, int], what: str) -> None:
    lo, hi = year_range
    _keys(c, COND_KEYS, COND_KEYS, what)
    if c["variable"] not in BASE_VARS + DERIVED_VARS:
        raise ValueError(f"{what}: unknown variable {c['variable']!r}")
    if c["stat"] not in STATS:
        raise ValueError(f"{what}: unknown stat {c['stat']!r}")
    if c["stat"].startswith("days_") != (c["day_threshold"] is not None):
        raise ValueError(f"{what}: day_threshold is required for days_above/days_below and only then")
    _year_pair(c["years"], lo, hi, what)
    months = c["months"]
    if not isinstance(months, list) or not months or any(m not in range(1, 13) for m in months):
        raise ValueError(f"{what}: months must be a non-empty list of 1..12")
    _validate_test(c["test"], lo, hi, what)


def _validate_test(t, lo: int, hi: int, what: str) -> None:
    if not isinstance(t, dict) or t.get("kind") not in ("absolute", "history_rank", "area_rank"):
        raise ValueError(f"{what}: test kind must be absolute, history_rank or area_rank")
    if t["kind"] == "absolute":
        _keys(t, {"kind", "op", "value"}, {"kind", "op", "value"}, f"{what}.test")
        if t["op"] not in (">=", "<="):
            raise ValueError(f"{what}: op must be >= or <=")
        return
    need = {"kind", "side", "pct"} | ({"history"} if t["kind"] == "history_rank" else set())
    _keys(t, need, need, f"{what}.test")
    if t["side"] not in ("high", "low") or not 0 < t["pct"] < 100:
        raise ValueError(f"{what}: side must be high/low and 0 < pct < 100")
    if t["kind"] == "history_rank":
        _year_pair(t["history"], lo, hi, f"{what}.history")


def _year_pair(pair, lo: int, hi: int, what: str) -> None:
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(y, int) for y in pair)
        or pair[0] > pair[1]
        or pair[0] < lo
        or pair[1] > hi
    ):
        raise ValueError(f"{what}: years must be [start, end] within {lo}..{hi}")


@dataclass
class AreaContext:
    """What a plan needs to run in one study area."""

    leaves: list[str]
    groups: dict[str, list[str]]
    res_m: float
    power: xr.Dataset  # daily variables on the POWER grid
    area_cells: np.ndarray  # (lat, lon) bool: cells covering the area's tiles
    pixel_cell: dict[str, tuple[np.ndarray, np.ndarray]]  # tile -> (row index, col index) of POWER cell
    cache: dict = field(default_factory=dict)

    def climate(self, cond: dict) -> np.ndarray:
        key = repr(sorted((k, repr(v)) for k, v in cond.items()))
        if key not in self.cache:
            self.cache[key] = condition_cells(self.power, cond, self.area_cells)
        return self.cache[key]

    def climate_values(self, cond: dict) -> dict:
        """The per-cell numbers behind ``climate(cond)`` (see ``power.condition_values``)."""
        key = "values|" + repr(sorted((k, repr(v)) for k, v in cond.items()))
        if key not in self.cache:
            self.cache[key] = condition_values(self.power, cond, self.area_cells)
        return self.cache[key]


def union(leaf_map: np.ndarray, names: list[str], ctx: AreaContext) -> np.ndarray:
    return np.logical_or.reduce([class_mask(leaf_map, n, ctx.leaves, ctx.groups) for n in names])


def min_patch(mask: np.ndarray, min_px: float) -> np.ndarray:
    """Keep 8-connected patches with at least ``min_px`` pixels."""
    lab, n = label(mask, structure=EIGHT)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def within(mask: np.ndarray, distance_m: float, res_m: float) -> np.ndarray:
    """Pixels whose centre is within ``distance_m`` of a True pixel (see ``askmap.geo.within``)."""
    return _within(mask, distance_m, res_m)


def execute_tile(plan: dict, leaf_map: np.ndarray, tile: str, ctx: AreaContext) -> np.ndarray:
    """Answer mask of ``plan`` on one tile's land map."""
    land = plan["land"]
    out = union(leaf_map, land["classes"], ctx)
    if land["min_patch_ha"] is not None:
        out = min_patch(out, land["min_patch_ha"] * 10_000 / ctx.res_m**2)
    for n in plan["near"]:
        near = within(union(leaf_map, n["classes"], ctx), n["distance_m"], ctx.res_m)
        out &= ~near if n["negate"] else near
    if plan["climate"]:
        iy, ix = ctx.pixel_cell[tile]
        for c in plan["climate"]:
            out &= ctx.climate(c)[iy, ix]
    return out


def execute(plan: dict, leaf_maps: dict[str, np.ndarray], ctx: AreaContext) -> dict[str, np.ndarray]:
    """Answer masks of ``plan`` per tile."""
    return {t: execute_tile(plan, m, t, ctx) for t, m in leaf_maps.items()}
