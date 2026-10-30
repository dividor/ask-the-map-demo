"""Demo search engine: answers a plain-English question in one study area with one of three methods.

- ``aligned`` (Method 1): the plan in caption wording without its distance rules (land, patch size and
  climate as text) against every pixel's projected vector (A3 stage 2 model); every distance rule is
  measured with geometry on the zero-shot land map of the aligned space (amendment A8).
- ``aligned_land_power_climate`` (Method 2): the plan's land wording through the aligned space, distance
  rules by the same geometry, climate by NASA POWER numbers.
- ``classifier``: the open language model writes a plan; one AlphaEarth classifier per land class, A2's
  geometry and NASA POWER numbers answer it (A3 C3). Only places that meet every step of the plan are shown
  (land probability above 0.5, every distance rule, every climate condition), with the count left after each
  step, the POWER numbers behind each climate condition and, for an empty answer, the step that removed
  everything (A4).
- ``best``: whichever combination stage 3 measured best (recorded in the demo metadata).

Everything heavy (text encoder, language model, POWER contexts) is passed in, so the engine can be tested with
stand-ins.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from askmap.aether_us import climate_ok_tile
from askmap.captions import patch_text
from askmap.demo.core import (
    cell_rects,
    class_shares,
    climate_words,
    describe_plan,
    land_words,
    near_words,
    overlay_rgba,
    png_bytes,
    quantity_words,
    shown_bounds,
)
from askmap.layers import class_indices
from askmap.plan import min_patch, within
from askmap.zeroshot import climate_text, distance_ok, leaf_grid, leaf_phrases, zero_shot_leaf
from askmap.zeroshot import plan_text as caption_plan_text

LAND_THRESHOLD = 0.5  # the same threshold the plan's distance rules use on the class probabilities

# aligned: Method 1; aligned_land_power_climate: Method 2 (aligned space for land, NASA POWER numbers for
# climate); classifier: Method 3; best: the method chosen by the stage 3 tests (Method 3)
METHODS = ("aligned", "aligned_land_power_climate", "classifier", "best")


class UserError(ValueError):
    """A problem with the request that can be shown to the user as is."""


class Unsupported(UserError):
    """The planner says a part of the question cannot be expressed with the data (configs/language.md). The
    classifier method needs a plan, so it refuses; the aligned space can score any text, so Methods 1 and 2
    search it with the question's own words instead."""

    def __init__(self, reason: str):
        super().__init__(f"This data cannot answer that: {reason}")
        self.reason = reason


class Unavailable(RuntimeError):
    """A method that cannot run right now (for example, the language model is not configured)."""


def target_mask(t: TileData, classes: list[str], ctx) -> np.ndarray:
    """Cells of ``classes`` on AlphaEarth's maps (probability above ``LAND_THRESHOLD``): what a distance rule
    measures from."""
    probs = t.probs.astype(np.float32)
    return probs[class_indices(classes, ctx.leaves, ctx.groups)].max(axis=0) > LAND_THRESHOLD


@dataclass
class TileData:
    """One tile on the demo's coarse grid (every ``stride``-th pixel)."""

    name: str
    bounds: tuple[float, float, float, float]  # west, south, east, north
    index: np.ndarray  # display grid -> flat pixel index (-1 outside)
    valid: np.ndarray  # (side, side) bool
    z: np.ndarray  # (valid pixels, dim) projected unit vectors
    probs: np.ndarray  # (classes, side, side) classifier probabilities
    leaf: np.ndarray  # (side, side) official class index, for the facts panel


class Engine:
    def __init__(
        self,
        areas: dict[str, list[TileData]],
        leaves: list[str],
        encode: Callable[[list[str]], np.ndarray],
        planner: Callable[[str, str], dict] | None,
        contexts: dict,
        best: str | None,
        cap: dict,
        top_share: float = 0.02,
        not_ready: dict[str, str] | None = None,
    ):
        """``best`` is the measured best combination (None until stage 3 has run); ``not_ready`` maps a
        method to why it cannot run yet. A method that is not ready is refused with that reason, never
        replaced by another."""
        if best not in (None, "aligned", "classifier", "aligned_land_power_climate"):
            raise ValueError(f"unknown best method {best!r}")
        self.areas, self.leaves, self.encode, self.planner = areas, leaves, encode, planner
        self.contexts, self.best, self.cap, self.top_share = contexts, best, cap, top_share
        self.not_ready = dict(not_ready or {})
        if best is None:
            self.not_ready.setdefault(
                "best", "The best combination is not known yet: stage 3 has not finished."
            )
        if planner is None:
            for m in ("aligned", "aligned_land_power_climate", "classifier"):
                self.not_ready.setdefault(m, "The language model is not configured.")
        # zero-shot land maps of the aligned space (A8), built on first use: candidate leaves, their phrases
        self._candidates, self._phrases = leaf_phrases(cap, leaves)
        self._phrase_vectors: np.ndarray | None = None
        self._zero_shot: dict[str, np.ndarray] = {}

    def _ranked(self, text: str, tiles: list[TileData], share: float) -> tuple[list[np.ndarray], float]:
        """Similarity of every valid cell to ``text`` and the score at which the most similar ``share`` of the
        study area begins."""
        scores = self._text_scores(text, tiles)
        cut = np.quantile(np.concatenate([s[t.valid] for s, t in zip(scores, tiles, strict=True)]), 1 - share)
        return scores, float(cut)

    def _text_scores(self, text: str, tiles: list[TileData]) -> list[np.ndarray]:
        q = self.encode([text])[0]
        q = q / np.linalg.norm(q)
        out = []
        for t in tiles:
            s = np.full(t.valid.shape, -np.inf, dtype=np.float32)
            s[t.valid] = t.z.astype(np.float32) @ q
            out.append(s)
        return out

    def _plan(self, question: str, area: str) -> dict:
        if self.planner is None:
            raise Unavailable("this method needs the language model, which is not configured")
        return self.planner(question, area)

    def zero_shot_map(self, t: TileData) -> np.ndarray:
        """(rows, cols) zero-shot land map of a tile from its projected vectors: the leaf whose wording-0
        phrase is most similar (-1 outside the valid cells). Computed once per tile."""
        if t.name not in self._zero_shot:
            if self._phrase_vectors is None:
                v = np.asarray(self.encode(self._phrases), dtype=np.float32)
                self._phrase_vectors = v / np.linalg.norm(v, axis=1, keepdims=True)
            sim = self._phrase_vectors @ t.z.astype(np.float32).T
            self._zero_shot[t.name] = leaf_grid(t.valid, zero_shot_leaf(sim, self._candidates))
        return self._zero_shot[t.name]

    def plan_steps(
        self, plan: dict, t: TileData, ctx, word_land: tuple[np.ndarray, float] | None = None
    ) -> tuple[np.ndarray, list[tuple[str, str, np.ndarray]]]:
        """Land probability of the plan's classes on one tile, and one (kind, label, mask) per step: land
        (probability above ``LAND_THRESHOLD``, patch size), each distance rule, each climate condition. A plan
        with no land class ranks the land by similarity to the question's own words instead: ``word_land`` is
        (this tile's similarity scores, the area-wide cut)."""
        probs = t.probs.astype(np.float32)
        if plan["land"]["classes"]:
            land_idx = class_indices(plan["land"]["classes"], ctx.leaves, ctx.groups)
            land_p = probs[land_idx].max(axis=0)
            land = t.valid & (land_p > LAND_THRESHOLD)
            if plan["land"]["min_patch_ha"] is not None:
                land &= min_patch(
                    land_p > LAND_THRESHOLD, plan["land"]["min_patch_ha"] * 10_000 / ctx.res_m**2
                )
        else:
            if word_land is None:
                raise ValueError("a plan without land classes needs the words' similarity scores")
            land_p, cut = word_land
            land = t.valid & (land_p >= cut)
            if plan["land"]["min_patch_ha"] is not None:
                land &= min_patch(land, plan["land"]["min_patch_ha"] * 10_000 / ctx.res_m**2)
        steps = [("land", land_words(plan), land)]
        for n in plan["near"]:
            near = within(target_mask(t, n["classes"], ctx), n["distance_m"], ctx.res_m)
            steps.append(("near", near_words(n), ~near if n["negate"] else near))
        iy, ix = ctx.pixel_cell[t.name]
        for c in plan["climate"]:
            steps.append(("climate", climate_words(c), ctx.climate(c)[iy, ix]))
        return land_p, steps

    def _climate_facts(self, plan: dict, ctx) -> tuple[list[dict], list[dict]]:
        """Per climate condition, the POWER value of every cell of the area with its threshold; and the cells
        as map boxes with whether they pass every condition."""
        rects = cell_rects(ctx.power.lat.values, ctx.power.lon.values, ctx.area_cells)
        facts, passes = [], np.ones(len(rects), bool)
        for c in plan["climate"]:
            v = ctx.climate_values(c)
            what, unit = quantity_words(c)
            cells = []
            for k, r in enumerate(rects):
                val, thr = float(v["value"][r["iy"], r["ix"]]), float(v["threshold"][r["iy"], r["ix"]])
                ok = val >= thr if v["op"] == ">=" else val <= thr
                passes[k] &= ok
                cells.append(
                    {
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "value": round(val, 3),
                        "threshold": round(thr, 3),
                        "pass": bool(ok),
                    }
                )
            facts.append(
                {"label": climate_words(c), "quantity": what, "unit": unit, "op": v["op"], "cells": cells}
            )
        boxes = [{"bounds": r["bounds"], "pass": bool(p)} for r, p in zip(rects, passes, strict=True)]
        return facts, boxes

    @staticmethod
    def _empty_reason(steps: list[dict], climate: list[dict]) -> str:
        """Why an answer is empty: the first step that left no places."""
        k = next(i for i, st in enumerate(steps) if st["places"] == 0)
        st = steps[k]
        if st["kind"] == "land":
            return f"The land classifier found no {st['label']} in this area."
        if st["kind"] == "near":
            return f"None of the {steps[0]['label']} places is {st['label']}."
        c = climate[[i for i, x in enumerate(steps) if x["kind"] == "climate"].index(k)]
        if not any(cell["pass"] for cell in c["cells"]):
            return Engine._no_cell_reason(c)
        return (
            f'Some NASA POWER cells meet "{c["label"]}", but none of the places left after the earlier '
            "steps lies in them."
        )

    @staticmethod
    def _no_cell_reason(c: dict) -> str:
        """Why no NASA POWER cell meets a climate condition: the closest cell's value and its threshold."""
        sign = 1 if c["op"] == ">=" else -1
        best = max(c["cells"], key=lambda x: sign * (x["value"] - x["threshold"]))
        return (
            f'No NASA POWER cell in this area meets "{c["label"]}": the closest cell had '
            f"{best['value']:g} against a threshold of {best['threshold']:g} ({c['unit']})."
        )

    def search(self, question: str, area: str, method: str, share: float | None = None) -> dict:
        """``share``: the most similar share of the study area that Methods 1 and 2 (and a word-ranked land
        step of Method 3) show; default ``top_share``."""
        if area not in self.areas:
            raise UserError(f"Unknown area {area!r}.")
        if share is None:
            share = self.top_share
        elif not isinstance(share, int | float) or not 0 < share <= 0.2:
            raise UserError("The share of places to show must be between 0 and 20%.")
        if method not in METHODS:
            raise UserError(f"Unknown method {method!r}.")
        if method in self.not_ready:
            raise Unavailable(self.not_ready[method])
        t0 = time.perf_counter()
        tiles = self.areas[area]
        how = self.best if method == "best" else method
        if how is None:
            raise Unavailable("The best combination is not known yet.")
        if how in self.not_ready:
            raise Unavailable(f"The best combination uses '{how}', which is not ready: {self.not_ready[how]}")
        shown_plan: dict | None
        try:
            plan = self._plan(question, area)
        except Unsupported as e:
            if how == "classifier":
                raise
            shown_plan, out = None, self._answer_free_text(question, tiles, e.reason, share)
        else:
            shown_plan = plan
            if how == "classifier":
                out = self._answer_plan(plan, tiles, self.contexts[area], share)
            else:
                out = self._answer_top_share(plan, tiles, area, how, share)
        out.update(
            {
                "area": area,
                "method": method,
                "used": how,
                "plan": shown_plan,
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
        return out

    def answer_masks(self, plan: dict, area: str, how: str = "classifier") -> dict[str, np.ndarray]:
        """The places shown for ``plan`` in ``area`` by engine method ``how``, per tile, on the demo grid:
        every step met (``classifier``), or the aligned space's candidates that pass the plan's filters
        (``aligned``, ``aligned_land_power_climate``)."""
        if how != "classifier":
            tiles = self.areas[area]
            _, shows, _ = self._top_share_shows(plan, tiles, area, how, self.top_share)
            return {t.name: show for t, show in zip(tiles, shows, strict=True)}
        ctx = self.contexts[area]
        out = {}
        for t in self.areas[area]:
            keep = t.valid.copy()
            for _, _, mask in self.plan_steps(plan, t, ctx, self._word_land_for(plan, t, area))[1]:
                keep &= mask
            out[t.name] = keep
        return out

    def _word_land_for(self, plan: dict, t: TileData, area: str) -> tuple[np.ndarray, float] | None:
        """For ``answer_masks``: the word ranking of one tile when the plan has no land class."""
        if plan["land"]["classes"]:
            return None
        tiles = self.areas[area]
        scores, cut = self._ranked(plan["land"]["words"], tiles, self.top_share)
        return scores[[x.name for x in tiles].index(t.name)], cut

    def _answer_plan(self, plan: dict, tiles: list[TileData], ctx, share: float | None = None) -> dict:
        """Places that meet every step of the plan. A plan with no land class takes the most similar
        ``share`` of the area to the question's own land words as its land step."""
        share = self.top_share if share is None else share
        word_scores: list[np.ndarray] | None = None
        cut = 0.0
        if not plan["land"]["classes"]:
            word_scores, cut = self._ranked(plan["land"]["words"], tiles, share)
        out_tiles, shown_leaf = [], []
        steps: list[dict[str, Any]] | None = None
        for i, t in enumerate(tiles):
            _, tile_steps = self.plan_steps(plan, t, ctx, (word_scores[i], cut) if word_scores else None)
            if steps is None:
                steps = [{"kind": k, "label": lab, "places": 0} for k, lab, _ in tile_steps]
            keep = t.valid.copy()
            tile_counts = []
            for st, (_, _, mask) in zip(steps, tile_steps, strict=True):
                keep &= mask
                st["places"] += int(keep.sum())
                tile_counts.append({"label": st["label"], "places": int(keep.sum())})
            out_tiles.append(self._tile_out(t, keep))
            # the same details for this tile alone (shown from the tile's "Match information" tab)
            shares = class_shares(t.leaf[keep].astype(int), self.leaves)
            out_tiles[-1]["facts"] = {
                "steps": tile_counts,
                "land_of_shown_pixels": shares,
                "cells": int(t.valid.sum()),
            }
            if plan["climate"]:
                out_tiles[-1]["count_meeting_climate"] = int(keep.sum())  # every shown place meets them
            shown_leaf.append(t.leaf[keep])
        codes = np.concatenate(shown_leaf).astype(int)
        climate, boxes = self._climate_facts(plan, ctx) if plan["climate"] else ([], [])
        facts = {
            "shown_pixels": int(codes.size),
            "land_of_shown_pixels": class_shares(codes, self.leaves),
            "steps": steps,
            "climate": climate,
            "plan_words": describe_plan(plan),
            "rule": "every step",
            "empty_reason": self._empty_reason(steps or [], climate) if codes.size == 0 else None,
        }
        if word_scores is not None:
            facts["word_land"] = {"words": plan["land"]["words"], "share": share}
        if plan["climate"]:
            facts["share_meeting_climate_conditions"] = 1.0 if codes.size else None
        return {"tiles": out_tiles, "facts": facts, "power_cells": boxes}

    def _answer_free_text(
        self, question: str, tiles: list[TileData], reason: str, share: float | None = None
    ) -> dict:
        """Methods 1 and 2 without a plan: the most similar ``top_share`` of the study area to the question's
        own words in the aligned space, with no distance or climate filter (none can be derived from the
        text)."""
        share = self.top_share if share is None else share
        scores, cut = self._ranked(question, tiles, share)
        out_tiles, shown_leaf = [], []
        for s, t in zip(scores, tiles, strict=True):
            show = t.valid & (s >= cut)
            out_tiles.append(self._tile_out(t, show))
            out_tiles[-1]["facts"] = {
                "land_of_shown_pixels": class_shares(t.leaf[show].astype(int), self.leaves),
                "cells": int(t.valid.sum()),
            }
            shown_leaf.append(t.leaf[show])
        codes = np.concatenate(shown_leaf).astype(int)
        facts = {
            "shown_pixels": int(codes.size),
            "land_of_shown_pixels": class_shares(codes, self.leaves),
            "rule": f"top {100 * share:g}%",
            "plan_words": f"Free-text search of the aligned space for \u201c{question}\u201d; no plan, so no "
            f"distance or climate filter ({reason}).",
            "text": question,
            "free_text": True,
            "distance_gate": False,
            "climate_gate": False,
            "climate": [],
        }
        return {"tiles": out_tiles, "facts": facts, "power_cells": []}

    def _answer_top_share(
        self, plan: dict, tiles: list[TileData], area: str, how: str, share: float | None = None
    ) -> dict:
        """The aligned space has no class map to threshold, so its candidates are the most similar
        ``top_share`` of the whole study area, and the plan's geometry (A8) then filters them.

        Method 1 (``aligned``): the plan's land, patch size and climate in caption wording pick the
        candidates; only candidates that pass every distance rule (measured on the zero-shot land map) are
        shown. The plan's NASA POWER cells are shown for comparison, not used. Method 2
        (``aligned_land_power_climate``): the plan's land wording picks the candidates; only candidates that
        pass every distance rule and lie in NASA POWER cells that meet every climate condition are shown."""
        ctx = self.contexts[area]
        share = self.top_share if share is None else share
        text, shows, n = self._top_share_shows(plan, tiles, area, how, share)
        out_tiles, shown_leaf, clim = [], [], []
        for show, t in zip(shows, tiles, strict=True):
            out_tiles.append(self._tile_out(t, show))
            shares = class_shares(t.leaf[show].astype(int), self.leaves)
            out_tiles[-1]["facts"] = {"land_of_shown_pixels": shares, "cells": int(t.valid.sum())}
            shown_leaf.append(t.leaf[show])
            if plan["climate"]:
                ok = climate_ok_tile(plan, t.name, ctx)[show]
                clim.append(ok)
                out_tiles[-1]["count_meeting_climate"] = int(ok.sum())
        codes = np.concatenate(shown_leaf).astype(int)
        facts = {
            "shown_pixels": int(codes.size),
            "land_of_shown_pixels": class_shares(codes, self.leaves),
            "rule": f"top {100 * share:g}%",
            "plan_words": describe_plan(plan),
            "text": text,
            "distance_gate": bool(plan["near"]),
            "climate_gate": how != "aligned" and bool(plan["climate"]),
        }
        if clim:
            c = np.concatenate(clim)
            facts["share_meeting_climate_conditions"] = round(float(c.mean()), 3) if c.size else None
        boxes: list[dict] = []
        if plan["climate"]:
            facts["climate"], boxes = self._climate_facts(plan, ctx)
        if how == "aligned" and plan["climate"]:
            facts["comparison_only"] = True  # cells and values are shown beside the answer, not used by it
        if codes.size == 0:
            facts["empty_reason"] = self._top_share_empty_reason(
                plan, facts, n["candidates"], n["distance"], n["candidates_distance"]
            )
        return {"tiles": out_tiles, "facts": facts, "power_cells": boxes}

    def _search_text(self, plan: dict, how: str) -> str:
        """The text Methods 1 and 2 embed: the question's own land words when the plan carries them (the
        aligned space can score any text, so the wording the user chose is kept), else the plan's land in
        caption wording; plus the patch size and, for Method 1, the climate conditions in caption wording."""
        words = plan["land"].get("words")
        if not words:
            return caption_plan_text(plan, self.cap, distances=False, climate=how == "aligned")
        extras = []
        if plan["land"]["min_patch_ha"] is not None:
            extras.append(patch_text(plan["land"]["min_patch_ha"], self.cap, 0))
        if how == "aligned":
            extras += [climate_text(c, self.cap) for c in plan["climate"]]
        return words + ("" if not extras else " " + " and ".join(extras))

    def _top_share_shows(
        self, plan: dict, tiles: list[TileData], area: str, how: str, share: float
    ) -> tuple[str, list[np.ndarray], dict[str, int]]:
        """Methods 1 and 2: the text, the shown places per tile (the most similar ``share`` of the study area
        that pass the plan's filters) and the counts behind an empty answer's reason."""
        ctx = self.contexts[area]
        text = self._search_text(plan, how)
        scores, cut = self._ranked(text, tiles, share)
        shows, n = [], {"candidates": 0, "distance": 0, "candidates_distance": 0}
        for s, t in zip(scores, tiles, strict=True):
            candidate = t.valid & (s >= cut)
            ok = t.valid.copy()
            if plan["near"]:
                ok &= distance_ok(plan, self.zero_shot_map(t), ctx)
            n["candidates"] += int(candidate.sum())
            n["distance"] += int(ok.sum())
            n["candidates_distance"] += int((candidate & ok).sum())
            if how != "aligned":
                ok &= climate_ok_tile(plan, t.name, ctx)
            shows.append(candidate & ok)
        return text, shows, n

    @staticmethod
    def _top_share_empty_reason(
        plan: dict, facts: dict, n_candidates: int, n_distance: int, n_candidates_distance: int
    ) -> str | None:
        """Why an aligned-space answer is empty: no place passes the distance rules, or no candidate does,
        or (Method 2) no NASA POWER cell meets a climate condition, or no candidate that passes the distance
        rules lies in a passing cell."""
        rules = " and ".join(near_words(n) for n in plan["near"])
        if plan["near"] and n_distance == 0:
            return f"No place in this area is {rules} on the aligned space's land map."
        if plan["near"] and n_candidates_distance == 0:
            return (
                f"None of the {n_candidates:,} places most similar to “{facts['text']}” is {rules} "
                "on the aligned space's land map."
            )
        if not facts["climate_gate"]:
            return None
        failing = [c for c in facts["climate"] if not any(x["pass"] for x in c["cells"])]
        if failing:
            return Engine._no_cell_reason(failing[0])
        return (
            f"Some NASA POWER cells meet the climate conditions, but none of the {n_candidates:,} places "
            f"most similar to “{facts['text']}”{' that pass the distance rules' if plan['near'] else ''} "
            "lies in them."
        )

    @staticmethod
    def _tile_out(t: TileData, show: np.ndarray) -> dict:
        rgba = overlay_rgba(show, t.index)
        inside = t.index >= 0
        sel = np.zeros(t.index.shape, bool)
        sel[inside] = show.ravel()[t.index[inside]]
        return {
            "name": t.name,
            "bounds": list(t.bounds),
            "png": base64.b64encode(png_bytes(rgba)).decode("ascii"),
            "count": int(show.sum()),
            "match_bounds": shown_bounds(sel, t.bounds),
        }
