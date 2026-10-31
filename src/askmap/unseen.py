"""OpenStreetMap answer keys for words the shared space never saw, and the scores used on them.

Unlike ``askmap.osm`` (areas only, as used in training), an answer key here can also come from lines
(railways, power lines, streams) and points (wind turbines, silos), each buffered by a fixed distance before
rasterising.
"""

from __future__ import annotations

import re

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point
from sklearn.metrics import average_precision_score

from askmap.geo import EQUAL_AREA, Tile
from askmap.osm import element_polygon, matches


def overpass_query_selectors(
    bbox: tuple[float, float, float, float], selectors: list[dict], timeout_s: int
) -> str:
    """Overpass QL for every selector ``{"filter": '["k"="v"]', "types": [node, way, relation]}`` in a
    (west, south, east, north) box."""
    w, s, e, n = bbox
    box = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    parts = []
    for sel in selectors:
        for kind in sel["types"]:
            if kind not in ("node", "way", "relation"):
                raise ValueError(f"unknown element type {kind!r}")
            parts.append(f"{kind}{sel['filter']}{box};")
    return f"[out:json][timeout:{timeout_s}];({''.join(parts)});out geom;"


def element_shape(el: dict, kinds: list[str]):
    """``(kind, geometry)`` in lon/lat for an element, as allowed by ``kinds`` ("area", "line", "point"),
    or None.

    Nodes are points. A closed way is an area if areas are allowed, otherwise a line; an open way is a line.
    Multipolygon relations are areas.
    """
    if el["type"] == "node":
        return ("point", Point(el["lon"], el["lat"])) if "point" in kinds else None
    if el["type"] == "relation":
        poly = element_polygon(el) if "area" in kinds else None
        return ("area", poly) if poly is not None and not poly.is_empty else None
    pts = [(c["lon"], c["lat"]) for c in el.get("geometry", [])]
    if len(pts) < 2:
        return None
    closed = len(pts) >= 4 and pts[0] == pts[-1]
    if closed and "area" in kinds:
        poly = element_polygon(el)
        return ("area", poly) if poly is not None and not poly.is_empty else None
    return ("line", LineString(pts)) if "line" in kinds else None


def rule_mask(
    elements: list[dict], tile: Tile, rule: dict, line_buffer_m: float, point_buffer_m: float, stride: int
) -> np.ndarray:
    """Answer key of one word on the demo grid (every ``stride``-th pixel of the tile's grid): pixels whose
    centre is inside a matching area, or within the buffer of a matching line or point."""
    from rasterio.features import rasterize

    tr = Transformer.from_crs(4326, EQUAL_AREA, always_xy=True)

    def to_tile(xy: np.ndarray) -> np.ndarray:
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    buffer = {"area": 0.0, "line": line_buffer_m, "point": point_buffer_m}
    shapes = []
    for el in elements:
        if not matches(el.get("tags", {}), rule["match"]):
            continue
        got = element_shape(el, rule["kinds"])
        if got is None:
            continue
        kind, geom = got
        geom = shapely.transform(geom, to_tile)
        if buffer[kind] > 0:
            geom = geom.buffer(buffer[kind])
        shapes.append(geom)
    full = np.zeros((tile.size_px, tile.size_px), dtype=bool)
    if shapes:
        full = rasterize(
            [(g, 1) for g in shapes], out_shape=full.shape, transform=tile.transform, fill=0, dtype="uint8"
        ).astype(bool)
    return full[::stride, ::stride]


def terms_in_captions(captions: list[str], terms: list[str]) -> set[str]:
    """The terms (whole words or phrases, any case) that occur in at least one caption."""
    text = "\n".join(captions).lower()
    return {t for t in terms if re.search(rf"\b{re.escape(t.lower())}\b", text)}


def ap_and_chance(truth: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Average precision of ``scores`` ranking ``truth``, and its chance level (the share of true cells)."""
    truth = np.asarray(truth, dtype=bool)
    if not truth.any() or truth.all():
        raise ValueError("the answer key must have both true and false cells")
    return float(average_precision_score(truth, scores)), float(truth.mean())


def block_shuffle(mask: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """``mask`` with its whole ``block`` x ``block`` blocks moved to random block positions; the edge strips
    that do not fill a whole block stay where they are."""
    out = mask.copy()
    nr, nc = mask.shape[0] // block, mask.shape[1] // block
    if nr * nc < 2:
        raise ValueError("the mask is too small to shuffle")
    blocks = [
        mask[i * block : (i + 1) * block, j * block : (j + 1) * block] for i in range(nr) for j in range(nc)
    ]
    for k, src in enumerate(rng.permutation(len(blocks))):
        i, j = divmod(k, nc)
        out[i * block : (i + 1) * block, j * block : (j + 1) * block] = blocks[src]
    return out


def bootstrap_median_by_word(
    words: np.ndarray, values: np.ndarray, n: int, seed: int
) -> tuple[float, float, float]:
    """Median of ``values`` and its 95% interval from resampling words (a word brings all of its rows)."""
    words, values = np.asarray(words), np.asarray(values, dtype=float)
    uniq = np.unique(words)
    if len(uniq) < 2:
        raise ValueError("need at least two words to bootstrap")
    rows = {w: values[words == w] for w in uniq}
    rng = np.random.default_rng(seed)
    meds = np.array(
        [np.median(np.concatenate([rows[w] for w in rng.choice(uniq, size=len(uniq))])) for _ in range(n)]
    )
    return float(np.median(values)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def bootstrap_median_difference(
    words_a: np.ndarray, values_a: np.ndarray, words_b: np.ndarray, values_b: np.ndarray, n: int, seed: int
) -> tuple[float, float, float]:
    """Median of ``values_a`` minus median of ``values_b``, with a 95% interval from resampling the words of
    each group independently (a word brings all of its rows)."""
    groups = []
    for words, values in ((words_a, values_a), (words_b, values_b)):
        words, values = np.asarray(words), np.asarray(values, dtype=float)
        uniq = np.unique(words)
        if len(uniq) < 2:
            raise ValueError("need at least two words in each group to bootstrap")
        groups.append((uniq, {w: values[words == w] for w in uniq}))
    rng = np.random.default_rng(seed)

    def med(uniq, rows):
        return np.median(np.concatenate([rows[w] for w in rng.choice(uniq, size=len(uniq))]))

    diffs = np.array([med(*groups[0]) - med(*groups[1]) for _ in range(n)])
    point = float(np.median(np.asarray(values_a, float)) - np.median(np.asarray(values_b, float)))
    return point, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
