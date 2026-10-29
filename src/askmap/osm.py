"""OpenStreetMap land-use tags for the study tiles: one Overpass request per tile, polygons, raster layers.

Closed ways become polygons; multipolygon relations become their outer rings (assembled from the member ways)
minus their inner rings. Open ways (roads, streams, tree rows) are ignored. A tag "vocabulary" entry names one
caption fact and lists the tag conditions that make it true (any condition; within a condition, every key must
match one of its values).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import requests
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from askmap.geo import EQUAL_AREA, Tile


def overpass_query(bbox: tuple[float, float, float, float], keys: list[str], timeout_s: int) -> str:
    """Overpass QL for ways and relations carrying any of ``keys`` in a (west, south, east, north) box."""
    w, s, e, n = bbox
    box = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    parts = "".join(f'way["{k}"]{box};relation["{k}"]{box};' for k in keys)
    return f"[out:json][timeout:{timeout_s}];({parts});out geom;"


def fetch_overpass(
    query: str, url: str, path: Path, retries: int, wait_s: float, user_agent: str, log=print
) -> dict:
    """Run an Overpass query once and keep the JSON reply at ``path``; a saved reply is reused.

    The public server answers HTTP 406 to the default python-requests User-Agent, so one is always sent.
    """
    if path.exists():
        return json.loads(path.read_text())
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, data={"data": query}, headers={"User-Agent": user_agent}, timeout=300)
            if r.status_code == 200:
                data = r.json()
                if "remark" in data and "error" in str(data["remark"]).lower():
                    raise RuntimeError(f"Overpass error: {data['remark']}")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                tmp.write_text(json.dumps(data))
                tmp.replace(path)
                return data
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = str(e)
        log(f"Overpass attempt {attempt + 1} failed ({last}); waiting {wait_s:.0f}s")
        time.sleep(wait_s * (attempt + 1))
    raise RuntimeError(f"Overpass request failed after {retries + 1} attempts: {last}")


def _ring(coords: list[dict]) -> list[tuple[float, float]]:
    return [(c["lon"], c["lat"]) for c in coords]


def element_polygon(el: dict):
    """Polygon (lon/lat) of a closed way or a multipolygon relation; None for anything else."""
    if el["type"] == "way":
        return _way_polygon(el)
    if el["type"] == "relation" and el.get("tags", {}).get("type") == "multipolygon":
        return _multipolygon(el)
    return None


def _way_polygon(el: dict):
    pts = _ring(el.get("geometry", []))
    if len(pts) >= 4 and pts[0] == pts[-1]:
        p = Polygon(pts)
        return p if p.is_valid else p.buffer(0)
    return None


def _multipolygon(el: dict):
    lines: dict[str, list[LineString]] = {"outer": [], "inner": []}
    for m in el.get("members", []):
        role = m.get("role", "outer")
        if m.get("type") == "way" and m.get("geometry") and role in ("outer", "inner", ""):
            lines["inner" if role == "inner" else "outer"].append(LineString(_ring(m["geometry"])))
    outer = unary_union(list(polygonize(lines["outer"])))
    if outer.is_empty:
        return None
    inner = unary_union(list(polygonize(lines["inner"])))
    return outer if inner.is_empty else outer.difference(inner)


def matches(tags: dict, conditions: list[dict]) -> bool:
    """True if the tags meet any condition; a condition maps keys to allowed values (every key must match)."""
    return any(all(tags.get(k) in vals for k, vals in cond.items()) for cond in conditions)


def tag_layers(elements: list[dict], tile: Tile, vocab: dict[str, dict]) -> dict[str, np.ndarray]:
    """One boolean (rows, cols) layer per vocabulary entry: pixels whose centre lies in a matching polygon."""
    from rasterio.features import rasterize

    tr = Transformer.from_crs(4326, EQUAL_AREA, always_xy=True)

    def to_tile(xy: np.ndarray) -> np.ndarray:
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    shapes: dict[str, list] = {name: [] for name in vocab}
    for el in elements:
        tags = el.get("tags", {})
        hit = [name for name, v in vocab.items() if matches(tags, v["match"])]
        if not hit:
            continue
        geom = element_polygon(el)
        if geom is None or geom.is_empty:
            continue
        geom = shapely.transform(geom, to_tile)
        for name in hit:
            shapes[name].append(geom)
    out = {}
    for name, geoms in shapes.items():
        if geoms:
            out[name] = rasterize(
                [(g, 1) for g in geoms],
                out_shape=(tile.size_px, tile.size_px),
                transform=tile.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
        else:
            out[name] = np.zeros((tile.size_px, tile.size_px), dtype=bool)
    return out


def tag_pixel_counts(elements: list[dict], tile: Tile, keys: list[str]) -> dict[str, int]:
    """Pixels of the tile covered by each ``key=value`` tag of the given keys (areas only, as
    ``tag_layers``)."""
    found = {f"{k}={el['tags'][k]}" for el in elements for k in keys if k in el.get("tags", {})}
    vocab = {t: {"match": [{t.split("=", 1)[0]: [t.split("=", 1)[1]]}]} for t in found}
    return {t: int(m.sum()) for t, m in tag_layers(elements, tile, vocab).items()}


def new_tag_entries(
    counts: dict[str, int], vocab: dict[str, dict], excluded: list[dict], min_px: int
) -> list[str]:
    """``key=value`` tags covering at least ``min_px`` pixels that no vocabulary entry matches with that key
    and value alone, and that no excluded condition names (the same key with that value, whatever else it
    needs); largest first (ties by name)."""
    alone = {
        (k, v)
        for e in vocab.values()
        for cond in e["match"]
        if len(cond) == 1
        for k, vs in cond.items()
        for v in vs
    }
    named = {(k, v) for cond in excluded for k, vs in cond.items() for v in vs}
    out = []
    for t, c in counts.items():
        k, v = t.split("=", 1)
        if c >= min_px and (k, v) not in alone and (k, v) not in named:
            out.append(t)
    return sorted(out, key=lambda t: (-counts[t], t))
