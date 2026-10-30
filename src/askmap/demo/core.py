"""Pure helpers for the demo: map grids, colouring scores, PNG overlays and summaries of the top matches."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from pyproj import Transformer

from askmap.geo import EQUAL_AREA, Tile, tile_bbox_4326


def display_index(
    tile: Tile, stride: int, n_out: int
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Map a regular lon/lat grid (``n_out`` x ``n_out``) over the tile's box to its stride-``stride`` pixels.

    Returns the flat pixel index of every display cell (-1 outside the tile) and the box (west, south, east,
    north), so a browser map can show the overlay as a lon/lat image without reprojection.
    """
    w, s, e, n = tile_bbox_4326(tile)
    lon = w + (np.arange(n_out) + 0.5) * (e - w) / n_out
    lat = n - (np.arange(n_out) + 0.5) * (n - s) / n_out
    glon, glat = np.meshgrid(lon, lat)
    x, y = Transformer.from_crs(4326, EQUAL_AREA, always_xy=True).transform(glon, glat)
    step = tile.res_m * stride
    side = int(np.ceil(tile.size_px / stride))
    col = np.floor((np.asarray(x) - tile.x0) / step).astype(int)
    row = np.floor((tile.y1 - np.asarray(y)) / step).astype(int)
    inside = (col >= 0) & (col < side) & (row >= 0) & (row < side)
    return np.where(inside, row * side + col, -1), (w, s, e, n)


def top_mask(scores: np.ndarray, valid: np.ndarray, share: float) -> np.ndarray:
    """True for the valid pixels whose score is in the top ``share`` of all valid pixels given."""
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    v = scores[valid]
    if v.size == 0:
        return np.zeros_like(valid)
    cut = np.quantile(v, 1 - share)
    return valid & (scores >= cut)


MATCH_COLOR = (255, 23, 23)  # bright red; how see-through it is depends on the map zoom (set on the page)


def overlay_rgba(show: np.ndarray, index: np.ndarray, color=MATCH_COLOR) -> np.ndarray:
    """RGBA display image on the ``index`` display grid: shown pixels fully opaque in ``color``; the rest
    transparent."""
    flat_show = show.ravel()
    out = np.zeros((*index.shape, 4), np.uint8)
    inside = index >= 0
    sel = np.zeros(index.shape, bool)
    sel[inside] = flat_show[index[inside]]
    out[sel, 0], out[sel, 1], out[sel, 2] = color
    out[sel, 3] = 255
    return out


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def class_shares(codes: np.ndarray, names: list[str], top: int = 5) -> list[dict]:
    """The most common classes among ``codes`` (indices into ``names``) with their shares."""
    if codes.size == 0:
        return []
    counts = np.bincount(codes, minlength=len(names))
    order = np.argsort(-counts)[:top]
    return [{"class": names[i], "share": round(float(counts[i] / codes.size), 3)} for i in order if counts[i]]


# Plain-English wording of plans (shown on the page next to the plan's JSON)
VAR_WORDS = {
    "T2M": ("temperature", "°C"),  # the daily mean; "mean" comes from the statistic
    "T2M_MAX": ("daily high temperature", "°C"),
    "T2M_MIN": ("daily low temperature", "°C"),
    "PRECTOTCORR": ("rainfall", "mm"),
    "GWETROOT": ("root-zone soil wetness", "(0 to 1)"),
    "WS50M": ("wind speed at 50 m", "m/s"),
    "RH2M": ("relative humidity", "%"),
    "ALLSKY_SFC_SW_DWN": ("sunlight", "kWh/m² per day"),
    "GDD10": ("growing degree days (base 10 °C)", "degree days"),
    "HDD18": ("heating degree days", "degree days"),
    "CDD18": ("cooling degree days", "degree days"),
    "LOW_WIND_SUN": ("days with both low wind and low sunlight", "days"),
}
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def class_words(names: list[str]) -> str:
    """'corn or soybeans' from class names ('solar_farm' -> 'solar farm')."""
    words = [n.replace("_", " ") for n in names]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def period_words(months: list[int], years: list[int]) -> str:
    """'July 2021', 'April to June 2022', 'June to August, 2021 to 2023 (average)' or '2021, all year'."""
    ms = sorted(months)
    if ms == list(range(1, 13)):
        m = "all year"
    elif ms == list(range(ms[0], ms[-1] + 1)):
        m = MONTHS[ms[0] - 1] if len(ms) == 1 else f"{MONTHS[ms[0] - 1]} to {MONTHS[ms[-1] - 1]}"
    else:
        m = ", ".join(MONTHS[i - 1] for i in ms)
    y0, y1 = years
    if y0 == y1:
        return f"{y0}, all year" if m == "all year" else f"{m} {y0}"
    return f"{y0} to {y1}, all year (average)" if m == "all year" else f"{m}, {y0} to {y1} (average)"


def quantity_words(cond: dict) -> tuple[str, str]:
    """What is measured, and its unit (e.g. ('days with daily high temperature above 35 °C', 'days'))."""
    label, unit = VAR_WORDS[cond["variable"]]
    stat = cond["stat"]
    if stat in ("days_above", "days_below"):
        side = "above" if stat == "days_above" else "below"
        return f"days with {label} {side} {cond['day_threshold']:g} {unit}", "days"
    if stat == "total":
        return f"total {label}", unit
    if cond["variable"] == "PRECTOTCORR":
        return f"mean daily {label}", "mm per day"
    return f"mean {label}", unit


def climate_words(cond: dict) -> str:
    """One NASA POWER condition in plain English."""
    when = period_words(cond["months"], cond["years"])
    what, unit = quantity_words(cond)
    test = cond["test"]
    if test["kind"] == "absolute":
        bound = "at least" if test["op"] == ">=" else "at most"
        if cond["stat"] in ("days_above", "days_below"):
            return f"{when} had {bound} {test['value']:g} {what}"
        return f"{when}: {what} {bound} {test['value']:g} {unit}"
    side = "highest" if test["side"] == "high" else "lowest"
    if test["kind"] == "history_rank":
        h0, h1 = test["history"]
        return f"{when}: {what} in the {side} {test['pct']:g}% of years {h0} to {h1}"
    return f"{when}: {what} in the {side} {test['pct']:g}% of this area's places"


def near_words(rule: dict) -> str:
    if rule["distance_m"] == 0:  # touching
        return f"{'not ' if rule['negate'] else ''}next to {class_words(rule['classes'])}"
    if rule["negate"]:
        return f"more than {rule['distance_m']:g} m from any {class_words(rule['classes'])}"
    return f"within {rule['distance_m']:g} m of {class_words(rule['classes'])}"


def land_words(plan: dict) -> str:
    land = plan["land"]
    if not land["classes"]:
        text = f"places most like \u201c{land['words']}\u201d (no land class fits)"
    else:
        text = class_words(land["classes"])
    if plan["land"]["min_patch_ha"] is not None:
        text += f" in patches of at least {plan['land']['min_patch_ha']:g} ha"
    return text


def describe_plan(plan: dict) -> str:
    """The whole plan as one plain-English sentence."""
    parts = [land_words(plan)]
    parts += [near_words(n) for n in plan["near"]]
    text = ", ".join(parts)
    if plan["climate"]:
        text += ", where " + " and ".join(climate_words(c) for c in plan["climate"]) + " (NASA POWER)"
    return text[0].upper() + text[1:] + "."


def shown_bounds(show_display: np.ndarray, bounds: tuple[float, float, float, float]) -> list[float] | None:
    """Lon/lat box (west, south, east, north) of the shown cells of a display grid laid over ``bounds``."""
    rows, cols = np.nonzero(show_display)
    if rows.size == 0:
        return None
    w, s, e, n = bounds
    h, wd = show_display.shape
    dx, dy = (e - w) / wd, (n - s) / h
    return [w + cols.min() * dx, n - (rows.max() + 1) * dy, w + (cols.max() + 1) * dx, n - rows.min() * dy]


def cell_rects(lat: np.ndarray, lon: np.ndarray, cells: np.ndarray) -> list[dict]:
    """Lon/lat boxes of the grid cells marked in ``cells`` (lat, lon), each with its indices."""
    dlat = float(np.diff(lat).mean()) / 2 if lat.size > 1 else 0.25
    dlon = float(np.diff(lon).mean()) / 2 if lon.size > 1 else 0.3125
    out = []
    for iy, ix in zip(*np.nonzero(cells), strict=True):
        la, lo = float(lat[iy]), float(lon[ix])
        out.append(
            {
                "iy": int(iy),
                "ix": int(ix),
                "lat": la,
                "lon": lo,
                "bounds": [lo - dlon, la - dlat, lo + dlon, la + dlat],
            }
        )
    return out
