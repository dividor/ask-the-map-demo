"""Climate facts of every universe cell (for captions), the summary-statistics climate encoder, and the
mapping from a reference plan's climate condition to a fact.

Fact ids (strings):
- ``hist|VAR|SEASON|YEAR|high/low|P``: the season's value that year is in the cell's highest/lowest P% of
  2001-2025 (same rule as ``power.condition_cells`` with ``history_rank``);
- ``heat|SEASON|T|YEAR|N``: at least N days with a daily maximum above T C;
- ``rain|SEASON|YEAR|ge/le|MM``: season rainfall total at least / at most MM;
- ``gdd|YEAR|ge/le|G``: May-September growing degree days (base 10 C) at least / at most G;
- ``long|VAR|high/low|TIER``: 2001-2025 mean in the highest/lowest third or half of the training universe's
  cells;
- ``sun|ge|X``: 2001-2025 mean sunlight at least X kWh/m2/day.
"""

from __future__ import annotations

import numpy as np

from askmap.power import yearly_values

ALL_MONTHS = list(range(1, 13))


def seasons(cap: dict) -> dict[str, list[int]]:
    s = {k: v["months"] for k, v in cap["climate_year"]["seasons"].items()}
    s["all"] = ALL_MONTHS
    return s


def _cells(da_years, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """(years, lat, lon) DataArray -> (years, cells) at the given cells."""
    return np.asarray(da_years.values)[:, iy, ix]


def yearly(
    universe, var: str, months: list[int], stat: str, years: list[int], thr: float | None = None
) -> np.ndarray:
    """(years, cells) yearly values for every universe cell (boxes stacked in universe order)."""
    parts = []
    for box, g in universe.cells.groupby("box", sort=False):
        v = yearly_values(universe.daily[box][var], years, months, stat, thr)
        parts.append((g.index.to_numpy(), _cells(v, g.iy.to_numpy(), g.ix.to_numpy())))
    out = np.empty((len(years), len(universe.cells)))
    for idx, vals in parts:
        out[:, idx] = vals
    if not np.isfinite(out).all():
        raise ValueError(f"missing yearly values for {var}")
    return out


def long_term(universe, var: str, months: list[int], stat: str, years: list[int]) -> np.ndarray:
    """(cells,) mean over ``years`` of the yearly values."""
    return yearly(universe, var, months, stat, years).mean(axis=0)


def history_flags(vals: np.ndarray, side: str, pct: float) -> np.ndarray:
    """(years, cells): value in the cell's own highest/lowest ``pct``% of its years."""
    if side == "high":
        return vals >= np.quantile(vals, 1 - pct / 100, axis=0)
    return vals <= np.quantile(vals, pct / 100, axis=0)


def tier_flags(values: np.ndarray, reference: np.ndarray, side: str, pct: float) -> np.ndarray:
    """(cells,): value in the highest/lowest ``pct``% of ``reference`` values."""
    if side == "high":
        return values >= np.quantile(reference, 1 - pct / 100)
    return values <= np.quantile(reference, pct / 100)


Facts = list[tuple[str, np.ndarray]]  # (fact id, (cells,) flags), in the order the facts are defined


def fact_table(universe, cap: dict, years: list[int]) -> tuple[list[str], np.ndarray]:
    """All climate facts: ids and a (cells, facts) boolean table (tiers relative to non-held-out cells)."""
    ss = seasons(cap)
    facts = (
        _history_facts(universe, cap, ss, years)
        + _heat_facts(universe, cap, ss, years)
        + _rain_facts(universe, cap, ss, years)
        + _gdd_facts(universe, cap, ss, years)
        + _long_term_facts(universe, cap, ss, years)
    )
    ids = [fid for fid, _ in facts]
    return ids, np.stack([np.asarray(col, dtype=bool) for _, col in facts], axis=1)


def _history_facts(universe, cap: dict, ss: dict, years: list[int]) -> Facts:
    cy = cap["climate_year"]
    out: Facts = []
    for var, vc in cy["variables"].items():
        for sk, months in ((k, ss[k]) for k in cy["seasons"]):
            vals = yearly(universe, var, months, vc["stat"], years)
            for side in ("high", "low"):
                for p in cy["pcts"]:
                    f = history_flags(vals, side, p)
                    out += [(f"hist|{var}|{sk}|{y}|{side}|{p}", f[k]) for k, y in enumerate(years)]
    return out


def _heat_facts(universe, cap: dict, ss: dict, years: list[int]) -> Facts:
    hd = cap["heat_days"]
    out: Facts = []
    for sk in hd["seasons"]:
        for t in hd["thresholds_c"]:
            vals = yearly(universe, "T2M_MAX", ss[sk], "days_above", years, t)
            for k, y in enumerate(years):
                out += [(f"heat|{sk}|{t}|{y}|{n}", vals[k] >= n) for n in hd["min_days"]]
    return out


def _rain_facts(universe, cap: dict, ss: dict, years: list[int]) -> Facts:
    rt = cap["rain_total"]
    out: Facts = []
    for sk in rt["seasons"]:
        vals = yearly(universe, "PRECTOTCORR", ss[sk], "total", years)
        for k, y in enumerate(years):
            out += [(f"rain|{sk}|{y}|ge|{mm}", vals[k] >= mm) for mm in rt["at_least_mm"]]
            out += [(f"rain|{sk}|{y}|le|{mm}", vals[k] <= mm) for mm in rt["at_most_mm"]]
    return out


def _gdd_facts(universe, cap: dict, ss: dict, years: list[int]) -> Facts:
    gd = cap["gdd"]
    vals = yearly(universe, "GDD10", ss["MJJAS"], "total", years)
    out: Facts = []
    for k, y in enumerate(years):
        out += [(f"gdd|{y}|le|{g}", vals[k] <= g) for g in gd["at_most"]]
        out += [(f"gdd|{y}|ge|{g}", vals[k] >= g) for g in gd["at_least"]]
    return out


def _long_term_facts(universe, cap: dict, ss: dict, years: list[int]) -> Facts:
    cl = cap["climate_long"]
    train = ~universe.cells.heldout.to_numpy()
    out: Facts = []
    for var, vc in cl["variables"].items():
        lt = long_term(universe, var, ss[vc["months"]], vc["stat"], years)
        for side in ("high", "low"):
            out += [
                (f"long|{var}|{side}|{tier}", tier_flags(lt, lt[train], side, pct))
                for tier, pct in cl["tiers"].items()
            ]
    sun = long_term(universe, "ALLSKY_SFC_SW_DWN", ALL_MONTHS, "mean", years)
    out += [(f"sun|ge|{x}", sun >= x) for x in cl["sunlight_at_least"]]
    return out


def summary_features(universe, cap: dict, years: list[int]) -> np.ndarray:
    """Summary-statistics climate encoder, (cells, features): per-year seasonal ranks within the cell's own
    record (rain, root-zone wetness, temperature), heat-day counts, rain totals, growing degree days, and
    long-term means (z-scored over non-held-out cells)."""
    ss = seasons(cap)
    feats = []
    cy = cap["climate_year"]
    n = len(years)
    for var, vc in cy["variables"].items():
        for sk in cy["seasons"]:
            vals = yearly(universe, var, ss[sk], vc["stat"], years)
            ranks = vals.argsort(axis=0).argsort(axis=0) / (n - 1)
            feats.append(ranks.T)
    hd = cap["heat_days"]
    for sk in hd["seasons"]:
        for t in hd["thresholds_c"]:
            feats.append(yearly(universe, "T2M_MAX", ss[sk], "days_above", years, t).T / 31)
    for sk in cap["rain_total"]["seasons"]:
        feats.append(yearly(universe, "PRECTOTCORR", ss[sk], "total", years).T / 500)
    feats.append(yearly(universe, "GDD10", ss["MJJAS"], "total", years).T / 2000)
    train = ~universe.cells.heldout.to_numpy()
    for vc_var, vc in cap["climate_long"]["variables"].items():
        lt = long_term(universe, vc_var, ss[vc["months"]], vc["stat"], years)
        feats.append(((lt - lt[train].mean()) / lt[train].std())[:, None])
    return np.concatenate(feats, axis=1).astype(np.float32)


def condition_fact(cond: dict, cap: dict) -> str:
    """Fact id that renders a reference plan's climate condition in caption wording."""
    ss = {tuple(v): k for k, v in seasons(cap).items()}
    months = tuple(cond["months"])
    if months not in ss:
        raise ValueError(f"no caption season for months {cond['months']}")
    sk = ss[months]
    var, stat, test = cond["variable"], cond["stat"], cond["test"]
    y0, y1 = cond["years"]
    if test["kind"] == "history_rank":
        return f"hist|{var}|{sk}|{y0}|{test['side']}|{int(test['pct'])}"
    if test["kind"] == "area_rank":
        if (
            var not in cap["climate_long"]["variables"]
            or cap["climate_long"]["variables"][var]["months"] != sk
        ):
            raise ValueError(f"no long-term caption fact for {var} over {sk}")
        tiers = {round(v, 1): k for k, v in cap["climate_long"]["tiers"].items()}
        pct = round(float(test["pct"]), 1)
        if pct not in tiers:
            raise ValueError(f"no long-term caption tier for {test['pct']}%")
        return f"long|{var}|{test['side']}|{tiers[pct]}"
    value = test["value"]
    if var == "T2M_MAX" and stat == "days_above":
        return f"heat|{sk}|{int(cond['day_threshold'])}|{y0}|{int(value)}"
    op = "ge" if test["op"] == ">=" else "le"
    if var == "PRECTOTCORR" and stat == "total":
        return f"rain|{sk}|{y0}|{op}|{int(value)}"
    if var == "GDD10":
        return f"gdd|{y0}|{op}|{int(value)}"
    if var == "ALLSKY_SFC_SW_DWN" and (y0, y1) == (2001, 2025) and op == "ge":
        return f"sun|ge|{value}"
    raise ValueError(f"no caption fact for condition {cond}")


def long_term_vector(universe, cap: dict, years: list[int]) -> np.ndarray:
    """(cells, 7) standardised long-term POWER climate summaries (the ``climate_long`` variables, 2001-2025
    means), z-scored over non-held-out cells: the climate part of "places like this point"."""
    ss = seasons(cap)
    train = ~universe.cells.heldout.to_numpy()
    cols = []
    for var, vc in cap["climate_long"]["variables"].items():
        lt = long_term(universe, var, ss[vc["months"]], vc["stat"], years)
        cols.append((lt - lt[train].mean()) / lt[train].std())
    return np.stack(cols, axis=1).astype(np.float32)
