"""Land-class maps: official ones from CDL, NWI and USPVDB, and the class vocabulary shared with predicted
maps.

A tile's land map is an integer array of *leaf* class indices (one per pixel). Query classes are either a leaf
("corn") or a group of leaves ("cropland").
"""

from __future__ import annotations

import numpy as np

from askmap.nwi import NWI_WATER, NWI_WETLAND  # NWI rasters: 1 = wetland, 2 = water


def annual_crop_codes(ranges: list[list[int]]) -> list[int]:
    """Expand inclusive ``[[lo, hi], ...]`` CDL class ranges into a sorted list of codes.

    Adapted from Matt Harris's earlier code (``cropint.one_in_four``)."""
    codes = sorted({c for lo, hi in ranges for c in range(lo, hi + 1)})
    if not codes:
        raise ValueError("no CDL codes given")
    return codes


# USPVDB rasters: 1 = farm built by the cut-off year, 2 = built later (left out of scoring)
SOLAR_EXISTING, SOLAR_LATER = 1, 2


def official_leaf_map(
    cdl: np.ndarray,
    leaves: list[str],
    classes_cfg: dict,
    nwi: np.ndarray | None = None,
    solar: np.ndarray | None = None,
) -> np.ndarray:
    """Leaf index per pixel from the official layers.

    Order (later wins): ``other``; CDL annual crops -> ``other_crop``; named CDL classes; CDL wetland classes
    (only where there is no NWI layer); NWI water; NWI wetland; USPVDB solar farms built by the cut-off.
    """
    idx = {name: i for i, name in enumerate(leaves)}
    for need in ("other", "other_crop"):
        if need not in idx:
            raise ValueError(f"leaf list must include {need!r}")
    cdl = np.asarray(cdl).astype(int)
    out = np.full(cdl.shape, idx["other"], dtype=np.int16)
    out[np.isin(cdl, annual_crop_codes(classes_cfg["cropland_ranges"]))] = idx["other_crop"]
    for name, codes in classes_cfg["cdl"].items():
        if name in idx:
            out[np.isin(cdl, codes)] = idx[name]
    if "wetland" in idx and nwi is None:
        out[np.isin(cdl, classes_cfg["cdl"]["wetland_cdl"])] = idx["wetland"]
    if nwi is not None:
        if "water" in idx:
            out[nwi == NWI_WATER] = idx["water"]
        if "wetland" in idx:
            out[nwi == NWI_WETLAND] = idx["wetland"]
    if solar is not None:
        if "solar_farm" not in idx:
            raise ValueError("solar layer given but 'solar_farm' is not a leaf class")
        out[solar == SOLAR_EXISTING] = idx["solar_farm"]
    return out


def class_names(leaves: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Every class a query may name in an area: its leaves (except ``other``) and groups with a leaf there."""
    names = [n for n in leaves if n != "other"]
    names += [g for g, members in groups.items() if any(m in leaves for m in members)]
    return names


def class_mask(
    leaf_map: np.ndarray, name: str, leaves: list[str], groups: dict[str, list[str]]
) -> np.ndarray:
    """Boolean mask of a leaf or group class; unknown names fail. A name that is a leaf of this class list is
    that leaf, even if the configured groups also know it as a group (data built before a leaf was split)."""
    if name in groups and name not in leaves:
        members = [leaves.index(m) for m in groups[name] if m in leaves]
        if not members:
            raise ValueError(f"group {name!r} has no leaf in this area")
        return np.isin(leaf_map, members)
    if name not in leaves or name == "other":
        raise ValueError(f"unknown class {name!r}")
    return leaf_map == leaves.index(name)


def class_indices(names: list[str], leaves: list[str], groups: dict[str, list[str]]) -> list[int]:
    """Sorted leaf indices covered by class names (leaves or groups); unknown names fail. A name that is a
    leaf of this class list is that leaf, even if the configured groups also know it as a group."""
    out = set()
    for name in names:
        members = [name] if name in leaves else groups.get(name, [name])
        idx = [leaves.index(m) for m in members if m in leaves]
        if not idx or (name not in groups and name == "other"):
            raise ValueError(f"unknown class {name!r}")
        out.update(idx)
    return sorted(out)
