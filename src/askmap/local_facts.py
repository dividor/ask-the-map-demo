"""Per-pixel land facts on an official land map: distance to each class and the size of the pixel's patch.

Distances use the same rule as the answer key (``askmap.geo.within``: Euclidean distance between pixel
centres, so "within d" means distance <= d). Patch sizes use 8-connected patches, as ``plan.min_patch``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from askmap.layers import class_mask
from askmap.plan import EIGHT

FAR = 1e9  # distance when a class is absent from the tile


def distances(
    leaf_map: np.ndarray, classes: list[str], leaves: list[str], groups: dict, res_m: float
) -> np.ndarray:
    """(rows, cols, classes) distance in metres to the nearest pixel of each class (0 on the class itself)."""
    out = np.full((*leaf_map.shape, len(classes)), FAR, dtype=np.float32)
    for k, c in enumerate(classes):
        m = class_mask(leaf_map, c, leaves, groups)
        if m.any():
            out[..., k] = distance_transform_edt(~m) * res_m
    return out


def patch_area_ha(mask: np.ndarray, res_m: float) -> np.ndarray:
    """Area (ha) of the 8-connected patch each True pixel belongs to; 0 elsewhere."""
    lab, n = label(mask, structure=EIGHT)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    sizes = np.bincount(lab.ravel()).astype(np.float32) * res_m**2 / 10_000
    sizes[0] = 0
    return sizes[lab]


def leaf_patch_ha(leaf_map: np.ndarray, n_leaves: int, res_m: float) -> np.ndarray:
    """Patch area (ha) of each pixel's own leaf class."""
    out = np.zeros(leaf_map.shape, dtype=np.float32)
    for c in range(n_leaves):
        m = leaf_map == c
        if m.any():
            out[m] = patch_area_ha(m, res_m)[m]
    return out


def sample_pixels(
    leaf_map: np.ndarray, valid: np.ndarray, n: int, n_leaves: int, seed: int, balanced_share: float = 0.5
) -> np.ndarray:
    """Flat indices of ``n`` distinct valid pixels: a uniform share plus a share spread evenly over the leaf
    classes present (all of a class if it has fewer)."""
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(valid.ravel())
    if len(ok) < n:
        raise ValueError(f"only {len(ok)} valid pixels")
    flat = leaf_map.ravel()[ok]
    n_bal = int(n * balanced_share)
    present = [c for c in range(n_leaves) if (flat == c).any()]
    per = max(1, n_bal // max(1, len(present)))
    chosen = []
    for c in present:
        idx = ok[flat == c]
        chosen.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
    picked = np.unique(np.concatenate(chosen)) if chosen else np.array([], dtype=np.int64)
    rest = np.setdiff1d(ok, picked)
    extra = rng.choice(rest, size=n - len(picked), replace=False)
    return np.sort(np.concatenate([picked, extra]))
