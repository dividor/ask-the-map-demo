"""Amendment A7: land maps that see each place's surroundings, and more training places.

- neighbourhood features: a pixel's AlphaEarth numbers plus the mean (and standard deviation) of the valid
  pixels in fixed square windows around it, the same for every class;
- a natural-proportion pixel sample drawn across many tiles without holding all their features in memory;
- a multinomial logistic regression on standardised features, used as argmax;
- a fixed rule for picking new training tiles (random points in the training states, and random solar farms).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LogisticRegression


def window_mean_std(ae: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each band over the valid pixels (no band missing) in the ``size`` x
    ``size`` window around every pixel, using only pixels inside the tile. ``ae`` is (bands, rows, cols);
    a window with no valid pixel gives NaN."""
    if size < 1 or size % 2 == 0:
        raise ValueError("window size must be a positive odd number")
    valid = np.isfinite(ae).all(axis=0)
    x = np.where(valid[None], ae, 0.0).astype(np.float64)
    count = uniform_filter(valid.astype(np.float64), size, mode="constant") * size * size
    s1 = uniform_filter(x, (1, size, size), mode="constant") * size * size
    s2 = uniform_filter(x * x, (1, size, size), mode="constant") * size * size
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / count
        var = np.maximum(s2 / count - mean * mean, 0.0)
    empty = count < 0.5
    mean[:, empty] = np.nan
    var[:, empty] = np.nan
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def context_features(ae: np.ndarray, mean_sizes: tuple[int, ...], std_size: int) -> np.ndarray:
    """(bands x (2 + len(mean_sizes)), rows, cols): the pixel's own numbers, the window means for each of
    ``mean_sizes`` and the window standard deviation for ``std_size``. Pixels with a missing band are NaN."""
    ae = ae.astype(np.float32)
    valid = np.isfinite(ae).all(axis=0)
    parts = [ae]
    for s in mean_sizes:
        parts.append(window_mean_std(ae, s)[0])
    parts.append(window_mean_std(ae, std_size)[1])
    out = np.concatenate(parts)
    out[:, ~valid] = np.nan
    return out


def natural_sample_tiles(
    names: list[str],
    load: Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    features: Callable[[np.ndarray], np.ndarray],
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` pixels drawn uniformly without replacement from the usable pixels of all tiles (classes keep
    their natural proportions). ``load(name)`` gives (embeddings, labels, keep); a pixel is usable where
    ``keep`` is true and every embedding band is finite. ``features(embeddings)`` gives (features, rows,
    cols); it is only computed once per tile, on the second pass."""
    counts = []
    for nm in names:
        ae, _, keep = load(nm)
        counts.append(int((keep & np.isfinite(ae).all(axis=0)).sum()))
    total = int(sum(counts))
    if n > total:
        raise ValueError(f"asked for {n} pixels but only {total} are usable")
    picks = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
    starts = np.cumsum([0, *counts])
    xs, ys = [], []
    for i, nm in enumerate(names):
        mine = picks[(picks >= starts[i]) & (picks < starts[i + 1])] - starts[i]
        if len(mine) == 0:
            continue
        ae, lab, keep = load(nm)
        ok = (keep & np.isfinite(ae).all(axis=0)).ravel()
        flat = np.flatnonzero(ok)[mine]
        f = features(ae)
        x = f.reshape(f.shape[0], -1)[:, flat].T
        if not np.isfinite(x).all():
            raise ValueError(f"tile {nm}: features are missing for usable pixels")
        xs.append(x.astype(np.float32))
        ys.append(lab.ravel()[flat])
    return np.concatenate(xs), np.concatenate(ys)


def fit_scaled_multinomial(X: np.ndarray, y: np.ndarray, n_classes: int, max_iter: int = 2000) -> dict:
    """A softmax logistic regression on features standardised by the sample's mean and standard deviation.
    Returns ``coef`` (classes, features), ``intercept``, ``mean`` and ``scale``. Every class must occur."""
    if n_classes < 3:
        raise ValueError("the multinomial model needs at least three classes")
    missing = sorted(set(range(n_classes)) - set(np.unique(y).tolist()))
    if missing:
        raise ValueError(f"classes {missing} have no pixels in the sample")
    mean, scale = X.mean(axis=0), X.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("a feature is constant in the sample")
    model = LogisticRegression(max_iter=max_iter).fit((X - mean) / scale, y)
    order = np.argsort(model.classes_)
    return {
        "coef": model.coef_[order].astype(np.float32),
        "intercept": model.intercept_[order].astype(np.float32),
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
    }


def scaled_onehot(model: dict, feats: np.ndarray) -> np.ndarray:
    """(classes, rows, cols) one-hot of the most likely class per pixel for (features, rows, cols); pixels
    with missing features get no class (all zeros)."""
    z = (feats - model["mean"][:, None, None]) / model["scale"][:, None, None]
    ok = np.isfinite(z).all(axis=0)
    lin = np.einsum("lf,frc->lrc", model["coef"], np.nan_to_num(z)) + model["intercept"][:, None, None]
    top = lin.argmax(axis=0)
    out = (np.arange(len(model["coef"]))[:, None, None] == top[None]).astype(np.float32)
    out[:, ~ok] = 0.0
    return out


def pick_new_tiles(
    states,
    state_codes: list[str],
    farms,
    n_random: int,
    n_farms: int,
    existing_lonlat: list[tuple[float, float]],
    min_km: float,
    seed: int,
) -> list[tuple[str, float, float]]:
    """New training tile centres (name, lon, lat): ``n_random`` uniformly random points inside the states with
    codes ``state_codes`` ("R1"...), then ``n_farms`` centred on randomly ordered farms (polygons) in those
    states ("F1"...). A candidate within ``min_km`` of an existing or already chosen centre is skipped.
    ``states`` and ``farms`` are GeoDataFrames (``STUSPS`` and ``p_state`` hold the state codes)."""
    from pyproj import Transformer

    eq = "EPSG:5070"
    to_ll = Transformer.from_crs(eq, 4326, always_xy=True)
    area = states[states["STUSPS"].isin(state_codes)].to_crs(eq)
    if area["STUSPS"].nunique() != len(set(state_codes)):
        raise ValueError("some state codes are not in the outlines")
    union = area.geometry.union_all()
    to_eq = Transformer.from_crs(4326, eq, always_xy=True)
    taken = [np.array(to_eq.transform(lon, lat)) for lon, lat in existing_lonlat]
    rng = np.random.default_rng(seed)

    def free(p: np.ndarray) -> bool:
        return all(np.hypot(*(p - q)) >= min_km * 1000 for q in taken)

    out: list[tuple[str, float, float]] = []
    xmin, ymin, xmax, ymax = union.bounds
    from shapely.geometry import Point

    tries = 0
    while len(out) < n_random:
        tries += 1
        if tries > 100_000:
            raise RuntimeError("could not place the random tiles")
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        if union.contains(Point(*p)) and free(p):
            taken.append(p)
            out.append((f"R{len(out) + 1}", *to_ll.transform(*p)))
    pool = farms[farms["p_state"].isin(state_codes)].to_crs(eq)
    for i in rng.permutation(len(pool)):
        if len(out) == n_random + n_farms:
            break
        c = pool.geometry.iloc[i].centroid
        p = np.array([c.x, c.y])
        if free(p):
            taken.append(p)
            out.append((f"F{len(out) - n_random + 1}", *to_ll.transform(*p)))
    if len(out) < n_random + n_farms:
        raise RuntimeError("not enough farms far enough apart")
    return out
