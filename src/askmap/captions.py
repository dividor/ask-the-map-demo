"""Training captions generated from a location's known facts, and questions rendered in caption wording.

A caption = optional prefix + land phrase + at most one extra fact (distance, patch size, or a climate fact)
+ optional area phrase. Each caption also carries a machine-readable spec, so that the training loss can tell
which other locations the same caption is true for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NONE, NEAR, PATCH, CLIMATE = 0, 1, 2, 3
FAMILIES = ("hist", "heat", "rain", "gdd")


def _dist_text(d: float) -> str:
    return f"{d / 1000:g} km" if d >= 1000 else f"{d:g} m"


@dataclass
class Vocab:
    """Everything the generator needs: caption wording, class lists and the climate fact ids. ``wordings`` is
    how many parallel wordings every phrase list holds (3 in ``configs/captions.yaml``; more after A8's
    paraphrases); each caption draws one uniformly."""

    cap: dict
    leaves: list[str]
    groups: dict[str, list[str]]
    fact_ids: list[str]
    wordings: int = 3
    near_classes: list[str] = field(init=False)
    land_names: list[str] = field(init=False)
    fact_index: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.near_classes = list(self.cap["near"]["classes"])
        self.land_names = [n for n in self.leaves if n != "other"] + list(self.groups)
        self.fact_index = {f: k for k, f in enumerate(self.fact_ids)}

    def leaf_set(self, name: str) -> list[int]:
        members = self.groups.get(name, [name])
        return [self.leaves.index(m) for m in members if m in self.leaves]


def fact_text(fid: str, cap: dict, v: int) -> str:
    """Wording ``v`` (0, 1, 2) of a climate fact."""
    p = fid.split("|")
    kind = p[0]
    cy = cap["climate_year"]
    if kind == "hist":
        _, var, sk, year, side, pct = p
        vc = cy["variables"][var]
        return cy["templates"][v].format(
            season=cy["seasons"][sk]["words"][v], year=year, var=vc["words"][v], adj=vc[side], p=pct
        )
    if kind == "heat":
        _, sk, t, year, n = p
        return cap["heat_days"]["templates"][v].format(
            season=cy["seasons"][sk]["words"][v], year=year, n=n, t=t
        )
    if kind == "rain":
        _, sk, year, op, mm = p
        key = "templates_at_least" if op == "ge" else "templates_at_most"
        return cap["rain_total"][key][v].format(season=cy["seasons"][sk]["words"][v], year=year, mm=mm)
    if kind == "gdd":
        _, year, op, g = p
        key = "templates_at_least" if op == "ge" else "templates_at_most"
        return cap["gdd"][key][v].format(year=year, g=g)
    cl = cap["climate_long"]
    if kind == "long":
        _, var, side, tier = p
        vc = cl["variables"][var]
        return cl["templates"][v].format(adj=vc[side], tier=tier, var=vc["words"][v])
    if kind == "sun":
        return cl["sunlight_templates"][v].format(x=p[2])
    raise ValueError(f"unknown fact {fid}")


def land_text(names: list[str], cap: dict, v: int, join: int = 0) -> str:
    return cap["union_join"][join].join(cap["land"][n][v] for n in names)


def near_text(classes: list[str], d: float, negate: bool, cap: dict, v: int) -> str:
    nc = cap["near"]
    c = " or ".join(nc["classes"][k][v] for k in classes)
    tpl = nc["far_templates" if negate else "near_templates"][v]
    return tpl.format(d=_dist_text(d), c=c)


def patch_text(h: float, cap: dict, v: int) -> str:
    return cap["patch"]["templates"][v].format(h=f"{h:g}")


def area_text(area: str, cap: dict, v: int) -> str:
    return cap["area_phrase"][v].format(area=cap["area_names"][area])


@dataclass
class Location:
    leaf: int
    dist: np.ndarray  # (near classes,) metres
    patch_leaf: float
    patch_crop: float
    true_facts: dict  # family -> array of climate fact indices true for the location's cell
    area: str


def _spec(vocab: Vocab) -> dict:
    return {
        "land": np.zeros(len(vocab.leaves), bool),
        "type": NONE,
        "near": np.zeros(len(vocab.near_classes), bool),
        "d": 0.0,
        "neg": False,
        "patch_kind": 0,
        "h": 0.0,
        "fact": -1,
    }


def make_caption(loc: Location, vocab: Vocab, rng: np.random.Generator) -> dict:
    """One random caption that is true for ``loc``: texts (full, land part, climate part) and its spec.

    The random draws happen in a fixed order (land names, wordings, then the extra clause, then the prefix and
    area phrase), so the same seed gives the same captions."""
    cap = vocab.cap
    comp = cap["composition"]
    if vocab.leaves[loc.leaf] == "other":
        raise ValueError("no captions for 'other' pixels")
    names = _land_names(loc, vocab, rng)
    spec = _spec(vocab)
    for n in names:
        spec["land"][vocab.leaf_set(n)] = True
    v = int(rng.integers(vocab.wordings))
    land = land_text(names, cap, v, int(rng.integers(2)))
    extra, clim = "", ""
    kinds = ["none", "near", "patch", "climate_year", "climate_long"]
    probs = np.array([comp[k] for k in kinds], dtype=float)
    for kind in rng.choice(kinds, size=len(kinds), replace=False, p=probs / probs.sum()):
        if kind == "none":
            break
        if kind == "near":
            extra = _near_clause(loc, vocab, rng, spec)
            break
        if kind == "patch":
            extra = _patch_clause(loc, vocab, rng, spec, names)
        else:
            extra = clim = _climate_clause(loc, vocab, rng, spec, kind)
        if extra:
            break
    prefix = str(rng.choice(cap["prefixes"]))
    full = prefix + land + (" " + extra if extra else "")
    if rng.random() < comp["area_phrase_prob"]:
        full += " " + area_text(loc.area, cap, int(rng.integers(2)))
    return {"full": full, "land": prefix + land, "clim": (prefix + clim) if clim else "", "spec": spec}


def _land_names(loc: Location, vocab: Vocab, rng: np.random.Generator) -> list[str]:
    """The land names a caption describes: the leaf (or sometimes its group), sometimes joined by another."""
    leaf_name = vocab.leaves[loc.leaf]
    names = [leaf_name]
    for g, members in vocab.groups.items():
        if leaf_name in members and rng.random() < 0.3:
            names = [g]
    if rng.random() < vocab.cap["composition"]["land_union_prob"]:
        others = [
            n
            for n in vocab.land_names
            if n not in names and vocab.leaves[loc.leaf] not in [vocab.leaves[i] for i in vocab.leaf_set(n)]
        ]
        names = [*names, str(rng.choice(others))]
        rng.shuffle(names)
    return names


def _near_clause(loc: Location, vocab: Vocab, rng: np.random.Generator, spec: dict) -> str:
    """A distance clause ("within 300 m of a wetland") that is true for the location."""
    nc = vocab.cap["near"]
    k = [int(rng.integers(len(vocab.near_classes)))]
    if rng.random() < nc["union_prob"]:
        k.append(int(rng.choice([i for i in range(len(vocab.near_classes)) if i != k[0]])))
    d = float(rng.choice(nc["distances_m"]))
    neg = bool(loc.dist[k].min() > d)
    spec.update(type=NEAR, d=d, neg=neg)
    spec["near"][k] = True
    v_near = int(rng.integers(vocab.wordings))
    return near_text([vocab.near_classes[i] for i in k], d, neg, vocab.cap, v_near)


def _patch_clause(loc: Location, vocab: Vocab, rng: np.random.Generator, spec: dict, names: list[str]) -> str:
    """A patch-size clause, or "" when the location's patch is too small or the land is not a single class."""
    cap = vocab.cap
    if len(names) != 1 or names[0] not in cap["patch"]["classes"]:
        return ""
    crop = names[0] in vocab.groups
    area = loc.patch_crop if crop else loc.patch_leaf
    ok = [h for h in cap["patch"]["sizes_ha"] if area >= h]
    if not ok:
        return ""
    h = float(rng.choice(ok))
    spec.update(type=PATCH, patch_kind=int(crop), h=h)
    return patch_text(h, cap, int(rng.integers(vocab.wordings)))


def _climate_clause(loc: Location, vocab: Vocab, rng: np.random.Generator, spec: dict, kind: str) -> str:
    """A climate clause from one of the facts true for the location's cell, or "" when it has none."""
    fams = list(FAMILIES) if kind == "climate_year" else ["long"]
    fam = str(rng.choice(fams))
    pool = loc.true_facts.get(fam, np.array([], dtype=int))
    if len(pool) == 0:
        return ""
    f = int(rng.choice(pool))
    spec.update(type=CLIMATE, fact=f)
    return fact_text(vocab.fact_ids[f], vocab.cap, int(rng.integers(vocab.wordings)))


def render_plan(plan: dict, cap: dict, fact_of_condition) -> str:
    """A reference plan in caption wording (first wording of every phrase, no area name)."""
    land = land_text(plan["land"]["classes"], cap, 0)
    extras = []
    if plan["land"]["min_patch_ha"] is not None:
        extras.append(patch_text(plan["land"]["min_patch_ha"], cap, 0))
    for n in plan["near"]:
        extras.append(near_text(n["classes"], n["distance_m"], n["negate"], cap, 0))
    for c in plan["climate"]:
        extras.append(fact_text(fact_of_condition(c), cap, 0))
    return cap["prefixes"][0] + land + ("" if not extras else " " + " and ".join(extras))


def n_extra_conditions(plan: dict) -> int:
    return len(plan["near"]) + len(plan["climate"]) + int(plan["land"]["min_patch_ha"] is not None)
