"""AETHER's recipe on US farm country, with NASA POWER and OpenStreetMap facts.

A location = one 30 m pixel: its AlphaEarth vector plus the POWER summary vector of its cell, projected by
AETHER's MLP into a frozen text space. Captions come from ``askmap.captions`` plus OpenStreetMap tags. With
binary, standardised facts and AETHER's sigma 0.1 their soft target equals "caption j is true for location
i" to within exp(-66) (see tests), so targets are built from ``askmap.captions.truth`` plus the OpenStreetMap
check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from askmap.aether import concept_index, soft_contrastive_from_targets
from askmap.captions import CLIMATE, NONE, Location, Vocab, make_caption
from askmap.space import Items, Specs, truth


class Projector(nn.Module):
    """AETHER's MLPProjector: ``layers`` linear layers with ReLU between, ``hidden`` wide."""

    def __init__(self, d_in: int, d_out: int, layers: int, hidden: int):
        super().__init__()
        mods, d = [], d_in
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def osm_caption(
    base: dict,
    osm_on: np.ndarray,
    words: list[list[str]],
    joins: list[str],
    ocfg: dict,
    rng: np.random.Generator,
) -> tuple[str, dict, int]:
    """Add an OpenStreetMap fact to a caption (or replace it by a tag-only caption).

    ``base`` is ``make_caption``'s output; ``osm_on`` lists the location's tag indices. Returns the text, the
    spec (land set to every class for a tag-only caption) and the tag index (-1 if none).
    """
    if len(osm_on) == 0 or rng.random() >= ocfg["osm_prob"]:
        return base["full"], base["spec"], -1
    k = int(rng.choice(osm_on))
    v = int(rng.integers(len(words[k])))
    if rng.random() < ocfg["osm_only_prob"]:
        spec = dict(base["spec"])
        spec.update(
            land=np.ones_like(spec["land"]),
            type=NONE,
            near=np.zeros_like(spec["near"]),
            d=0.0,
            neg=False,
            patch_kind=0,
            h=0.0,
            fact=-1,
        )
        return words[k][v], spec, k
    return f"{base['full']}, {joins[v % len(joins)]} {words[k][v]}", base["spec"], k


def captions_for(
    locs: list[Location],
    osm: np.ndarray,
    vocab: Vocab,
    words: list[list[str]],
    joins: list[str],
    ocfg: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """``per_location`` captions for every location: rows of {item, text, spec, osm}."""
    rows = []
    for j, loc in enumerate(locs):
        on = np.flatnonzero(osm[j])
        for _ in range(ocfg["per_location"]):
            text, spec, k = osm_caption(make_caption(loc, vocab, rng), on, words, joins, ocfg, rng)
            rows.append({"item": j, "text": text, "spec": spec, "osm": k})
    return rows


def climate_only_spec(n_leaves: int, n_near: int, fact: int) -> dict:
    """Spec of a caption that states only a climate fact (true for any land)."""
    return {
        "land": np.ones(n_leaves, bool),
        "type": CLIMATE,
        "near": np.zeros(n_near, bool),
        "d": 0.0,
        "neg": False,
        "patch_kind": 0,
        "h": 0.0,
        "fact": fact,
    }


def targets(
    s: Specs,
    cap_osm: torch.Tensor,
    items: Items,
    osm: torch.Tensor,
    item_idx: torch.Tensor,
    cell_facts: torch.Tensor,
) -> torch.Tensor:
    """(locations, captions) float: 1 where caption j is true for location i (its facts and its OSM tag)."""
    t = truth(s, items, item_idx, cell_facts)  # (captions, locations)
    has = cap_osm >= 0
    tag_ok = osm[item_idx][:, cap_osm.clamp(min=0)].T  # (captions, locations)
    t = t & (~has[:, None] | tag_ok)
    return t.T.float()


@dataclass
class Bank:
    """Training material on one device: per-location inputs and facts, and every caption."""

    x: torch.Tensor  # (N, d_in) standardised location vectors
    items: Items
    osm: torch.Tensor  # (N, tags) bool
    cell_facts: torch.Tensor  # (cells, facts) bool
    cap_item: torch.Tensor  # (M,) location of each caption
    cap_text: torch.Tensor  # (M, text dim) frozen text vectors
    specs: Specs
    cap_osm: torch.Tensor  # (M,) tag index or -1


def epoch_captions(cap_item: np.ndarray, items: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One caption per location in ``items``, chosen at random among its captions (as AETHER's per-batch
    template draw); locations without a caption are dropped."""
    order = rng.permutation(len(cap_item))
    first: dict[int, int] = {}
    for c in order:
        first.setdefault(int(cap_item[c]), int(c))
    chosen = np.array([first[i] for i in items if i in first], dtype=np.int64)
    return chosen[rng.permutation(len(chosen))]


def batch_loss(model: Projector, log_temp: torch.Tensor, b: Bank, caps: torch.Tensor) -> torch.Tensor:
    it = b.cap_item[caps]
    t = targets(b.specs.take(caps), b.cap_osm[caps], b.items, b.osm, it, b.cell_facts)
    return soft_contrastive_from_targets(model(b.x[it]), b.cap_text[caps], t, log_temp)


def project(model: Projector, x: np.ndarray, device, batch: int = 65536) -> np.ndarray:
    """Unit-length projected location vectors."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            z = model(torch.as_tensor(x[s : s + batch], dtype=torch.float32, device=device))
            out.append(F.normalize(z, dim=-1).cpu().numpy())
    return np.concatenate(out)


def concept_indices(sim: np.ndarray, truth_cn: np.ndarray, min_pos: int, max_share: float) -> np.ndarray:
    """AETHER index per concept (rows of ``sim`` and ``truth_cn``); NaN if too few or too many are true."""
    out = np.full(len(sim), np.nan)
    n = truth_cn.shape[1]
    for c in range(len(sim)):
        k = int(truth_cn[c].sum())
        if k < min_pos or k > max_share * n:
            continue
        out[c] = concept_index(truth_cn[c].astype(float), sim[c], 0.5, True, k / n * 100)
    return out


def group_mean(index: np.ndarray, groups: np.ndarray) -> float:
    """Mean over groups of the mean index within each group (NaN concepts skipped; empty groups dropped)."""
    means = [
        np.nanmean(index[groups == g]) for g in np.unique(groups) if np.isfinite(index[groups == g]).any()
    ]
    if not means:
        raise ValueError("no concept could be scored")
    return float(np.mean(means))


def train(
    b: Bank, train_idx: np.ndarray, val_idx: np.ndarray, val_eval, cfg: dict, device, log=print
) -> tuple[Projector, torch.Tensor, list[dict]]:
    """AETHER training: Adam, plateau schedule on the validation loss, early stopping on the validation index
    (``val_eval(model) -> float``); returns the best model, its log temperature and the history."""
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = Projector(b.x.shape[1], b.cap_text.shape[1], cfg["layers"], cfg["hidden"]).to(device)
    log_temp = nn.Parameter(torch.log(torch.tensor(cfg["temperature"], device=device)))
    opt = torch.optim.Adam([*model.parameters(), log_temp], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg["plateau_factor"], patience=cfg["plateau_patience"]
    )
    cap_item = b.cap_item.cpu().numpy()
    best, best_state, wait, hist = -np.inf, None, 0, []
    bs = cfg["batch"]
    for ep in range(cfg["max_epochs"]):
        model.train()
        caps = epoch_captions(cap_item, train_idx, rng)
        tot, steps = 0.0, 0
        for s in range(0, len(caps) - bs + 1, bs):
            loss = batch_loss(model, log_temp, b, torch.as_tensor(caps[s : s + bs], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, steps = tot + loss.item(), steps + 1
        model.eval()
        vcaps = epoch_captions(cap_item, val_idx, np.random.default_rng(cfg["seed"]))
        if len(vcaps) < 2:
            raise ValueError("validation needs at least 2 locations with captions")
        with torch.no_grad():
            vl = [
                batch_loss(model, log_temp, b, torch.as_tensor(vcaps[s : s + bs], device=device)).item()
                for s in range(0, len(vcaps), bs)
            ]
        val_loss = float(np.mean(vl))
        sched.step(val_loss)
        score = val_eval(model)
        hist.append(
            {
                "epoch": ep,
                "train_loss": tot / steps,
                "val_loss": val_loss,
                "val_index": score,
                "temp": log_temp.exp().item(),
                "lr": opt.param_groups[0]["lr"],
            }
        )
        log(f"epoch {ep}: train {tot / steps:.3f} val {val_loss:.3f} index {score:.3f}")
        if score > best + 1e-3:
            best, wait = score, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_state = (state, log_temp.detach().clone())
        else:
            wait += 1
            if wait >= cfg["early_stop_patience"]:
                break
    if best_state is None:
        raise RuntimeError("training produced no scored epoch")
    model.load_state_dict(best_state[0])
    return model, best_state[1], hist


@dataclass
class TileScores:
    """One concept's scores split by tile: all scores sorted, and the true pixels' scores sorted."""

    all_sorted: list[np.ndarray]
    pos_sorted: list[np.ndarray]

    @classmethod
    def of(cls, score: np.ndarray, truth: np.ndarray, tile_id: np.ndarray, n_tiles: int) -> TileScores:
        return cls(
            [np.sort(score[tile_id == t]) for t in range(n_tiles)],
            [np.sort(score[(tile_id == t) & truth]) for t in range(n_tiles)],
        )


def _count_at_least(sorted_arrays: list[np.ndarray], mult: np.ndarray, v: float) -> int:
    return int(
        sum(
            m * (len(a) - np.searchsorted(a, v, side="left"))
            for a, m in zip(sorted_arrays, mult, strict=True)
            if m
        )
    )


def multiset_index(ts: TileScores, mult: np.ndarray, min_pos: int, max_share: float) -> float:
    """AETHER's index (k = true pixels) on tiles drawn ``mult`` times each; NaN if the concept is skipped.

    The top-k threshold is found by bisection on the score value, so no pixel arrays are rebuilt per draw.
    Pixels tied at the threshold all count (true hits are capped at k).
    """
    n = int(sum(m * len(a) for a, m in zip(ts.all_sorted, mult, strict=True)))
    k = int(sum(m * len(a) for a, m in zip(ts.pos_sorted, mult, strict=True)))
    if k < min_pos or k > max_share * n:
        return np.nan
    used = [a for a, m in zip(ts.all_sorted, mult, strict=True) if m and len(a)]
    lo, hi = min(a[0] for a in used), max(a[-1] for a in used)
    # largest v with count(score >= v) >= k
    for _ in range(60):
        mid = (lo + hi) / 2
        if _count_at_least(ts.all_sorted, mult, mid) >= k:
            lo = mid
        else:
            hi = mid
    hits = _count_at_least(ts.pos_sorted, mult, lo)
    base = k / n
    return float((min(hits, k) / k - base) / (1 - base))


def clip_features(texts: list[str], tokenizer, model, device, batch: int = 256) -> np.ndarray:
    """CLIP text features as AETHER's ClipTextEncoder makes them: padding, truncation at 77 tokens, the
    model's ``get_text_features``; not normalised."""
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            tok = tokenizer(
                texts[s : s + batch], padding=True, truncation=True, max_length=77, return_tensors="pt"
            )
            f = model.get_text_features(**{k: v.to(device) for k, v in tok.items()})
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


# Hugging Face model commits the results were made with; a model not listed here cannot be loaded
PINNED_REVISIONS = {
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
}


class TextEncoder:
    """A frozen text encoder loaded once; ``kind`` is "clip" (Hugging Face CLIP, AETHER's) or
    "sentence_transformer". The model is loaded at its pinned commit (``PINNED_REVISIONS``)."""

    def __init__(self, name: str, kind: str, device, cache_dir: str | None):
        self.kind, self.device = kind, device
        if name not in PINNED_REVISIONS:
            raise ValueError(f"no pinned commit for text encoder {name!r}; add it to PINNED_REVISIONS")
        rev = PINNED_REVISIONS[name]
        if kind == "clip":
            from transformers import CLIPModel, CLIPTokenizer

            self.tok = CLIPTokenizer.from_pretrained(name, cache_dir=cache_dir, revision=rev)
            self.model = CLIPModel.from_pretrained(name, cache_dir=cache_dir, revision=rev).to(device)
        elif kind == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(name, device=str(device), cache_folder=cache_dir, revision=rev)
        else:
            raise ValueError(f"unknown text encoder kind {kind!r}")

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self.kind == "clip":
            return clip_features(texts, self.tok, self.model, self.device)
        return self.model.encode(
            texts, batch_size=128, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)


def text_vectors(texts: list[str], name: str, kind: str, device, cache_dir: str | None) -> np.ndarray:
    """Frozen text vectors from a freshly loaded :class:`TextEncoder`."""
    return TextEncoder(name, kind, device, cache_dir)(texts)


def _members(name: str, leaves: list[str], groups: dict[str, list[str]]) -> list[int]:
    idx = [leaves.index(m) for m in groups.get(name, [name]) if m in leaves]
    if not idx:
        raise ValueError(f"unknown class {name!r}")
    return idx


def climate_ok_tile(plan: dict, tile: str, ctx) -> np.ndarray:
    """(rows, cols) bool: pixels of ``tile`` whose POWER cell meets every climate condition of the plan."""
    iy, ix = ctx.pixel_cell[tile]
    ok = np.ones(iy.shape, dtype=bool)
    for c in plan["climate"]:
        ok &= ctx.climate(c)[iy, ix]
    return ok


def plan_scores_tile(plan: dict, probs: np.ndarray, tile: str, ctx) -> np.ndarray:
    """Score of a plan on one tile from per-class probability maps ``probs`` (leaves, rows, cols).

    The highest probability among the plan's land classes, times 1 where every distance, patch and climate
    condition holds (class maps for the conditions are the probabilities thresholded at 0.5).
    """
    from askmap.plan import min_patch, within

    leaves, groups = ctx.leaves, ctx.groups
    land_idx = sorted({i for n in plan["land"]["classes"] for i in _members(n, leaves, groups)})
    land_p = probs[land_idx].max(axis=0)
    ok = np.ones(land_p.shape, dtype=bool)
    if plan["land"]["min_patch_ha"] is not None:
        ok &= min_patch(land_p > 0.5, plan["land"]["min_patch_ha"] * 10_000 / ctx.res_m**2)
    for n in plan["near"]:
        idx = sorted({i for c in n["classes"] for i in _members(c, leaves, groups)})
        near = within(probs[idx].max(axis=0) > 0.5, n["distance_m"], ctx.res_m)
        ok &= ~near if n["negate"] else near
    ok &= climate_ok_tile(plan, tile, ctx)
    return land_p * ok


def similarities(
    model: Projector,
    ae: np.ndarray,
    cell: np.ndarray,
    summary: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    text: np.ndarray,
    device,
    chunk: int = 100_000,
) -> np.ndarray:
    """(texts, pixels) cosine similarity of unit ``text`` vectors and the projected pixels, in chunks, so
    neither the standardised inputs nor the projected vectors of all pixels are held at once."""
    out = np.empty((len(text), len(ae)), dtype=np.float32)
    t = torch.as_tensor(text, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(ae), chunk):
            x = ((np.c_[ae[s : s + chunk], summary[cell[s : s + chunk]]] - mu) / sd).astype(np.float32)
            z = F.normalize(model(torch.as_tensor(x, device=device)), dim=-1)
            out[:, s : s + chunk] = (t @ z.T).cpu().numpy()
    return out
