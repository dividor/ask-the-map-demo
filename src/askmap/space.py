"""The shared text / land / climate space: projection heads, the multi-positive contrastive loss, training and
search.

Positives: a caption's positives are all items in the batch for which the caption is true (its land classes
and its one extra fact), not only the item it was written for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from askmap.captions import CLIMATE, NEAR, PATCH


def mlp(d_in: int, hidden: int, d_out: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))


class Space(nn.Module):
    """Text projection, land head and (optional) climate head into one ``dim``-sized space."""

    def __init__(
        self, text_dim: int, land_dim: int, clim_dim: int | None, dim: int, hidden: int, dropout: float
    ):
        super().__init__()
        self.text = nn.Linear(text_dim, dim)
        self.land = mlp(land_dim, hidden, dim, dropout)
        self.clim = mlp(clim_dim, hidden, dim, dropout) if clim_dim else None

    def t(self, e: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(e), dim=-1)

    def zl(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.land(x), dim=-1)

    def zc(self, c: torch.Tensor) -> torch.Tensor:
        if self.clim is None:
            raise ValueError("this space has no climate head")
        return F.normalize(self.clim(c), dim=-1)

    def joint(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        if self.clim is None:
            return self.zl(x)
        if c is None:
            raise ValueError("climate features are needed for a space with a climate head")
        return F.normalize(self.zl(x) + self.zc(c), dim=-1)


def multi_positive_loss(sim: torch.Tensor, pos: torch.Tensor, tau: float) -> torch.Tensor:
    """InfoNCE with several positives, both directions; rows/columns without a positive are skipped."""
    logits = sim / tau
    neg_inf = torch.finfo(logits.dtype).min
    losses = []
    for lg, ps in ((logits, pos), (logits.T, pos.T)):
        has = ps.any(dim=1)
        if not has.any():
            continue
        num = torch.logsumexp(lg.masked_fill(~ps, neg_inf), dim=1)
        den = torch.logsumexp(lg, dim=1)
        losses.append((den - num)[has].mean())
    return torch.stack(losses).mean()


@dataclass
class Items:
    """Location attributes needed to decide which captions are true for them (tensors on one device)."""

    leaf: torch.Tensor  # (N,) long
    dist: torch.Tensor  # (N, near classes) float
    patch_leaf: torch.Tensor  # (N,) float
    patch_crop: torch.Tensor  # (N,) float
    cell: torch.Tensor  # (N,) long, row in the climate fact table


@dataclass
class Specs:
    """Caption specs as tensors (see captions.make_caption)."""

    land: torch.Tensor  # (M, leaves) bool
    type: torch.Tensor  # (M,) long
    near: torch.Tensor  # (M, near classes) bool
    d: torch.Tensor  # (M,) float
    neg: torch.Tensor  # (M,) bool
    patch_kind: torch.Tensor  # (M,) long
    h: torch.Tensor  # (M,) float
    fact: torch.Tensor  # (M,) long (-1 = none)

    def take(self, idx: torch.Tensor) -> Specs:
        return Specs(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))


def specs_from(spec_list: list[dict], device) -> Specs:
    def arr(key, dtype):
        return torch.as_tensor(np.array([s[key] for s in spec_list]), dtype=dtype, device=device)

    return Specs(
        arr("land", torch.bool),
        arr("type", torch.long),
        arr("near", torch.bool),
        arr("d", torch.float32),
        arr("neg", torch.bool),
        arr("patch_kind", torch.long),
        arr("h", torch.float32),
        arr("fact", torch.long),
    )


def truth(
    s: Specs,
    items: Items,
    item_idx: torch.Tensor,
    cell_facts: torch.Tensor,
    land_only: bool = False,
    climate_only: bool = False,
) -> torch.Tensor:
    """(captions, items) bool: is each caption true for each item. ``land_only`` checks just the land classes;
    ``climate_only`` just the climate fact."""
    n = len(item_idx)
    if climate_only:
        cells = items.cell[item_idx]
        return cell_facts[cells][:, s.fact.clamp(min=0)].T
    ok = s.land[:, items.leaf[item_idx]]
    if land_only:
        return ok
    extra = torch.ones_like(ok)
    dist = items.dist[item_idx]  # (n, C)
    big = torch.finfo(dist.dtype).max
    dmin = torch.where(s.near[:, None, :], dist[None, :, :], big).amin(dim=2)  # (m, n)
    near_ok = torch.where(s.neg[:, None], dmin > s.d[:, None], dmin <= s.d[:, None])
    area = torch.where(
        s.patch_kind[:, None] == 1, items.patch_crop[item_idx][None, :], items.patch_leaf[item_idx][None, :]
    )
    patch_ok = area >= s.h[:, None]
    cells = items.cell[item_idx]
    clim_ok = cell_facts[cells][:, s.fact.clamp(min=0)].T
    t = s.type[:, None].expand(-1, n)
    extra = torch.where(t == NEAR, near_ok, extra)
    extra = torch.where(t == PATCH, patch_ok, extra)
    extra = torch.where(t == CLIMATE, clim_ok, extra)
    return ok & extra


def search(
    model: Space, text_emb: np.ndarray, land: np.ndarray, clim: np.ndarray | None, device, batch: int = 65536
) -> np.ndarray:
    """(questions, items) cosine similarity between question texts and item embeddings."""
    model.eval()
    with torch.no_grad():
        t = model.t(torch.as_tensor(text_emb, device=device))
        out = []
        for s in range(0, len(land), batch):
            x = torch.as_tensor(land[s : s + batch], device=device)
            c = (
                None
                if clim is None or model.clim is None
                else torch.as_tensor(clim[s : s + batch], device=device)
            )
            out.append((t @ model.joint(x, c).T).cpu().numpy())
    return np.concatenate(out, axis=1)


@dataclass
class TrainData:
    """Everything training needs, on one device."""

    items: Items
    land: torch.Tensor  # (N, land features)
    clim: torch.Tensor | None  # (cells, climate features)
    cap_item: torch.Tensor  # (M,) item of each caption
    t_full: torch.Tensor  # (M, text dim)
    t_land: torch.Tensor
    t_clim: torch.Tensor
    has_clim: torch.Tensor  # (M,) bool
    specs: Specs
    co_cell: torch.Tensor  # (K,) climate-only captions: cell
    co_text: torch.Tensor  # (K, text dim)
    co_fact: torch.Tensor  # (K,) fact
    cell_facts: torch.Tensor  # (cells, facts) bool


def train_space(d: TrainData, cfg: dict, land_only: bool, device, log=print) -> tuple[Space, list[float]]:
    """Train a shared space; ``land_only`` drops the climate head and the climate terms."""
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    if land_only:
        clim: torch.Tensor | None = None
    elif d.clim is None:
        raise ValueError("climate features are needed unless land_only")
    else:
        clim = d.clim
    clim_dim = None if clim is None else clim.shape[1]
    model = Space(d.t_full.shape[1], d.land.shape[1], clim_dim, cfg["dim"], cfg["hidden"], cfg["dropout"]).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    tau, bs = cfg["temperature"], cfg["batch"]
    m, k = len(d.cap_item), len(d.co_cell)
    history = []
    for ep in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(m, generator=gen).to(device)
        total, steps = 0.0, 0
        for s in range(0, m - bs + 1, bs):
            idx = order[s : s + bs]
            it = d.cap_item[idx]
            sp = d.specs.take(idx)
            x = d.land[it]
            c = None if clim is None else clim[d.items.cell[it]]
            terms = []
            z = model.joint(x, c)
            terms.append(
                multi_positive_loss(model.t(d.t_full[idx]) @ z.T, truth(sp, d.items, it, d.cell_facts), tau)
            )
            terms.append(
                multi_positive_loss(
                    model.t(d.t_land[idx]) @ model.zl(x).T,
                    truth(sp, d.items, it, d.cell_facts, land_only=True),
                    tau,
                )
            )
            if clim is not None and c is not None:
                hc = d.has_clim[idx]
                if hc.sum() > 1:
                    zc = model.zc(c)
                    pos = truth(
                        sp.take(hc.nonzero().squeeze(1)), d.items, it, d.cell_facts, climate_only=True
                    )
                    terms.append(multi_positive_loss(model.t(d.t_clim[idx][hc]) @ zc.T, pos, tau))
                j = torch.randint(0, k, (bs,), generator=gen).to(device)
                cells = d.co_cell[j]
                pos = d.cell_facts[cells][:, d.co_fact[j]].T
                terms.append(multi_positive_loss(model.t(d.co_text[j]) @ model.zc(clim[cells]).T, pos, tau))
            loss = torch.stack(terms).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        history.append(total / max(steps, 1))
        log(f"epoch {ep + 1}: loss {history[-1]:.4f}")
    model.eval()
    return model, history
