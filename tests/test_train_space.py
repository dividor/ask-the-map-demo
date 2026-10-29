import numpy as np
import torch

from askmap.captions import CLIMATE, NONE
from askmap.space import Items, Specs, TrainData, search, train_space

CFG = {
    "dim": 16,
    "hidden": 32,
    "dropout": 0.0,
    "temperature": 0.1,
    "lr": 0.01,
    "weight_decay": 0.0,
    "batch": 64,
    "epochs": 30,
    "seed": 0,
}


def toy(n_items=200, n_leaves=3, n_cells=4, n_facts=2):
    """Land class readable from the land vector, climate fact from the cell vector; text = class/fact id."""
    rng = np.random.default_rng(0)
    leaf = rng.integers(0, n_leaves, n_items)
    cell = rng.integers(0, n_cells, n_items)
    land = np.eye(n_leaves)[leaf] + 0.05 * rng.normal(size=(n_items, n_leaves))
    cell_facts = np.array([[c % 2 == 0, c % 2 == 1] for c in range(n_cells)])
    clim = cell_facts.astype(float) + 0.05 * rng.normal(size=(n_cells, n_facts))
    # text vectors: one-hot of (land class, fact or none)
    caps_item = np.repeat(np.arange(n_items), 2)
    kind = np.tile([NONE, CLIMATE], n_items)
    fact = np.where(
        kind == CLIMATE, np.array([np.flatnonzero(cell_facts[cell[i]])[0] for i in caps_item]), -1
    )
    text = np.zeros((len(caps_item), n_leaves + n_facts))
    text[np.arange(len(caps_item)), leaf[caps_item]] = 1
    has = fact >= 0
    text[np.flatnonzero(has), n_leaves + fact[has]] = 1
    t = torch.as_tensor
    land_mask = np.eye(n_leaves, dtype=bool)[leaf[caps_item]]
    specs = Specs(
        t(land_mask),
        t(kind),
        t(np.zeros((len(caps_item), 1), bool)),
        t(np.zeros(len(caps_item))).float(),
        t(np.zeros(len(caps_item), bool)),
        t(np.zeros(len(caps_item), dtype=np.int64)),
        t(np.zeros(len(caps_item))).float(),
        t(fact),
    )
    items = Items(
        t(leaf),
        t(np.zeros((n_items, 1))).float(),
        t(np.zeros(n_items)).float(),
        t(np.zeros(n_items)).float(),
        t(cell),
    )
    clim_text = np.zeros_like(text)
    clim_text[np.flatnonzero(has), n_leaves + fact[has]] = 1
    co_cell = np.repeat(np.arange(n_cells), 5)
    co_fact = np.array([np.flatnonzero(cell_facts[c])[0] for c in co_cell])
    co_text = np.zeros((len(co_cell), n_leaves + n_facts))
    co_text[np.arange(len(co_cell)), n_leaves + co_fact] = 1
    d = TrainData(
        items=items,
        land=t(land).float(),
        clim=t(clim).float(),
        cap_item=t(caps_item),
        t_full=t(text).float(),
        t_land=t(np.eye(n_leaves + n_facts)[leaf[caps_item]]).float(),
        t_clim=t(clim_text).float(),
        has_clim=t(has),
        specs=specs,
        co_cell=t(co_cell),
        co_text=t(co_text).float(),
        co_fact=t(co_fact),
        cell_facts=t(cell_facts),
    )
    return d, leaf, cell, land, clim, cell_facts


def test_train_space_learns_land_and_climate():
    d, leaf, cell, land, clim, cell_facts = toy()
    model, hist = train_space(d, CFG, land_only=False, device="cpu", log=lambda *_: None)
    assert hist[-1] < hist[0]
    q = np.zeros((1, 5), np.float32)
    q[0, 1] = 1  # class 1 ...
    q[0, 3 + 1] = 1  # ... in cells with fact 1
    s = search(model, q, land.astype(np.float32), clim[cell].astype(np.float32), "cpu")[0]
    target = (leaf == 1) & cell_facts[cell, 1]
    top = np.argsort(-s)[: target.sum()]
    assert target[top].mean() > 0.8
    lo, _ = train_space(d, CFG, land_only=True, device="cpu", log=lambda *_: None)
    assert lo.clim is None
