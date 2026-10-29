import numpy as np
import pytest
import torch

from askmap.aether import soft_target_matrix
from askmap.aether_us import (
    Bank,
    Projector,
    TileScores,
    climate_ok_tile,
    climate_only_spec,
    clip_features,
    concept_indices,
    epoch_captions,
    group_mean,
    multiset_index,
    osm_caption,
    plan_scores_tile,
    project,
    similarities,
    targets,
    text_vectors,
    train,
)
from askmap.captions import CLIMATE, NONE
from askmap.space import Items, specs_from

CFG = {
    "layers": 3,
    "hidden": 32,
    "temperature": 0.07,
    "lr": 0.01,
    "weight_decay": 0.0,
    "batch": 64,
    "max_epochs": 25,
    "plateau_patience": 10,
    "plateau_factor": 0.1,
    "early_stop_patience": 15,
    "seed": 0,
}


def toy_bank(n=240, n_leaves=3, n_cells=4, n_tags=2):
    """Land class readable from the vector, a climate fact from the cell, an OSM tag on some locations."""
    rng = np.random.default_rng(0)
    leaf = rng.integers(0, n_leaves, n)
    cell = rng.integers(0, n_cells, n)
    osm = np.zeros((n, n_tags), bool)
    osm[:, 0] = leaf == 0
    cell_facts = np.array([[c % 2 == 0, c % 2 == 1] for c in range(n_cells)])
    x = np.c_[np.eye(n_leaves)[leaf], cell_facts[cell].astype(float)] + 0.05 * rng.normal(
        size=(n, n_leaves + 2)
    )
    specs, items_of, text, cap_osm = [], [], [], []
    for i in range(n):
        f = int(np.flatnonzero(cell_facts[cell[i]])[0])
        land = np.eye(n_leaves, dtype=bool)[leaf[i]]
        base = {
            "land": land,
            "type": NONE,
            "near": np.zeros(1, bool),
            "d": 0.0,
            "neg": False,
            "patch_kind": 0,
            "h": 0.0,
            "fact": -1,
        }
        specs += [base, dict(base, type=CLIMATE, fact=f)]
        e = np.zeros(n_leaves + 2)
        e[leaf[i]] = 1
        text += [e, e + np.eye(n_leaves + 2)[n_leaves + f]]
        items_of += [i, i]
        cap_osm += [-1, -1]
    t = torch.as_tensor
    items = Items(t(leaf), torch.zeros(n, 1), torch.zeros(n), torch.zeros(n), t(cell))
    return Bank(
        x=t(x).float(),
        items=items,
        osm=t(osm),
        cell_facts=t(cell_facts),
        cap_item=t(np.array(items_of)),
        cap_text=t(np.array(text)).float(),
        specs=specs_from(specs, "cpu"),
        cap_osm=t(np.array(cap_osm)),
    ), leaf


def test_projector_layers_and_shape():
    p = Projector(5, 7, layers=3, hidden=11)
    assert len([m for m in p.net if isinstance(m, torch.nn.Linear)]) == 3
    assert p(torch.zeros(4, 5)).shape == (4, 7)


def test_osm_caption_adds_a_clause_or_replaces_the_caption():
    base = {
        "full": "corn fields",
        "spec": {
            "land": np.array([True, False]),
            "type": 1,
            "near": np.ones(2, bool),
            "d": 100.0,
            "neg": False,
            "patch_kind": 0,
            "h": 0.0,
            "fact": -1,
        },
    }
    words = [["a quarry", "a pit", "a mine"], ["a golf course", "links", "a club"]]
    rng = np.random.default_rng(0)
    text, spec, k = osm_caption(
        base, np.array([1]), words, ["tagged as"], {"osm_prob": 1.0, "osm_only_prob": 0.0}, rng
    )
    assert text.startswith("corn fields, tagged as ") and k == 1 and spec is base["spec"]
    text, spec, k = osm_caption(
        base, np.array([0]), words, ["tagged as"], {"osm_prob": 1.0, "osm_only_prob": 1.0}, rng
    )
    assert text in words[0] and spec["land"].all() and spec["type"] == NONE and not spec["near"].any()
    assert (
        osm_caption(base, np.array([], int), words, ["x"], {"osm_prob": 1.0, "osm_only_prob": 0.0}, rng)[2]
        == -1
    )


def test_targets_apply_the_osm_tag_and_match_the_ported_soft_target():
    b, leaf = toy_bank(n=12)
    caps = torch.arange(0, 6)
    t = targets(
        b.specs.take(caps),
        torch.tensor([0, -1, -1, -1, -1, -1]),
        b.items,
        b.osm,
        b.cap_item[caps],
        b.cell_facts,
    )
    assert t.shape == (6, 6)
    # caption 0 needs OSM tag 0 as well as its land class
    want0 = (leaf[b.cap_item[caps].numpy()] == leaf[b.cap_item[0]]) & b.osm.numpy()[
        b.cap_item[caps].numpy(), 0
    ]
    np.testing.assert_array_equal(t[:, 0].numpy(), want0.astype(float))
    # the ported AETHER soft target on standardised binary facts equals the truth matrix within exp(-66)
    facts = np.c_[np.eye(3)[leaf], b.cell_facts.numpy()[b.items.cell.numpy()]].astype(float)
    z = torch.as_tensor((facts - facts.mean(0)) / facts.std(0)).float()
    ids = [[int(leaf[i])] for i in range(12)]
    soft = soft_target_matrix(z, ids, sigma=0.1)
    hard = torch.as_tensor(leaf[:, None] == leaf[None, :]).float()
    assert torch.max(torch.abs(soft - hard)) < np.exp(-66)


def test_climate_only_spec_is_true_for_any_land():
    s = climate_only_spec(4, 2, fact=3)
    assert s["land"].all() and s["type"] == CLIMATE and s["fact"] == 3


def test_epoch_captions_one_per_location():
    cap_item = np.array([0, 0, 1, 1, 2, 3])
    got = epoch_captions(cap_item, np.array([0, 1, 3]), np.random.default_rng(0))
    assert sorted(cap_item[got].tolist()) == [0, 1, 3]


def test_concept_indices_and_group_mean():
    truth_cn = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], bool)
    sim = np.array([[0.9, 0.8, 0.1, 0.2, 0.0, 0.3], [0.1, 0.9, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    idx = concept_indices(sim, truth_cn, min_pos=2, max_share=0.95)
    assert idx[0] == pytest.approx(1.0) and np.isnan(idx[1]) and np.isnan(idx[2])
    assert group_mean(np.array([1.0, 0.0, 0.5]), np.array(["a", "a", "b"])) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        group_mean(np.array([np.nan]), np.array(["a"]))


def test_train_learns_land_and_climate_wording():
    b, leaf = toy_bank()
    idx = np.arange(len(leaf))
    tv = b.cap_text.numpy()

    def val_eval(model):
        z = project(model, b.x.numpy(), "cpu")
        q = tv[[0, 1]] / np.linalg.norm(tv[[0, 1]], axis=1, keepdims=True)
        return float(np.corrcoef((q @ z.T)[0], leaf == leaf[0])[0, 1])

    _model, log_temp, hist = train(b, idx[:200], idx[200:], val_eval, CFG, "cpu", log=lambda *_: None)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert max(h["val_index"] for h in hist) > 0.8
    assert float(log_temp.exp()) > 0


def test_multiset_index_matches_direct_computation():
    rng = np.random.default_rng(1)
    score = rng.normal(size=600)
    truth_v = score + rng.normal(size=600) > 0.8
    tile = np.repeat(np.arange(3), 200)
    ts = TileScores.of(score, truth_v, tile, 3)
    direct = concept_indices(score[None], truth_v[None], 5, 0.95)[0]
    assert multiset_index(ts, np.array([1, 1, 1]), 5, 0.95) == pytest.approx(direct, abs=1e-9)
    # drawing tile 0 twice and tile 2 once equals the index on that pixel multiset
    sel = np.r_[np.flatnonzero(tile == 0), np.flatnonzero(tile == 0), np.flatnonzero(tile == 2)]
    want = concept_indices(score[sel][None], truth_v[sel][None], 5, 0.95)[0]
    got = multiset_index(ts, np.array([2, 0, 1]), 5, 0.95)
    assert got == pytest.approx(want, abs=0.02)  # duplicated scores tie at the threshold
    assert np.isnan(multiset_index(ts, np.array([1, 0, 0]), 10_000, 0.95))


class _Tok:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding and truncation and max_length == 77 and return_tensors == "pt"
        return {"input_ids": torch.tensor([[len(t)] for t in texts])}


class _Model(torch.nn.Module):
    def get_text_features(self, input_ids):
        return input_ids.float().repeat(1, 3)


def test_clip_features_batches_and_keeps_order():
    got = clip_features(["a", "bbb", "cc"], _Tok(), _Model(), "cpu", batch=2)
    np.testing.assert_allclose(got, [[1, 1, 1], [3, 3, 3], [2, 2, 2]])
    with pytest.raises(ValueError):
        text_vectors(["x"], "m", "other", "cpu", None)


class _Ctx:
    res_m = 30.0

    def __init__(self, cells_ok):
        self.leaves = ["corn", "wetland", "other"]
        self.groups = {"crops": ["corn"]}
        self.pixel_cell = {"t": (np.zeros((4, 6), int), np.arange(24).reshape(4, 6) % 2)}
        self._ok = cells_ok

    def climate(self, cond):
        return self._ok


def test_plan_scores_tile_combines_land_distance_and_climate():
    probs = np.zeros((3, 4, 6))
    probs[0, :, :3] = 0.9  # corn on the left half
    probs[1, :, 5] = 0.8  # wetland in the last column
    base = {"land": {"classes": ["crops"], "min_patch_ha": None}, "near": [], "climate": []}
    ctx = _Ctx(np.array([[True, False]]))
    s = plan_scores_tile(base, probs, "t", ctx)
    np.testing.assert_allclose(s[:, :3], 0.9)
    near = dict(base, near=[{"classes": ["wetland"], "distance_m": 100, "negate": False}])
    s = plan_scores_tile(near, probs, "t", ctx)
    assert (s[:, 2] > 0).all() and (s[:, 0] == 0).all()  # 90 m from the wetland column vs 150 m
    clim = dict(base, climate=[{"variable": "x"}])
    s = plan_scores_tile(clim, probs, "t", ctx)
    assert (s[:, [0, 2]] > 0).all() and (s[:, 1] == 0).all()  # odd columns fall in a cell that fails
    assert climate_ok_tile(clim, "t", ctx)[:, 1].sum() == 0
    with pytest.raises(ValueError):
        plan_scores_tile(dict(base, land={"classes": ["rice"], "min_patch_ha": None}), probs, "t", ctx)


def test_similarities_match_projecting_everything_at_once():
    rng = np.random.default_rng(3)
    model = Projector(6, 5, layers=2, hidden=8)
    ae, cell = rng.normal(size=(37, 4)).astype(np.float32), rng.integers(0, 3, 37)
    summary = rng.normal(size=(3, 2)).astype(np.float32)
    mu, sd = np.zeros(6), np.ones(6)
    text = rng.normal(size=(2, 5))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    got = similarities(model, ae, cell, summary, mu, sd, text, "cpu", chunk=10)
    want = text @ project(model, np.c_[ae, summary[cell]].astype(np.float32), "cpu").T
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
