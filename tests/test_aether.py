import math
import random

import numpy as np
import pytest
import torch

from askmap.aether import (
    PERCENT_WORDS,
    baseline_percent,
    concept_index,
    dynamic_k,
    elbow_point,
    extract_tokens,
    fill,
    percent_word,
    soft_contrastive_loss,
    soft_target_matrix,
    topk_agreement,
)


def test_soft_target_matrix_uses_only_the_columns_each_caption_mentions():
    aux = torch.tensor([[0.0, 5.0], [1.0, -5.0]])
    t = soft_target_matrix(aux, [[0], [0]], sigma=1.0)
    off = math.exp(-0.5)
    np.testing.assert_allclose(t.numpy(), [[1.0, off], [off, 1.0]], rtol=1e-6)
    # a caption that mentions no column treats every location as equally similar
    t2 = soft_target_matrix(aux, [[], [1]], sigma=1.0)
    np.testing.assert_allclose(t2[:, 0].numpy(), [1.0, 1.0], rtol=1e-6)
    assert t2[0, 1] < 1e-20


def test_soft_contrastive_loss_prefers_matching_pairs():
    eo = torch.eye(3)
    aux = torch.tensor([[0.0], [10.0], [20.0]])
    ids = [[0], [0], [0]]
    log_temp = torch.log(torch.tensor(0.07))
    matched = soft_contrastive_loss(eo, eo.clone(), aux, ids, log_temp, sigma=0.1)
    swapped = soft_contrastive_loss(eo, eo[[1, 2, 0]], aux, ids, log_temp, sigma=0.1)
    assert matched < 1e-3 < swapped
    # the temperature is capped at 100
    big = soft_contrastive_loss(eo, eo.clone(), aux, ids, torch.tensor(10.0), sigma=0.1)
    capped = soft_contrastive_loss(eo, eo.clone(), aux, ids, torch.log(torch.tensor(100.0)), sigma=0.1)
    assert big == pytest.approx(capped.item())


def test_dynamic_k_agreement_baseline_and_index():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    assert dynamic_k(truth, 2.0, True) == 2
    assert dynamic_k(truth, 1.0, False) == 2
    with pytest.raises(ValueError):
        dynamic_k(truth, 9.0, True)
    assert topk_agreement(truth, truth, 2, True) == 100.0
    assert topk_agreement(truth, -truth, 2, True) == 0.0
    assert topk_agreement(truth, np.array([0.0, 1.0, 3.0, 2.0]), 1, False) == 100.0
    assert baseline_percent(truth, 2.0, True) == 50.0
    assert baseline_percent(truth, -1.0, True) == 75.0  # all 4 beyond the threshold -> (n - 1) / n
    assert concept_index(truth, truth, 2.0, True, 50.0) == 1.0
    assert concept_index(truth, -truth, 2.0, True, 50.0) == -1.0


def test_elbow_point_finds_the_bend():
    vals = np.r_[np.zeros(50), np.linspace(0.1, 1.0, 90), np.linspace(2.0, 20.0, 10)]
    e = elbow_point(vals)
    assert 1.0 <= e <= 4.0
    with pytest.raises(ValueError):
        elbow_point(np.zeros(5))


def test_tokens_fill_and_percent_words():
    tpl = "area with <corn> near <wetland>."
    assert extract_tokens(tpl) == ["corn", "wetland"]
    assert fill(tpl, {"corn": "much corn", "wetland": "a marsh"}) == "area with much corn near a marsh."
    rng = random.Random(0)
    assert percent_word(0, rng) in PERCENT_WORDS["none"]
    assert percent_word(55, rng) in PERCENT_WORDS["considerable"]
    assert percent_word(100, rng) in PERCENT_WORDS["complete"]
    with pytest.raises(ValueError):
        percent_word(101, rng)
