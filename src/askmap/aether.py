"""Port of AETHER's soft contrastive loss, chance-corrected concept-retrieval index and caption sampling.

Source: Tijūnaitytė, van der Plas, Belaid, Knapen, Soja and Athanasiadis, "Domain-specific language
alignment of multimodal embeddings towards explainable predictions", arXiv 2610.23110; code
github.com/WUR-AI/aether at commit 7b382c55645fb5a1d3f11ec8ba81a46c523fca48 (MIT licence, see
THIRD_PARTY_NOTICES.md). Ported from
``src/models/components/loss_fns/soft_contrastive_loss.py``,
``src/models/components/metrics/contrastive_validation.py``, ``src/models/text_alignment_model.py``
(index), ``src/data/base_datamodule.py`` (baselines, elbow threshold) and ``src/data/base_caption_builder.py``
(templates, percentage adjectives). Checked against their code on their data (PLAN.md, R1b).
"""

from __future__ import annotations

import random
import re

import numpy as np
import torch
import torch.nn.functional as F


def soft_target_matrix(aux_std: torch.Tensor, aux_ids: list[list[int]], sigma: float) -> torch.Tensor:
    """Gaussian similarity of locations i (rows) and j, over the standardised facts caption j mentions."""
    n, n_cols = aux_std.shape
    diffs = (aux_std.unsqueeze(1) - aux_std.unsqueeze(0)) ** 2
    mask = torch.zeros((n, n_cols), dtype=aux_std.dtype, device=aux_std.device)
    for j, cols in enumerate(aux_ids):
        if len(cols) > 0:
            mask[j, cols] = 1.0
    mask = mask.unsqueeze(0)
    dist = (diffs * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    return torch.exp(-dist / (2 * sigma**2))


def soft_contrastive_loss(
    eo: torch.Tensor,
    text: torch.Tensor,
    aux_std: torch.Tensor,
    aux_ids: list[list[int]],
    log_temp: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Symmetric cross-entropy against the soft target matrix (temperature exp(log_temp), capped at 100)."""
    return soft_contrastive_from_targets(eo, text, soft_target_matrix(aux_std, aux_ids, sigma), log_temp)


def soft_contrastive_from_targets(
    eo: torch.Tensor, text: torch.Tensor, t: torch.Tensor, log_temp: torch.Tensor
) -> torch.Tensor:
    """The same loss for a given target matrix ``t`` (rows: locations, columns: captions)."""
    t_eo2text = t / t.sum(dim=1, keepdim=True)
    t_text2eo = t / t.sum(dim=0, keepdim=True)
    eo = F.normalize(eo, dim=-1)
    text = F.normalize(text, dim=-1)
    logits = (eo @ text.T) / torch.clamp(log_temp.exp(), max=100)
    loss_eo2text = -(t_eo2text * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_text2eo = -(t_text2eo * F.log_softmax(logits, dim=0)).sum(dim=0).mean()
    return (loss_eo2text + loss_text2eo) / 2


def dynamic_k(truth: np.ndarray, theta_k: float, is_max: bool) -> int:
    """Number of locations at or beyond the concept threshold (the k used for that concept)."""
    k = int((truth >= theta_k).sum() if is_max else (truth <= theta_k).sum())
    if k == 0:
        raise ValueError("no location reaches the concept threshold")
    return k


def topk_agreement(truth: np.ndarray, score: np.ndarray, k: int, is_max: bool) -> float:
    """Percentage of the true top-k locations (by fact value) that are also in the top-k by score.

    As in AETHER, both rankings are descending; for a minimum concept the bottom k of both are compared.
    Ties are broken by a stable sort (their torch.argsort is not stable; with continuous values ties are
    rare).
    """
    n = len(truth)
    gt_pos = np.empty(n, dtype=np.int64)
    gt_pos[np.argsort(-truth, kind="stable")] = np.arange(n)
    pred_pos = np.empty(n, dtype=np.int64)
    pred_pos[np.argsort(-score, kind="stable")] = np.arange(n)
    if is_max:
        both = (gt_pos < k) & (pred_pos < k)
    else:
        both = (gt_pos >= n - k) & (pred_pos >= n - k)
    return float(both.sum() / k * 100)


def baseline_percent(truth: np.ndarray, theta_k: float, is_max: bool) -> float:
    """Chance level of :func:`topk_agreement` at the dynamic k: share (%) of locations beyond the threshold.

    A share of 100% is set to (n - 1) / n, as in AETHER, so the index stays defined.
    """
    n = len(truth)
    k = dynamic_k(truth, theta_k, is_max)
    if k == n:
        k = n - 1
    return k / n * 100


def concept_index(
    truth: np.ndarray, score: np.ndarray, theta_k: float, is_max: bool, baseline: float
) -> float:
    """AETHER's chance-corrected index: (agreement - baseline) / (100 - baseline); 0 = chance, 1 = perfect."""
    acc = topk_agreement(truth, score, dynamic_k(truth, theta_k, is_max), is_max)
    return (acc - baseline) / (100 - baseline)


def elbow_point(values: np.ndarray) -> float:
    """AETHER's concept threshold: the sorted value farthest from the straight line joining first and last.

    Values equal to the minimum are dropped first (so a fact that is mostly zero gets a threshold among the
    non-zero values).
    """
    v = np.sort(values[~np.isnan(values)])
    v = v[v > v[0]]
    if len(v) == 0:
        raise ValueError("all values are equal")
    x = np.arange(len(v)) / len(v)
    if x[0] == x[-1]:
        return float(v[0])
    slope = (v[-1] - v[0]) / (x[-1] - x[0])
    intercept = v[0] - slope * x[0]
    ortho = -1 / slope
    cross = ((v - ortho * x) - intercept) / (slope - ortho)
    dist = np.sqrt((x - cross) ** 2 + (v - (slope * x + intercept)) ** 2)
    return float(v[int(np.argmax(dist))])


def extract_tokens(template: str) -> list[str]:
    """Names inside <...> in a caption template."""
    return re.findall(r"<([^<>]+)>", template)


def fill(template: str, fillers: dict[str, str]) -> str:
    """Replace each <token> once with its text."""
    for t, f in fillers.items():
        template = template.replace(f"<{t}>", f, 1)
    return template


PERCENT_WORDS = {
    "none": ["none", "zero", "absent", "nonexistent"],
    "negligible": ["negligible", "trivial", "trace", "barely any", "scarcely any"],
    "minimal": ["minimal", "tiny", "very little", "marginal", "meager"],
    "slight": ["slight", "small", "modest", "limited", "faint"],
    "some": ["some", "a bit of", "a portion of", "partial", "a measure of"],
    "moderate": ["moderate", "fair", "reasonable", "middling", "decent"],
    "considerable": ["considerable", "notable", "meaningful", "appreciable", "marked"],
    "substantial": ["substantial", "solid", "sizable", "hefty", "goodly"],
    "significant": ["significant", "large", "strong", "pronounced", "prominent"],
    "major": ["major", "great", "high", "intense", "serious"],
    "extensive": ["extensive", "vast", "sweeping", "far-reaching", "immense"],
    "complete": ["complete", "total", "full", "entire", "absolute"],
}
_PERCENT_STEPS = [
    (10, "negligible"),
    (20, "minimal"),
    (30, "slight"),
    (40, "some"),
    (50, "moderate"),
    (60, "considerable"),
    (70, "substantial"),
    (80, "significant"),
    (90, "major"),
    (100, "extensive"),
]


def percent_word(percent: float, rng: random.Random) -> str:
    """AETHER's word for a percentage (0 to 100), drawn at random from its synonyms."""
    if not 0 <= percent <= 100:
        raise ValueError(f"percentage must be between 0 and 100, got {percent}")
    if percent == 0:
        key = "none"
    elif percent >= 100:
        key = "complete"
    else:
        key = next(name for top, name in _PERCENT_STEPS if percent < top)
    return rng.choice(PERCENT_WORDS[key])
