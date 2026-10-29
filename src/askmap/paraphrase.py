"""Amendment A8 (test V): more wordings for the training captions, written once by the open language model.

Every wording list of ``configs/captions.yaml`` that the caption generator indexes in parallel (land phrases,
near classes and templates, patch templates, climate templates with their variable and season words) gets the
same number of new wordings, so a caption can draw any wording index for every part. Candidates are checked
before use: the same ``{placeholders}`` as the original, not an existing wording, and no A6 never-seen term.
"""

from __future__ import annotations

import copy
import json
import string
from pathlib import Path

from askmap.unseen import terms_in_captions

SYSTEM = (
    "You write alternative wordings for phrases used in training captions of a map search over US farm "
    "country (land cover, distances and NASA POWER climate). Reply with one JSON object only, no other text."
)


def parallel_lists(cap: dict) -> dict[str, list[str]]:
    """The caption wording lists indexed in parallel by the generator, keyed by their path in ``cap``."""
    out = {f"land.{k}": v for k, v in cap["land"].items()}
    out |= {f"near.classes.{k}": v for k, v in cap["near"]["classes"].items()}
    out["near.near_templates"] = cap["near"]["near_templates"]
    out["near.far_templates"] = cap["near"]["far_templates"]
    out["patch.templates"] = cap["patch"]["templates"]
    cy = cap["climate_year"]
    out |= {f"climate_year.variables.{k}.words": v["words"] for k, v in cy["variables"].items()}
    out |= {f"climate_year.seasons.{k}.words": v["words"] for k, v in cy["seasons"].items()}
    out["climate_year.templates"] = cy["templates"]
    out["heat_days.templates"] = cap["heat_days"]["templates"]
    for k in ("templates_at_least", "templates_at_most"):
        out[f"rain_total.{k}"] = cap["rain_total"][k]
    for k in ("templates_at_most", "templates_at_least"):
        out[f"gdd.{k}"] = cap["gdd"][k]
    cl = cap["climate_long"]
    out |= {f"climate_long.variables.{k}.words": v["words"] for k, v in cl["variables"].items()}
    out["climate_long.templates"] = cl["templates"]
    out["climate_long.sunlight_templates"] = cl["sunlight_templates"]
    sizes = {len(v) for v in out.values()}
    if len(sizes) != 1:
        raise ValueError(f"parallel wording lists differ in length: {sorted(sizes)}")
    return {k: list(v) for k, v in out.items()}


def placeholders(text: str) -> list[str]:
    """The ``{name}`` fields of a template, sorted (a field used twice appears twice)."""
    return sorted(name for _, name, _, _ in string.Formatter().parse(text) if name is not None)


def keep_valid(
    original: list[str], candidates: list[str], banned_terms: list[str], need: int, prefix: bool = False
) -> list[str]:
    """The first ``need`` usable candidates: the originals' placeholders, not a repeat of an original or an
    earlier candidate (any case), no banned term (whole word, any case). With ``prefix`` a candidate is a
    caption prefix and is given exactly one trailing space. Fewer than ``need`` usable candidates fail."""
    fields = {tuple(placeholders(o)) for o in original}
    if len(fields) > 1:
        raise ValueError(f"the original wordings use different placeholders: {original}")
    want = fields.pop() if fields else ()
    seen = {o.strip().lower() for o in original}
    out: list[str] = []
    for c in candidates:
        if not isinstance(c, str):
            continue
        text = c.strip() + " " if prefix else c.strip()
        if not text.strip() or text.strip().lower() in seen:
            continue
        try:
            if tuple(placeholders(text)) != want:
                continue
        except ValueError:  # an unbalanced brace is not a usable template
            continue
        if terms_in_captions([text], banned_terms):
            continue
        seen.add(text.strip().lower())
        out.append(text)
        if len(out) == need:
            return out
    raise ValueError(f"only {len(out)} usable candidates of {len(candidates)} for {original}; {need} needed")


def extended_cap(cap: dict, extra: dict[str, list[str]]) -> dict:
    """A copy of ``cap`` with ``extra[path]`` appended to the list at each dotted path."""
    out = copy.deepcopy(cap)
    for path, words in extra.items():
        node = out
        *parents, last = path.split(".")
        for p in parents:
            node = node[p]
        if not isinstance(node[last], list):
            raise ValueError(f"{path} is not a wording list")
        node[last] = [*node[last], *words]
    return out


def request_text(lists: dict[str, list[str]], n: int, banned_terms: list[str], note: str) -> str:
    """The user message asking for ``n`` new wordings of every list."""
    return (
        f"{note}\n\nFor every key of the JSON object below, write exactly {n} NEW wordings that mean exactly "
        "the same as the existing ones, in plain US English, the way a farmer, planner or analyst might type "
        "them. Keep the same grammatical role (a noun phrase stays a noun phrase; a clause that follows a "
        "land phrase stays such a clause). Keep every {placeholder} exactly as written, once each, and add "
        "no new ones. Keep meanings exact: the same months, the same direction (more or less, within or "
        "beyond, highest or lowest). Do not repeat an existing wording. Never use any of these words: "
        f"{', '.join(banned_terms)}.\n\nReply with a JSON object with the same keys, each mapped to a list "
        f"of {n} strings.\n\n{json.dumps(lists, indent=1)}"
    )


def ask(
    client,
    model: str,
    lists: dict[str, list[str]],
    n: int,
    banned_terms: list[str],
    note: str,
    log_path: Path,
    llm_cfg: dict,
    max_tokens: int,
) -> dict[str, list[str]]:
    """The model's candidates for every list (one request; a saved record is reused, never re-asked). A reply
    that is not a JSON object with every key fails."""
    from askmap.llm import call, parse_plan

    if log_path.exists():
        rec = json.loads(log_path.read_text())
    else:
        rec = call(
            client,
            model,
            SYSTEM,
            request_text(lists, n, banned_terms, note),
            max_tokens,
            llm_cfg["temperature"],
            llm_cfg["transport_retries"],
            llm_cfg["retry_wait_s"],
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = log_path.with_suffix(".part")
        tmp.write_text(json.dumps(rec, indent=1, default=str))
        tmp.replace(log_path)
    got = parse_plan(rec["content"])
    missing = sorted(set(lists) - set(got))
    if missing:
        raise ValueError(f"the reply in {log_path.name} has no candidates for {missing}")
    return {k: list(got[k]) for k in lists}
