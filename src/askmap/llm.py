"""Turn a plain-English question into a plan with an open language model behind an OpenAI-compatible API (the
Hugging Face router by default).

The token comes from the ``HF_TOKEN`` variable or the local Hugging Face login file
(``~/.cache/huggingface/token``); it is never written to a file or printed. Every request, the raw response,
the model id and the provider that answered are saved. A reply that is not a JSON object is a malformed plan:
it is recorded as an error and never re-asked. Only transport failures (HTTP or network errors) are retried,
a fixed number of times.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def system_prompt(language: str, catalogue: str) -> str:
    return (
        "You translate a user's question about land and climate into a plan in the query language below. "
        "Reply with the plan only: one JSON object that follows the language exactly, with no other text.\n\n"
        f"# Query language\n\n{language}\n\n# Catalogue\n\n{catalogue}\n"
    )


def hf_token(login_file: Path | None = None) -> str:
    """``HF_TOKEN`` if set, else the local Hugging Face login (``~/.cache/huggingface/token``). The login file
    is read directly because this project's ``.env`` moves ``HF_HOME`` (the model cache) elsewhere."""
    token = os.environ.get("HF_TOKEN")
    path = login_file or Path.home() / ".cache" / "huggingface" / "token"
    if not token and path.exists():
        token = path.read_text().strip()
    if not token:
        raise RuntimeError("no Hugging Face token: log in with `hf auth login` or set HF_TOKEN")
    return token


def make_client(base_url: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=hf_token())


def parse_plan(content: str | None) -> dict:
    """The JSON object in a reply. Only surrounding whitespace and one Markdown code fence are removed."""
    if not content or not content.strip():
        raise ValueError("empty reply")
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    plan = json.loads(text)
    if not isinstance(plan, dict):
        raise ValueError("reply is not a JSON object")
    return plan


def call(
    client,
    model: str,
    system: str,
    question: str,
    max_tokens: int,
    temperature: float,
    transport_retries: int,
    wait_s: float,
) -> dict:
    """One chat request; returns the record (request, raw response, provider, content). Transport errors are
    retried up to ``transport_retries`` times; the last one is raised."""
    import openai

    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    for attempt in range(transport_retries + 1):
        try:
            t0 = time.time()
            raw = client.chat.completions.with_raw_response.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            r = raw.parse()
            msg = r.choices[0].message.model_dump()
            return {
                "request": {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                "response": r.model_dump(),
                "model_returned": r.model,
                "provider": raw.headers.get("x-inference-provider"),
                "seconds": round(time.time() - t0, 2),
                "content": msg.get("content"),
                "finish_reason": r.choices[0].finish_reason,
                "transport_attempts": attempt + 1,
            }
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError,
        ):
            if attempt == transport_retries:
                raise
            time.sleep(wait_s * (attempt + 1))
    raise AssertionError("unreachable")


def translate(client, model: str, system: str, question: str, log_path: Path, cfg: dict) -> dict:
    """Plan for ``question`` or the error that made it malformed; a saved record is reused, never re-asked."""
    if log_path.exists():
        return json.loads(log_path.read_text())
    rec = call(
        client,
        model,
        system,
        question,
        cfg["max_tokens"],
        cfg["temperature"],
        cfg["transport_retries"],
        cfg["retry_wait_s"],
    )
    rec["question"] = question
    try:
        rec["plan"], rec["error"] = parse_plan(rec["content"]), None
    except (ValueError, json.JSONDecodeError) as e:
        rec["plan"], rec["error"] = None, f"malformed plan: {type(e).__name__}: {e}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = log_path.with_suffix(".part")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    tmp.replace(log_path)
    return rec


def _norm(obj):
    """Plan in a canonical form: numbers as floats, string lists sorted, lists of objects sorted."""
    if isinstance(obj, dict):
        return {k: _norm(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        items = [_norm(v) for v in obj]
        if all(isinstance(v, str) for v in items):
            return sorted(items)
        if all(isinstance(v, dict) for v in items):
            return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
        return items
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    return float(obj)


def same_plan(a: dict, b: dict) -> bool:
    """True if two plans are identical apart from key order, the order of classes and rules, and int/float."""
    return _norm(a) == _norm(b)
