"""The static version of the demo for the blog: the same page, with pre-answered questions and no server.

``write_site`` saves the page (switched to static mode), ``data/meta.json`` (areas and the list of
answers) and one JSON file per answer, exactly as the live server returned it.
"""

from __future__ import annotations

import json
from pathlib import Path

LIVE_BODY = '<body data-mode="live">'
STATIC_BODY = '<body data-mode="static">'


def static_page(html: str) -> str:
    """The demo page switched to static mode (it then reads ``data/*.json`` instead of calling the server)."""
    if html.count(LIVE_BODY) != 1:
        raise ValueError("the demo page must contain exactly one live-mode <body> tag")
    return html.replace(LIVE_BODY, STATIC_BODY)


def write_site(out_dir: Path, page_html: str, areas: dict, answers: list[dict]) -> list[Path]:
    """Write the static site. ``answers`` are ``{"question", "area", "result"}`` with the server's result."""
    if not answers:
        raise ValueError("no answers to write")
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    written, index = [], []
    for i, a in enumerate(answers, start=1):
        if a["area"] not in areas:
            raise KeyError(f"unknown area {a['area']!r}")
        name = f"q{i:02d}.json"
        (data / name).write_text(json.dumps(a["result"], separators=(",", ":")))
        written.append(data / name)
        index.append({"question": a["question"], "area": a["area"], "file": name})
    (data / "meta.json").write_text(json.dumps({"areas": areas, "answers": index}, indent=1))
    (out_dir / "index.html").write_text(static_page(page_html))
    return [*written, data / "meta.json", out_dir / "index.html"]
