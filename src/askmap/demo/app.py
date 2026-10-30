"""Local web app for the demo: a map page, area metadata and a search endpoint.

Run with ``uv run python -m askmap.demo.serve`` (loads the engine from the precomputed demo data).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from askmap.demo.engine import Unavailable, UserError

PAGE = Path(__file__).with_name("index.html")


class Query(BaseModel):
    question: str
    area: str
    method: str
    share: float | None = None  # the most similar share of the study area to show (default: the engine's)


@dataclass
class Warmup:
    """What the server is doing before it can answer: the page shows this while a cold container loads
    the tiles and the text encoder, instead of a blank wait. ``ready()`` hands over the engine and the
    metadata."""

    stage: str = "starting"
    started: float = field(default_factory=time.time)
    engine: object | None = None
    meta: dict | None = None
    error: str | None = None

    def ready(self, engine, meta: dict) -> None:
        self.engine, self.meta, self.stage = engine, meta, "ready"

    def status(self) -> dict:
        return {
            "loading": True,
            "stage": self.stage,
            "seconds": round(time.time() - self.started),
            "error": self.error,
        }


def _answer(engine, q: Query) -> dict:
    """One search, with every failure turned into a readable HTTP error."""
    text = q.question.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please type a question.")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Please keep the question under 500 characters.")
    try:
        return engine.search(text, q.area, q.method, q.share)
    except UserError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Unavailable as e:
        raise HTTPException(status_code=503, detail=f"Not available yet: {e}") from e
    except (ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
        # a fault in the engine: logged in full here, shown to the user as a readable message, never a bare
        # "Internal Server Error" the page cannot read
        logging.getLogger("askmap.demo").exception("search failed for %r in %r (%s)", text, q.area, q.method)
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong answering that ({type(e).__name__}); it has been logged.",
        ) from e


def create_app(engine, meta: dict | None, warmup: Warmup | None = None) -> FastAPI:
    """``engine`` answers ``search(question, area, method)``; ``meta`` lists areas, methods and examples.
    With a ``warmup``, both may be None at first: the app serves the page and reports the loading stage
    until ``warmup.ready()`` is called (the server listens while a cold container loads)."""
    app = FastAPI(title="Ask the map")
    warm = warmup or Warmup()
    if engine is not None:
        warm.ready(engine, meta or {})

    @app.get("/", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        # never cached, so a reload always shows the current page
        return HTMLResponse(PAGE.read_text(), headers={"Cache-Control": "no-store"})

    @app.get("/api/meta")
    def get_meta() -> dict:
        return warm.status() if warm.engine is None else (warm.meta or {})

    @app.post("/api/search")
    def search(q: Query) -> dict:
        if warm.engine is None:
            raise HTTPException(
                status_code=503, detail=f"Still waking up: {warm.stage}. Try again in a moment."
            )
        return _answer(warm.engine, q)

    return app
