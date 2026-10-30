"""Local web app for the demo: a map page, area metadata and a search endpoint.

Run with ``uv run python -m askmap.demo.serve`` (loads the engine from the precomputed demo data).
"""

from __future__ import annotations

import logging
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


def create_app(engine, meta: dict) -> FastAPI:
    """``engine`` answers ``search(question, area, method)``; ``meta`` lists areas, methods and examples."""
    app = FastAPI(title="Ask the map")

    @app.get("/", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        # never cached, so a reload always shows the current page
        return HTMLResponse(PAGE.read_text(), headers={"Cache-Control": "no-store"})

    @app.get("/api/meta")
    def get_meta() -> dict:
        return meta

    @app.post("/api/search")
    def search(q: Query) -> dict:
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
            # a fault in the engine: logged in full here, shown to the user as a readable message, never a
            # bare "Internal Server Error" the page cannot read
            logging.getLogger("askmap.demo").exception(
                "search failed for %r in %r (%s)", text, q.area, q.method
            )
            raise HTTPException(
                status_code=500,
                detail=f"Something went wrong answering that ({type(e).__name__}); it has been logged.",
            ) from e

    return app
