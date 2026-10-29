"""Earth Engine: log in with gcloud application-default credentials and download small tiles with
``computePixels`` (no asset storage needed).

Adapted from Matt Harris's earlier code (``cropint.earth_engine``, ``cropint.tiles_ee`` and
``cropint.one_in_four_ee`` in the climate-embeddings-crop-integrity project). Requests stay under Earth
Engine's per-request size limit by splitting bands into groups, and rate-limit errors are retried with a
growing wait.
"""

from __future__ import annotations

import os
import time

import numpy as np

from askmap.geo import EQUAL_AREA, Tile

CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
AE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
CDL_COLLECTION = "USDA/NASS/CDL"
MAX_REQUEST_BYTES = 30_000_000  # stay well under Earth Engine's ~48 MB computePixels limit

EE_SETUP_HELP = """Earth Engine is not set up. MANUAL STEPS:
  1. gcloud auth application-default login
  2. Copy .env.example to .env and set EE_PROJECT to your Earth Engine Cloud project ID.
See README.md, "Manual steps"."""


def resolve_project(project: str | None = None) -> str:
    """Project from the argument or the EE_PROJECT environment variable; no default."""
    resolved = project or os.environ.get("EE_PROJECT")
    if not resolved:
        raise RuntimeError(EE_SETUP_HELP)
    return resolved


def init_earth_engine(project: str | None = None) -> str:
    """Initialise Earth Engine with application-default credentials (cloud-platform scope).

    Returns the project ID.

    ``earthengine authenticate`` asks for an Earth Engine-specific scope that Google blocks for some accounts,
    so the standard cloud-platform scope is used instead.
    """
    import ee
    import google.auth

    resolved = resolve_project(project)
    try:
        creds, _ = google.auth.default(scopes=[CLOUD_SCOPE], quota_project_id=resolved)
        ee.Initialize(credentials=creds, project=resolved)
        ee.Number(1).getInfo()
    except Exception as e:  # google.auth and ee raise several unrelated exception types
        raise RuntimeError(f"{EE_SETUP_HELP}\n\nEarth Engine said: {e}") from e
    return resolved


def is_rate_limited(exc: Exception) -> bool:
    """True for Earth Engine 'too many requests' / concurrency-limit errors (HTTP 429)."""
    text = str(exc)
    return "429" in text or "Too Many Requests" in text or "concurrency limit" in text


def with_backoff(fn, retries: int, first_wait_s: float, log=print):
    """Call ``fn()``; on a rate-limit error wait and retry (doubling the wait). Other errors propagate."""
    wait = first_wait_s
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # ee and googleapiclient raise several unrelated types for HTTP 429
            if not is_rate_limited(exc) or attempt == retries:
                raise
            log(f"    rate limited; retry {attempt + 1}/{retries} in {wait:.0f}s")
            time.sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")


def band_groups(
    bands: list[str], bytes_per_value: int, pixels: int, max_bytes: int = MAX_REQUEST_BYTES
) -> list[list[str]]:
    """Split ``bands`` into consecutive groups whose request size stays under ``max_bytes``."""
    per_band = bytes_per_value * pixels
    if per_band > max_bytes:
        raise ValueError(f"a single band ({per_band:,} bytes) exceeds the request limit; use a smaller tile")
    size = max(1, max_bytes // per_band)
    return [bands[i : i + size] for i in range(0, len(bands), size)]


def grid_spec(tile: Tile) -> dict:
    """Earth Engine pixel grid for a tile (EPSG:5070, north-up)."""
    return {
        "dimensions": {"width": tile.size_px, "height": tile.size_px},
        "affineTransform": {
            "scaleX": tile.res_m,
            "shearX": 0,
            "translateX": tile.x0,
            "shearY": 0,
            "scaleY": -tile.res_m,
            "translateY": tile.y1,
        },
        "crsCode": EQUAL_AREA,
    }


def fetch_tile(image, bands: list[str], tile: Tile, bytes_per_value: int = 4, log=print) -> np.ndarray:
    """Pixel values of ``bands`` over ``tile`` as a float array ``(n_bands, rows, cols)``."""
    import ee

    out: list[np.ndarray] = []
    for group in band_groups(bands, bytes_per_value, tile.size_px**2):
        # cast to 32-bit on the server; Earth Engine otherwise sends 64-bit values (twice the request size)
        req = {
            "expression": image.select(group).toFloat(),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": grid_spec(tile),
        }
        arr = with_backoff(lambda req=req: ee.data.computePixels(req), retries=6, first_wait_s=30, log=log)
        out.extend(np.asarray(arr[b], dtype=np.float32) for b in group)
    return np.stack(out)


def alphaearth_image(year: int):
    import ee

    return ee.ImageCollection(AE_COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()


def cdl_image(years: list[int]):
    import ee

    return ee.Image.cat(
        [ee.Image(f"{CDL_COLLECTION}/{y}").select("cropland").unmask(0).rename(f"cdl_{y}") for y in years]
    )
