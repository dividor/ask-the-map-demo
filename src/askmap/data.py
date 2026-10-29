"""Downloads recorded in a manifest (source and SHA-256), and the USPVDB reader."""

from __future__ import annotations

from pathlib import Path

from askmap.manifest import Manifest


def fetch_file(url: str, path: Path, manifest: Manifest, timeout: int = 600) -> Path:
    """Download ``url`` to ``path`` once; later calls reuse it if the manifest says it is complete.

    The body is streamed to disk in chunks (the National Wetlands Inventory file for North Dakota is
    1.5 GB), and the download fails if fewer bytes arrive than the server announced."""
    import requests

    if manifest.is_current(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        expected = r.headers.get("Content-Length")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    got = tmp.stat().st_size
    if expected is not None and got != int(expected):
        raise OSError(f"{url}: got {got:,} bytes, expected {int(expected):,}")
    tmp.replace(path)
    manifest.record(path, source=url, downloaded_bytes=got)
    return path


def read_uspvdb(zip_path: Path, member: str):
    """USPVDB facility polygons (EPSG:4326) with their install year ``p_year`` and ``case_id``."""
    import geopandas as gpd

    gdf = gpd.read_file(f"zip://{zip_path}!{member}")
    missing = {"p_year", "case_id", "geometry"} - set(gdf.columns)
    if missing:
        raise KeyError(f"USPVDB is missing columns {sorted(missing)}")
    return gdf
