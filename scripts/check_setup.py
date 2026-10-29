"""Check the local set-up before running the pipeline: `uv run python scripts/check_setup.py`.

Checks the data folder, the Hugging Face login and (unless --no-earth-engine) Earth Engine. Nothing secret is
printed: not the Earth Engine project ID and not the token.
"""

from __future__ import annotations

import argparse
import os
import sys

from askmap.config import data_root, load_env
from askmap.llm import hf_token


def check_data_root() -> str:
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    return f"data folder: {root}"


def check_hugging_face() -> str:
    hf_token()  # raises with the fix if there is no token
    cache = os.environ.get("HF_HOME")
    if not cache:
        raise RuntimeError("HF_HOME is not set in .env (the folder for the text-encoder downloads)")
    return f"Hugging Face: token found; model cache {cache}"


def check_earth_engine() -> str:
    import ee

    from askmap.earth_engine import CDL_COLLECTION, init_earth_engine

    init_earth_engine()  # raises RuntimeError with the manual steps when the login or project is missing
    try:
        bands = ee.ImageCollection(CDL_COLLECTION).first().bandNames().getInfo()
    except ee.EEException as e:
        raise RuntimeError(f"Earth Engine refused the request: {e}") from e
    return f"Earth Engine: logged in; {CDL_COLLECTION} has bands {bands}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-earth-engine", action="store_true", help="skip the Earth Engine check")
    args = parser.parse_args()
    load_env()
    checks = [check_data_root, check_hugging_face]
    if not args.no_earth_engine:
        checks.append(check_earth_engine)
    failed = False
    for check in checks:
        try:
            print("OK   ", check())
        except (RuntimeError, OSError, ValueError) as e:  # each check names its own fix in the message
            failed = True
            print("FAIL ", type(e).__name__, "-", str(e).splitlines()[0])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
