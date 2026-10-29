"""Experiment configuration (YAML) and the data folder (``ASKMAP_DATA_ROOT`` in ``.env``)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with Path(path).open() as fh:
        return yaml.safe_load(fh)


def load_env() -> None:
    """Load ``.env`` (data root, Earth Engine project) into the environment."""
    load_dotenv(PROJECT_ROOT / ".env")


def data_root() -> Path:
    """Folder for downloaded data; must be set in ``.env``."""
    load_env()
    value = os.environ.get("ASKMAP_DATA_ROOT")
    if not value:
        raise RuntimeError("ASKMAP_DATA_ROOT is not set; copy .env.example to .env and set it")
    root = Path(value)
    root.mkdir(parents=True, exist_ok=True)
    return root
