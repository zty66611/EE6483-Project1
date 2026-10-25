"""Shared paths, loading and text cleaning used by every model so all members see identical data."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

TEXT_COL = "reviews"
LABEL_COL = "sentiments"
CLEAN_COL = "text"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a raw JSON array file, keeping the original row order as ``orig_index``."""
    df = pd.read_json(path)
    df.insert(0, "orig_index", range(len(df)))
    return df


def clean_text(text: str) -> str:
    """Remove HTML remnants and redundant whitespace; casing and negations are left untouched."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def load_split(name: str) -> pd.DataFrame:
    """Load a processed split: ``train``, ``val`` or ``full`` (deduplicated train + val)."""
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.prepare_data` first.")
    return pd.read_csv(path, keep_default_na=False)


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
