"""Stable pastel colors for compound Role / Identifier-type grid cells."""

from __future__ import annotations

import hashlib
from typing import Sequence

import pandas as pd

from core.compound_baseline import NONE_SENTINEL

# Light pastels — always readable with black text.
CATEGORICAL_PALETTE: list[str] = [
    "#c6f6d5",  # soft green
    "#bee3f8",  # soft blue
    "#fefcbf",  # soft yellow
    "#feebc8",  # soft orange
    "#e9d8fd",  # soft purple
    "#fed7e2",  # soft pink
    "#b2f5ea",  # soft teal
    "#faf089",  # soft gold
]

ABSENT_GRAY = "#e2e8f0"
TEXT_BLACK = "#000000"


def color_for_value(
    value: object,
    palette: Sequence[str] | None = None,
) -> str:
    """
    Map a categorical cell value to a stable pastel hex color.

    Blank / ``(none)`` / NaN → gray. Otherwise ``palette[stable_hash % len]``
    so the same label always gets the same color across rows and columns.
    """
    colors = list(palette) if palette is not None else CATEGORICAL_PALETTE
    if not colors:
        return ABSENT_GRAY
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ABSENT_GRAY
    text = str(value).strip()
    if not text or text.casefold() == NONE_SENTINEL.casefold():
        return ABSENT_GRAY
    digest = hashlib.md5(text.casefold().encode("utf-8")).hexdigest()
    return colors[int(digest, 16) % len(colors)]


def css_cell(background: str) -> str:
    """Background + black text for light compound-grid cells."""
    return f"background-color: {background}; color: {TEXT_BLACK}"
