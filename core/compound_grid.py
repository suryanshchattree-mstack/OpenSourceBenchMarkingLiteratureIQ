"""Helpers for keeping Presence / Role / Identifier-type compound grids aligned."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

COMPOUND_GRID_FIELDS = ("presence", "role", "identifier_type")


def compound_key(value: object) -> str:
    """Strip + casefold key for Compound display-string matching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().casefold()


def drop_compound_rows(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """
    Drop rows whose ``Compound`` display string is in ``names``.

    Matching is strip + casefold against the stored display string.
    Unrelated rows are preserved; index is reset.
    """
    if df.empty or "Compound" not in df.columns or not names:
        return df.copy()

    drop_folded = {
        compound_key(name)
        for name in names
        if name is not None and str(name).strip()
    }
    drop_folded.discard("")
    if not drop_folded:
        return df.copy()

    keep = ~df["Compound"].map(compound_key).isin(drop_folded)
    return df.loc[keep].reset_index(drop=True)


def drop_compounds_from_frames(
    frames: Mapping[str, pd.DataFrame],
    names: list[str],
) -> dict[str, pd.DataFrame]:
    """Apply ``drop_compound_rows`` to each named frame (e.g. presence/role/id_type)."""
    return {key: drop_compound_rows(frame, names) for key, frame in frames.items()}
