"""Helpers for keeping Presence / Reaction class / Product grids aligned."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

REACTION_GRID_FIELDS = ("presence", "reaction_class", "product")


def reaction_key(value: object) -> str:
    """Strip + casefold key for Reaction display-string matching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().casefold()


def drop_reaction_rows(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """
    Drop rows whose ``Reaction`` display string is in ``names``.

    Matching is strip + casefold against the stored display string.
    """
    if df.empty or "Reaction" not in df.columns or not names:
        return df.copy()

    drop_folded = {
        reaction_key(name)
        for name in names
        if name is not None and str(name).strip()
    }
    drop_folded.discard("")
    if not drop_folded:
        return df.copy()

    keep = ~df["Reaction"].map(reaction_key).isin(drop_folded)
    return df.loc[keep].reset_index(drop=True)


def drop_reactions_from_frames(
    frames: Mapping[str, pd.DataFrame],
    names: list[str],
) -> dict[str, pd.DataFrame]:
    """Apply ``drop_reaction_rows`` to each named frame."""
    return {key: drop_reaction_rows(frame, names) for key, frame in frames.items()}
