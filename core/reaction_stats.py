"""Editable-grid seeding and stats for reactions benchmark (Streamlit-free)."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from core.compound_baseline import NONE_SENTINEL
from core.compound_matching import NWayCluster
from core.reaction_baseline import ReactionBaselineDefaults, product_display_value
from core.reaction_grid import reaction_key
from core.reaction_nway import format_match_tier
from core.reaction_parsing import ReactionEntry
from core.reaction_report import cluster_display_label

_METRIC_COLS = ("presence_f1", "reaction_class_accuracy", "product_accuracy")
_DEFAULT_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "presence_f1": 1.0,
        "reaction_class_accuracy": 0.25,
        "product_accuracy": 0.50,
    }
)


def _field_cell(entry: ReactionEntry, field: str) -> str:
    if field == "product":
        value = product_display_value(entry)
    else:
        raw = getattr(entry, field, None)
        value = None if raw is None else str(raw).strip() or None
    if value is None:
        return NONE_SENTINEL
    return value


def seed_view_dataframe(
    clusters: list[NWayCluster[ReactionEntry]],
    model_labels: list[str],
    field: str,
    *,
    defaults: Mapping[str, ReactionBaselineDefaults],
    preferred_label: str,
) -> pd.DataFrame:
    """
    Seed an editable grid for ``field`` in ``presence`` | ``reaction_class`` | ``product``.

    Columns: Reaction, Match tier, one column per model label, Baseline.
    """
    if field not in {"presence", "reaction_class", "product"}:
        raise ValueError(f"Unsupported field: {field!r}")

    rows: list[dict[str, object]] = []
    for cluster in clusters:
        display = cluster_display_label(cluster, preferred_label)
        default = defaults.get(display)
        row: dict[str, object] = {
            "Reaction": display,
            "Match tier": format_match_tier(cluster.match_tier),
        }
        if field == "presence":
            for label in model_labels:
                row[label] = label in cluster.membership
            row["Baseline"] = default.present if default is not None else False
        else:
            for label in model_labels:
                if label not in cluster.membership:
                    row[label] = NONE_SENTINEL
                    continue
                row[label] = _field_cell(cluster.representatives[label], field)
            baseline_value = getattr(default, field) if default is not None else None
            if baseline_value is None or not str(baseline_value).strip():
                row["Baseline"] = NONE_SENTINEL
            else:
                row["Baseline"] = str(baseline_value).strip()
        rows.append(row)

    columns = ["Reaction", "Match tier", *model_labels, "Baseline"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def _is_none_sentinel(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.casefold() == NONE_SENTINEL.casefold()


def filter_by_presence_baseline(
    field_df: pd.DataFrame,
    presence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only rows whose Reaction has Presence Baseline == True."""
    if field_df is None or field_df.empty or "Reaction" not in field_df.columns:
        return field_df.copy() if field_df is not None else pd.DataFrame()
    if (
        presence_df is None
        or presence_df.empty
        or "Reaction" not in presence_df.columns
        or "Baseline" not in presence_df.columns
    ):
        return field_df.iloc[0:0].copy()

    present_keys: set[str] = set()
    for _, row in presence_df.iterrows():
        key = reaction_key(row.get("Reaction"))
        if not key:
            continue
        baseline = row.get("Baseline")
        if baseline is None or (isinstance(baseline, float) and pd.isna(baseline)):
            continue
        if bool(baseline):
            present_keys.add(key)

    keep = field_df["Reaction"].map(reaction_key).isin(present_keys)
    return field_df.loc[keep].reset_index(drop=True)


def compute_presence_stats(
    presence_df: pd.DataFrame,
    model_labels: list[str],
) -> pd.DataFrame:
    """Per model: precision / recall / F1 of that model's bool column vs Baseline."""
    rows: list[dict[str, object]] = []
    baseline_col = presence_df["Baseline"].fillna(False).astype(bool)
    for label in model_labels:
        model_col = presence_df[label].fillna(False).astype(bool)
        tp = int(((baseline_col) & (model_col)).sum())
        fp = int(((~baseline_col) & (model_col)).sum())
        fn = int(((baseline_col) & (~model_col)).sum())
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        if precision is None or recall is None:
            f1 = None
        elif (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "model": label,
                "presence_precision": precision,
                "presence_recall": recall,
                "presence_f1": f1,
            }
        )
    return pd.DataFrame(rows)


def compute_field_accuracy(
    field_df: pd.DataFrame,
    model_labels: list[str],
) -> pd.DataFrame:
    """Per model: accuracy = matches vs Baseline / rows where Baseline is not ``(none)``."""
    rows: list[dict[str, object]] = []
    for label in model_labels:
        comparable = 0
        matches = 0
        for _, row in field_df.iterrows():
            baseline = row.get("Baseline")
            if _is_none_sentinel(baseline):
                continue
            comparable += 1
            model_value = row.get(label)
            if _is_none_sentinel(model_value):
                continue
            if str(model_value).strip().casefold() == str(baseline).strip().casefold():
                matches += 1
        accuracy = matches / comparable if comparable else None
        rows.append({"model": label, "accuracy": accuracy})
    return pd.DataFrame(rows)


def _sort_key_nan_safe(value: object) -> float:
    if value is None:
        return float("-inf")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if number != number:  # NaN
        return float("-inf")
    return number


def _tie_keys_equal(left: tuple[float, float, float], right: tuple[float, float, float]) -> bool:
    for a, b in zip(left, right):
        if a != a and b != b:
            continue
        if a != b:
            return False
    return True


def _weighted_nanmean(row: pd.Series, weights: Mapping[str, float]) -> float:
    numer = 0.0
    denom = 0.0
    any_present = False
    for col, weight in weights.items():
        if col not in row.index:
            continue
        value = pd.to_numeric(row[col], errors="coerce")
        if value is None or (isinstance(value, float) and value != value):
            continue
        any_present = True
        w = float(weight)
        numer += w * float(value)
        denom += w
    if not any_present:
        return float("nan")
    if denom == 0.0:
        present_vals = [
            float(pd.to_numeric(row[col], errors="coerce"))
            for col in weights
            if col in row.index and pd.notna(pd.to_numeric(row[col], errors="coerce"))
        ]
        return float(np.mean(present_vals)) if present_vals else float("nan")
    return numer / denom


def rank_models(
    stats_df: pd.DataFrame,
    weights: Mapping[str, float] = _DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """
    Add composite_score + Rank; sort by Rank ascending.

    Ties broken by presence_f1 then reaction_class_accuracy (both descending).
    """
    if stats_df is None or stats_df.empty:
        out = pd.DataFrame(
            columns=["Rank", "composite_score", *list(stats_df.columns if stats_df is not None else [])]
        )
        return out

    effective = dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)
    if all(float(effective.get(col, 0.0)) == 0.0 for col in _METRIC_COLS):
        effective = {col: 1.0 for col in _METRIC_COLS}

    out = stats_df.copy()
    present_metrics = [col for col in _METRIC_COLS if col in out.columns]
    if present_metrics:
        metric_weights = {col: float(effective.get(col, 1.0)) for col in present_metrics}
        out["composite_score"] = out[present_metrics].apply(
            lambda row: _weighted_nanmean(row, metric_weights),
            axis=1,
        )
    else:
        out["composite_score"] = float("nan")

    out["_sort_composite"] = out["composite_score"].map(_sort_key_nan_safe)
    out["_sort_f1"] = (
        out["presence_f1"].map(_sort_key_nan_safe) if "presence_f1" in out.columns else 0.0
    )
    out["_sort_class"] = (
        out["reaction_class_accuracy"].map(_sort_key_nan_safe)
        if "reaction_class_accuracy" in out.columns
        else 0.0
    )
    out = out.sort_values(
        by=["_sort_composite", "_sort_f1", "_sort_class"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    ranks: list[int] = []
    prev_key: tuple[float, float, float] | None = None
    for index, row in out.iterrows():
        key = (
            float(row["_sort_composite"]),
            float(row["_sort_f1"]),
            float(row["_sort_class"]),
        )
        if prev_key is None:
            ranks.append(1)
        elif _tie_keys_equal(key, prev_key):
            ranks.append(ranks[-1])
        else:
            ranks.append(int(index) + 1)
        prev_key = key
    out = out.drop(columns=["_sort_composite", "_sort_f1", "_sort_class"])
    out.insert(0, "Rank", ranks)

    cols = list(out.columns)
    cols.remove("composite_score")
    cols.insert(1, "composite_score")
    return out[cols]
