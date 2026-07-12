"""Shared report formatting for M2 compound diff (CLI + Streamlit)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.compound_matching import CompoundDiffResult, NWayDiffResult, canonicalize_name
from core.compound_parsing import CompoundEntry


def _format_aliases(aliases: tuple[str, ...]) -> str:
    return ", ".join(aliases) if aliases else ""


def entry_to_dict(entry: CompoundEntry) -> dict[str, Any]:
    return {
        "identifier": entry.identifier,
        "identifier_type": entry.identifier_type,
        "aliases": list(entry.aliases),
        "section_label": entry.section_label,
        "role": entry.role,
        "resolved": entry.resolved,
        "unresolved_reference": entry.unresolved_reference,
    }


def entries_to_dataframe(entries: list[CompoundEntry]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame(
            columns=[
                "identifier",
                "identifier_type",
                "aliases",
                "section_label",
                "role",
                "unresolved_reference",
            ]
        )
    rows = []
    for entry in entries:
        rows.append(
            {
                "identifier": entry.identifier,
                "identifier_type": entry.identifier_type,
                "aliases": _format_aliases(entry.aliases),
                "section_label": entry.section_label or "",
                "role": entry.role or "",
                "unresolved_reference": entry.unresolved_reference,
            }
        )
    return pd.DataFrame(rows)


def matched_pairs_to_dataframe(
    pairs: list[tuple[CompoundEntry, CompoundEntry]],
    claude_label: str,
    benchmark_label: str,
) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(
            columns=[
                f"{claude_label} identifier",
                f"{claude_label} type",
                f"{benchmark_label} identifier",
                f"{benchmark_label} type",
            ]
        )
    rows = []
    for claude_entry, benchmark_entry in pairs:
        rows.append(
            {
                f"{claude_label} identifier": claude_entry.identifier,
                f"{claude_label} type": claude_entry.identifier_type,
                f"{benchmark_label} identifier": benchmark_entry.identifier,
                f"{benchmark_label} type": benchmark_entry.identifier_type,
            }
        )
    return pd.DataFrame(rows)


def identifier_type_counts(entries: list[CompoundEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.identifier_type] = counts.get(entry.identifier_type, 0) + 1
    return counts


def build_diff_json_payload(
    result: CompoundDiffResult,
    claude_label: str,
    benchmark_label: str,
) -> dict[str, Any]:
    recall = (
        result.common / result.deduped_claude_count
        if result.deduped_claude_count > 0
        else None
    )
    precision = (
        result.common / result.deduped_benchmark_count
        if result.deduped_benchmark_count > 0
        else None
    )
    return {
        "summary": {
            "claude_label": claude_label,
            "benchmark_label": benchmark_label,
            "raw_claude_count": result.raw_claude_count,
            "raw_benchmark_count": result.raw_benchmark_count,
            "deduped_claude_count": result.deduped_claude_count,
            "deduped_benchmark_count": result.deduped_benchmark_count,
            "common": result.common,
            "claude_only": result.claude_only,
            "benchmark_only": result.benchmark_only,
            "recall_vs_claude": recall,
            "precision_vs_claude": precision,
        },
        "matched_pairs": [
            {
                "claude": entry_to_dict(left),
                "benchmark": entry_to_dict(right),
            }
            for left, right in result.matched_pairs
        ],
        "claude_only": [entry_to_dict(entry) for entry in result.claude_only_entries],
        "benchmark_only": [entry_to_dict(entry) for entry in result.benchmark_only_entries],
    }


def build_upset_memberships(nway_result: NWayDiffResult) -> list[frozenset[str]]:
    """
    One membership frozenset per cluster for ``upsetplot.from_memberships``.

    Cluster count (after within-label dedupe) is the compound count encoded in
    the UpSet plot — identical memberships appear once per matching cluster.
    """
    return [cluster.membership for cluster in nway_result.clusters]


def nway_label_counts_dataframe(nway_result: NWayDiffResult) -> pd.DataFrame:
    """Per-label raw/deduped counts and singleton-only cluster counts."""
    rows = []
    for label in nway_result.labels:
        only_count = sum(
            1 for cluster in nway_result.clusters if cluster.membership == frozenset({label})
        )
        rows.append(
            {
                "label": label,
                "raw_count": nway_result.raw_counts.get(label, 0),
                "deduped_count": nway_result.deduped_counts.get(label, 0),
                "only_count": only_count,
            }
        )
    return pd.DataFrame(rows)


def nway_clusters_dataframe(
    nway_result: NWayDiffResult,
    *,
    labels: list[str] | None = None,
    min_labels: int = 1,
) -> pd.DataFrame:
    """
    Cluster table with one identifier column per label.

    Empty cells mean that label is absent from the cluster. Set ``min_labels=2``
    to keep only multi-model (common) clusters.
    """
    ordered_labels = list(labels) if labels is not None else list(nway_result.labels)
    columns = ["membership", "size", *ordered_labels]
    rows = []
    for cluster in nway_result.clusters:
        if len(cluster.membership) < min_labels:
            continue
        row: dict[str, Any] = {
            "membership": ", ".join(sorted(cluster.membership)),
            "size": len(cluster.membership),
        }
        for label in ordered_labels:
            entry = cluster.representatives.get(label)
            row[label] = entry.identifier if entry is not None else ""
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -int(row["size"]),
            min(
                (
                    canonicalize_name(str(row[label])) or ""
                    for label in ordered_labels
                    if row.get(label)
                ),
                default="",
            ),
        )
    )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def nway_pairwise_summary_dataframe(
    nway_result: NWayDiffResult,
    baseline: str,
) -> pd.DataFrame:
    """Recall/precision of each non-baseline label against ``baseline``."""
    if baseline not in nway_result.deduped_counts:
        raise KeyError(f"Baseline label {baseline!r} not in n-way result")
    rows = []
    for label in nway_result.labels:
        if label == baseline:
            continue
        common, baseline_only, other_only, recall, precision = nway_result.pairwise_metrics(
            baseline, label
        )
        rows.append(
            {
                "label": label,
                "common": common,
                "baseline_only": baseline_only,
                "model_only": other_only,
                "recall_vs_baseline": recall,
                "precision_vs_baseline": precision,
                "deduped_count": nway_result.deduped_counts.get(label, 0),
            }
        )
    return pd.DataFrame(rows)
