"""Shared report formatting for M2 compound diff (CLI + Streamlit)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.compound_matching import CompoundDiffResult
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
