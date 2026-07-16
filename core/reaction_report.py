"""Shared report formatting for reaction pairwise + N-way comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.compound_matching import NWayCluster, NWayDiffResult, canonicalize_name
from core.reaction_matching import (
    ReactionBenchmarkReport,
    ReactionMatchDetail,
)
from core.reaction_nway import (
    MATCH_WATERFALL,
    compound_smiles_set,
    product_smiles_key,
    reaction_row_label,
)
from core.reaction_parsing import ReactionEntry


def _format_list(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else ""


def cluster_display_label(cluster: NWayCluster[ReactionEntry], preferred_label: str) -> str:
    """Preferred model's row label if present, else alphabetically-first model's."""
    if preferred_label in cluster.representatives:
        return reaction_row_label(cluster.representatives[preferred_label])
    if not cluster.representatives:
        return "(empty)"
    first_label = min(cluster.representatives.keys())
    return reaction_row_label(cluster.representatives[first_label])


def clusters_sorted_by_consensus(
    nway_result: NWayDiffResult[ReactionEntry],
) -> list[NWayCluster[ReactionEntry]]:
    """Clusters ordered by membership size descending, then by canonical row label."""

    def sort_key(cluster: NWayCluster[ReactionEntry]) -> tuple[int, str]:
        names = [
            canonicalize_name(reaction_row_label(entry)) or ""
            for entry in cluster.representatives.values()
        ]
        return (-len(cluster.membership), min(names) if names else "")

    return sorted(nway_result.clusters, key=sort_key)


def build_upset_memberships(
    nway_result: NWayDiffResult[ReactionEntry],
) -> list[frozenset[str]]:
    """One membership frozenset per cluster for ``upsetplot.from_memberships``."""
    return [cluster.membership for cluster in nway_result.clusters]


def enrichment_coverage_caption(
    entries_by_label: dict[str, list[ReactionEntry]],
) -> str:
    """Per-model % with non-empty compound_smiles (clustering key coverage)."""
    parts: list[str] = []
    for label, entries in entries_by_label.items():
        if not entries:
            parts.append(f"{label}: n/a")
            continue
        n = len(entries)
        with_set = sum(1 for e in entries if compound_smiles_set(e))
        parts.append(
            f"{label}: compound_smiles {with_set}/{n} "
            f"({100.0 * with_set / n:.0f}%)"
        )
    return "Enrichment coverage — " + "; ".join(parts)


def entry_to_dict(entry: ReactionEntry) -> dict[str, Any]:
    return {
        "reaction_id": entry.reaction_id,
        "canonical_rxn": entry.canonical_rxn,
        "product_name": entry.product_name,
        "product_smiles": entry.product_smiles,
        "compound_smiles": sorted(compound_smiles_set(entry)),
        "reactant_names": list(entry.reactant_names),
        "reactant_smiles": list(entry.reactant_smiles),
        "product_yield_pct": entry.product_yield_pct,
        "procedure_text": entry.procedure_text,
        "has_procedure_vector": entry.procedure_vector is not None,
        "has_reaction_vector": entry.reaction_vector is not None,
        "temperature_c": entry.temperature_c,
        "room_temperature": entry.room_temperature,
        "time_h": entry.time_h,
        "atmosphere": entry.atmosphere,
        "reaction_class": entry.reaction_class,
        "non_synthetic": entry.non_synthetic,
        "section_label": entry.section_label,
        "step_label": entry.step_label,
        "step_index": entry.step_index,
        "start_line": entry.start_line,
        "end_line": entry.end_line,
        "line_join": entry.line_join,
    }


def _model_payload(entry: ReactionEntry) -> dict[str, Any]:
    """Prefer the original upload payload; overlay canonical SMILES + R1 lines."""
    if entry.raw:
        payload = dict(entry.raw)
    else:
        payload = entry_to_dict(entry)
    if entry.product_smiles is not None:
        payload["product_smiles"] = entry.product_smiles
    payload["compound_smiles"] = sorted(compound_smiles_set(entry))
    payload["start_line"] = entry.start_line
    payload["end_line"] = entry.end_line
    payload["step_index"] = entry.step_index
    payload["line_join"] = entry.line_join
    return payload


def _cluster_line_span(cluster: NWayCluster[ReactionEntry]) -> dict[str, int] | None:
    """Union of members' line spans when any lines are present (R1 traceability)."""
    starts: list[int] = []
    ends: list[int] = []
    for entry in cluster.representatives.values():
        if entry.start_line is not None and entry.end_line is not None:
            starts.append(entry.start_line)
            ends.append(entry.end_line)
    if not starts:
        return None
    return {"start_line": min(starts), "end_line": max(ends)}


def _cluster_product_smiles(cluster: NWayCluster[ReactionEntry]) -> str | None:
    """Shared canonical product SMILES key when present on any member."""
    keys = {
        product_smiles_key(entry)
        for entry in cluster.representatives.values()
    }
    keys.discard(None)
    if len(keys) == 1:
        return next(iter(keys))
    return None


def _cluster_compound_smiles(cluster: NWayCluster[ReactionEntry]) -> list[str]:
    """Sorted union of members' compound SMILES sets (inspectability)."""
    union: set[str] = set()
    for entry in cluster.representatives.values():
        union |= compound_smiles_set(entry)
    return sorted(union)


def build_reaction_groups_json(
    nway_result: NWayDiffResult[ReactionEntry],
    *,
    patent_id: str,
    baseline_label: str,
) -> dict[str, Any]:
    """Serialize compound-Jaccard N-way clusters with each model's payload + lines."""
    preferred = (
        baseline_label
        if baseline_label in nway_result.labels
        else (nway_result.labels[0] if nway_result.labels else baseline_label)
    )
    clusters_payload: list[dict[str, Any]] = []
    for cluster in nway_result.clusters:
        item: dict[str, Any] = {
            "display_name": cluster_display_label(cluster, preferred),
            "match_tier": cluster.match_tier,
            "membership": sorted(cluster.membership),
            "compound_smiles": _cluster_compound_smiles(cluster),
            "models": {
                label: _model_payload(entry)
                for label, entry in sorted(cluster.representatives.items())
            },
        }
        shared_smiles = _cluster_product_smiles(cluster)
        if shared_smiles is not None:
            item["product_smiles"] = shared_smiles
        line_span = _cluster_line_span(cluster)
        if line_span is not None:
            item["line_span"] = line_span
        clusters_payload.append(item)
    return {
        "patent_id": patent_id,
        "baseline_label": baseline_label,
        "match_waterfall": list(MATCH_WATERFALL),
        "models": list(nway_result.labels),
        "cluster_count": len(nway_result.clusters),
        "clusters": clusters_payload,
    }


def summary_row(result: ReactionBenchmarkReport) -> dict[str, Any]:
    coverage = result.axis_coverage
    return {
        "baseline": result.baseline_label,
        "candidate": result.candidate_label,
        "baseline_count": result.baseline_reaction_count,
        "candidate_count": result.candidate_reaction_count,
        "tp": result.true_positives,
        "fp": result.false_positives,
        "fn": result.false_negatives,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "synonym_near_miss": result.synonym_near_miss,
        "avg_reactant_jaccard": result.avg_reactant_jaccard,
        "perfect_reactant_match_pct": result.perfect_reactant_match_pct,
        "yield_mae": result.yield_mae,
        "yield_within_5ppt_pct": result.yield_within_5ppt_pct,
        "reaction_class_accuracy": result.reaction_class_accuracy,
        "avg_procedure_similarity": result.avg_procedure_similarity,
        "legacy_label_precision": result.legacy_label_precision,
        "legacy_label_recall": result.legacy_label_recall,
        "axis_coverage_name": None if coverage is None else coverage.product_name,
        "axis_coverage_smiles": None if coverage is None else coverage.product_smiles,
        "axis_coverage_reactants": None if coverage is None else coverage.reactant_jaccard,
        "axis_coverage_procedure": None if coverage is None else coverage.procedure_cosine,
        "axis_coverage_yield": None if coverage is None else coverage.yield_score,
        "axis_coverage_conditions": None if coverage is None else coverage.conditions,
        "non_synthetic_skipped_baseline": result.non_synthetic_skipped_baseline,
        "non_synthetic_skipped_candidate": result.non_synthetic_skipped_candidate,
    }


def summary_to_dataframe(result: ReactionBenchmarkReport) -> pd.DataFrame:
    return pd.DataFrame([summary_row(result)])


def _entry_label(entry: ReactionEntry | None) -> str:
    if entry is None:
        return ""
    section = entry.section_label or ""
    step = entry.step_label or ""
    if section and step:
        return f"{section} | {step}"
    return section or step


def matched_pairs_to_dataframe(result: ReactionBenchmarkReport) -> pd.DataFrame:
    baseline = result.baseline_label
    candidate = result.candidate_label
    columns = [
        f"{baseline} label",
        f"{baseline} product",
        f"{candidate} label",
        f"{candidate} product",
        "composite",
        "axis_name",
        "axis_smiles",
        "axis_reactants",
        "axis_procedure",
        "axis_yield",
        "axis_conditions",
        "label_match",
        "product_smiles_match",
    ]
    rows: list[dict[str, Any]] = []
    for detail in result.match_details:
        if detail.match_type != "CONTENT_MATCH":
            continue
        axes = detail.axis_scores
        rows.append(
            {
                f"{baseline} label": _entry_label(detail.baseline),
                f"{baseline} product": (
                    "" if detail.baseline is None else (detail.baseline.product_name or "")
                ),
                f"{candidate} label": _entry_label(detail.candidate),
                f"{candidate} product": (
                    "" if detail.candidate is None else (detail.candidate.product_name or "")
                ),
                "composite": detail.composite_score,
                "axis_name": None if axes is None else axes.product_name,
                "axis_smiles": None if axes is None else axes.product_smiles,
                "axis_reactants": None if axes is None else axes.reactant_jaccard,
                "axis_procedure": None if axes is None else axes.procedure_cosine,
                "axis_yield": None if axes is None else axes.yield_score,
                "axis_conditions": None if axes is None else axes.conditions,
                "label_match": detail.label_match,
                "product_smiles_match": detail.product_smiles_match,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _side_only_rows(
    details: list[ReactionMatchDetail],
    *,
    match_type: str,
    side: str,
) -> pd.DataFrame:
    columns = ["label", "product_name", "product_smiles", "reactant_names", "yield_pct"]
    rows: list[dict[str, Any]] = []
    for detail in details:
        if detail.match_type != match_type:
            continue
        entry = detail.baseline if side == "baseline" else detail.candidate
        if entry is None:
            continue
        rows.append(
            {
                "label": _entry_label(entry),
                "product_name": entry.product_name or "",
                "product_smiles": entry.product_smiles or "",
                "reactant_names": _format_list(entry.reactant_names),
                "yield_pct": entry.product_yield_pct,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def false_positives_to_dataframe(result: ReactionBenchmarkReport) -> pd.DataFrame:
    """Candidate-only reactions (unmatched against baseline)."""
    return _side_only_rows(
        result.match_details, match_type="FALSE_POSITIVE", side="candidate"
    )


def false_negatives_to_dataframe(result: ReactionBenchmarkReport) -> pd.DataFrame:
    """Baseline-only reactions (missed by candidate)."""
    return _side_only_rows(
        result.match_details, match_type="FALSE_NEGATIVE", side="baseline"
    )


def build_compare_json_payload(result: ReactionBenchmarkReport) -> dict[str, Any]:
    coverage = result.axis_coverage
    return {
        "summary": summary_row(result),
        "axis_coverage": None
        if coverage is None
        else {
            "product_name": coverage.product_name,
            "product_smiles": coverage.product_smiles,
            "reactant_jaccard": coverage.reactant_jaccard,
            "procedure_cosine": coverage.procedure_cosine,
            "yield_score": coverage.yield_score,
            "conditions": coverage.conditions,
        },
        "matched_pairs": [
            {
                "composite_score": detail.composite_score,
                "label_match": detail.label_match,
                "product_smiles_match": detail.product_smiles_match,
                "axis_scores": None
                if detail.axis_scores is None
                else {
                    "product_name": detail.axis_scores.product_name,
                    "product_smiles": detail.axis_scores.product_smiles,
                    "reactant_jaccard": detail.axis_scores.reactant_jaccard,
                    "procedure_cosine": detail.axis_scores.procedure_cosine,
                    "yield_score": detail.axis_scores.yield_score,
                    "conditions": detail.axis_scores.conditions,
                },
                "baseline": None if detail.baseline is None else entry_to_dict(detail.baseline),
                "candidate": None
                if detail.candidate is None
                else entry_to_dict(detail.candidate),
            }
            for detail in result.match_details
            if detail.match_type == "CONTENT_MATCH"
        ],
        "false_positives": [
            entry_to_dict(detail.candidate)
            for detail in result.match_details
            if detail.match_type == "FALSE_POSITIVE" and detail.candidate is not None
        ],
        "false_negatives": [
            entry_to_dict(detail.baseline)
            for detail in result.match_details
            if detail.match_type == "FALSE_NEGATIVE" and detail.baseline is not None
        ],
    }
