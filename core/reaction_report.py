"""Shared report formatting for reaction pairwise comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.reaction_matching import ReactionBenchmarkReport, ReactionMatchDetail
from core.reaction_parsing import ReactionEntry


def _format_list(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else ""


def entry_to_dict(entry: ReactionEntry) -> dict[str, Any]:
    return {
        "product_name": entry.product_name,
        "product_smiles": entry.product_smiles,
        "reactant_names": list(entry.reactant_names),
        "reactant_smiles": list(entry.reactant_smiles),
        "product_yield_pct": entry.product_yield_pct,
        "procedure_text": entry.procedure_text,
        "has_procedure_vector": entry.procedure_vector is not None,
        "temperature_c": entry.temperature_c,
        "room_temperature": entry.room_temperature,
        "time_h": entry.time_h,
        "atmosphere": entry.atmosphere,
        "reaction_class": entry.reaction_class,
        "non_synthetic": entry.non_synthetic,
        "section_label": entry.section_label,
        "step_label": entry.step_label,
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
