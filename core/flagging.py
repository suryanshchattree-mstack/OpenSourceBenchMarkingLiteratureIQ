"""Flag contiguous line regions where benchmarks disagree with the reference."""

from __future__ import annotations

from dataclasses import dataclass

from core.line_arrays import LineArrays
from core.scoring import MultiRunScores


@dataclass
class FlagRegion:
    start_line: int
    end_line: int
    run_types: dict[str, str | None]
    run_labels: dict[str, str | None]
    avg_cum_sim: float
    issue_kind: str  # type_mismatch | low_similarity | both
    disagreeing_models: list[str]


def _dominant_value(values: list[str | None]) -> str | None:
    """Pick the most common non-None value in a list."""
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _line_type_mismatch(scores: MultiRunScores, line: int) -> bool:
    return any(not run_scores.type_agree[line] for run_scores in scores.per_run.values())


def _line_low_similarity(scores: MultiRunScores, line: int, threshold: float) -> bool:
    return any(run_scores.cum_sim[line] < threshold for run_scores in scores.per_run.values())


def _line_worst_cum_sim(scores: MultiRunScores, line: int) -> float:
    if not scores.per_run:
        return 1.0
    return min(run_scores.cum_sim[line] for run_scores in scores.per_run.values())


def build_disagree_mask(
    scores: MultiRunScores,
    threshold: float,
) -> list[bool]:
    """Per-line disagreement mask vs reference (index 0 unused)."""
    total = scores.total_lines
    disagree: list[bool] = [False] * (total + 1)
    for i in range(1, total + 1):
        type_mismatch = _line_type_mismatch(scores, i)
        low_sim = _line_low_similarity(scores, i, threshold)
        disagree[i] = type_mismatch or low_sim
    return disagree


def collapse_flag_regions(
    disagree: list[bool],
    runs: list[LineArrays],
    run_labels: list[str],
    scores: MultiRunScores,
    threshold: float,
) -> list[FlagRegion]:
    """Collapse per-line disagree mask into contiguous flag regions."""
    total = scores.total_lines
    regions: list[FlagRegion] = []
    i = 1
    while i <= total:
        if not disagree[i]:
            i += 1
            continue
        start = i
        while i <= total and disagree[i]:
            i += 1
        end = i - 1

        run_types: dict[str, str | None] = {}
        run_label_values: dict[str, str | None] = {}
        for label, arrays in zip(run_labels, runs):
            types_in_region = [arrays.line_type[j] for j in range(start, end + 1)]
            labels_in_region = [arrays.line_label[j] for j in range(start, end + 1)]
            run_types[label] = _dominant_value(types_in_region)
            run_label_values[label] = _dominant_value(labels_in_region)

        has_type_mismatch = any(_line_type_mismatch(scores, j) for j in range(start, end + 1))
        has_low_sim = any(_line_low_similarity(scores, j, threshold) for j in range(start, end + 1))
        if has_type_mismatch and has_low_sim:
            issue_kind = "both"
        elif has_type_mismatch:
            issue_kind = "type_mismatch"
        else:
            issue_kind = "low_similarity"

        disagreeing_models = sorted(
            label
            for label, run_scores in scores.per_run.items()
            if any(
                not run_scores.type_agree[j] or run_scores.cum_sim[j] < threshold
                for j in range(start, end + 1)
            )
        )

        worst_sims = [_line_worst_cum_sim(scores, j) for j in range(start, end + 1)]
        avg_sim = sum(worst_sims) / len(worst_sims)

        regions.append(
            FlagRegion(
                start_line=start,
                end_line=end,
                run_types=run_types,
                run_labels=run_label_values,
                avg_cum_sim=avg_sim,
                issue_kind=issue_kind,
                disagreeing_models=disagreeing_models,
            )
        )

    return regions
