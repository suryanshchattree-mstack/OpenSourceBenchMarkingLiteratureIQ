"""Compute per-line and summary scores vs a reference run (e.g. Claude)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.line_arrays import LineArrays


@dataclass
class RunVsReferenceScores:
    label: str
    type_agree: list[bool]
    sim_model_1: list[float]
    sim_model_2: list[float]
    sim_model_3: list[float]
    cum_sim: list[float]
    score_type_agreement: float
    score_model_1: float
    score_model_2: float
    score_model_3: float
    score_cumulative: float


@dataclass
class MultiRunScores:
    reference_label: str
    reference_index: int
    benchmark_labels: list[str]
    per_run: dict[str, RunVsReferenceScores]
    total_lines: int
    run_labels: list[str]


def resolve_reference_index(run_labels: list[str], reference_label: str | None = None) -> int:
    if reference_label is not None:
        try:
            return run_labels.index(reference_label)
        except ValueError as exc:
            raise ValueError(
                f"Reference label '{reference_label}' not found among runs: {run_labels}"
            ) from exc
    if "Claude" in run_labels:
        return run_labels.index("Claude")
    return 0


def _types_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return a == b


def _mean_line_values(values: list[float], total_lines: int) -> float:
    if total_lines <= 0:
        return 0.0
    return float(np.mean([values[i] for i in range(1, total_lines + 1)]))


def compute_multi_run_scores(
    runs: list[LineArrays],
    run_labels: list[str],
    reference_index: int,
    sims_per_benchmark: list[list[list[float]]],
) -> MultiRunScores:
    if len(runs) < 2:
        raise ValueError("At least two runs are required for comparison")
    if len(run_labels) != len(runs):
        raise ValueError("run_labels must match the number of runs")
    if reference_index < 0 or reference_index >= len(runs):
        raise ValueError("reference_index is out of range")

    total_lines = runs[0].total_lines
    for run in runs[1:]:
        if run.total_lines != total_lines:
            raise ValueError("All runs must have the same total_lines")

    reference = runs[reference_index]
    benchmark_indices = [i for i in range(len(runs)) if i != reference_index]
    if len(benchmark_indices) != len(sims_per_benchmark):
        raise ValueError("sims_per_benchmark must have one entry per non-reference run")

    per_run: dict[str, RunVsReferenceScores] = {}
    for bench_idx, sims in zip(benchmark_indices, sims_per_benchmark):
        label = run_labels[bench_idx]
        benchmark = runs[bench_idx]

        type_agree: list[bool] = [False] * (total_lines + 1)
        cum_sim: list[float] = [0.0] * (total_lines + 1)
        for line in range(1, total_lines + 1):
            type_agree[line] = _types_match(reference.line_type[line], benchmark.line_type[line])
            cum_sim[line] = (sims[0][line] + sims[1][line] + sims[2][line]) / 3.0

        agree_count = sum(1 for line in range(1, total_lines + 1) if type_agree[line])
        per_run[label] = RunVsReferenceScores(
            label=label,
            type_agree=type_agree,
            sim_model_1=sims[0],
            sim_model_2=sims[1],
            sim_model_3=sims[2],
            cum_sim=cum_sim,
            score_type_agreement=agree_count / total_lines if total_lines else 0.0,
            score_model_1=_mean_line_values(sims[0], total_lines),
            score_model_2=_mean_line_values(sims[1], total_lines),
            score_model_3=_mean_line_values(sims[2], total_lines),
            score_cumulative=_mean_line_values(cum_sim, total_lines),
        )

    return MultiRunScores(
        reference_label=run_labels[reference_index],
        reference_index=reference_index,
        benchmark_labels=[run_labels[i] for i in benchmark_indices],
        per_run=per_run,
        total_lines=total_lines,
        run_labels=list(run_labels),
    )
