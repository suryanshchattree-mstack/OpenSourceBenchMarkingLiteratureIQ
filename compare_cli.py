#!/usr/bin/env python3
"""One-off CLI to run pre-pass comparison on local files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.embeddings import MODEL_NAMES, compute_vs_reference_similarities, load_models, resolve_device
from core.flagging import build_disagree_mask, collapse_flag_regions
from core.models import build_prepass_run
from core.parsing import total_lines_from_markdown
from core.scoring import compute_multi_run_scores, resolve_reference_index


def _parse_run(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Each --run must be LABEL:path/to/file.json (e.g. Claude:baseline.json)"
        )
    label, path_str = value.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    path = Path(path_str)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple labeled pre-pass JSON outputs against a reference",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        help="LABEL:path/to/prepass.json (repeat for each output, minimum 2)",
    )
    parser.add_argument("--markdown", required=True, help="Enriched markdown file")
    parser.add_argument(
        "--reference",
        help="Reference label to compare against (defaults to Claude if present, else first run)",
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    if len(args.run) < 2:
        parser.error("Provide at least two --run arguments")

    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        parser.error("Each --run label must be unique")

    markdown_bytes = Path(args.markdown).read_bytes()
    total_lines = total_lines_from_markdown(markdown_bytes)
    runs = [
        build_prepass_run(label, path.name, path.read_bytes(), total_lines)
        for label, path in args.run
    ]

    reference_index = resolve_reference_index(labels, args.reference)
    reference_label = labels[reference_index]
    reference_arrays = runs[reference_index].arrays
    benchmark_arrays = [run.arrays for i, run in enumerate(runs) if i != reference_index]

    device = resolve_device()
    print(f"Loading embedding models on {device}...")
    models = load_models(device)
    sims_per_benchmark = compute_vs_reference_similarities(
        models,
        reference_arrays,
        benchmark_arrays,
    )
    scores = compute_multi_run_scores(
        [run.arrays for run in runs],
        labels,
        reference_index,
        sims_per_benchmark,
    )

    disagree = build_disagree_mask(scores, args.threshold)
    flags = collapse_flag_regions(
        disagree,
        [run.arrays for run in runs],
        labels,
        scores,
        args.threshold,
    )

    print("\n=== SCORES ===")
    print(f"Total lines (markdown):     {total_lines}")
    print(f"Reference:                  {reference_label}")
    for run in runs:
        role = "reference" if run.label == reference_label else "benchmark"
        print(f"{run.label:20s} [{role:9s}] sections: {len(run.sections):3d}  file: {run.filename}")

    print(f"\n=== SCORES VS {reference_label.upper()} ===")
    for label in scores.benchmark_labels:
        run_scores = scores.per_run[label]
        print(f"\n{label}:")
        print(f"  Type agreement:           {run_scores.score_type_agreement:.2%}")
        print(f"  Model 1 ({MODEL_NAMES[0]}): {run_scores.score_model_1:.4f}")
        print(f"  Model 2 ({MODEL_NAMES[1]}): {run_scores.score_model_2:.4f}")
        print(f"  Model 3 ({MODEL_NAMES[2]}): {run_scores.score_model_3:.4f}")
        print(f"  Cumulative similarity:    {run_scores.score_cumulative:.4f}")

    print("\n=== TYPE DISTRIBUTION ===")
    all_types = sorted({t for run in runs for t in run.type_counts})
    for section_type in all_types:
        counts = "  ".join(f"{run.label}={run.type_counts.get(section_type, 0):3d}" for run in runs)
        print(f"  {section_type:35s}  {counts}")

    print(f"\n=== FLAGGED REGIONS VS {reference_label.upper()} (threshold={args.threshold}) — {len(flags)} total ===")
    for index, region in enumerate(flags, 1):
        print(
            f"  #{index:2d} lines {region.start_line:4d}-{region.end_line:4d} "
            f"({region.end_line - region.start_line + 1:3d} lines) "
            f"[{region.issue_kind:15s}] worst_avg_sim={region.avg_cum_sim:.3f} "
            f"models={', '.join(region.disagreeing_models)}"
        )
        for label in labels:
            print(
                f"       {label:20s} {region.run_types.get(label)} / {region.run_labels.get(label)}"
            )


if __name__ == "__main__":
    main()
