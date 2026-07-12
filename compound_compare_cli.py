#!/usr/bin/env python3
"""One-off CLI to deterministically diff two M2 compound outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.compound_matching import diff_compounds
from core.compound_parsing import parse_compounds_json
from core.compound_report import build_diff_json_payload


def _parse_side(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Each side must be LABEL:path/to/file.json (e.g. Claude:claude-m2.json)"
        )
    label, path_str = value.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Side label cannot be empty")
    path = Path(path_str)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return label, path


def _format_aliases(aliases: tuple[str, ...]) -> str:
    if not aliases:
        return ""
    return ", ".join(aliases)


def _print_entry_table(title: str, entries, label: str) -> None:
    print(f"\n=== {title} ({len(entries)}) ===")
    if not entries:
        print("  (none)")
        return
    for index, entry in enumerate(entries, 1):
        print(f"  #{index:3d} [{entry.identifier_type:16s}] {entry.identifier}")
        if entry.aliases:
            print(f"       aliases: {_format_aliases(entry.aliases)}")
        if entry.section_label:
            print(f"       section: {entry.section_label}")
        if entry.unresolved_reference:
            print("       unresolved_reference: true")


def _print_matched_pairs(pairs: list[tuple], claude_label: str, benchmark_label: str) -> None:
    print(f"\n=== MATCHED PAIRS ({len(pairs)}) ===")
    if not pairs:
        print("  (none)")
        return
    for index, (claude_entry, benchmark_entry) in enumerate(pairs, 1):
        print(
            f"  #{index:3d} {claude_label}: {claude_entry.identifier} "
            f"<-> {benchmark_label}: {benchmark_entry.identifier}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically diff two M2 compound JSON outputs (no PubChem)",
    )
    parser.add_argument(
        "--claude",
        required=True,
        type=_parse_side,
        help="LABEL:path/to/claude-m2.json (baseline)",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=_parse_side,
        help="LABEL:path/to/benchmark-m2.json",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write full diff result as JSON",
    )
    args = parser.parse_args()

    claude_label, claude_path = args.claude
    benchmark_label, benchmark_path = args.benchmark

    claude_compounds = parse_compounds_json(
        claude_path.read_bytes(),
        source_label=f"{claude_label} ({claude_path.name})",
    )
    benchmark_compounds = parse_compounds_json(
        benchmark_path.read_bytes(),
        source_label=f"{benchmark_label} ({benchmark_path.name})",
    )

    result = diff_compounds(claude_compounds, benchmark_compounds)

    print("\n=== COMPOUND DIFF SUMMARY ===")
    print(f"{claude_label:20s} raw entries:      {result.raw_claude_count}")
    print(f"{claude_label:20s} deduped unique:   {result.deduped_claude_count}")
    print(f"{benchmark_label:20s} raw entries:      {result.raw_benchmark_count}")
    print(f"{benchmark_label:20s} deduped unique:   {result.deduped_benchmark_count}")
    print(f"{'common (matched)':20s}                  {result.common}")
    print(f"{(claude_label + ' only'):20s}                  {result.claude_only}")
    print(f"{(benchmark_label + ' only'):20s}                  {result.benchmark_only}")

    if result.deduped_claude_count > 0:
        recall = result.common / result.deduped_claude_count
        print(f"{'recall vs ' + claude_label:20s}                  {recall:.1%}")
    if result.deduped_benchmark_count > 0:
        precision = result.common / result.deduped_benchmark_count
        print(f"{'precision vs ' + claude_label:20s}                  {precision:.1%}")

    _print_matched_pairs(result.matched_pairs, claude_label, benchmark_label)
    _print_entry_table(f"{claude_label.upper()} ONLY", result.claude_only_entries, claude_label)
    _print_entry_table(
        f"{benchmark_label.upper()} ONLY",
        result.benchmark_only_entries,
        benchmark_label,
    )

    if args.json_out:
        payload = build_diff_json_payload(result, claude_label, benchmark_label)
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {out_path}")


if __name__ == "__main__":
    main()
