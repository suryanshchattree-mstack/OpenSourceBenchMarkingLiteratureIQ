"""Shared data structures for multi-run pre-pass comparison."""

from __future__ import annotations

from dataclasses import dataclass

from core.line_arrays import LineArrays, build_line_arrays
from core.parsing import Section, parse_prepass_json, type_distribution


@dataclass(frozen=True)
class PrepassRun:
    """One labeled pre-pass output aligned to a shared markdown line count."""

    label: str
    filename: str
    sections: list[Section]
    arrays: LineArrays
    type_counts: dict[str, int]


def build_prepass_run(
    label: str,
    filename: str,
    raw: bytes,
    total_lines: int,
) -> PrepassRun:
    sections = parse_prepass_json(
        raw,
        source_label=f"{label} ({filename})",
    )
    arrays = build_line_arrays(sections, total_lines)
    return PrepassRun(
        label=label,
        filename=filename,
        sections=sections,
        arrays=arrays,
        type_counts=type_distribution(sections),
    )
