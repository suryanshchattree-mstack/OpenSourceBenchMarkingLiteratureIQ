"""Build per-line type and label arrays from section lists."""

from __future__ import annotations

from dataclasses import dataclass

from core.parsing import Section


@dataclass
class LineArrays:
    """Per-line arrays indexed 1..total_lines (index 0 unused)."""

    line_type: list[str | None]
    line_label: list[str | None]

    @property
    def total_lines(self) -> int:
        return len(self.line_type) - 1


def build_line_arrays(sections: list[Section], total_lines: int) -> LineArrays:
    """
    Paint each line with the section_type and section_label of the section covering it.
    Later sections overwrite earlier ones if ranges overlap (shouldn't happen in well-formed data).
    """
    if total_lines < 1:
        raise ValueError("total_lines must be >= 1")

    line_type: list[str | None] = [None] * (total_lines + 1)
    line_label: list[str | None] = [None] * (total_lines + 1)

    for sec in sections:
        start = max(1, sec.start_line)
        end = min(total_lines, sec.end_line)
        for i in range(start, end + 1):
            line_type[i] = sec.section_type
            line_label[i] = sec.section_label

    return LineArrays(line_type=line_type, line_label=line_label)
