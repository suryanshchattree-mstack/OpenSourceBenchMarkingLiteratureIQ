"""Join ReactionEntry records to R1 step line spans (per model label)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from core.reaction_parsing import ReactionEntry


def normalize_step_label(label: str | None) -> str | None:
    """Casefold, ``_``→space, collapse whitespace."""
    if label is None:
        return None
    text = " ".join(str(label).casefold().replace("_", " ").split())
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _index_r1_steps(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int], Mapping[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    """Index by (section_label, step_index) then (section_label, normalized step_label)."""
    by_index: dict[tuple[str, int], Mapping[str, Any]] = {}
    by_label: dict[tuple[str, str], Mapping[str, Any]] = {}
    for step in steps:
        section = _optional_str(step.get("section_label"))
        if section is None:
            continue
        step_index = _optional_int(step.get("step_index"))
        if step_index is not None:
            key = (section, step_index)
            by_index.setdefault(key, step)
        step_label = normalize_step_label(_optional_str(step.get("step_label")))
        if step_label is not None:
            label_key = (section, step_label)
            by_label.setdefault(label_key, step)
    return by_index, by_label


def _join_one(
    entry: ReactionEntry,
    by_index: Mapping[tuple[str, int], Mapping[str, Any]],
    by_label: Mapping[tuple[str, str], Mapping[str, Any]],
) -> ReactionEntry:
    section = (entry.section_label or "").strip() or None
    if section is None:
        return replace(entry, start_line=None, end_line=None, line_join=None)

    step: Mapping[str, Any] | None = None
    join_kind: str | None = None

    if entry.step_index is not None:
        step = by_index.get((section, entry.step_index))
        if step is not None:
            join_kind = "exact_index"

    if step is None:
        label_key = normalize_step_label(entry.step_label)
        if label_key is not None:
            step = by_label.get((section, label_key))
            if step is not None:
                join_kind = "exact_label"

    if step is None:
        return replace(entry, start_line=None, end_line=None, line_join=None)

    start_line = _optional_int(step.get("start_line"))
    end_line = _optional_int(step.get("end_line"))
    if start_line is None or end_line is None:
        return replace(entry, start_line=None, end_line=None, line_join=None)

    return replace(
        entry,
        start_line=start_line,
        end_line=end_line,
        line_join=join_kind,
    )


def join_reactions_to_r1(
    entries_by_label: Mapping[str, list[ReactionEntry]],
    r1_steps_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[ReactionEntry]]:
    """
    Annotate each reaction with R1 ``start_line`` / ``end_line`` for its model.

    Lookup order per entry: ``(section_label, step_index)`` then
    ``(section_label, normalized step_label)``. Labels with reactions but no R1
    leave all lines null.
    """
    joined: dict[str, list[ReactionEntry]] = {}
    for label, entries in entries_by_label.items():
        steps = r1_steps_by_label.get(label) or ()
        by_index, by_label = _index_r1_steps(steps)
        joined[label] = [_join_one(entry, by_index, by_label) for entry in entries]
    return joined


def line_join_caption(
    entries_by_label: Mapping[str, list[ReactionEntry]],
    r1_labels: set[str] | None = None,
) -> str:
    """Caption: joined vs orphaned counts per model; note missing R1."""
    parts: list[str] = []
    missing_r1: list[str] = []
    for label, entries in entries_by_label.items():
        if not entries:
            parts.append(f"{label}: n/a")
            continue
        if r1_labels is not None and label not in r1_labels:
            missing_r1.append(label)
        joined = sum(1 for e in entries if e.start_line is not None and e.end_line is not None)
        orphaned = len(entries) - joined
        parts.append(f"{label}: {joined}/{len(entries)} joined ({orphaned} orphaned)")
    caption = "R1 line join — " + "; ".join(parts)
    if missing_r1:
        caption += f". Missing R1 for: {', '.join(sorted(missing_r1))}"
    return caption
