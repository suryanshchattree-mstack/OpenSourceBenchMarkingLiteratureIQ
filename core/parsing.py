"""Parse pre-pass JSON and enriched markdown inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Section:
    section_index: int
    section_label: str
    section_type: str
    start_line: int
    end_line: int
    estimated_tokens: int | None = None


def _decode_text(raw: str | bytes) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    return raw.strip()


def _section_from_dict(sec: dict[str, Any]) -> Section:
    return Section(
        section_index=int(sec["section_index"]),
        section_label=str(sec.get("section_label") or ""),
        section_type=str(sec.get("section_type") or ""),
        start_line=int(sec["start_line"]),
        end_line=int(sec["end_line"]),
        estimated_tokens=(
            int(sec["estimated_tokens"])
            if sec.get("estimated_tokens") is not None
            else None
        ),
    )


def _groups_from_flat_sections(sections: list[Any]) -> list[dict[str, Any]]:
    by_group: dict[int, list[dict[str, Any]]] = {}
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        group_index = int(sec.get("group_index", 0))
        by_group.setdefault(group_index, []).append(sec)
    return [
        {"group_index": group_index, "sections": group_sections}
        for group_index, group_sections in sorted(by_group.items())
    ]


def _coerce_prepass_payload(data: Any, *, source_label: str) -> list[dict[str, Any]]:
    """Normalize supported pre-pass JSON shapes to a group list."""
    if data is None:
        raise ValueError(
            f"{source_label} is null. The benchmark pre-pass may not exist yet — "
            "run extraction first, then download the raw pre-pass-*.json file."
        )

    if isinstance(data, str):
        inner = data.strip()
        if not inner:
            raise ValueError(f"{source_label} contains an empty JSON string.")
        return _coerce_prepass_payload(json.loads(inner), source_label=source_label)

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError(
            f"{source_label} must be a JSON array of groups, not {type(data).__name__}."
        )

    for side_key in ("baseline", "benchmark"):
        if side_key in data and isinstance(data[side_key], dict):
            side = data[side_key]
            if side.get("found") is False or side.get("json") in (None, ""):
                raise ValueError(
                    f"{source_label} compare export has no data for '{side_key}' "
                    "(found=false). Run benchmark extraction or upload the raw "
                    "pre-pass-*.json from blob/local fallback storage."
                )
            return _coerce_prepass_payload(side.get("json"), source_label=source_label)

    if "json" in data:
        if data.get("found") is False or data.get("json") in (None, ""):
            raise ValueError(
                f"{source_label} compare export has no pre-pass data (found=false or "
                "json is null). Run benchmark extraction or use the raw pre-pass-*.json file."
            )
        return _coerce_prepass_payload(data.get("json"), source_label=source_label)

    if isinstance(data.get("sections"), list):
        return _groups_from_flat_sections(data["sections"])

    if isinstance(data.get("groups"), list):
        return data["groups"]

    raise ValueError(
        f"{source_label} has an unrecognized format. Expected a top-level array of groups "
        "like `[{\"group_index\": 0, \"sections\": [...]}]`, or the inner `json` field "
        "from the literatureiq benchmark compare API."
    )


def parse_prepass_json(raw: str | bytes, *, source_label: str = "Pre-pass JSON") -> list[Section]:
    """Flatten groups[].sections[] from a pre-pass JSON array into Section objects."""
    text = _decode_text(raw)
    if not text:
        raise ValueError(
            f"{source_label} file is empty. Upload the raw pre-pass-*.json output — "
            "not the benchmark dashboard \"Not found\" placeholder text."
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:120].replace("\n", " ")
        raise ValueError(
            f"{source_label} is not valid JSON ({e.msg} at line {e.lineno}, column {e.colno}). "
            "Common causes: empty file, wrong file uploaded, or text copied from the "
            f"benchmark dashboard instead of the JSON blob. Preview: {preview!r}"
        ) from e

    groups = _coerce_prepass_payload(data, source_label=source_label)
    if not groups:
        raise ValueError(f"{source_label} contains no section groups.")

    sections: list[Section] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_sections = group.get("sections")
        if not group_sections:
            continue
        for sec in group_sections:
            if not isinstance(sec, dict):
                continue
            sections.append(_section_from_dict(sec))

    if not sections:
        raise ValueError(
            f"{source_label} parsed successfully but contains no sections. "
            "Check that the file is a completed pre-pass output."
        )

    sections.sort(key=lambda s: s.section_index)
    return sections


def total_lines_from_markdown(raw: str | bytes) -> int:
    """Return line count of enriched markdown (1-indexed lines map to array indices 1..N)."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    # Preserve trailing empty line semantics consistent with split("\n", -1) in Java pipeline.
    return len(raw.split("\n", -1))


def type_distribution(sections: list[Section]) -> dict[str, int]:
    """Count sections per section_type."""
    counts: dict[str, int] = {}
    for sec in sections:
        counts[sec.section_type] = counts.get(sec.section_type, 0) + 1
    return counts
