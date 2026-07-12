"""Parse R1 (reaction pass 1 — step boundaries) JSON into pre-pass Section objects.

R1 steps share the same geometry as pre-pass sections (document-global start_line/
end_line), so downstream line_arrays / scoring / flagging / visuals can be reused
unchanged after this field mapping.
"""

from __future__ import annotations

import json
from typing import Any

from core.parsing import Section


def _decode_text(raw: str | bytes) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    return raw.strip()


def _section_from_step(step: dict[str, Any]) -> Section:
    """Map an R1 step dict onto the existing Section dataclass."""
    return Section(
        section_index=int(step["step_index"]),
        section_label=str(step.get("step_label") or ""),
        section_type=str(step.get("section_type") or ""),
        start_line=int(step["start_line"]),
        end_line=int(step["end_line"]),
        estimated_tokens=(
            int(step["estimated_tokens"])
            if step.get("estimated_tokens") is not None
            else None
        ),
    )


def _is_step_dict(item: Any) -> bool:
    return isinstance(item, dict) and "step_index" in item and "start_line" in item


def _coerce_steps_list(value: Any, *, source_label: str) -> list[dict[str, Any]]:
    """Normalize one section's R1 payload (string / list / null) to step dicts."""
    if value is None:
        return []

    if isinstance(value, str):
        inner = value.strip()
        if not inner:
            return []
        return _coerce_steps_list(json.loads(inner), source_label=source_label)

    if isinstance(value, list):
        steps: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                steps.extend(_coerce_steps_list(item, source_label=source_label))
            elif _is_step_dict(item):
                steps.append(item)
        return steps

    if _is_step_dict(value):
        return [value]

    raise ValueError(
        f"{source_label} contains an unrecognized per-section R1 entry of type "
        f"{type(value).__name__}."
    )


def _flatten_r1_payload(data: Any, *, source_label: str) -> list[dict[str, Any]]:
    """Normalize supported R1 JSON shapes to a flat list of step dicts.

    Supported shapes:
    - Double-encoded consolidated: ``[\"[...steps...]\", \"[]\", ...]``
      (outer array = one entry per pre-pass section; each entry a JSON-string or
      list of step dicts).
    - Already-decoded consolidated: ``[[...steps...], [], ...]``
    - Flat list of step dicts: ``[{step_index, step_label, ...}, ...]``
    - Nested object wrappers with ``json`` / ``steps`` keys (same coercion spirit
      as pre-pass compare exports).
    """
    if data is None:
        raise ValueError(
            f"{source_label} is null. Upload a completed reaction-pass-1 JSON output."
        )

    if isinstance(data, str):
        inner = data.strip()
        if not inner:
            raise ValueError(f"{source_label} contains an empty JSON string.")
        return _flatten_r1_payload(json.loads(inner), source_label=source_label)

    if isinstance(data, list):
        if not data:
            return []

        # Flat list of step dicts.
        if all(_is_step_dict(item) or not isinstance(item, (dict, list, str)) for item in data):
            flat_steps = [item for item in data if _is_step_dict(item)]
            if flat_steps:
                return flat_steps

        # Consolidated: one entry per section (stringified list, list, or empty).
        flat: list[dict[str, Any]] = []
        for item in data:
            flat.extend(_coerce_steps_list(item, source_label=source_label))
        return flat

    if not isinstance(data, dict):
        raise ValueError(
            f"{source_label} must be a JSON array of R1 steps (or consolidated "
            f"per-section entries), not {type(data).__name__}."
        )

    for side_key in ("baseline", "benchmark"):
        if side_key in data and isinstance(data[side_key], dict):
            side = data[side_key]
            if side.get("found") is False or side.get("json") in (None, ""):
                raise ValueError(
                    f"{source_label} compare export has no data for '{side_key}' "
                    "(found=false). Upload the raw reaction-pass-1-*.json file."
                )
            return _flatten_r1_payload(side.get("json"), source_label=source_label)

    if "json" in data:
        if data.get("found") is False or data.get("json") in (None, ""):
            raise ValueError(
                f"{source_label} compare export has no R1 data (found=false or "
                "json is null). Upload the raw reaction-pass-1-*.json file."
            )
        return _flatten_r1_payload(data.get("json"), source_label=source_label)

    if isinstance(data.get("steps"), list):
        return _flatten_r1_payload(data["steps"], source_label=source_label)

    raise ValueError(
        f"{source_label} has an unrecognized format. Expected a top-level array of "
        "step objects, or the double-encoded consolidated shape "
        "`[\"[{...}]\", \"[]\", ...]` from reaction-pass-1-consolidated.json."
    )


def parse_r1_json(raw: str | bytes, *, source_label: str = "R1 JSON") -> list[Section]:
    """Flatten R1 step JSON into Section objects for the existing pre-pass pipeline.

    Field mapping:
        step_index  -> section_index
        step_label  -> section_label
        section_type -> section_type  (inherited from parent pre-pass section)
        start_line / end_line -> unchanged (document-global)

    Returns steps sorted by ``(start_line, end_line, section_index)`` so document
    order is preserved even when ``step_index`` resets per section.
    """
    text = _decode_text(raw)
    if not text:
        raise ValueError(
            f"{source_label} file is empty. Upload the raw reaction-pass-1 JSON "
            "output (consolidated or flat)."
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:120].replace("\n", " ")
        raise ValueError(
            f"{source_label} is not valid JSON ({e.msg} at line {e.lineno}, column "
            f"{e.colno}). Preview: {preview!r}"
        ) from e

    steps = _flatten_r1_payload(data, source_label=source_label)
    if not steps:
        raise ValueError(
            f"{source_label} parsed successfully but contains no steps. "
            "Check that the file is a completed reaction-pass-1 output."
        )

    sections = [_section_from_step(step) for step in steps]
    sections.sort(key=lambda s: (s.start_line, s.end_line, s.section_index))
    return sections
