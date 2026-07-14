"""Parse M2 (molecule pass 2) JSON into CompoundEntry objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CompoundEntry:
    identifier: str
    identifier_type: str
    aliases: tuple[str, ...]
    resolved: bool
    unresolved_reference: bool
    section_label: str | None
    role: str | None
    smiles: str | None = None
    inchi_key: str | None = None
    molecular_formula: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


def _decode_text(raw: str | bytes) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    return raw.strip()


def _aliases_from_value(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(alias).strip() for alias in value if alias is not None and str(alias).strip())


def _optional_str(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _entry_from_dict(item: dict[str, Any]) -> CompoundEntry | None:
    identifier = item.get("identifier")
    if identifier is None or not str(identifier).strip():
        return None
    return CompoundEntry(
        identifier=str(identifier).strip(),
        identifier_type=str(item.get("identifier_type") or "other"),
        aliases=_aliases_from_value(item.get("aliases")),
        resolved=bool(item.get("resolved", False)),
        unresolved_reference=bool(item.get("unresolved_reference", False)),
        section_label=(
            str(item["section_label"]).strip()
            if item.get("section_label") is not None and str(item.get("section_label")).strip()
            else None
        ),
        role=(
            str(item["role"]).strip()
            if item.get("role") is not None and str(item.get("role")).strip()
            else None
        ),
        smiles=_optional_str(item, "smiles"),
        inchi_key=_optional_str(item, "inchi_key"),
        molecular_formula=_optional_str(item, "molecular_formula"),
        raw=dict(item),
    )


def _flatten_compound_payload(data: Any, *, source_label: str) -> list[dict[str, Any]]:
    """Normalize supported M2 JSON shapes to a flat list of compound dicts."""
    if data is None:
        raise ValueError(
            f"{source_label} is null. Upload a completed molecule-pass-2 JSON output."
        )

    if isinstance(data, str):
        inner = data.strip()
        if not inner:
            raise ValueError(f"{source_label} contains an empty JSON string.")
        return _flatten_compound_payload(json.loads(inner), source_label=source_label)

    if isinstance(data, list):
        flat: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and "identifier" in item:
                flat.append(item)
            elif isinstance(item, dict) and isinstance(item.get("compounds"), list):
                for compound in item["compounds"]:
                    if isinstance(compound, dict):
                        flat.append(compound)
            elif isinstance(item, dict) and isinstance(item.get("sections"), list):
                for section in item["sections"]:
                    if not isinstance(section, dict):
                        continue
                    section_compounds = section.get("compounds")
                    if isinstance(section_compounds, list):
                        for compound in section_compounds:
                            if isinstance(compound, dict):
                                flat.append(compound)
        if flat:
            return flat
        if all(isinstance(item, dict) for item in data):
            return [item for item in data if isinstance(item, dict)]
        raise ValueError(
            f"{source_label} is a JSON array but contains no compound entries with an "
            "'identifier' field."
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"{source_label} must be a JSON array of compounds, not {type(data).__name__}."
        )

    for side_key in ("baseline", "benchmark", "claude"):
        if side_key in data and isinstance(data[side_key], dict):
            side = data[side_key]
            if side.get("found") is False or side.get("json") in (None, ""):
                raise ValueError(
                    f"{source_label} compare export has no data for '{side_key}' "
                    "(found=false). Upload the raw molecule-pass-2 JSON file."
                )
            return _flatten_compound_payload(side.get("json"), source_label=source_label)

    if "json" in data:
        if data.get("found") is False or data.get("json") in (None, ""):
            raise ValueError(
                f"{source_label} compare export has no compound data (found=false or "
                "json is null). Upload the raw molecule-pass-2 JSON file."
            )
        return _flatten_compound_payload(data.get("json"), source_label=source_label)

    if isinstance(data.get("compounds"), list):
        return [item for item in data["compounds"] if isinstance(item, dict)]

    raise ValueError(
        f"{source_label} has an unrecognized format. Expected a top-level JSON array of "
        "compound objects with an 'identifier' field, or the inner `json` field from a "
        "benchmark compare export."
    )


def parse_compounds_json(raw: str | bytes, *, source_label: str = "M2 JSON") -> list[CompoundEntry]:
    """Parse a molecule pass 2 JSON file into a flat list of CompoundEntry objects."""
    text = _decode_text(raw)
    if not text:
        raise ValueError(
            f"{source_label} file is empty. Upload the raw molecule-pass-2-*.json output."
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:120].replace("\n", " ")
        raise ValueError(
            f"{source_label} is not valid JSON ({e.msg} at line {e.lineno}, column {e.colno}). "
            f"Preview: {preview!r}"
        ) from e

    compound_dicts = _flatten_compound_payload(data, source_label=source_label)
    if not compound_dicts:
        raise ValueError(f"{source_label} contains no compound entries.")

    entries: list[CompoundEntry] = []
    skipped = 0
    for item in compound_dicts:
        if not isinstance(item, dict):
            skipped += 1
            continue
        entry = _entry_from_dict(item)
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)

    if not entries:
        raise ValueError(
            f"{source_label} parsed successfully but contains no valid compound entries "
            f"(skipped {skipped} items without an identifier)."
        )

    return entries
