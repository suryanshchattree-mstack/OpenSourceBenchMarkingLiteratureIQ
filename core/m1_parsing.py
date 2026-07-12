"""Parse M1 (molecule pass 1) JSON into M1CompoundEntry objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

QUANTITY_FIELDS = ("mass_g", "volume_ml", "mmol", "equivalents", "yield_pct")


@dataclass(frozen=True)
class M1CompoundEntry:
    identifier: str
    identifier_type: str
    aliases: tuple[str, ...]
    role: str | None
    is_section_product: bool
    commercially_available: bool
    quantity: dict[str, float | None]
    ms_mz: float | None
    notes: str | None
    section_label: str | None


def _decode_text(raw: str | bytes) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    return raw.strip()


def _aliases_from_value(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(alias).strip() for alias in value if alias is not None and str(alias).strip())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantity_from_value(value: Any) -> dict[str, float | None]:
    raw = value if isinstance(value, dict) else {}
    return {field: _optional_float(raw.get(field)) for field in QUANTITY_FIELDS}


def _entry_from_dict(item: dict[str, Any]) -> M1CompoundEntry | None:
    identifier = item.get("identifier")
    if identifier is None or not str(identifier).strip():
        return None
    return M1CompoundEntry(
        identifier=str(identifier).strip(),
        identifier_type=str(item.get("identifier_type") or "other"),
        aliases=_aliases_from_value(item.get("aliases")),
        role=_optional_str(item.get("role")),
        is_section_product=bool(item.get("is_section_product", False)),
        commercially_available=bool(item.get("commercially_available", False)),
        quantity=_quantity_from_value(item.get("quantity")),
        ms_mz=_optional_float(item.get("ms_mz")),
        notes=_optional_str(item.get("notes")),
        section_label=_optional_str(item.get("section_label")),
    )


def _flatten_m1_payload(data: Any, *, source_label: str) -> list[dict[str, Any]]:
    """Normalize supported M1 JSON shapes to a flat list of compound dicts."""
    if data is None:
        raise ValueError(
            f"{source_label} is null. Upload a completed molecule-pass-1 JSON output."
        )

    if isinstance(data, str):
        inner = data.strip()
        if not inner:
            raise ValueError(f"{source_label} contains an empty JSON string.")
        return _flatten_m1_payload(json.loads(inner), source_label=source_label)

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
                    "(found=false). Upload the raw molecule-pass-1 JSON file."
                )
            return _flatten_m1_payload(side.get("json"), source_label=source_label)

    if "json" in data:
        if data.get("found") is False or data.get("json") in (None, ""):
            raise ValueError(
                f"{source_label} compare export has no compound data (found=false or "
                "json is null). Upload the raw molecule-pass-1 JSON file."
            )
        return _flatten_m1_payload(data.get("json"), source_label=source_label)

    if isinstance(data.get("compounds"), list):
        return [item for item in data["compounds"] if isinstance(item, dict)]

    if isinstance(data.get("sections"), list):
        flat = []
        for section in data["sections"]:
            if not isinstance(section, dict):
                continue
            section_compounds = section.get("compounds")
            if isinstance(section_compounds, list):
                for compound in section_compounds:
                    if isinstance(compound, dict):
                        flat.append(compound)
        if flat:
            return flat

    raise ValueError(
        f"{source_label} has an unrecognized format. Expected a top-level JSON array of "
        "compound objects with an 'identifier' field, nested sections[].compounds[], "
        "or the inner `json` field from a benchmark compare export."
    )


def parse_m1_json(raw: str | bytes, *, source_label: str = "M1 JSON") -> list[M1CompoundEntry]:
    """Parse a molecule pass 1 JSON file into a flat list of M1CompoundEntry objects."""
    text = _decode_text(raw)
    if not text:
        raise ValueError(
            f"{source_label} file is empty. Upload the raw molecule-pass-1-*.json output."
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:120].replace("\n", " ")
        raise ValueError(
            f"{source_label} is not valid JSON ({e.msg} at line {e.lineno}, column {e.colno}). "
            f"Preview: {preview!r}"
        ) from e

    compound_dicts = _flatten_m1_payload(data, source_label=source_label)
    if not compound_dicts:
        raise ValueError(f"{source_label} contains no compound entries.")

    entries: list[M1CompoundEntry] = []
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
