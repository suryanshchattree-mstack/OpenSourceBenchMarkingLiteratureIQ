"""Parse enriched ReactionRecord JSON into ReactionEntry objects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ReactionEntry:
    product_name: str | None
    product_smiles: str | None
    reactant_names: tuple[str, ...]
    reactant_smiles: tuple[str, ...]
    product_yield_pct: float | None
    procedure_text: str | None
    procedure_vector: tuple[float, ...] | None
    temperature_c: float | None
    room_temperature: bool | None
    time_h: float | None
    atmosphere: str | None
    reaction_class: str | None
    non_synthetic: bool
    section_label: str | None
    step_label: str | None
    reaction_id: str | None = None
    canonical_rxn: str | None = None
    reaction_vector: tuple[float, ...] | None = None
    step_index: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    line_join: str | None = None
    compound_smiles: frozenset[str] = field(default_factory=frozenset)
    raw: Mapping[str, Any] = field(default_factory=dict)


def _decode_text(raw: str | bytes) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    return raw.strip()


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


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            items.append(text)
    return tuple(items)


def _float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        return None
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return tuple(out) if out else None


def _conditions_block(item: dict[str, Any]) -> dict[str, Any]:
    conditions = item.get("conditions")
    return conditions if isinstance(conditions, dict) else {}


def _temperature_c(item: dict[str, Any], conditions: dict[str, Any]) -> float | None:
    top = _optional_float(item.get("temperature_c"))
    if top is not None:
        return top
    temperature = conditions.get("temperature")
    if isinstance(temperature, dict):
        return _optional_float(temperature.get("value_c"))
    return _optional_float(conditions.get("temperature_c"))


def _room_temperature(item: dict[str, Any], conditions: dict[str, Any]) -> bool | None:
    if "room_temperature" in item:
        return _optional_bool(item.get("room_temperature"))
    temperature = conditions.get("temperature")
    if isinstance(temperature, dict):
        temp_type = _optional_str(temperature.get("type"))
        if temp_type is not None:
            return temp_type == "room_temperature"
        if "room_temperature" in temperature:
            return _optional_bool(temperature.get("room_temperature"))
    if "room_temperature" in conditions:
        return _optional_bool(conditions.get("room_temperature"))
    return None


def _time_h(item: dict[str, Any], conditions: dict[str, Any]) -> float | None:
    top = _optional_float(item.get("time_h"))
    if top is not None:
        return top
    return _optional_float(conditions.get("time_h"))


def _atmosphere(item: dict[str, Any], conditions: dict[str, Any]) -> str | None:
    top = _optional_str(item.get("atmosphere"))
    if top is not None:
        return top
    return _optional_str(conditions.get("atmosphere"))


def _entry_from_dict(item: dict[str, Any]) -> ReactionEntry:
    conditions = _conditions_block(item)
    return ReactionEntry(
        product_name=_optional_str(item.get("product_name")),
        product_smiles=_optional_str(item.get("product_smiles")),
        reactant_names=_string_tuple(item.get("reactant_names")),
        reactant_smiles=_string_tuple(item.get("reactant_smiles")),
        product_yield_pct=_optional_float(item.get("product_yield_pct")),
        procedure_text=_optional_str(item.get("procedure_text")),
        procedure_vector=_float_tuple(item.get("procedure_vector")),
        temperature_c=_temperature_c(item, conditions),
        room_temperature=_room_temperature(item, conditions),
        time_h=_time_h(item, conditions),
        atmosphere=_atmosphere(item, conditions),
        reaction_class=_optional_str(item.get("reaction_class")),
        non_synthetic=bool(item.get("non_synthetic", False)),
        section_label=_optional_str(item.get("section_label")),
        step_label=_optional_str(item.get("step_label")),
        reaction_id=_optional_str(item.get("reaction_id")),
        canonical_rxn=_optional_str(
            item.get("canonical_rxn") or item.get("canonical_rxn_smiles")
        ),
        reaction_vector=_float_tuple(item.get("reaction_vector")),
        step_index=_optional_int(item.get("step_index")),
        start_line=None,
        end_line=None,
        line_join=None,
        raw=dict(item),
    )


def _flatten_reaction_payload(data: Any, *, source_label: str) -> list[dict[str, Any]]:
    """Normalize supported reactions JSON shapes to a flat list of reaction dicts."""
    if data is None:
        raise ValueError(
            f"{source_label} is null. Upload a completed enriched reactions JSON output."
        )

    if isinstance(data, str):
        inner = data.strip()
        if not inner:
            raise ValueError(f"{source_label} contains an empty JSON string.")
        return _flatten_reaction_payload(json.loads(inner), source_label=source_label)

    if isinstance(data, list):
        flat: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and (
                "product_name" in item
                or "product_smiles" in item
                or "section_label" in item
                or "reactant_names" in item
                or "procedure_text" in item
            ):
                flat.append(item)
            elif isinstance(item, dict) and isinstance(item.get("reactions"), list):
                for reaction in item["reactions"]:
                    if isinstance(reaction, dict):
                        flat.append(reaction)
            elif isinstance(item, str):
                try:
                    decoded = json.loads(item)
                except json.JSONDecodeError:
                    continue
                flat.extend(_flatten_reaction_payload(decoded, source_label=source_label))
        if flat:
            return flat
        if all(isinstance(item, dict) for item in data):
            return [item for item in data if isinstance(item, dict)]
        raise ValueError(
            f"{source_label} is a JSON array but contains no reaction entries."
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"{source_label} must be a JSON array of reactions, not {type(data).__name__}."
        )

    for side_key in ("baseline", "benchmark", "claude", "candidate"):
        if side_key in data and isinstance(data[side_key], dict):
            side = data[side_key]
            if side.get("found") is False or side.get("json") in (None, ""):
                raise ValueError(
                    f"{source_label} compare export has no data for '{side_key}' "
                    "(found=false). Upload the raw reactions JSON file."
                )
            return _flatten_reaction_payload(side.get("json"), source_label=source_label)

    if "json" in data:
        if data.get("found") is False or data.get("json") in (None, ""):
            raise ValueError(
                f"{source_label} compare export has no reaction data (found=false or "
                "json is null). Upload the raw reactions JSON file."
            )
        return _flatten_reaction_payload(data.get("json"), source_label=source_label)

    if isinstance(data.get("reactions"), list):
        return [item for item in data["reactions"] if isinstance(item, dict)]

    raise ValueError(
        f"{source_label} has an unrecognized format. Expected a top-level JSON array of "
        "enriched ReactionRecord objects."
    )


def parse_reactions_json(
    raw: str | bytes,
    *,
    source_label: str = "Reactions JSON",
) -> list[ReactionEntry]:
    """Parse an enriched reactions JSON file into ReactionEntry objects."""
    text = _decode_text(raw)
    if not text:
        raise ValueError(
            f"{source_label} file is empty. Upload the raw reactions JSON output."
        )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:120].replace("\n", " ")
        raise ValueError(
            f"{source_label} is not valid JSON ({e.msg} at line {e.lineno}, column {e.colno}). "
            f"Preview: {preview!r}"
        ) from e

    reaction_dicts = _flatten_reaction_payload(data, source_label=source_label)
    if not reaction_dicts:
        raise ValueError(f"{source_label} contains no reaction entries.")

    return [_entry_from_dict(item) for item in reaction_dicts if isinstance(item, dict)]
