"""Ensure product_smiles + role-filtered compound_smiles sets on ReactionEntry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import replace
from typing import Any, Optional

from core.reaction_parsing import ReactionEntry
from core.smiles_resolve import (
    canonicalize_smiles_rdkit,
    looks_like_markush,
    resolve_name_to_smiles,
)

ResolveFn = Callable[[str], Optional[str]]

# Roles included in compound-set Jaccard clustering.
COMPOUND_SET_ROLES = frozenset(
    {
        "product",
        "reactant",
        "reagent",
        "catalyst",
        "ligand",
        "base",
        "acid",
        "oxidant",
        "reductant",
    }
)

# Explicit exclusions (solvents / workup noise — kept for documentation / filters).
COMPOUND_SET_EXCLUDED_ROLES = frozenset(
    {
        "solvent",
        "drying_agent",
        "additive",
        "by_product",
        "other",
    }
)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _compound_smiles_name(item: Mapping[str, Any]) -> tuple[str | None, str | None]:
    smiles = (
        _optional_str(item.get("smiles"))
        or _optional_str(item.get("canonical_smiles"))
        or _optional_str(item.get("product_smiles"))
    )
    name = (
        _optional_str(item.get("identifier"))
        or _optional_str(item.get("name"))
        or _optional_str(item.get("product_name"))
    )
    return smiles, name


def _product_from_compounds(raw: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(smiles, name)`` from the first ``is_product`` compound in ``raw``."""
    compounds = raw.get("compounds")
    if not isinstance(compounds, list):
        return None, None
    for item in compounds:
        if not isinstance(item, dict):
            continue
        if not item.get("is_product"):
            continue
        return _compound_smiles_name(item)
    return None, None


def _role_eligible(item: Mapping[str, Any]) -> bool:
    """True when compound should contribute to the role-filtered SMILES set."""
    role = _optional_str(item.get("role"))
    role_norm = role.lower() if role else None
    if role_norm in COMPOUND_SET_EXCLUDED_ROLES:
        return False
    if role_norm in COMPOUND_SET_ROLES:
        return True
    # Product compounds even when role is missing.
    return bool(item.get("is_product"))


def _collect_compound_candidates(
    entry: ReactionEntry,
) -> list[tuple[str | None, str | None]]:
    """Top-level product/reactants plus role-eligible ``raw.compounds``."""
    pairs: list[tuple[str | None, str | None]] = []
    pairs.append((entry.product_smiles, entry.product_name))

    n_react = max(len(entry.reactant_smiles), len(entry.reactant_names))
    for index in range(n_react):
        smiles = entry.reactant_smiles[index] if index < len(entry.reactant_smiles) else None
        name = entry.reactant_names[index] if index < len(entry.reactant_names) else None
        pairs.append((smiles, name))

    compounds = entry.raw.get("compounds") if entry.raw else None
    if isinstance(compounds, list):
        for item in compounds:
            if not isinstance(item, dict):
                continue
            if not _role_eligible(item):
                continue
            pairs.append(_compound_smiles_name(item))
    return pairs


def _resolve_to_canonical_smiles(
    smiles: str | None,
    name: str | None,
    *,
    resolve: ResolveFn | None,
    cache: MutableMapping[str, str | None],
) -> str | None:
    """Canonicalize ``smiles`` if present; else resolve ``name`` → SMILES."""
    canon = canonicalize_smiles_rdkit(smiles)
    if canon is not None:
        return canon
    if not name or not str(name).strip():
        return None
    key = str(name).strip()
    if looks_like_markush(key):
        if key not in cache:
            cache[key] = None
        return None
    if resolve is not None:
        if key in cache:
            return cache[key]
        try:
            resolved = resolve(key)
        except Exception:
            resolved = None
        canon = canonicalize_smiles_rdkit(resolved) if resolved else None
        cache[key] = canon
        return canon
    return resolve_name_to_smiles(key, cache=cache)


# Back-compat alias used by older tests / imports.
_resolve_product_smiles = _resolve_to_canonical_smiles


def ensure_product_canonical_smiles(
    entries: Sequence[ReactionEntry],
    *,
    resolve_fn: ResolveFn | None = None,
    cache: MutableMapping[str, str | None] | None = None,
) -> list[ReactionEntry]:
    """
    Return copies with RDKit-canonical ``product_smiles`` and ``compound_smiles``.

    Product ladder:
      1. Existing ``product_smiles`` → RDKit canonicalize
      2. ``product_name`` → PubChem → OPSIN → RDKit (or injected ``resolve_fn``)
      3. First ``raw["compounds"]`` with ``is_product`` → SMILES / identifier
      4. Otherwise leave ``product_smiles`` unchanged (may stay None)

    Compound set: resolve every top-level product/reactant plus role-allowlisted
    ``raw.compounds`` entries into a ``frozenset`` of canonical SMILES.

    Inject ``resolve_fn`` / share ``cache`` in tests to avoid live network.
    """
    name_cache: MutableMapping[str, str | None] = cache if cache is not None else {}
    out: list[ReactionEntry] = []
    for entry in entries:
        product = _resolve_to_canonical_smiles(
            entry.product_smiles,
            entry.product_name,
            resolve=resolve_fn,
            cache=name_cache,
        )
        if product is None and entry.raw:
            compound_smiles, compound_name = _product_from_compounds(entry.raw)
            product = _resolve_to_canonical_smiles(
                compound_smiles,
                compound_name,
                resolve=resolve_fn,
                cache=name_cache,
            )

        resolved: set[str] = set()
        for smiles, name in _collect_compound_candidates(entry):
            canon = _resolve_to_canonical_smiles(
                smiles,
                name,
                resolve=resolve_fn,
                cache=name_cache,
            )
            if canon is not None:
                resolved.add(canon)
        # Ensure resolved product is always in the set when present.
        if product is not None:
            resolved.add(product)

        compound_set = frozenset(resolved)
        # Leave product_smiles unchanged when unresolvable (may stay None).
        new_product = product if product is not None else entry.product_smiles
        if new_product != entry.product_smiles or compound_set != entry.compound_smiles:
            out.append(
                replace(
                    entry,
                    product_smiles=new_product,
                    compound_smiles=compound_set,
                )
            )
        else:
            out.append(entry)
    return out


def ensure_compound_smiles_sets(
    entries: Sequence[ReactionEntry],
    *,
    resolve_fn: ResolveFn | None = None,
    cache: MutableMapping[str, str | None] | None = None,
) -> list[ReactionEntry]:
    """Alias for :func:`ensure_product_canonical_smiles` (product + set enrich)."""
    return ensure_product_canonical_smiles(
        entries, resolve_fn=resolve_fn, cache=cache
    )


def ensure_product_canonical_smiles_by_label(
    entries_by_label: Mapping[str, list[ReactionEntry]],
    *,
    resolve_fn: ResolveFn | None = None,
    cache: MutableMapping[str, str | None] | None = None,
) -> dict[str, list[ReactionEntry]]:
    """Enrich every label; share one name→SMILES cache across models."""
    name_cache: MutableMapping[str, str | None] = cache if cache is not None else {}
    return {
        label: ensure_product_canonical_smiles(
            entries, resolve_fn=resolve_fn, cache=name_cache
        )
        for label, entries in entries_by_label.items()
    }
