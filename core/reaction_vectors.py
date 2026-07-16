"""Fill missing ReactionEntry.reaction_vector via a SMILES → rxnfp ladder."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import replace
from typing import Optional

from core.reaction_parsing import ReactionEntry
from core.rxnfp_embed import embed_reaction_smiles
from core.smiles_resolve import (
    canonicalize_smiles_rdkit,
    resolve_name_to_smiles,
)

RxnEmbedFn = Callable[[Sequence[str]], list[Optional[tuple[float, ...]]]]
ResolveFn = Callable[[str], Optional[str]]


def assemble_canonical_rxn(
    reactant_smiles: Sequence[str],
    product_smiles: str,
) -> str | None:
    """Assemble ``reacs>>product`` (dot-joined reactants) when both sides exist."""
    reactants = [s.strip() for s in reactant_smiles if s and str(s).strip()]
    product = product_smiles.strip() if product_smiles else ""
    if not reactants or not product:
        return None
    return ".".join(reactants) + ">>" + product


def _species_smiles(
    smiles: str | None,
    name: str | None,
    *,
    resolve: ResolveFn | None,
    cache: MutableMapping[str, str | None],
    allow_resolve: bool,
) -> str | None:
    canon = canonicalize_smiles_rdkit(smiles)
    if canon is not None:
        return canon
    if not allow_resolve or not name or not str(name).strip():
        return None
    key = str(name).strip()
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


def _build_rxn_smiles(
    entry: ReactionEntry,
    *,
    resolve: ResolveFn | None,
    cache: MutableMapping[str, str | None],
    allow_resolve: bool,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """
    Return ``(canonical_rxn, product_smiles, reactant_smiles)``.

    When ``allow_resolve`` is False, only existing SMILES fields are used
    (after RDKit canonicalize). When True, missing species are resolved by name.
    """
    product = _species_smiles(
        entry.product_smiles,
        entry.product_name,
        resolve=resolve,
        cache=cache,
        allow_resolve=allow_resolve,
    )
    n = max(len(entry.reactant_smiles), len(entry.reactant_names) if allow_resolve else 0)
    reactants: list[str] = []
    for index in range(n):
        smi = entry.reactant_smiles[index] if index < len(entry.reactant_smiles) else None
        name = entry.reactant_names[index] if index < len(entry.reactant_names) else None
        resolved = _species_smiles(
            smi,
            name,
            resolve=resolve,
            cache=cache,
            allow_resolve=allow_resolve,
        )
        if resolved is not None:
            reactants.append(resolved)
    # Also pick up trailing SMILES when names are shorter and resolve is off.
    if not allow_resolve and len(entry.reactant_smiles) > n:
        for smi in entry.reactant_smiles[n:]:
            resolved = canonicalize_smiles_rdkit(smi)
            if resolved is not None:
                reactants.append(resolved)

    if product is None or not reactants:
        return None, product, tuple(reactants)
    rxn = assemble_canonical_rxn(reactants, product)
    return rxn, product, tuple(reactants)


def ensure_reaction_vectors(
    entries: Sequence[ReactionEntry],
    *,
    embed_fn: RxnEmbedFn | None = None,
    resolve_fn: ResolveFn | None = None,
    name_cache: MutableMapping[str, str | None] | None = None,
) -> list[ReactionEntry]:
    """
    Return copies of ``entries`` with ``reaction_vector`` set when possible.

    Ladder per reaction:
      1. Existing ``reaction_vector``
      2. Embed existing ``canonical_rxn`` with rxnfp
      3. product_smiles + ≥1 reactant_smiles → RDKit → assemble → embed
      4. Resolve missing names (PubChem → OPSIN) then assemble → embed
      5. Otherwise leave without a reaction vector

    Embeds via the pip ``rxnfp`` package (bundled ``bert_ft`` weights).
    Inject ``embed_fn`` / ``resolve_fn`` in tests to avoid network / model load.
    """
    embed = embed_fn or embed_reaction_smiles
    cache: MutableMapping[str, str | None] = name_cache if name_cache is not None else {}

    out = list(entries)
    need_embed_indices: list[int] = []
    need_embed_smiles: list[str] = []
    pending_meta: list[tuple[str | None, str | None, tuple[str, ...]]] = []

    for index, entry in enumerate(entries):
        if entry.reaction_vector is not None:
            continue

        # Step 2: existing canonical_rxn
        if entry.canonical_rxn and str(entry.canonical_rxn).strip():
            need_embed_indices.append(index)
            need_embed_smiles.append(str(entry.canonical_rxn).strip())
            pending_meta.append((entry.canonical_rxn.strip(), None, ()))
            continue

        # Step 3: SMILES-only assemble (no network)
        rxn, product, reactants = _build_rxn_smiles(
            entry, resolve=resolve_fn, cache=cache, allow_resolve=False
        )
        if rxn is not None:
            need_embed_indices.append(index)
            need_embed_smiles.append(rxn)
            pending_meta.append((rxn, product, reactants))
            continue

        # Steps 4–5: resolve names then assemble
        rxn, product, reactants = _build_rxn_smiles(
            entry, resolve=resolve_fn, cache=cache, allow_resolve=True
        )
        if rxn is not None:
            need_embed_indices.append(index)
            need_embed_smiles.append(rxn)
            pending_meta.append((rxn, product, reactants))

    if not need_embed_indices:
        return out

    unique_smiles = list(dict.fromkeys(need_embed_smiles))
    vectors = embed(unique_smiles)
    if len(vectors) != len(unique_smiles):
        raise ValueError(
            f"embed_fn returned {len(vectors)} vectors for {len(unique_smiles)} reaction SMILES"
        )
    smiles_to_vector = dict(zip(unique_smiles, vectors))

    for index, rxn_smiles, meta in zip(need_embed_indices, need_embed_smiles, pending_meta):
        vector = smiles_to_vector.get(rxn_smiles)
        if vector is None:
            continue
        built_rxn, product, reactants = meta
        updates: dict = {"reaction_vector": vector}
        if built_rxn:
            updates["canonical_rxn"] = built_rxn
        if product and not out[index].product_smiles:
            updates["product_smiles"] = product
        if reactants and not out[index].reactant_smiles:
            updates["reactant_smiles"] = reactants
        out[index] = replace(out[index], **updates)
    return out


def ensure_reaction_vectors_by_label(
    entries_by_label: dict[str, list[ReactionEntry]],
    *,
    embed_fn: RxnEmbedFn | None = None,
    resolve_fn: ResolveFn | None = None,
    name_cache: MutableMapping[str, str | None] | None = None,
) -> dict[str, list[ReactionEntry]]:
    """Ensure reaction vectors for every label; share one name cache across labels."""
    cache: MutableMapping[str, str | None] = name_cache if name_cache is not None else {}
    flat: list[ReactionEntry] = []
    spans: list[tuple[str, int]] = []
    for label, entries in entries_by_label.items():
        spans.append((label, len(entries)))
        flat.extend(entries)
    filled = ensure_reaction_vectors(
        flat, embed_fn=embed_fn, resolve_fn=resolve_fn, name_cache=cache
    )
    result: dict[str, list[ReactionEntry]] = {}
    offset = 0
    for label, count in spans:
        result[label] = filled[offset : offset + count]
        offset += count
    return result
