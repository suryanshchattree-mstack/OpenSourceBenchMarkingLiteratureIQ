"""Multi-signal reaction pair scoring for tiered N-way clustering.

The clustering key for reactions is fundamentally a *provenance* question —
two extractions describe the same reaction when they describe the same passage
in the source document, regardless of how each model named the compounds. Set
Jaccard over canonical SMILES (the original single signal) fails whenever a
model emits a generic / Markush name ("trihalobenzene" vs "1,2,4-trichloro-
benzene"): the generic name never resolves to the same SMILES, so the sets
diverge and the pair splits.

This module scores a pair of reactions across three signals, ordered strongest
(most provenance-anchored, naming-independent) to weakest:

    1. provenance       — overlap of R1 line spans (naming-independent)
    2. compound_jaccard — role-filtered canonical-SMILES set Jaccard over the
                          WHOLE reaction (product + reactants/reagents/
                          catalysts/...). Deliberately not an exact-product-
                          SMILES check: a process patent commonly describes
                          several distinct routes that converge on the same
                          final compound (e.g. three synthetic paths all
                          ending in the same target), and those routes are
                          different reactions despite sharing a product. Only
                          requiring the FULL compound set to overlap (not just
                          the product) rejects that false convergence, since
                          each route's reactants differ.
    3. combined         — renormalized weighted blend of product-name equality,
                          procedure-text cosine, compound Jaccard, and reaction
                          fingerprint cosine (catches generic-name pairs that
                          still lack usable provenance)

``pair_match`` returns the strongest tier that fires (or ``None``). The caller
(``core.reaction_nway``) merges greedily, strongest tier first, under a
one-reaction-per-model constraint.

A stricter complete-linkage variant (require mutual compatibility with every
existing group member, not just one) was tried and measured against a real
6-model export: it fragmented many genuinely-identical reactions into
singletons (real multi-model agreement lost to heterogeneous per-model
completeness) while still failing to prevent the one concrete false merge it
targeted. It cost far more recall than it bought in precision, so greedy
single-linkage is what's implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.compound_matching import canonicalize_name
from core.reaction_parsing import ReactionEntry

# Tier names, strongest → weakest. Index in this tuple is the merge priority
# (lower = merged first) and the ranking used to pick a component's match_tier.
TIER_PROVENANCE = "provenance"
TIER_COMPOUND_JACCARD = "compound_jaccard"
TIER_COMBINED = "combined"

TIER_ORDER = (
    TIER_PROVENANCE,
    TIER_COMPOUND_JACCARD,
    TIER_COMBINED,
)


@dataclass(frozen=True)
class MatchConfig:
    """Thresholds and weights for the tiered reaction matcher.

    Thresholds gate the decisive tiers; weights blend the ``combined`` fallback.
    Enable flags let callers (and the offline tuner) ablate individual tiers.
    """

    # Decisive-tier thresholds.
    tau_provenance: float = 0.50  # min line-span interval Jaccard to merge
    tau_jaccard: float = 0.85  # min compound-set Jaccard to merge
    tau_combined: float = 0.70  # min weighted blend to merge

    # ``combined`` fallback weights (renormalized over available signals).
    w_product_name: float = 0.30
    w_procedure: float = 0.30
    w_compound: float = 0.25
    w_reaction: float = 0.15

    # Tier toggles (for ablation / tuning).
    enable_provenance: bool = True
    enable_compound_jaccard: bool = True
    enable_combined: bool = True


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def interval_jaccard(start1: int, end1: int, start2: int, end2: int) -> float:
    """Inclusive line-interval Jaccard; 0.0 when no overlap or invalid spans."""
    if end1 < start1 or end2 < start2:
        return 0.0
    left = max(start1, start2)
    right = min(end1, end2)
    inter = max(0, right - left + 1)
    if inter == 0:
        return 0.0
    len1 = end1 - start1 + 1
    len2 = end2 - start2 + 1
    union = len1 + len2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def smiles_set_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """``|A∩B|/|A∪B|``; 0.0 when either set is empty."""
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def cosine_similarity(
    left: tuple[float, ...] | None,
    right: tuple[float, ...] | None,
) -> float | None:
    """Cosine of two equal-length vectors; ``None`` when unusable.

    Returns ``None`` if either vector is missing, empty, length-mismatched, or
    zero-norm — those pairs contribute no signal rather than a misleading 0.
    """
    if not left or not right or len(left) != len(right):
        return None
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left <= 0.0 or norm_right <= 0.0:
        return None
    return dot / math.sqrt(norm_left * norm_right)


def provenance_overlap(a: ReactionEntry, b: ReactionEntry) -> float | None:
    """Line-span interval Jaccard, or ``None`` when either span is unknown.

    Assumes both models' R1 line numbers index the *same* source document, so
    the spans are directly comparable across models. Entries without a joined R1
    span (``start_line``/``end_line`` None) contribute no provenance signal.
    """
    if (
        a.start_line is None
        or a.end_line is None
        or b.start_line is None
        or b.end_line is None
    ):
        return None
    return interval_jaccard(a.start_line, a.end_line, b.start_line, b.end_line)


def product_name_similarity(a: ReactionEntry, b: ReactionEntry) -> float | None:
    """1.0 when normalized product names match, 0.0 when both present and differ.

    ``None`` when either name is missing (contributes no signal). Deliberately
    exact-after-normalize: fuzzy name matching is where generic-vs-specific
    collisions live, so we leave that to provenance rather than guess here.
    """
    left = canonicalize_name(a.product_name)
    right = canonicalize_name(b.product_name)
    if left is None or right is None:
        return None
    return 1.0 if left == right else 0.0


def compound_jaccard(a: ReactionEntry, b: ReactionEntry) -> float:
    """Role-filtered canonical-SMILES set Jaccard for the two reactions."""
    return smiles_set_jaccard(a.compound_smiles, b.compound_smiles)


def combined_score(a: ReactionEntry, b: ReactionEntry, config: MatchConfig) -> float:
    """Weighted blend of available soft signals, renormalized over what exists.

    Signals absent for a pair (missing names, no procedure/reaction vector, an
    empty compound set) are dropped and the remaining weights renormalized, so
    a pair is never penalized for a signal neither side carries. Returns 0.0
    when no signal is available.
    """
    parts: list[tuple[float, float]] = []

    name_sim = product_name_similarity(a, b)
    if name_sim is not None:
        parts.append((config.w_product_name, name_sim))

    procedure_cos = cosine_similarity(a.procedure_vector, b.procedure_vector)
    if procedure_cos is not None:
        parts.append((config.w_procedure, _clamp01(procedure_cos)))

    if a.compound_smiles and b.compound_smiles:
        parts.append((config.w_compound, smiles_set_jaccard(a.compound_smiles, b.compound_smiles)))

    reaction_cos = cosine_similarity(a.reaction_vector, b.reaction_vector)
    if reaction_cos is not None:
        parts.append((config.w_reaction, _clamp01(reaction_cos)))

    total_weight = sum(weight for weight, _ in parts)
    if total_weight <= 0.0:
        return 0.0
    return sum(weight * value for weight, value in parts) / total_weight


def pair_match(
    a: ReactionEntry,
    b: ReactionEntry,
    config: MatchConfig,
) -> tuple[str | None, float]:
    """Return ``(tier, score)`` for the strongest tier that fires, else ``(None, 0.0)``.

    Tiers are evaluated strongest → weakest and short-circuit on the first hit:
    provenance, compound-set Jaccard, then the weighted combined fallback.
    ``score`` is that tier's own similarity value, used only for deterministic
    tie-breaking among candidate merges of equal tier.
    """
    if config.enable_provenance:
        overlap = provenance_overlap(a, b)
        if overlap is not None and overlap >= config.tau_provenance:
            return TIER_PROVENANCE, overlap

    if config.enable_compound_jaccard and a.compound_smiles and b.compound_smiles:
        jaccard = smiles_set_jaccard(a.compound_smiles, b.compound_smiles)
        if jaccard >= config.tau_jaccard:
            return TIER_COMPOUND_JACCARD, jaccard

    if config.enable_combined:
        score = combined_score(a, b, config)
        if score >= config.tau_combined:
            return TIER_COMBINED, score

    return None, 0.0
