"""N-way reaction clustering by a tiered multi-signal matcher (one-per-model).

Merges cross-model reaction pairs greedily, strongest tier first, under a
one-reaction-per-model constraint. Tiers (see ``core.reaction_scoring``):
provenance (R1 line-span overlap) → compound-set Jaccard (whole reaction, not
just the product) → weighted combined fallback. Provenance and the combined
fallback are what make the matcher robust to generic / Markush names that
break pure SMILES matching.

This is single-linkage: once any one qualifying edge joins two components,
they merge, regardless of whether every other cross-pair between them would
also qualify. A stricter complete-linkage variant (require mutual
compatibility with every existing member) was tried and measured against a
real 6-model patent export: it fragmented many genuinely-identical multi-model
reactions into singletons — because real per-model extractions are
heterogeneous enough (different SMILES resolved, different line-boundary
drift) that requiring universal pairwise agreement rejects far more correct
merges than it prevents incorrect ones — while still failing to prevent the
one concrete false merge (two different reactions sharing an overlapping line
span) it targeted. Single-linkage measurably tracks real-world recall much
more closely, so that's what's implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

from core.compound_matching import (
    NWayCluster,
    NWayDiffResult,
    canonicalize_name,
)
from core.reaction_matching import canonicalize_smiles
from core.reaction_parsing import ReactionEntry
from core.reaction_product_enrich import (
    ResolveFn,
    ensure_product_canonical_smiles_by_label,
)
from core.reaction_scoring import (
    TIER_COMBINED,
    TIER_COMPOUND_JACCARD,
    TIER_ORDER,
    TIER_PROVENANCE,
    MatchConfig,
    interval_jaccard,
    pair_match,
    smiles_set_jaccard,
)

# Strongest → weakest. A cluster's match_tier is the weakest tier that
# contributed a merge (mirrors core.compound_matching semantics).
MATCH_TIERS = TIER_ORDER
MATCH_WATERFALL = TIER_ORDER

DEFAULT_COMPOUND_JACCARD_TAU = 0.85

MATCH_TIER_DISPLAY = {
    TIER_PROVENANCE: "Provenance (R1 lines)",
    TIER_COMPOUND_JACCARD: "Compound-set Jaccard",
    TIER_COMBINED: "Weighted signals",
}


def format_match_tier(match_tier: str | None) -> str:
    """Human-readable match-tier label for tables / UI."""
    if match_tier is None or match_tier == "single-model":
        return "—"
    return MATCH_TIER_DISPLAY.get(match_tier, match_tier)


def filter_synthetic(entries: list[ReactionEntry]) -> tuple[list[ReactionEntry], int]:
    """Drop ``non_synthetic`` records; return (kept, skipped_count)."""
    kept: list[ReactionEntry] = []
    skipped = 0
    for entry in entries:
        if entry.non_synthetic:
            skipped += 1
            continue
        kept.append(entry)
    return kept, skipped


def rxn_smiles_key(entry: ReactionEntry) -> str | None:
    """
    Hard key: canon(product) + '<<' + sorted(canon(reactants)) joined by '|'.

    Kept for diagnostics / pairwise tooling; unused by N-way clustering.
    """
    product = canonicalize_smiles(entry.product_smiles)
    if product is None:
        return None
    reactants: list[str] = []
    for smiles in entry.reactant_smiles:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            reactants.append(canonical)
    if not reactants:
        return None
    return product + "<<" + "|".join(sorted(reactants))


def product_smiles_key(entry: ReactionEntry) -> str | None:
    return canonicalize_smiles(entry.product_smiles)


def product_name_key(entry: ReactionEntry) -> str | None:
    return canonicalize_name(entry.product_name)


def compound_smiles_set(entry: ReactionEntry) -> frozenset[str]:
    """Role-filtered canonical SMILES set used for Jaccard clustering."""
    return entry.compound_smiles if entry.compound_smiles else frozenset()


def reaction_row_label(entry: ReactionEntry) -> str:
    """Display name for a reaction row (preferred model pick / inspector)."""
    if entry.reaction_id and str(entry.reaction_id).strip():
        return str(entry.reaction_id).strip()
    section = (entry.section_label or "").strip()
    step = (entry.step_label or "").strip()
    if section and step:
        return f"{section} | {step}"
    if section:
        return section
    if step:
        return step
    if entry.product_name and entry.product_name.strip():
        return entry.product_name.strip()
    if entry.product_smiles and entry.product_smiles.strip():
        smiles = entry.product_smiles.strip()
        return smiles if len(smiles) <= 48 else smiles[:45] + "..."
    return "(unnamed reaction)"


def _pick_representative(indices: list[int], reactions: list[ReactionEntry]) -> ReactionEntry:
    """Prefer longest row label, then longest product_name."""

    def sort_key(entry: ReactionEntry) -> tuple[int, int]:
        return (len(reaction_row_label(entry)), len(entry.product_name or ""))

    return max((reactions[i] for i in indices), key=sort_key)


def _weaker_tier(left: str | None, right: str | None) -> str | None:
    """The weaker (higher-index in TIER_ORDER) of two tiers; ``None``-tolerant."""
    if left is None:
        return right
    if right is None:
        return left
    return left if TIER_ORDER.index(left) >= TIER_ORDER.index(right) else right


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1
        return True


def _models_in_component(uf: _UnionFind, sides: Sequence[str], root: int) -> set[str]:
    return {sides[index] for index in range(len(sides)) if uf.find(index) == root}


def _can_merge_one_per_model(
    uf: _UnionFind,
    sides: Sequence[str],
    left: int,
    right: int,
) -> bool:
    root_left = uf.find(left)
    root_right = uf.find(right)
    if root_left == root_right:
        return False
    return _models_in_component(uf, sides, root_left).isdisjoint(
        _models_in_component(uf, sides, root_right)
    )


def _greedy_multifeature_buckets(
    uf: _UnionFind,
    sides: Sequence[str],
    reactions: Sequence[ReactionEntry],
    *,
    config: MatchConfig,
    component_tier: dict[int, str | None],
) -> None:
    """Greedy single-linkage cross-model merge over the tiered matcher (one reaction
    per model).

    Builds every cross-model edge that fires at some tier, then merges them in
    priority order — strongest tier first, higher score first, index order to
    break ties — so decisive provenance edges form clusters before weaker
    Jaccard / combined edges can. Each component tracks the *weakest* tier that
    contributed a merge, exposed later as the cluster's ``match_tier``.
    """
    n = len(reactions)
    edges: list[tuple[int, float, int, int, str]] = []
    for left in range(n):
        for right in range(left + 1, n):
            if sides[left] == sides[right]:
                continue
            tier, score = pair_match(reactions[left], reactions[right], config)
            if tier is None:
                continue
            edges.append((TIER_ORDER.index(tier), -score, left, right, tier))
    edges.sort()
    for _tier_rank, _neg_score, left, right, tier in edges:
        if not _can_merge_one_per_model(uf, sides, left, right):
            continue
        root_left = uf.find(left)
        root_right = uf.find(right)
        merged = _weaker_tier(
            _weaker_tier(component_tier.get(root_left), component_tier.get(root_right)),
            tier,
        )
        uf.union(left, right)
        new_root = uf.find(left)
        for old_root in (root_left, root_right):
            if old_root != new_root:
                component_tier.pop(old_root, None)
        component_tier[new_root] = merged


def _cluster_sort_key(cluster: NWayCluster[ReactionEntry]) -> tuple[str, ...]:
    names = [
        canonicalize_name(reaction_row_label(entry)) or ""
        for entry in cluster.representatives.values()
    ]
    return (min(names) if names else "", *sorted(cluster.membership))


def prepare_reaction_entries(
    entries_by_label: Mapping[str, list[ReactionEntry]],
    *,
    resolve_fn: ResolveFn | None = None,
    name_cache: MutableMapping[str, str | None] | None = None,
) -> dict[str, list[ReactionEntry]]:
    """
    Filter ``non_synthetic``, then enrich product + compound SMILES sets.

    Shares one name→SMILES cache across models so network resolve runs once per name.
    """
    filtered: dict[str, list[ReactionEntry]] = {}
    for label, entries in entries_by_label.items():
        kept, _skipped = filter_synthetic(list(entries))
        filtered[label] = kept
    return ensure_product_canonical_smiles_by_label(
        filtered,
        resolve_fn=resolve_fn,
        cache=name_cache,
    )


def diff_reactions_nway(
    entries_by_label: Mapping[str, list[ReactionEntry]],
    *,
    config: MatchConfig | None = None,
    tau_jaccard: float | None = None,
    skip_ensure: bool = False,
    resolve_fn: ResolveFn | None = None,
    name_cache: MutableMapping[str, str | None] | None = None,
) -> NWayDiffResult[ReactionEntry]:
    """
    Deterministically compare N labeled reaction lists with the tiered matcher.

    Filters ``non_synthetic``, optionally enriches compound SMILES sets, then
    greedily unions cross-model pairs strongest tier first (provenance →
    compound-set Jaccard → weighted combined), one reaction per model. Each
    multi-model component records the weakest tier that merged it as
    ``match_tier``. Reactions that fire no tier against anything stay singletons.

    ``config`` supplies all thresholds/weights; when omitted a default
    :class:`MatchConfig` is used. ``tau_jaccard`` is a back-compat shortcut that
    overrides only the compound-Jaccard threshold on that default config.
    """
    if config is None:
        config = (
            MatchConfig()
            if tau_jaccard is None
            else MatchConfig(tau_jaccard=tau_jaccard)
        )
    labels = tuple(entries_by_label.keys())
    filtered_by_label: dict[str, list[ReactionEntry]] = {}
    raw_counts: dict[str, int] = {}
    for label, entries in entries_by_label.items():
        kept, _skipped = filter_synthetic(list(entries))
        filtered_by_label[label] = kept
        raw_counts[label] = len(kept)

    if skip_ensure:
        working = filtered_by_label
    else:
        working = ensure_product_canonical_smiles_by_label(
            filtered_by_label,
            resolve_fn=resolve_fn,
            cache=name_cache,
        )

    all_reactions: list[ReactionEntry] = []
    sides: list[str] = []
    for label, entries in working.items():
        all_reactions.extend(entries)
        sides.extend([label] * len(entries))

    if not all_reactions:
        return NWayDiffResult(
            labels=labels,
            clusters=[],
            raw_counts=raw_counts,
            deduped_counts={label: 0 for label in labels},
        )

    uf = _UnionFind(len(all_reactions))
    component_tier: dict[int, str | None] = {}
    _greedy_multifeature_buckets(
        uf, sides, all_reactions, config=config, component_tier=component_tier
    )

    groups: dict[int, dict[str, list[int]]] = {}
    for index, side in enumerate(sides):
        root = uf.find(index)
        groups.setdefault(root, {}).setdefault(side, []).append(index)

    deduped_counts = {label: 0 for label in labels}
    clusters: list[NWayCluster[ReactionEntry]] = []
    for root, by_label in groups.items():
        membership = frozenset(by_label.keys())
        representatives = {
            label: _pick_representative(indices, all_reactions)
            for label, indices in by_label.items()
        }
        for label in by_label:
            deduped_counts[label] += 1
        match_tier = component_tier.get(root) if len(membership) > 1 else None
        clusters.append(
            NWayCluster(
                membership=membership,
                representatives=representatives,
                match_tier=match_tier,
            )
        )

    clusters.sort(key=_cluster_sort_key)
    return NWayDiffResult(
        labels=labels,
        clusters=clusters,
        raw_counts=raw_counts,
        deduped_counts=deduped_counts,
    )
