"""Deterministic compound matching for M1/M2 benchmark comparison."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Generic, Mapping, Protocol, TypeVar

from core.compound_parsing import CompoundEntry

_WHITESPACE_RUN = re.compile(r"\s+")
_HYPHEN_SPACES = re.compile(r"\s*-\s*")
_TRAILING_PUNCT = re.compile(r"[.,;]+$")

# Strongest → weakest. Cluster match_tier is the weakest tier that contributed a merge.
MATCH_TIERS = ("inchi_key", "smiles", "molecular_formula", "name")

MATCH_TIER_DISPLAY = {
    "inchi_key": "InChIKey",
    "smiles": "SMILES",
    "molecular_formula": "Formula (weak)",
    "name": "Name",
}


class HasNamePool(Protocol):
    """Minimal shape needed for identifier/alias union-find matching."""

    identifier: str
    aliases: tuple[str, ...]


TEntry = TypeVar("TEntry", bound=HasNamePool)


def canonicalize_name(raw: str | None) -> str | None:
    """Port of BenchmarkTextNormalizer.canonicalizeName from the Java repo."""
    if raw is None:
        return None
    text = unicodedata.normalize("NFKC", raw)
    text = _WHITESPACE_RUN.sub(" ", text).strip()
    if not text:
        return None
    text = text.lower()
    text = _HYPHEN_SPACES.sub("-", text)
    text = _TRAILING_PUNCT.sub("", text)
    return text or None


def canonicalize_smiles(raw_smiles: str | None) -> str | None:
    """Canonical RDKit SMILES, or None if missing/unparseable."""
    if raw_smiles is None:
        return None
    text = str(raw_smiles).strip()
    if not text:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return canonical or None


def build_name_pool(entry: HasNamePool) -> set[str]:
    """All normalized names for a compound (identifier + aliases)."""
    pool: set[str] = set()
    if name := canonicalize_name(entry.identifier):
        pool.add(name)
    for alias in entry.aliases:
        if name := canonicalize_name(alias):
            pool.add(name)
    return pool


def format_match_tier(match_tier: str | None) -> str:
    """Human-readable match-tier label for tables / UI."""
    if match_tier is None or match_tier == "single-model":
        return "—"
    return MATCH_TIER_DISPLAY.get(match_tier, match_tier)


def _tier_key(compound: HasNamePool, tier: str) -> str | None:
    if tier == "inchi_key":
        value = getattr(compound, "inchi_key", None)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    if tier == "smiles":
        return canonicalize_smiles(getattr(compound, "smiles", None))
    if tier == "molecular_formula":
        value = getattr(compound, "molecular_formula", None)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    return None


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
        """Unite two nodes. Returns True if they were previously separate."""
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


@dataclass(frozen=True)
class CompoundDiffResult:
    total_claude: int
    total_benchmark: int
    common: int
    claude_only: int
    benchmark_only: int
    matched_pairs: list[tuple[CompoundEntry, CompoundEntry]]
    claude_only_entries: list[CompoundEntry]
    benchmark_only_entries: list[CompoundEntry]
    raw_claude_count: int
    raw_benchmark_count: int
    deduped_claude_count: int
    deduped_benchmark_count: int


@dataclass(frozen=True)
class NWayCluster(Generic[TEntry]):
    """One union-find component after cross-label identifier/alias matching."""

    membership: frozenset[str]
    representatives: Mapping[str, TEntry]
    match_tier: str | None = None


@dataclass(frozen=True)
class NWayDiffResult(Generic[TEntry]):
    """N-way compound diff: clusters with per-label membership and representatives."""

    labels: tuple[str, ...]
    clusters: list[NWayCluster[TEntry]]
    raw_counts: Mapping[str, int]
    deduped_counts: Mapping[str, int]

    def clusters_for_labels(self, required: frozenset[str]) -> list[NWayCluster[TEntry]]:
        """Clusters whose membership equals exactly ``required``."""
        return [cluster for cluster in self.clusters if cluster.membership == required]

    def clusters_containing(self, label: str) -> list[NWayCluster[TEntry]]:
        return [cluster for cluster in self.clusters if label in cluster.membership]

    def only_entries(self, label: str) -> list[TEntry]:
        """Representative entries for clusters that contain only ``label``."""
        singleton = frozenset({label})
        entries = [
            cluster.representatives[label]
            for cluster in self.clusters
            if cluster.membership == singleton
        ]
        return sorted(entries, key=lambda entry: canonicalize_name(entry.identifier) or "")

    def pairwise_metrics(
        self,
        baseline: str,
        other: str,
    ) -> tuple[int, int, int, float | None, float | None]:
        """
        Pairwise counts and recall/precision for ``other`` vs ``baseline``.

        Returns (common, baseline_only, other_only, recall, precision) where
        recall = common / deduped_baseline and precision = common / deduped_other.
        """
        if baseline not in self.deduped_counts or other not in self.deduped_counts:
            raise KeyError(f"Unknown label(s): baseline={baseline!r}, other={other!r}")
        common = 0
        baseline_only = 0
        other_only = 0
        for cluster in self.clusters:
            has_baseline = baseline in cluster.membership
            has_other = other in cluster.membership
            if has_baseline and has_other:
                common += 1
            elif has_baseline and not has_other:
                baseline_only += 1
            elif has_other and not has_baseline:
                other_only += 1
        baseline_total = self.deduped_counts[baseline]
        other_total = self.deduped_counts[other]
        recall = common / baseline_total if baseline_total > 0 else None
        precision = common / other_total if other_total > 0 else None
        return common, baseline_only, other_only, recall, precision


def _pick_representative(indices: list[int], compounds: list[TEntry]) -> TEntry:
    """Pick the entry with the longest identifier as the group representative."""
    return max((compounds[i] for i in indices), key=lambda entry: len(entry.identifier))


def _cluster_sort_key(cluster: NWayCluster[TEntry]) -> tuple[str, ...]:
    names = [
        canonicalize_name(entry.identifier) or ""
        for entry in cluster.representatives.values()
    ]
    return (min(names) if names else "", *sorted(cluster.membership))


def _weaker_tier(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None:
        return left
    return left if MATCH_TIERS.index(left) >= MATCH_TIERS.index(right) else right


def _union_with_tier(
    uf: _UnionFind,
    left: int,
    right: int,
    tier: str,
    component_tier: dict[int, str | None],
) -> bool:
    root_left = uf.find(left)
    root_right = uf.find(right)
    if root_left == root_right:
        return False
    left_tier = component_tier.get(root_left)
    right_tier = component_tier.get(root_right)
    uf.union(left, right)
    new_root = uf.find(left)
    merged = _weaker_tier(left_tier, right_tier)
    merged = _weaker_tier(merged, tier)
    for old_root in (root_left, root_right):
        component_tier.pop(old_root, None)
    component_tier[new_root] = merged
    return True


def _apply_key_groups(
    uf: _UnionFind,
    key_to_indices: Mapping[str, list[int]],
    tier: str,
    component_tier: dict[int, str | None],
) -> None:
    for indices in key_to_indices.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for other in indices[1:]:
            _union_with_tier(uf, anchor, other, tier, component_tier)


def diff_compounds_nway(
    entries_by_label: Mapping[str, list[TEntry]],
) -> NWayDiffResult[TEntry]:
    """
    Deterministically compare N labeled compound lists.

    Matching waterfall (strongest → weakest): inchi_key → smiles → molecular_formula → name.
    Within-label duplicates are merged before cross-label diff. Each cluster records
    ``match_tier`` as the weakest tier that contributed a successful union
    (``None`` when the cluster is a single unmerged entry).
    """
    labels = tuple(entries_by_label.keys())
    raw_counts = {label: len(entries) for label, entries in entries_by_label.items()}

    all_compounds: list[TEntry] = []
    sides: list[str] = []
    for label, entries in entries_by_label.items():
        all_compounds.extend(entries)
        sides.extend([label] * len(entries))

    if not all_compounds:
        return NWayDiffResult(
            labels=labels,
            clusters=[],
            raw_counts=raw_counts,
            deduped_counts={label: 0 for label in labels},
        )

    uf = _UnionFind(len(all_compounds))
    component_tier: dict[int, str | None] = {}

    for tier in MATCH_TIERS:
        if tier == "name":
            name_to_indices: dict[str, list[int]] = {}
            for index, compound in enumerate(all_compounds):
                for name in build_name_pool(compound):
                    name_to_indices.setdefault(name, []).append(index)
            _apply_key_groups(uf, name_to_indices, tier, component_tier)
            continue

        key_to_indices: dict[str, list[int]] = {}
        for index, compound in enumerate(all_compounds):
            key = _tier_key(compound, tier)
            if key is None:
                continue
            key_to_indices.setdefault(key, []).append(index)
        _apply_key_groups(uf, key_to_indices, tier, component_tier)

    # root -> label -> compound indices
    groups: dict[int, dict[str, list[int]]] = {}
    for index, side in enumerate(sides):
        root = uf.find(index)
        groups.setdefault(root, {}).setdefault(side, []).append(index)

    deduped_counts = {label: 0 for label in labels}
    clusters: list[NWayCluster[TEntry]] = []
    for root, by_label in groups.items():
        membership = frozenset(by_label.keys())
        representatives = {
            label: _pick_representative(indices, all_compounds)
            for label, indices in by_label.items()
        }
        for label in by_label:
            deduped_counts[label] += 1
        entry_count = sum(len(indices) for indices in by_label.values())
        match_tier = component_tier.get(root) if entry_count > 1 else None
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


def _nway_to_pairwise(
    nway: NWayDiffResult[CompoundEntry],
    claude_label: str,
    benchmark_label: str,
    raw_claude_count: int,
    raw_benchmark_count: int,
) -> CompoundDiffResult:
    matched_pairs: list[tuple[CompoundEntry, CompoundEntry]] = []
    claude_only_entries: list[CompoundEntry] = []
    benchmark_only_entries: list[CompoundEntry] = []

    for cluster in nway.clusters:
        has_claude = claude_label in cluster.membership
        has_benchmark = benchmark_label in cluster.membership
        if has_claude and has_benchmark:
            matched_pairs.append(
                (
                    cluster.representatives[claude_label],
                    cluster.representatives[benchmark_label],
                )
            )
        elif has_claude:
            claude_only_entries.append(cluster.representatives[claude_label])
        elif has_benchmark:
            benchmark_only_entries.append(cluster.representatives[benchmark_label])

    matched_pairs.sort(key=lambda pair: canonicalize_name(pair[0].identifier) or "")
    claude_only_entries.sort(key=lambda entry: canonicalize_name(entry.identifier) or "")
    benchmark_only_entries.sort(key=lambda entry: canonicalize_name(entry.identifier) or "")

    deduped_claude = nway.deduped_counts.get(claude_label, 0)
    deduped_benchmark = nway.deduped_counts.get(benchmark_label, 0)
    return CompoundDiffResult(
        total_claude=deduped_claude,
        total_benchmark=deduped_benchmark,
        common=len(matched_pairs),
        claude_only=len(claude_only_entries),
        benchmark_only=len(benchmark_only_entries),
        matched_pairs=matched_pairs,
        claude_only_entries=claude_only_entries,
        benchmark_only_entries=benchmark_only_entries,
        raw_claude_count=raw_claude_count,
        raw_benchmark_count=raw_benchmark_count,
        deduped_claude_count=deduped_claude,
        deduped_benchmark_count=deduped_benchmark,
    )


def diff_compounds(
    claude_compounds: list[CompoundEntry],
    benchmark_compounds: list[CompoundEntry],
) -> CompoundDiffResult:
    """
    Deterministically compare two M2 compound lists.

    Thin wrapper over :func:`diff_compounds_nway` with fixed side labels
    ``\"claude\"`` / ``\"benchmark\"``.
    """
    nway = diff_compounds_nway(
        {
            "claude": claude_compounds,
            "benchmark": benchmark_compounds,
        }
    )
    return _nway_to_pairwise(
        nway,
        "claude",
        "benchmark",
        raw_claude_count=len(claude_compounds),
        raw_benchmark_count=len(benchmark_compounds),
    )
