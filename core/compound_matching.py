"""Deterministic compound matching for M2 benchmark comparison."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from core.compound_parsing import CompoundEntry

_WHITESPACE_RUN = re.compile(r"\s+")
_HYPHEN_SPACES = re.compile(r"\s*-\s*")
_TRAILING_PUNCT = re.compile(r"[.,;]+$")


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


def build_name_pool(entry: CompoundEntry) -> set[str]:
    """All normalized names for a compound (identifier + aliases)."""
    pool: set[str] = set()
    if name := canonicalize_name(entry.identifier):
        pool.add(name)
    for alias in entry.aliases:
        if name := canonicalize_name(alias):
            pool.add(name)
    return pool


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


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


def _group_compounds(
    compounds: list[CompoundEntry],
    sides: list[str],
    side: str,
    uf: _UnionFind,
) -> dict[int, list[int]]:
    """Group compound indices by union-find root for one side."""
    groups: dict[int, list[int]] = {}
    for index, (compound, compound_side) in enumerate(zip(compounds, sides)):
        if compound_side != side:
            continue
        root = uf.find(index)
        groups.setdefault(root, []).append(index)
    return groups


def _pick_representative(indices: list[int], compounds: list[CompoundEntry]) -> CompoundEntry:
    """Pick the entry with the longest identifier as the group representative."""
    return max((compounds[i] for i in indices), key=lambda entry: len(entry.identifier))


def diff_compounds(
    claude_compounds: list[CompoundEntry],
    benchmark_compounds: list[CompoundEntry],
) -> CompoundDiffResult:
    """
    Deterministically compare two M2 compound lists.

    Matching rule: two compounds match if any normalized identifier or alias overlaps.
    Within-model duplicates (overlapping name pools) are merged before cross-model diff.
    """
    all_compounds = claude_compounds + benchmark_compounds
    sides = ["claude"] * len(claude_compounds) + ["benchmark"] * len(benchmark_compounds)
    if not all_compounds:
        return CompoundDiffResult(
            total_claude=0,
            total_benchmark=0,
            common=0,
            claude_only=0,
            benchmark_only=0,
            matched_pairs=[],
            claude_only_entries=[],
            benchmark_only_entries=[],
            raw_claude_count=0,
            raw_benchmark_count=0,
            deduped_claude_count=0,
            deduped_benchmark_count=0,
        )

    name_to_indices: dict[str, list[int]] = {}
    for index, compound in enumerate(all_compounds):
        for name in build_name_pool(compound):
            name_to_indices.setdefault(name, []).append(index)

    uf = _UnionFind(len(all_compounds))
    for indices in name_to_indices.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for other in indices[1:]:
            uf.union(anchor, other)

    component_sides: dict[int, set[str]] = {}
    for index, side in enumerate(sides):
        root = uf.find(index)
        component_sides.setdefault(root, set()).add(side)

    matched_pairs: list[tuple[CompoundEntry, CompoundEntry]] = []
    claude_only_entries: list[CompoundEntry] = []
    benchmark_only_entries: list[CompoundEntry] = []

    claude_groups = _group_compounds(all_compounds, sides, "claude", uf)
    benchmark_groups = _group_compounds(all_compounds, sides, "benchmark", uf)

    deduped_claude_count = len(claude_groups)
    deduped_benchmark_count = len(benchmark_groups)

    for root, side_set in component_sides.items():
        has_claude = "claude" in side_set
        has_benchmark = "benchmark" in side_set
        if has_claude and has_benchmark:
            claude_indices = claude_groups[root]
            benchmark_indices = benchmark_groups[root]
            matched_pairs.append(
                (
                    _pick_representative(claude_indices, all_compounds),
                    _pick_representative(benchmark_indices, all_compounds),
                )
            )
        elif has_claude:
            claude_only_entries.append(
                _pick_representative(claude_groups[root], all_compounds)
            )
        elif has_benchmark:
            benchmark_only_entries.append(
                _pick_representative(benchmark_groups[root], all_compounds)
            )

    matched_pairs.sort(key=lambda pair: canonicalize_name(pair[0].identifier) or "")
    claude_only_entries.sort(key=lambda entry: canonicalize_name(entry.identifier) or "")
    benchmark_only_entries.sort(key=lambda entry: canonicalize_name(entry.identifier) or "")

    common = len(matched_pairs)
    claude_only = len(claude_only_entries)
    benchmark_only = len(benchmark_only_entries)

    return CompoundDiffResult(
        total_claude=deduped_claude_count,
        total_benchmark=deduped_benchmark_count,
        common=common,
        claude_only=claude_only,
        benchmark_only=benchmark_only,
        matched_pairs=matched_pairs,
        claude_only_entries=claude_only_entries,
        benchmark_only_entries=benchmark_only_entries,
        raw_claude_count=len(claude_compounds),
        raw_benchmark_count=len(benchmark_compounds),
        deduped_claude_count=deduped_claude_count,
        deduped_benchmark_count=deduped_benchmark_count,
    )
