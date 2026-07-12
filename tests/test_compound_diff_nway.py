"""Tests for N-way compound union-find matching."""

from __future__ import annotations

import unittest

from core.compound_matching import diff_compounds_nway
from core.compound_parsing import CompoundEntry


class DiffCompoundsNwayTest(unittest.TestCase):
    def _entry(
        self,
        identifier: str,
        *,
        aliases: tuple[str, ...] = (),
        section_label: str | None = None,
    ) -> CompoundEntry:
        return CompoundEntry(
            identifier=identifier,
            identifier_type="iupac",
            aliases=aliases,
            resolved=True,
            unresolved_reference=False,
            section_label=section_label,
            role="reagent",
        )

    def test_three_labels_shared_alias_cluster(self) -> None:
        """DIPEA linked across three models via identifier/alias overlap."""
        result = diff_compounds_nway(
            {
                "claude": [self._entry("N,N-diisopropylethylamine", aliases=("DIPEA",))],
                "gpt": [self._entry("DIPEA")],
                "deepseek": [self._entry("N,N-diisopropylethylamine")],
            }
        )
        memberships = {cluster.membership for cluster in result.clusters}
        self.assertEqual(memberships, {frozenset({"claude", "gpt", "deepseek"})})
        self.assertEqual(result.deduped_counts["claude"], 1)
        self.assertEqual(result.deduped_counts["gpt"], 1)
        self.assertEqual(result.deduped_counts["deepseek"], 1)

        common, baseline_only, other_only, recall, precision = result.pairwise_metrics(
            "claude", "gpt"
        )
        self.assertEqual((common, baseline_only, other_only), (1, 0, 0))
        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 1.0)

    def test_four_labels_partial_overlap_membership(self) -> None:
        """Four labels: one compound shared by three, one singleton, one pair."""
        result = diff_compounds_nway(
            {
                "a": [
                    self._entry("methanol", aliases=("MeOH",)),
                    self._entry("Compound Unique A"),
                ],
                "b": [
                    self._entry("MeOH"),
                    self._entry("toluene"),
                ],
                "c": [
                    self._entry("methanol"),
                ],
                "d": [
                    self._entry("toluene"),
                    self._entry("Compound Unique D"),
                ],
            }
        )
        memberships = {cluster.membership for cluster in result.clusters}
        self.assertIn(frozenset({"a", "b", "c"}), memberships)  # methanol / MeOH
        self.assertIn(frozenset({"b", "d"}), memberships)  # toluene
        self.assertIn(frozenset({"a"}), memberships)  # Compound Unique A
        self.assertIn(frozenset({"d"}), memberships)  # Compound Unique D
        self.assertEqual(len(result.clusters), 4)

        only_a = result.only_entries("a")
        self.assertEqual(len(only_a), 1)
        self.assertEqual(only_a[0].identifier, "Compound Unique A")

        common, baseline_only, other_only, _, _ = result.pairwise_metrics("a", "b")
        self.assertEqual(common, 1)  # methanol
        self.assertEqual(baseline_only, 1)  # Compound Unique A
        self.assertEqual(other_only, 1)  # toluene

    def test_within_label_dedupe_before_cross_label(self) -> None:
        result = diff_compounds_nway(
            {
                "claude": [
                    self._entry("DIPEA", section_label="Ex 1"),
                    self._entry("N,N-diisopropylethylamine", aliases=("DIPEA",), section_label="Ex 2"),
                ],
                "gpt": [self._entry("DIPEA")],
                "gemini": [self._entry("unrelated")],
            }
        )
        self.assertEqual(result.raw_counts["claude"], 2)
        self.assertEqual(result.deduped_counts["claude"], 1)

        dipea_clusters = [
            c for c in result.clusters if "claude" in c.membership and "gpt" in c.membership
        ]
        self.assertEqual(len(dipea_clusters), 1)
        self.assertEqual(dipea_clusters[0].membership, frozenset({"claude", "gpt"}))

        singleton = result.clusters_for_labels(frozenset({"gemini"}))
        self.assertEqual(len(singleton), 1)
        self.assertEqual(singleton[0].representatives["gemini"].identifier, "unrelated")

    def test_empty_inputs(self) -> None:
        result = diff_compounds_nway({"a": [], "b": [], "c": []})
        self.assertEqual(result.clusters, [])
        self.assertEqual(dict(result.deduped_counts), {"a": 0, "b": 0, "c": 0})


if __name__ == "__main__":
    unittest.main()
