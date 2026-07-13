"""Tests for M1 N-way cluster consensus classification."""

from __future__ import annotations

import unittest

from core.compound_matching import diff_compounds_nway
from core.m1_consensus import (
    compute_cluster_consensus,
    consensus_rows_to_dataframe,
)
from core.m1_parsing import QUANTITY_FIELDS, M1CompoundEntry
from core.m1_visuals import build_consensus_chart


class M1ConsensusTest(unittest.TestCase):
    def _entry(
        self,
        identifier: str,
        *,
        identifier_type: str = "iupac",
        aliases: tuple[str, ...] = (),
        role: str | None = "reagent",
        is_section_product: bool = False,
        quantity: dict[str, float | None] | None = None,
    ) -> M1CompoundEntry:
        qty = {field: None for field in QUANTITY_FIELDS}
        if quantity:
            qty.update(quantity)
        return M1CompoundEntry(
            identifier=identifier,
            identifier_type=identifier_type,
            aliases=aliases,
            role=role,
            is_section_product=is_section_product,
            commercially_available=False,
            quantity=qty,
            ms_mz=None,
            notes=None,
            section_label="Example 1",
        )

    def _role_row(self, rows, *, identifier: str = "methanol"):
        matches = [
            row
            for row in rows
            if row.cluster_identifier == identifier and row.field == "role"
        ]
        self.assertEqual(len(matches), 1, msg=f"expected one role row, got {matches}")
        return matches[0]

    def test_unanimous(self) -> None:
        nway = diff_compounds_nway(
            {
                "claude": [self._entry("methanol", role="solvent")],
                "gpt": [self._entry("methanol", role="solvent")],
                "gemini": [self._entry("methanol", role="solvent")],
                "llama": [self._entry("methanol", role="solvent")],
            }
        )
        rows = compute_cluster_consensus(nway, baseline="claude")
        role = self._role_row(rows)
        self.assertEqual(role.pattern, "unanimous")
        self.assertEqual(role.majority_value, "solvent")
        self.assertEqual(role.baseline_value, "solvent")
        self.assertEqual(role.majority_supporters, frozenset({"gpt", "gemini", "llama"}))
        self.assertEqual(role.not_in_majority, frozenset())

    def test_majority_vs_baseline(self) -> None:
        """2 of 3 non-baseline models agree against Claude → flagged."""
        nway = diff_compounds_nway(
            {
                "claude": [self._entry("methanol", role="solvent")],
                "gpt": [self._entry("methanol", role="reagent")],
                "gemini": [self._entry("methanol", role="reagent")],
                "llama": [self._entry("methanol", role="catalyst")],
            }
        )
        rows = compute_cluster_consensus(nway, baseline="claude")
        role = self._role_row(rows)
        self.assertEqual(role.pattern, "majority_vs_baseline")
        self.assertEqual(role.majority_value, "reagent")
        self.assertEqual(role.baseline_value, "solvent")
        self.assertEqual(role.majority_supporters, frozenset({"gpt", "gemini"}))
        self.assertEqual(role.not_in_majority, frozenset({"claude", "llama"}))

    def test_baseline_majority(self) -> None:
        """Baseline + 1 other agree; one other differs."""
        nway = diff_compounds_nway(
            {
                "claude": [self._entry("methanol", role="solvent")],
                "gpt": [self._entry("methanol", role="solvent")],
                "gemini": [self._entry("methanol", role="reagent")],
            }
        )
        rows = compute_cluster_consensus(nway, baseline="claude")
        role = self._role_row(rows)
        self.assertEqual(role.pattern, "baseline_majority")
        self.assertEqual(role.majority_value, "solvent")
        self.assertEqual(role.majority_supporters, frozenset({"gpt"}))
        self.assertEqual(role.not_in_majority, frozenset({"gemini"}))

    def test_split(self) -> None:
        """No plurality among non-baseline models."""
        nway = diff_compounds_nway(
            {
                "claude": [self._entry("methanol", role="solvent")],
                "gpt": [self._entry("methanol", role="reagent")],
                "gemini": [self._entry("methanol", role="catalyst")],
                "llama": [self._entry("methanol", role="product")],
            }
        )
        rows = compute_cluster_consensus(nway, baseline="claude")
        role = self._role_row(rows)
        self.assertEqual(role.pattern, "split")
        self.assertEqual(
            role.not_in_majority,
            frozenset({"claude", "gpt", "gemini", "llama"}),
        )

    def test_fewer_than_three_models_are_single_model(self) -> None:
        nway = diff_compounds_nway(
            {
                "claude": [self._entry("methanol", role="solvent")],
                "gpt": [self._entry("methanol", role="reagent")],
            }
        )
        rows = compute_cluster_consensus(nway, baseline="claude")
        self.assertTrue(rows)
        self.assertTrue(all(row.pattern == "single_model" for row in rows))

        fig = build_consensus_chart(rows)
        # single_model rows are excluded from chart counts
        total = sum(sum(trace.y) for trace in fig.data)
        self.assertEqual(total, 0)

    def test_consensus_rows_to_dataframe(self) -> None:
        nway = diff_compounds_nway(
            {
                "claude": [self._entry("methanol", role="solvent")],
                "gpt": [self._entry("methanol", role="reagent")],
                "gemini": [self._entry("methanol", role="reagent")],
            }
        )
        rows = compute_cluster_consensus(nway, baseline="claude")
        flagged = [row for row in rows if row.pattern == "majority_vs_baseline"]
        self.assertTrue(flagged)
        df = consensus_rows_to_dataframe(flagged, model_labels=nway.labels)
        self.assertIn("cluster_identifier", df.columns)
        self.assertIn("field", df.columns)
        self.assertIn("claude", df.columns)
        self.assertIn("gpt", df.columns)
        self.assertIn("gemini", df.columns)
        self.assertIn("baseline_value", df.columns)
        self.assertIn("majority_value", df.columns)
        self.assertIn("majority_supporters", df.columns)
        self.assertIn("not_in_majority", df.columns)
        self.assertIn("pattern", df.columns)
        self.assertEqual(df.iloc[0]["pattern"], "majority_vs_baseline")
        self.assertEqual(df.iloc[0]["majority_supporters"], "gemini, gpt")
        self.assertEqual(df.iloc[0]["not_in_majority"], "claude")


if __name__ == "__main__":
    unittest.main()
