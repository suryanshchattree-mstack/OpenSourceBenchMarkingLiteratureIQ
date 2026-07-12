"""Tests for M1 field-agreement metrics with deliberate mismatches."""

from __future__ import annotations

import unittest

from core.m1_agreement import compute_m1_agreement, filter_cluster_rows
from core.m1_parsing import QUANTITY_FIELDS, M1CompoundEntry


class M1AgreementTest(unittest.TestCase):
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

    def test_deliberate_type_role_and_quantity_mismatches(self) -> None:
        """Shared cluster with identifier_type, role, product flag, alias, quantity diffs."""
        baseline = [
            self._entry(
                "N,N-diisopropylethylamine",
                aliases=("DIPEA",),
                identifier_type="iupac",
                role="base",
                is_section_product=False,
                quantity={"mass_g": 1.0, "mmol": 5.0},
            ),
            self._entry(
                "methanol",
                aliases=("MeOH",),
                identifier_type="trivial_name",
                role="solvent",
                is_section_product=False,
                quantity={"volume_ml": 10.0},
            ),
        ]
        model = [
            self._entry(
                "DIPEA",
                aliases=("DIPEA", "Hunig's base"),
                identifier_type="abbreviation",  # mismatch vs iupac
                role="reagent",  # mismatch vs base
                is_section_product=True,  # mismatch
                quantity={"mass_g": 1.0},  # mmol missing vs populated
            ),
            self._entry(
                "MeOH",
                aliases=("MeOH",),
                identifier_type="trivial_name",
                role="solvent",
                is_section_product=False,
                quantity={"volume_ml": 10.0},
            ),
        ]

        result = compute_m1_agreement(
            {"claude": baseline, "gpt": model},
            baseline="claude",
        )

        self.assertEqual(result.baseline, "claude")
        self.assertEqual(len(result.summaries), 1)
        summary = result.summaries["gpt"]
        self.assertEqual(summary.common, 2)
        self.assertEqual(summary.baseline_only, 0)
        self.assertEqual(summary.model_only, 0)

        # One of two clusters disagrees on identifier_type / role / product flag.
        self.assertEqual(summary.identifier_type.agree, 1)
        self.assertEqual(summary.identifier_type.disagree, 1)
        self.assertIn("abbreviation<->iupac", summary.identifier_type.confusion_pairs)
        self.assertEqual(summary.role.agree, 1)
        self.assertEqual(summary.role.disagree, 1)
        self.assertIn("base<->reagent", summary.role.confusion_pairs)
        self.assertEqual(summary.is_section_product.agree, 1)
        self.assertEqual(summary.is_section_product.disagree, 1)

        self.assertIsNotNone(summary.alias_jaccard_mean)
        self.assertGreater(summary.alias_jaccard_mean, 0.0)
        self.assertLess(summary.alias_jaccard_mean, 1.0)

        # mmol presence disagrees on DIPEA pair; mass_g/volume_ml agree across both pairs.
        self.assertEqual(summary.quantity.field_disagree["mmol"], 1)
        self.assertEqual(summary.quantity.field_agree["mass_g"], 2)
        self.assertEqual(summary.quantity.field_agree["volume_ml"], 2)

        self.assertEqual(len(result.cluster_rows), 2)
        dipea_row = next(
            row for row in result.cluster_rows if row.model_identifier == "DIPEA"
        )
        self.assertFalse(dipea_row.identifier_type_agree)
        self.assertFalse(dipea_row.role_agree)
        self.assertFalse(dipea_row.is_section_product_agree)
        self.assertFalse(dipea_row.quantity_field_agree["mmol"])
        self.assertTrue(dipea_row.quantity_field_agree["mass_g"])
        self.assertEqual(dipea_row.baseline_aliases, ("DIPEA",))
        self.assertEqual(dipea_row.model_aliases, ("DIPEA", "Hunig's base"))
        self.assertEqual(dipea_row.baseline_quantity["mmol"], 5.0)
        self.assertIsNone(dipea_row.model_quantity["mmol"])
        self.assertIn("identifier_type", dipea_row.failed_fields)
        self.assertIn("role", dipea_row.failed_fields)
        self.assertIn("is_section_product", dipea_row.failed_fields)
        self.assertIn("aliases", dipea_row.failed_fields)
        self.assertIn("mmol", dipea_row.failed_fields)
        self.assertTrue(dipea_row.has_disagreement)

        methanol_row = next(
            row for row in result.cluster_rows if row.model_identifier == "MeOH"
        )
        self.assertFalse(methanol_row.has_disagreement)
        self.assertEqual(methanol_row.failed_fields, ())

    def test_filter_cluster_rows_combinations(self) -> None:
        result = compute_m1_agreement(
            {
                "claude": [
                    self._entry(
                        "DIPEA",
                        aliases=("DIPEA",),
                        identifier_type="iupac",
                        role="base",
                        quantity={"mmol": 1.0},
                    ),
                    self._entry(
                        "methanol",
                        aliases=("MeOH",),
                        identifier_type="trivial_name",
                        role="solvent",
                    ),
                ],
                "gpt": [
                    self._entry(
                        "DIPEA",
                        aliases=("DIPEA", "Hunig"),
                        identifier_type="abbreviation",
                        role="reagent",
                        quantity={},
                    ),
                    self._entry(
                        "methanol",
                        aliases=("MeOH",),
                        identifier_type="trivial_name",
                        role="solvent",
                    ),
                ],
                "glm": [
                    self._entry(
                        "DIPEA",
                        aliases=("DIPEA",),
                        identifier_type="iupac",
                        role="base",
                        quantity={"mmol": 1.0},
                    ),
                    self._entry(
                        "methanol",
                        aliases=("MeOH",),
                        identifier_type="trivial_name",
                        role="solvent",
                    ),
                ],
            },
            baseline="claude",
        )

        disagreements = filter_cluster_rows(
            result.cluster_rows,
            disagreements_only=True,
        )
        self.assertEqual(len(disagreements), 1)
        self.assertEqual(disagreements[0].model_label, "gpt")

        all_rows = filter_cluster_rows(
            result.cluster_rows,
            disagreements_only=False,
        )
        self.assertEqual(len(all_rows), 4)

        role_only = filter_cluster_rows(
            result.cluster_rows,
            field_filter="role",
            disagreements_only=False,
        )
        self.assertEqual(len(role_only), 1)
        self.assertIn("role", role_only[0].failed_fields)

        qty = filter_cluster_rows(
            result.cluster_rows,
            field_filter="mmol",
            disagreements_only=False,
        )
        self.assertEqual(len(qty), 1)

        searched = filter_cluster_rows(
            result.cluster_rows,
            identifier_query="dipea",
            disagreements_only=False,
        )
        self.assertEqual(len(searched), 2)

        model_filtered = filter_cluster_rows(
            result.cluster_rows,
            model_labels=["glm"],
            disagreements_only=False,
        )
        self.assertEqual(len(model_filtered), 2)
        self.assertTrue(all(row.model_label == "glm" for row in model_filtered))

    def test_perfect_agreement_on_matched_pair(self) -> None:
        shared = self._entry(
            "toluene",
            identifier_type="trivial_name",
            role="solvent",
            aliases=("PhMe",),
            quantity={"volume_ml": 5.0},
        )
        result = compute_m1_agreement(
            {"baseline": [shared], "model": [shared]},
            baseline="baseline",
        )
        summary = result.summaries["model"]
        self.assertEqual(summary.identifier_type.rate, 1.0)
        self.assertEqual(summary.role.rate, 1.0)
        self.assertEqual(summary.is_section_product.rate, 1.0)
        self.assertEqual(summary.alias_jaccard_mean, 1.0)
        self.assertEqual(summary.quantity.overall_rate, 1.0)
        self.assertEqual(result.cluster_rows[0].failed_fields, ())
        self.assertEqual(result.cluster_rows[0].baseline_aliases, ("PhMe",))

    def test_model_only_compound_does_not_inflate_agreement(self) -> None:
        result = compute_m1_agreement(
            {
                "claude": [self._entry("methanol")],
                "gpt": [
                    self._entry("methanol"),
                    self._entry("extra-only-in-gpt"),
                ],
            },
            baseline="claude",
        )
        summary = result.summaries["gpt"]
        self.assertEqual(summary.common, 1)
        self.assertEqual(summary.model_only, 1)
        self.assertEqual(summary.identifier_type.total, 1)
        self.assertEqual(len(result.cluster_rows), 1)

    def test_requires_baseline_and_two_labels(self) -> None:
        with self.assertRaises(KeyError):
            compute_m1_agreement({"a": [self._entry("x")]}, baseline="missing")
        with self.assertRaises(ValueError):
            compute_m1_agreement({"a": [self._entry("x")]}, baseline="a")


if __name__ == "__main__":
    unittest.main()
