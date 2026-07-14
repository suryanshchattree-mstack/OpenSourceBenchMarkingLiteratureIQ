"""Tests for tiered compound matching waterfall."""

from __future__ import annotations

import unittest

from core.compound_matching import canonicalize_smiles, diff_compounds_nway
from core.compound_parsing import CompoundEntry


def _entry(
    identifier: str,
    *,
    aliases: tuple[str, ...] = (),
    smiles: str | None = None,
    inchi_key: str | None = None,
    molecular_formula: str | None = None,
) -> CompoundEntry:
    return CompoundEntry(
        identifier=identifier,
        identifier_type="iupac",
        aliases=aliases,
        resolved=True,
        unresolved_reference=False,
        section_label="Example 1",
        role="reagent",
        smiles=smiles,
        inchi_key=inchi_key,
        molecular_formula=molecular_formula,
    )


class CanonicalizeSmilesTest(unittest.TestCase):
    def test_canonicalizes_equivalent_forms(self) -> None:
        left = canonicalize_smiles("C(O)C")
        right = canonicalize_smiles("CCO")
        self.assertIsNotNone(left)
        self.assertEqual(left, right)

    def test_bad_smiles_returns_none(self) -> None:
        self.assertIsNone(canonicalize_smiles("not-a-smiles%%%"))


class TieredMatchingTest(unittest.TestCase):
    def test_inchi_key_only_match(self) -> None:
        key = "OKKJLVBELUTLKV-UHFFFAOYSA-N"
        result = diff_compounds_nway(
            {
                "a": [_entry("methanol", inchi_key=key)],
                "b": [_entry("MeOH", inchi_key=key)],
            }
        )
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].membership, frozenset({"a", "b"}))
        self.assertEqual(result.clusters[0].match_tier, "inchi_key")

    def test_smiles_only_match_with_canonicalization(self) -> None:
        result = diff_compounds_nway(
            {
                "a": [_entry("ethanol", smiles="CCO")],
                "b": [_entry("EtOH", smiles="C(O)C")],
            }
        )
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].match_tier, "smiles")

    def test_formula_only_match(self) -> None:
        result = diff_compounds_nway(
            {
                "a": [_entry("Compound A", molecular_formula="C6H6")],
                "b": [_entry("Compound B", molecular_formula="C6H6")],
            }
        )
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].match_tier, "molecular_formula")

    def test_name_only_match_still_works(self) -> None:
        result = diff_compounds_nway(
            {
                "claude": [_entry("N,N-diisopropylethylamine", aliases=("DIPEA",))],
                "gpt": [_entry("DIPEA")],
            }
        )
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].match_tier, "name")

    def test_mixed_weakest_tier_recorded(self) -> None:
        """A–B via InChIKey, B–C via name ⇒ cluster match_tier is name."""
        key = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        result = diff_compounds_nway(
            {
                "a": [_entry("ethanol", inchi_key=key, aliases=())],
                "b": [_entry("EtOH", inchi_key=key, aliases=("ethanol",))],
                "c": [_entry("ethyl alcohol", aliases=("ethanol",))],
            }
        )
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].membership, frozenset({"a", "b", "c"}))
        self.assertEqual(result.clusters[0].match_tier, "name")

    def test_singleton_has_no_match_tier(self) -> None:
        result = diff_compounds_nway(
            {
                "a": [_entry("Unique A")],
                "b": [_entry("Unique B")],
            }
        )
        self.assertEqual(len(result.clusters), 2)
        for cluster in result.clusters:
            self.assertIsNone(cluster.match_tier)


if __name__ == "__main__":
    unittest.main()
