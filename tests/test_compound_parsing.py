"""Tests for CompoundEntry field round-trip from source JSON."""

from __future__ import annotations

import json
import unittest

from core.compound_parsing import parse_compounds_json


class CompoundParsingFieldsTest(unittest.TestCase):
    def test_raw_and_structure_fields_round_trip(self) -> None:
        payload = [
            {
                "identifier": "methanol",
                "identifier_type": "trivial_name",
                "aliases": ["MeOH"],
                "resolved": True,
                "unresolved_reference": False,
                "section_label": "Example 1",
                "role": "solvent",
                "smiles": "CO",
                "inchi_key": "OKKJLVBELUTLKV-UHFFFAOYSA-N",
                "molecular_formula": "CH4O",
                "quantity": "10 mL",
            }
        ]
        entries = parse_compounds_json(json.dumps(payload), source_label="test")
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.smiles, "CO")
        self.assertEqual(entry.inchi_key, "OKKJLVBELUTLKV-UHFFFAOYSA-N")
        self.assertEqual(entry.molecular_formula, "CH4O")
        self.assertEqual(entry.raw["quantity"], "10 mL")
        self.assertEqual(entry.raw["identifier"], "methanol")
        self.assertEqual(entry.raw["smiles"], "CO")

    def test_missing_structure_fields_are_none(self) -> None:
        payload = [
            {
                "identifier": "DIPEA",
                "identifier_type": "abbreviation",
                "aliases": [],
                "resolved": False,
                "unresolved_reference": False,
            }
        ]
        entries = parse_compounds_json(json.dumps(payload), source_label="test")
        entry = entries[0]
        self.assertIsNone(entry.smiles)
        self.assertIsNone(entry.inchi_key)
        self.assertIsNone(entry.molecular_formula)
        self.assertEqual(entry.raw["identifier"], "DIPEA")


if __name__ == "__main__":
    unittest.main()
