"""Tests for deterministic M2 compound diff."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.compound_matching import canonicalize_name, diff_compounds
from core.compound_parsing import CompoundEntry, parse_compounds_json


class CanonicalizeNameTest(unittest.TestCase):
    def test_matches_java_normalizer_behavior(self) -> None:
        self.assertEqual(
            canonicalize_name("N-tert- butyl"),
            "n-tert-butyl",
        )
        self.assertEqual(
            canonicalize_name("  DIPEA. "),
            "dipea",
        )
        self.assertIsNone(canonicalize_name(None))
        self.assertIsNone(canonicalize_name("   "))


class CompoundParsingTest(unittest.TestCase):
    def test_parses_flat_array(self) -> None:
        payload = [
            {
                "identifier": "N,N-diisopropylethylamine",
                "identifier_type": "iupac",
                "aliases": ["DIPEA"],
                "resolved": True,
                "unresolved_reference": False,
                "section_label": "Example 1",
                "role": "base",
            }
        ]
        entries = parse_compounds_json(json.dumps(payload), source_label="test")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].identifier, "N,N-diisopropylethylamine")
        self.assertEqual(entries[0].aliases, ("DIPEA",))


class CompoundMatchingTest(unittest.TestCase):
    def _entry(
        self,
        identifier: str,
        *,
        identifier_type: str = "iupac",
        aliases: tuple[str, ...] = (),
        section_label: str | None = None,
    ) -> CompoundEntry:
        return CompoundEntry(
            identifier=identifier,
            identifier_type=identifier_type,
            aliases=aliases,
            resolved=True,
            unresolved_reference=False,
            section_label=section_label,
            role="reagent",
        )

    def test_alias_overlap_counts_as_common(self) -> None:
        claude = [self._entry("N,N-diisopropylethylamine", aliases=("DIPEA",))]
        benchmark = [self._entry("DIPEA")]
        result = diff_compounds(claude, benchmark)
        self.assertEqual(result.common, 1)
        self.assertEqual(result.claude_only, 0)
        self.assertEqual(result.benchmark_only, 0)

    def test_within_model_dedupe(self) -> None:
        claude = [
            self._entry("DIPEA", section_label="Example 1"),
            self._entry("N,N-diisopropylethylamine", aliases=("DIPEA",), section_label="Example 2"),
        ]
        benchmark = [self._entry("DIPEA")]
        result = diff_compounds(claude, benchmark)
        self.assertEqual(result.raw_claude_count, 2)
        self.assertEqual(result.deduped_claude_count, 1)
        self.assertEqual(result.common, 1)

    def test_only_buckets(self) -> None:
        claude = [
            self._entry("Compound A"),
            self._entry("Compound B"),
        ]
        benchmark = [
            self._entry("Compound B"),
            self._entry("Compound C"),
        ]
        result = diff_compounds(claude, benchmark)
        self.assertEqual(result.common, 1)
        self.assertEqual(result.claude_only, 1)
        self.assertEqual(result.benchmark_only, 1)
        self.assertEqual(result.claude_only_entries[0].identifier, "Compound A")
        self.assertEqual(result.benchmark_only_entries[0].identifier, "Compound C")

    def test_normalization_match(self) -> None:
        claude = [self._entry("tert-Butanol")]
        benchmark = [self._entry("tert - butanol")]
        result = diff_compounds(claude, benchmark)
        self.assertEqual(result.common, 1)


class CompoundCliIntegrationTest(unittest.TestCase):
    def test_cli_end_to_end_with_temp_files(self) -> None:
        claude_payload = [
            {
                "identifier": "methanol",
                "identifier_type": "trivial_name",
                "aliases": [],
                "resolved": True,
                "unresolved_reference": False,
                "section_label": "Example 1",
                "role": "solvent",
            },
            {
                "identifier": "Compound X",
                "identifier_type": "local_label",
                "aliases": [],
                "resolved": False,
                "unresolved_reference": True,
                "section_label": "Example 2",
                "role": "product",
            },
        ]
        benchmark_payload = [
            {
                "identifier": "MeOH",
                "identifier_type": "abbreviation",
                "aliases": ["methanol"],
                "resolved": True,
                "unresolved_reference": False,
                "section_label": "Example 1",
                "role": "solvent",
            },
            {
                "identifier": "Compound Y",
                "identifier_type": "local_label",
                "aliases": [],
                "resolved": False,
                "unresolved_reference": True,
                "section_label": "Example 3",
                "role": "product",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            claude_path = Path(tmp) / "claude.json"
            benchmark_path = Path(tmp) / "benchmark.json"
            claude_path.write_text(json.dumps(claude_payload), encoding="utf-8")
            benchmark_path.write_text(json.dumps(benchmark_payload), encoding="utf-8")

            claude = parse_compounds_json(claude_path.read_bytes(), source_label="Claude")
            benchmark = parse_compounds_json(benchmark_path.read_bytes(), source_label="DeepSeek")
            result = diff_compounds(claude, benchmark)

            self.assertEqual(result.common, 1)
            self.assertEqual(result.claude_only, 1)
            self.assertEqual(result.benchmark_only, 1)


if __name__ == "__main__":
    unittest.main()
