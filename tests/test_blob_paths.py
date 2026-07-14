"""Unit tests for pure Azure blob path builders."""

from __future__ import annotations

import unittest
import uuid

from core.blob_paths import (
    BASELINE_PIPELINE_ID,
    base_path,
    compounds_fallback_paths,
    compounds_path,
    country_code,
    hash_bucket,
    markdown_paths,
    pipeline_prefix,
    prepass_prefix,
    r1_path,
    reactions_fallback_paths,
    reactions_path,
    uuid5_name,
)


class BlobPathsTest(unittest.TestCase):
    def test_uuid5_name_matches_dns_namespace(self) -> None:
        patent_id = "CN105884573B"
        self.assertEqual(
            uuid5_name(patent_id),
            str(uuid.uuid5(uuid.NAMESPACE_DNS, patent_id)),
        )
        self.assertEqual(
            uuid5_name(patent_id),
            "4506b18c-80c1-5905-ad34-cecfd06ac15b",
        )

    def test_hash_bucket_deterministic(self) -> None:
        patent_id = "CN105884573B"
        bucket = hash_bucket(patent_id)
        self.assertEqual(bucket, hash_bucket(patent_id))
        self.assertIsInstance(bucket, int)
        self.assertGreaterEqual(bucket, 0)
        self.assertLess(bucket, 128)
        # mmh3 signed murmur3_32 of UTF-8; mirrors Guava murmur3_32_fixed + floorMod
        self.assertEqual(bucket, 99)

    def test_country_code(self) -> None:
        self.assertEqual(country_code("CN105884573B"), "CN")
        self.assertEqual(country_code("US1234567B2"), "US")
        self.assertEqual(country_code("wo2017133990a1"), "WO")
        self.assertEqual(country_code("X"), "XX")
        self.assertEqual(country_code(""), "XX")

    def test_base_path_assembly(self) -> None:
        patent_id = "CN105884573B"
        path = base_path(patent_id)
        self.assertEqual(
            path,
            f"literature/patents/CN/99/{uuid5_name(patent_id)}",
        )
        self.assertEqual(
            base_path(patent_id, bucket=7),
            f"literature/patents/CN/7/{uuid5_name(patent_id)}",
        )

    def test_pipeline_kind_paths(self) -> None:
        base = "literature/patents/CN/99/abc"
        pipeline_id = "section-wise-v1-deepseek-flash"
        self.assertEqual(
            pipeline_prefix(base, pipeline_id),
            f"{base}/extraction/{pipeline_id}/",
        )
        self.assertEqual(
            prepass_prefix(base, pipeline_id),
            f"{base}/extraction/{pipeline_id}/pre-pass-",
        )
        self.assertEqual(
            r1_path(base, pipeline_id),
            f"{base}/extraction/{pipeline_id}/reaction-pass-1-consolidated.json",
        )

    def test_compounds_path_baseline_vs_other(self) -> None:
        base = "literature/patents/CN/99/abc"
        self.assertEqual(
            compounds_path(base, BASELINE_PIPELINE_ID),
            f"{base}/compounds.json",
        )
        other = "section-wise-v1-deepseek-flash"
        self.assertEqual(
            compounds_path(base, other),
            f"{base}/extraction/{other}/compounds.json",
        )

    def test_compounds_fallback_order(self) -> None:
        base = "literature/patents/CN/99/abc"
        patent_id = "CN105884573B"
        baseline_fallbacks = compounds_fallback_paths(
            base, patent_id, BASELINE_PIPELINE_ID
        )
        self.assertEqual(
            baseline_fallbacks,
            [
                f"{base}/compounds.json",
                f"{base}/extraction/{BASELINE_PIPELINE_ID}/compounds.json",
                f"persistent-store/{patent_id}/compounds.json",
            ],
        )
        other = "section-wise-v1-deepseek-flash"
        other_fallbacks = compounds_fallback_paths(base, patent_id, other)
        self.assertEqual(
            other_fallbacks,
            [
                f"{base}/extraction/{other}/compounds.json",
                f"{base}/compounds.json",
                f"persistent-store/{patent_id}/compounds.json",
            ],
        )

    def test_reactions_path_baseline_vs_other(self) -> None:
        base = "literature/patents/CN/99/abc"
        self.assertEqual(
            reactions_path(base, BASELINE_PIPELINE_ID),
            f"{base}/reactions.json",
        )
        other = "section-wise-v1-deepseek-flash"
        self.assertEqual(
            reactions_path(base, other),
            f"{base}/extraction/{other}/reactions.json",
        )

    def test_reactions_fallback_order(self) -> None:
        base = "literature/patents/CN/99/abc"
        patent_id = "CN105884573B"
        baseline_fallbacks = reactions_fallback_paths(
            base, patent_id, BASELINE_PIPELINE_ID
        )
        self.assertEqual(
            baseline_fallbacks,
            [
                f"{base}/reactions.json",
                f"{base}/extraction/{BASELINE_PIPELINE_ID}/reactions.json",
                f"persistent-store/{patent_id}/reactions.json",
            ],
        )
        other = "section-wise-v1-deepseek-flash"
        other_fallbacks = reactions_fallback_paths(base, patent_id, other)
        self.assertEqual(
            other_fallbacks,
            [
                f"{base}/extraction/{other}/reactions.json",
                f"{base}/reactions.json",
                f"persistent-store/{patent_id}/reactions.json",
            ],
        )

    def test_markdown_paths(self) -> None:
        base = "literature/patents/CN/99/abc"
        self.assertEqual(
            markdown_paths(base),
            [
                f"{base}/enriched/en/markdown.md",
                f"{base}/en/markdown.md",
            ],
        )


if __name__ == "__main__":
    unittest.main()
