"""Tests for Azure blob fetch orchestration (mocked SDK)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.blob_client import (
    FetchResult,
    fetch_markdown,
    fetch_pipeline_artifacts,
    latest_blob_by_prefix,
    resolve_base_path,
)
from core.blob_paths import (
    BASELINE_PIPELINE_ID,
    base_path,
    compounds_path,
    reactions_path,
    uuid5_name,
)


def _blob_item(name: str) -> MagicMock:
    item = MagicMock()
    item.name = name
    return item


class FakeBlobClient:
    def __init__(self, store: dict[str, bytes], name: str) -> None:
        self._store = store
        self.name = name

    def exists(self) -> bool:
        return self.name in self._store

    def download_blob(self) -> MagicMock:
        downloader = MagicMock()
        downloader.readall.return_value = self._store[self.name]
        return downloader


class FakeContainer:
    def __init__(self, store: dict[str, bytes]) -> None:
        self.store = store

    def list_blobs(self, name_starts_with: str | None = None, results_per_page: int | None = None):
        _ = results_per_page
        names = sorted(self.store)
        if name_starts_with:
            names = [name for name in names if name.startswith(name_starts_with)]
        return [_blob_item(name) for name in names]

    def get_blob_client(self, blob: str) -> FakeBlobClient:
        return FakeBlobClient(self.store, blob)


class BlobClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.patent_id = "CN105884573B"
        self.base = base_path(self.patent_id)
        self.pipeline_id = "section-wise-v1-deepseek-flash"

    def test_latest_blob_by_prefix_picks_lexicographic_max(self) -> None:
        prefix = f"{self.base}/extraction/{self.pipeline_id}/pre-pass-"
        store = {
            f"{prefix}20240101T000000Z.json": b"old",
            f"{prefix}20240615T120000Z.json": b"new",
            f"{prefix}20240301T000000Z.json": b"mid",
        }
        container = FakeContainer(store)
        latest = latest_blob_by_prefix(container, prefix)
        self.assertEqual(latest, f"{prefix}20240615T120000Z.json")

    def test_resolve_base_path_uses_computed_when_present(self) -> None:
        store = {f"{self.base}/enriched/en/markdown.md": b"# md"}
        container = FakeContainer(store)
        cache: dict[str, str] = {}
        resolved = resolve_base_path(self.patent_id, container=container, cache=cache)
        self.assertEqual(resolved, self.base)
        self.assertEqual(cache[self.patent_id], self.base)

    def test_resolve_base_path_bruteforces_wrong_bucket(self) -> None:
        uid = uuid5_name(self.patent_id)
        alt_base = f"literature/patents/CN/7/{uid}"
        store = {f"{alt_base}/en/markdown.md": b"# legacy"}
        container = FakeContainer(store)
        cache: dict[str, str] = {}
        resolved = resolve_base_path(self.patent_id, container=container, cache=cache)
        self.assertEqual(resolved, alt_base)
        # Second call hits cache without depending on container.
        self.assertEqual(
            resolve_base_path(self.patent_id, container=FakeContainer({}), cache=cache),
            alt_base,
        )

    def test_fetch_pipeline_artifacts_latest_prepass_and_missing_kinds(self) -> None:
        prefix = f"{self.base}/extraction/{self.pipeline_id}/pre-pass-"
        compounds = compounds_path(self.base, self.pipeline_id)
        reactions = reactions_path(self.base, self.pipeline_id)
        store = {
            f"{prefix}20240101T000000Z.json": b'[{"old": true}]',
            f"{prefix}20240615T120000Z.json": b'[{"new": true}]',
            compounds: b'{"compounds": true}',
            reactions: b'{"reactions": true}',
        }
        container = FakeContainer(store)
        results = fetch_pipeline_artifacts(
            self.patent_id,
            self.pipeline_id,
            container=container,
            resolve_base=lambda _: self.base,
        )

        self.assertTrue(results["prepass"].found)
        self.assertEqual(results["prepass"].content, b'[{"new": true}]')
        self.assertEqual(
            results["prepass"].blob_path,
            f"{prefix}20240615T120000Z.json",
        )
        self.assertEqual(results["prepass"].filename, "pre-pass-20240615T120000Z.json")

        self.assertTrue(results["compounds"].found)
        self.assertEqual(results["compounds"].content, b'{"compounds": true}')
        self.assertEqual(results["compounds"].blob_path, compounds)

        self.assertFalse(results["r1"].found)

        self.assertTrue(results["reactions"].found)
        self.assertEqual(results["reactions"].content, b'{"reactions": true}')
        self.assertEqual(results["reactions"].blob_path, reactions)

    def test_fetch_reactions_does_not_use_baseline_top_level(self) -> None:
        # Non-baseline: root reactions.json must not satisfy extraction/{id}/ path
        store = {f"{self.base}/reactions.json": b'[{"id": 1}]'}
        container = FakeContainer(store)
        results = fetch_pipeline_artifacts(
            self.patent_id,
            self.pipeline_id,
            container=container,
            resolve_base=lambda _: self.base,
        )
        self.assertFalse(results["reactions"].found)

    def test_fetch_reactions_baseline_primary_path(self) -> None:
        store = {f"{self.base}/reactions.json": b'[{"id": 1}]'}
        container = FakeContainer(store)
        results = fetch_pipeline_artifacts(
            self.patent_id,
            BASELINE_PIPELINE_ID,
            container=container,
            resolve_base=lambda _: self.base,
        )
        self.assertTrue(results["reactions"].found)
        self.assertEqual(results["reactions"].blob_path, f"{self.base}/reactions.json")

    def test_fetch_reactions_ignores_legacy_persistent_store(self) -> None:
        legacy = f"persistent-store/{self.patent_id}/reactions.json"
        store = {legacy: b'[{"legacy": true}]'}
        container = FakeContainer(store)
        results = fetch_pipeline_artifacts(
            self.patent_id,
            self.pipeline_id,
            container=container,
            resolve_base=lambda _: self.base,
        )
        self.assertFalse(results["reactions"].found)

    def test_fetch_compounds_ignores_legacy_persistent_store(self) -> None:
        legacy = f"persistent-store/{self.patent_id}/compounds.json"
        store = {legacy: b'[{"legacy": true}]'}
        container = FakeContainer(store)
        results = fetch_pipeline_artifacts(
            self.patent_id,
            self.pipeline_id,
            container=container,
            resolve_base=lambda _: self.base,
        )
        self.assertFalse(results["compounds"].found)

    def test_fetch_compounds_baseline_primary_path(self) -> None:
        store = {f"{self.base}/compounds.json": b'[{"id": 1}]'}
        container = FakeContainer(store)
        results = fetch_pipeline_artifacts(
            self.patent_id,
            BASELINE_PIPELINE_ID,
            container=container,
            resolve_base=lambda _: self.base,
        )
        self.assertTrue(results["compounds"].found)
        self.assertEqual(results["compounds"].blob_path, f"{self.base}/compounds.json")

    def test_fetch_markdown_prefers_enriched(self) -> None:
        enriched = f"{self.base}/enriched/en/markdown.md"
        legacy = f"{self.base}/en/markdown.md"
        store = {
            enriched: b"# enriched",
            legacy: b"# legacy",
        }
        container = FakeContainer(store)
        result = fetch_markdown(
            self.patent_id,
            container=container,
            resolve_base=lambda _: self.base,
        )
        self.assertTrue(result.found)
        self.assertEqual(result.content, b"# enriched")
        self.assertEqual(result.blob_path, enriched)

    def test_fetch_markdown_falls_back_to_legacy(self) -> None:
        legacy = f"{self.base}/en/markdown.md"
        store = {legacy: b"# legacy"}
        container = FakeContainer(store)
        result = fetch_markdown(
            self.patent_id,
            container=container,
            resolve_base=lambda _: self.base,
        )
        self.assertTrue(result.found)
        self.assertEqual(result.content, b"# legacy")

    def test_missing_kinds_do_not_raise(self) -> None:
        container = FakeContainer({})
        results = fetch_pipeline_artifacts(
            self.patent_id,
            self.pipeline_id,
            container=container,
            resolve_base=lambda _: self.base,
        )
        self.assertEqual(set(results), {"prepass", "compounds", "r1", "reactions"})
        for result in results.values():
            self.assertIsInstance(result, FetchResult)
            self.assertFalse(result.found)
            self.assertIsNone(result.content)


if __name__ == "__main__":
    unittest.main()
