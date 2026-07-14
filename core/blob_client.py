"""Azure Blob fetch orchestration for benchmark artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from dotenv import load_dotenv

from core.blob_paths import (
    HASH_BUCKET_COUNT,
    base_path,
    compounds_fallback_paths,
    country_code,
    markdown_paths,
    prepass_prefix,
    r1_path,
    reactions_fallback_paths,
    uuid5_name,
)

load_dotenv()

DEFAULT_CONTAINER = "datalake-raw-store"

FILE_KINDS = ("prepass", "compounds", "r1", "reactions")


@dataclass(frozen=True)
class FetchResult:
    found: bool
    blob_path: str | None
    content: bytes | None
    filename: str | None
    error: str | None = None


class BlobConfigError(RuntimeError):
    """Raised when Azure connection settings are missing or invalid."""


def _connection_string() -> str:
    value = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not value:
        raise BlobConfigError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. "
            "Copy .env.example to .env and add your connection string."
        )
    return value


def _container_name() -> str:
    return os.getenv("AZURE_STORAGE_CONTAINER", DEFAULT_CONTAINER).strip() or DEFAULT_CONTAINER


@lru_cache(maxsize=1)
def get_container_client():
    """Build a cached BlobContainerClient from env (or raise BlobConfigError)."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise BlobConfigError(
            "azure-storage-blob is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        service = BlobServiceClient.from_connection_string(_connection_string())
        return service.get_container_client(_container_name())
    except BlobConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface Azure auth/config failures clearly
        raise BlobConfigError(f"Failed to create Azure blob client: {exc}") from exc


def reset_blob_client_cache() -> None:
    """Clear the cached container client (useful in tests)."""
    get_container_client.cache_clear()
    _resolved_base_paths.clear()


# Patent ID -> resolved base path (avoids repeating brute-force bucket scan).
_resolved_base_paths: dict[str, str] = {}


def _prefix_has_blobs(container, prefix: str) -> bool:
    iterator = container.list_blobs(name_starts_with=prefix, results_per_page=1)
    return next(iter(iterator), None) is not None


def resolve_base_path(
    patent_id: str,
    *,
    container=None,
    cache: dict[str, str] | None = None,
) -> str:
    """
    Resolve ``literature/patents/{cc}/{bucket}/{uuid5}``.

    Tries the computed murmur3 bucket first; if no blobs exist under that prefix,
    scans all 128 buckets for the country code (bounded fallback for hash mismatch).
    """
    patent_id = patent_id.strip()
    store = _resolved_base_paths if cache is None else cache
    if patent_id in store:
        return store[patent_id]

    client = container if container is not None else get_container_client()
    computed = base_path(patent_id)
    if _prefix_has_blobs(client, computed + "/"):
        store[patent_id] = computed
        return computed
    # Also accept exact-prefix hits without trailing slash (rare).
    if _prefix_has_blobs(client, computed):
        store[patent_id] = computed
        return computed

    cc = country_code(patent_id)
    uid = uuid5_name(patent_id)
    for bucket in range(HASH_BUCKET_COUNT):
        candidate = f"literature/patents/{cc}/{bucket}/{uid}"
        if candidate == computed:
            continue
        if _prefix_has_blobs(client, candidate + "/") or _prefix_has_blobs(client, candidate):
            store[patent_id] = candidate
            return candidate

    store[patent_id] = computed
    return computed


def latest_blob_by_prefix(container, prefix: str) -> str | None:
    """Return the lexicographically-max blob name under ``prefix``, or None."""
    names = [blob.name for blob in container.list_blobs(name_starts_with=prefix)]
    if not names:
        return None
    return max(names)


def _download_blob(container, blob_path: str) -> bytes:
    return container.get_blob_client(blob_path).download_blob().readall()


def _filename_from_path(blob_path: str) -> str:
    return blob_path.rsplit("/", 1)[-1]


def _missing(blob_path: str | None = None, error: str | None = None) -> FetchResult:
    return FetchResult(
        found=False,
        blob_path=blob_path,
        content=None,
        filename=None,
        error=error,
    )


def _found(blob_path: str, content: bytes) -> FetchResult:
    return FetchResult(
        found=True,
        blob_path=blob_path,
        content=content,
        filename=_filename_from_path(blob_path),
        error=None,
    )


def _safe_download(container, blob_path: str) -> FetchResult:
    try:
        if not container.get_blob_client(blob_path).exists():
            return _missing(blob_path)
        content = _download_blob(container, blob_path)
        return _found(blob_path, content)
    except Exception as exc:  # noqa: BLE001
        return _missing(blob_path, error=str(exc))


def _fetch_prepass(container, base: str, pipeline_id: str) -> FetchResult:
    prefix = prepass_prefix(base, pipeline_id)
    try:
        blob_path = latest_blob_by_prefix(container, prefix)
        if blob_path is None:
            return _missing(prefix)
        content = _download_blob(container, blob_path)
        return _found(blob_path, content)
    except Exception as exc:  # noqa: BLE001
        return _missing(prefix, error=str(exc))


def _fetch_reactions(container, base: str, patent_id: str, pipeline_id: str) -> FetchResult:
    last: FetchResult | None = None
    for path in reactions_fallback_paths(base, patent_id, pipeline_id):
        result = _safe_download(container, path)
        if result.found:
            return result
        last = result
    return last if last is not None else _missing()


def _fetch_compounds(container, base: str, patent_id: str, pipeline_id: str) -> FetchResult:
    last: FetchResult | None = None
    for path in compounds_fallback_paths(base, patent_id, pipeline_id):
        result = _safe_download(container, path)
        if result.found:
            return result
        last = result
    return last if last is not None else _missing()


def fetch_pipeline_artifacts(
    patent_id: str,
    pipeline_id: str,
    *,
    container=None,
    resolve_base: Callable[[str], str] | None = None,
) -> dict[str, FetchResult]:
    """
    Fetch the four FILE_KINDS for ``pipeline_id``.

    Missing kinds return ``found=False`` without raising. Azure/auth errors are
    captured on each result's ``error`` field when possible; config errors raise
    ``BlobConfigError``.
    """
    patent_id = patent_id.strip()
    pipeline_id = pipeline_id.strip()
    if not patent_id:
        raise ValueError("Patent ID is required")
    if not pipeline_id:
        raise ValueError("Pipeline ID is required")

    client = container if container is not None else get_container_client()
    base = resolve_base(patent_id) if resolve_base else resolve_base_path(patent_id, container=client)

    results: dict[str, FetchResult] = {
        "prepass": _fetch_prepass(client, base, pipeline_id),
        "compounds": _fetch_compounds(client, base, patent_id, pipeline_id),
        "r1": _safe_download(client, r1_path(base, pipeline_id)),
        "reactions": _fetch_reactions(client, base, patent_id, pipeline_id),
    }
    return results


def fetch_markdown(
    patent_id: str,
    *,
    container=None,
    resolve_base: Callable[[str], str] | None = None,
) -> FetchResult:
    """Fetch shared enriched markdown (not pipeline-scoped)."""
    patent_id = patent_id.strip()
    if not patent_id:
        raise ValueError("Patent ID is required")

    client = container if container is not None else get_container_client()
    base = resolve_base(patent_id) if resolve_base else resolve_base_path(patent_id, container=client)

    last: FetchResult | None = None
    for path in markdown_paths(base):
        result = _safe_download(client, path)
        if result.found:
            return result
        last = result
    return last if last is not None else _missing()
