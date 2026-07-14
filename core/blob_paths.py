"""Pure path-building helpers for Azure literature blob layout (no network)."""

from __future__ import annotations

import uuid

import mmh3

BASELINE_PIPELINE_ID = "section-wise-v1"
HASH_BUCKET_COUNT = 128


def uuid5_name(patent_id: str) -> str:
    """DNS-namespace UUID5 of the patent ID (matches HashUtil.getUUID5)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, patent_id))


def hash_bucket(patent_id: str, buckets: int = HASH_BUCKET_COUNT) -> int:
    """
    Guava murmur3_32_fixed + floorMod(hash, buckets).

    Uses mmh3's signed 32-bit hash of UTF-8 bytes; Python ``%`` matches Java
    ``Math.floorMod`` for a positive modulus.
    """
    digest = mmh3.hash(patent_id.encode("utf-8"), signed=True)
    return digest % buckets


def country_code(patent_id: str) -> str:
    """First two letters of the patent number, uppercased (or ``XX``)."""
    if not patent_id or len(patent_id) < 2:
        return "XX"
    letters: list[str] = []
    for char in patent_id:
        if char.isalpha():
            letters.append(char)
            if len(letters) == 2:
                break
    return "".join(letters).upper() if len(letters) == 2 else "XX"


def base_path(patent_id: str, *, bucket: int | None = None) -> str:
    """``literature/patents/{cc}/{bucket}/{uuid5}``."""
    cc = country_code(patent_id)
    bucket_value = hash_bucket(patent_id) if bucket is None else bucket
    return f"literature/patents/{cc}/{bucket_value}/{uuid5_name(patent_id)}"


def pipeline_prefix(base: str, pipeline_id: str) -> str:
    return f"{base}/extraction/{pipeline_id}/"


def prepass_prefix(base: str, pipeline_id: str) -> str:
    return f"{base}/extraction/{pipeline_id}/pre-pass-"


def r1_path(base: str, pipeline_id: str) -> str:
    return f"{base}/extraction/{pipeline_id}/reaction-pass-1-consolidated.json"


def compounds_path(base: str, pipeline_id: str) -> str:
    """
    Baseline ``section-wise-v1`` stores compounds at ``{base}/compounds.json``.
    Other pipelines store under ``{base}/extraction/{pipelineId}/compounds.json``.
    """
    if pipeline_id == BASELINE_PIPELINE_ID:
        return f"{base}/compounds.json"
    return f"{base}/extraction/{pipeline_id}/compounds.json"


def compounds_fallback_paths(base: str, patent_id: str, pipeline_id: str) -> list[str]:
    """
    Candidate path(s) for compounds.json.

    Root ``{base}/compounds.json`` is baseline-only; other pipelines use only
    ``{base}/extraction/{pipelineId}/compounds.json``. No cross-layout or
    ``persistent-store`` fallback. ``patent_id`` is unused (kept for call-site
    compatibility).
    """
    _ = patent_id
    return [compounds_path(base, pipeline_id)]


def reactions_path(base: str, pipeline_id: str) -> str:
    """
    Baseline ``section-wise-v1`` stores reactions at ``{base}/reactions.json``.
    Other pipelines store under ``{base}/extraction/{pipelineId}/reactions.json``.
    """
    if pipeline_id == BASELINE_PIPELINE_ID:
        return f"{base}/reactions.json"
    return f"{base}/extraction/{pipeline_id}/reactions.json"


def reactions_fallback_paths(base: str, patent_id: str, pipeline_id: str) -> list[str]:
    """
    Candidate path(s) for reactions.json.

    Root ``{base}/reactions.json`` is baseline-only; other pipelines use only
    ``{base}/extraction/{pipelineId}/reactions.json``. No cross-layout or
    ``persistent-store`` fallback. ``patent_id`` is unused (kept for call-site
    compatibility).
    """
    _ = patent_id
    return [reactions_path(base, pipeline_id)]


def markdown_paths(base: str) -> list[str]:
    """Preferred enriched markdown, then legacy consolidated path."""
    return [
        f"{base}/enriched/en/markdown.md",
        f"{base}/en/markdown.md",
    ]
