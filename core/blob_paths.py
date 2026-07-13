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


def m1_path(base: str, pipeline_id: str) -> str:
    return f"{base}/extraction/{pipeline_id}/molecule-pass-1-consolidated.json"


def m2_path(base: str, pipeline_id: str) -> str:
    return f"{base}/extraction/{pipeline_id}/molecule-pass-2-consolidated.json"


def r1_path(base: str, pipeline_id: str) -> str:
    return f"{base}/extraction/{pipeline_id}/reaction-pass-1-consolidated.json"


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
    Ordered candidate paths for reactions.json.

    Primary path first, then the other layout (baseline vs pipeline), then legacy
    ``persistent-store/{patentId}/reactions.json``.
    """
    primary = reactions_path(base, pipeline_id)
    candidates = [primary]
    if pipeline_id == BASELINE_PIPELINE_ID:
        alt = f"{base}/extraction/{pipeline_id}/reactions.json"
    else:
        alt = f"{base}/reactions.json"
    if alt not in candidates:
        candidates.append(alt)
    legacy = f"persistent-store/{patent_id}/reactions.json"
    if legacy not in candidates:
        candidates.append(legacy)
    return candidates


def markdown_paths(base: str) -> list[str]:
    """Preferred enriched markdown, then legacy consolidated path."""
    return [
        f"{base}/enriched/en/markdown.md",
        f"{base}/en/markdown.md",
    ]
