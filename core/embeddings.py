"""Sentence-transformer embedding helpers with MPS support."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from core.line_arrays import LineArrays

MODEL_NAMES = [
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
    "all-mpnet-base-v2",
]


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_models(device: str | None = None) -> list[SentenceTransformer]:
    """Load all three embedding models on the given device."""
    dev = device or resolve_device()
    return [SentenceTransformer(name, device=dev) for name in MODEL_NAMES]


def _encode_labels(model: SentenceTransformer, labels: list[str]) -> dict[str, np.ndarray]:
    """Embed unique non-empty labels; empty/None -> zero vector placeholder key."""
    unique = sorted({lbl for lbl in labels if lbl})
    cache: dict[str, np.ndarray] = {}
    if unique:
        vecs = model.encode(unique, normalize_embeddings=True, show_progress_bar=False)
        for lbl, vec in zip(unique, vecs):
            cache[lbl] = np.asarray(vec, dtype=np.float32)
    return cache


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def embed_line_labels(
    model: SentenceTransformer,
    line_labels: list[str | None],
) -> list[np.ndarray | None]:
    """
    Return per-line embedding vectors (index 0 unused).
    Lines with None/empty labels get None.
    """
    total = len(line_labels) - 1
    cache = _encode_labels(model, line_labels)
    dim = next(iter(cache.values())).shape[0] if cache else 384

    result: list[np.ndarray | None] = [None] * (total + 1)
    for i in range(1, total + 1):
        lbl = line_labels[i]
        if not lbl:
            result[i] = None
        elif lbl in cache:
            result[i] = cache[lbl]
        else:
            result[i] = np.zeros(dim, dtype=np.float32)
    return result


def per_line_similarity(
    reference_vecs: list[np.ndarray | None],
    benchmark_vecs: list[np.ndarray | None],
) -> list[float]:
    """Cosine similarity per line vs reference (index 0 unused)."""
    total = len(reference_vecs) - 1
    sims: list[float] = [0.0] * (total + 1)
    for i in range(1, total + 1):
        ref_vec = reference_vecs[i]
        bench_vec = benchmark_vecs[i]
        if ref_vec is None or bench_vec is None:
            sims[i] = 0.0
        else:
            sims[i] = cosine_similarity(ref_vec, bench_vec)
    return sims


def compute_vs_reference_similarities(
    models: list[SentenceTransformer],
    reference: LineArrays,
    benchmarks: list[LineArrays],
) -> list[list[list[float]]]:
    """
    Return [benchmark_index][model_index][line_index] label similarity vs reference.
    Line index 0 is unused.
    """
    if not benchmarks:
        raise ValueError("At least one benchmark run is required besides the reference")

    total_lines = reference.total_lines
    for benchmark in benchmarks:
        if benchmark.total_lines != total_lines:
            raise ValueError("All runs must have the same total_lines")

    result: list[list[list[float]]] = []
    for benchmark in benchmarks:
        benchmark_sims: list[list[float]] = []
        for model in models:
            ref_vecs = embed_line_labels(model, reference.line_label)
            bench_vecs = embed_line_labels(model, benchmark.line_label)
            benchmark_sims.append(per_line_similarity(ref_vecs, bench_vecs))
        result.append(benchmark_sims)
    return result
