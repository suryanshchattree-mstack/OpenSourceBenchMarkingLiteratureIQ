"""Fill missing ReactionEntry.procedure_vector from procedure text via SciBERT."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from core.reaction_parsing import ReactionEntry
from core.scibert import embed_procedure_texts

EmbedFn = Callable[[Sequence[str]], list[tuple[float, ...]]]


def procedure_text_for_embed(entry: ReactionEntry) -> str | None:
    """Prefer ``procedure_summary`` from ``raw``; fall back to ``procedure_text``."""
    raw = entry.raw or {}
    summary = raw.get("procedure_summary")
    if summary is not None:
        text = str(summary).strip()
        if text:
            return text
    if entry.procedure_text and entry.procedure_text.strip():
        return entry.procedure_text.strip()
    return None


def ensure_procedure_vectors(
    entries: Sequence[ReactionEntry],
    *,
    embed_fn: EmbedFn | None = None,
) -> list[ReactionEntry]:
    """
    Return copies of ``entries`` with ``procedure_vector`` set when possible.

    Skips reactions that already have a vector, or that have neither
    ``procedure_summary`` nor ``procedure_text``. Unique texts are batch-embedded.
    """
    embed = embed_fn or embed_procedure_texts
    need_indices: list[int] = []
    texts_for_index: list[str] = []
    for index, entry in enumerate(entries):
        if entry.procedure_vector is not None:
            continue
        text = procedure_text_for_embed(entry)
        if text is None:
            continue
        need_indices.append(index)
        texts_for_index.append(text)

    if not need_indices:
        return list(entries)

    unique_texts = list(dict.fromkeys(texts_for_index))
    vectors = embed(unique_texts)
    if len(vectors) != len(unique_texts):
        raise ValueError(
            f"embed_fn returned {len(vectors)} vectors for {len(unique_texts)} texts"
        )
    text_to_vector = dict(zip(unique_texts, vectors))

    out = list(entries)
    for index, text in zip(need_indices, texts_for_index):
        vector = text_to_vector[text]
        out[index] = replace(out[index], procedure_vector=vector)
    return out


def ensure_procedure_vectors_by_label(
    entries_by_label: dict[str, list[ReactionEntry]],
    *,
    embed_fn: EmbedFn | None = None,
) -> dict[str, list[ReactionEntry]]:
    """Ensure vectors for every label; batch-embed all missing texts in one call."""
    flat: list[ReactionEntry] = []
    spans: list[tuple[str, int]] = []
    for label, entries in entries_by_label.items():
        spans.append((label, len(entries)))
        flat.extend(entries)
    filled = ensure_procedure_vectors(flat, embed_fn=embed_fn)
    result: dict[str, list[ReactionEntry]] = {}
    offset = 0
    for label, count in spans:
        result[label] = filled[offset : offset + count]
        offset += count
    return result
