"""rxnfp reaction SMILES embeddings via the pip ``rxnfp`` package.

Loads bundled BERT checkpoints from the installed package
(``rxnfp/models/transformers/bert_ft``) — not Hugging Face Hub
(``rxn4chemistry/rxnfp`` is no longer a public model). CLS pooling matches
``RXNBERTFingerprintGenerator`` / literatureiq's 256-d reaction vectors.

Install::

    pip install --no-deps rxnfp
    # needs setuptools for pkg_resources (rxnfp load path)
"""

from __future__ import annotations

from typing import Any, Sequence

from core.embeddings import resolve_device

# Package default fine-tuned fingerprint model (256-d CLS).
RXNFP_PACKAGE_MODEL = "bert_ft"

_generator: Any | None = None
_device: str | None = None


def _ensure_loaded(device: str | None = None) -> tuple[Any, str]:
    global _generator, _device
    if _generator is not None and _device is not None:
        return _generator, _device

    try:
        from rxnfp.transformer_fingerprints import (
            RXNBERTFingerprintGenerator,
            get_default_model_and_tokenizer,
        )
    except ImportError as exc:
        raise ImportError(
            "rxnfp is required for reaction-vector fill. Install with: "
            "pip install --no-deps rxnfp  (and setuptools for pkg_resources)"
        ) from exc

    # Prefer CPU when resolve_device says so (Streamlit/macOS often force_no_cuda).
    dev = device or resolve_device()
    force_no_cuda = not str(dev).startswith("cuda")
    model, tokenizer = get_default_model_and_tokenizer(
        RXNFP_PACKAGE_MODEL,
        force_no_cuda=force_no_cuda,
    )
    generator = RXNBERTFingerprintGenerator(
        model,
        tokenizer,
        force_no_cuda=force_no_cuda,
    )
    _generator = generator
    _device = "cpu" if force_no_cuda else "cuda"
    return generator, _device


def embed_reaction_smiles(
    reaction_smiles: Sequence[str],
    *,
    device: str | None = None,
) -> list[tuple[float, ...] | None]:
    """
    Batch-embed reaction SMILES with rxnfp CLS pooling.

    Empty / blank strings yield ``None`` at the corresponding index.
    """
    if not reaction_smiles:
        return []

    nonempty_indices: list[int] = []
    nonempty_texts: list[str] = []
    for index, text in enumerate(reaction_smiles):
        cleaned = str(text).strip() if text is not None else ""
        if cleaned:
            nonempty_indices.append(index)
            nonempty_texts.append(cleaned)

    out: list[tuple[float, ...] | None] = [None] * len(reaction_smiles)
    if not nonempty_texts:
        return out

    generator, _ = _ensure_loaded(device)
    # convert_batch → list of 256-d CLS embeddings
    embeddings = generator.convert_batch(nonempty_texts)
    for index, row in zip(nonempty_indices, embeddings):
        out[index] = tuple(float(x) for x in row)
    return out
