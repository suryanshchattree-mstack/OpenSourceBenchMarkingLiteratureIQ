"""SciBERT procedure-text embeddings (literatureiq ChemistryToolkit parity)."""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from core.embeddings import resolve_device

SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
_MAX_LENGTH = 512

_tokenizer: AutoTokenizer | None = None
_model: AutoModel | None = None
_device: str | None = None


def _ensure_loaded(device: str | None = None) -> tuple[AutoTokenizer, AutoModel, str]:
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None and _device is not None:
        return _tokenizer, _model, _device
    dev = device or resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME)
    model.eval()
    model.to(dev)
    _tokenizer = tokenizer
    _model = model
    _device = dev
    return tokenizer, model, dev


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-aware mean over tokens (matches single-seq mean when unpadded)."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_procedure_texts(texts: Sequence[str], *, device: str | None = None) -> list[tuple[float, ...]]:
    """
    Batch-embed procedure texts with SciBERT.

    Mean-pools ``last_hidden_state`` with ``max_length=512`` and truncation,
    matching literatureiq ``ChemistryToolkit.getProcedureVector``.
    """
    if not texts:
        return []
    tokenizer, model, dev = _ensure_loaded(device)
    inputs = tokenizer(
        list(texts),
        return_tensors="pt",
        max_length=_MAX_LENGTH,
        truncation=True,
        padding=True,
    )
    inputs = {key: value.to(dev) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return [tuple(float(x) for x in row.tolist()) for row in pooled.cpu()]
