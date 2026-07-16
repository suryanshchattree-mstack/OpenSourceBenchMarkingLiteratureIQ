"""Adjudicated baseline computation for N-way reaction clusters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from core.compound_baseline import NONE_SENTINEL, majority_with_tiebreak
from core.compound_matching import NWayDiffResult
from core.reaction_parsing import ReactionEntry
from core.reaction_report import cluster_display_label


@dataclass(frozen=True)
class ReactionBaselineDefaults:
    present: bool
    reaction_class: str | None
    product: str | None


def product_display_value(entry: ReactionEntry) -> str | None:
    """Product cell text: product_name, else short SMILES; None if absent."""
    if entry.product_name and entry.product_name.strip():
        return entry.product_name.strip()
    if entry.product_smiles and entry.product_smiles.strip():
        smiles = entry.product_smiles.strip()
        return smiles if len(smiles) <= 40 else smiles[:37] + "..."
    return None


def field_value_options(
    entries_by_label: Mapping[str, list[ReactionEntry]],
    field: str,
) -> list[str]:
    """
    Canonical-cased, case-fold-deduped, alphabetically sorted values for ``field``,
    prefixed with a ``(none)`` sentinel for absent/blank.

    ``field`` is ``reaction_class`` or ``product`` (product uses name/SMILES display).
    """
    display_by_folded: dict[str, str] = {}
    for entries in entries_by_label.values():
        for entry in entries:
            if field == "product":
                raw = product_display_value(entry)
            else:
                raw = getattr(entry, field, None)
            if raw is None:
                continue
            text = str(raw).strip()
            if not text:
                continue
            folded = text.casefold()
            display_by_folded.setdefault(folded, text)
    sorted_values = sorted(display_by_folded.values(), key=str.casefold)
    return [NONE_SENTINEL, *sorted_values]


def _field_value(entry: ReactionEntry, field: str) -> str | None:
    if field == "product":
        return product_display_value(entry)
    raw = getattr(entry, field, None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def compute_cluster_baselines(
    nway: NWayDiffResult[ReactionEntry],
    tiebreak_label: str = "Claude",
) -> dict[str, ReactionBaselineDefaults]:
    """
    Per-cluster adjudicated defaults keyed by display label.

    Presence: majority of models present; on a tie, present iff ``tiebreak_label``
    is in the cluster. Reaction class / product: majority with Claude tiebreak.
    """
    preferred = tiebreak_label if tiebreak_label in nway.labels else (
        nway.labels[0] if nway.labels else "Claude"
    )
    defaults: dict[str, ReactionBaselineDefaults] = {}
    for cluster in nway.clusters:
        display = cluster_display_label(cluster, preferred)
        n_models = len(nway.labels)
        present_count = sum(1 for label in nway.labels if label in cluster.membership)
        absent_count = n_models - present_count
        if present_count > absent_count:
            present = True
        elif present_count < absent_count:
            present = False
        else:
            present = tiebreak_label in cluster.membership

        classes: list[str | None] = []
        products: list[str | None] = []
        tiebreak_class: str | None = None
        tiebreak_product: str | None = None
        for label in nway.labels:
            if label not in cluster.membership:
                continue
            entry = cluster.representatives[label]
            rxn_class = _field_value(entry, "reaction_class")
            product = _field_value(entry, "product")
            classes.append(rxn_class)
            products.append(product)
            if label == tiebreak_label:
                tiebreak_class = rxn_class
                tiebreak_product = product

        defaults[display] = ReactionBaselineDefaults(
            present=present,
            reaction_class=majority_with_tiebreak(classes, tiebreak_class),
            product=majority_with_tiebreak(products, tiebreak_product),
        )
    return defaults
