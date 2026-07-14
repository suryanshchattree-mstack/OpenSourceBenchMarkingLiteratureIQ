"""Adjudicated baseline computation for N-way compound clusters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from core.compound_matching import NWayDiffResult
from core.compound_parsing import CompoundEntry
from core.compound_report import cluster_display_label

NONE_SENTINEL = "(none)"


@dataclass(frozen=True)
class BaselineDefaults:
    present: bool
    role: str | None
    identifier_type: str | None


def field_value_options(
    entries_by_label: Mapping[str, list[CompoundEntry]],
    field: str,
) -> list[str]:
    """
    Canonical-cased, case-fold-deduped, alphabetically sorted values for ``field``,
    prefixed with a ``(none)`` sentinel for absent/blank.
    """
    display_by_folded: dict[str, str] = {}
    for entries in entries_by_label.values():
        for entry in entries:
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


def _fold_vote(value: str | None) -> tuple[str, str]:
    """Map a raw vote to ``(fold_key, display)``; blank/None → ``(none)``."""
    if value is None or not str(value).strip():
        return NONE_SENTINEL.casefold(), NONE_SENTINEL
    text = str(value).strip()
    return text.casefold(), text


def majority_with_tiebreak(
    values: list[str | None],
    tiebreak_value: str | None,
) -> str | None:
    """
    Case-folded mode of values, counting blank/``None`` as explicit ``(none)`` votes.

    On a tie, prefer ``tiebreak_value`` if it is among the tied modes (blank
    tiebreak counts as ``(none)``); otherwise pick the alphabetically-first mode
    (by casefolded key). When the winning mode is ``(none)``, return ``None``.
    """
    if not values:
        return None

    folded_counts: Counter[str] = Counter()
    display_by_folded: dict[str, str] = {}
    for value in values:
        folded, display = _fold_vote(value)
        folded_counts[folded] += 1
        # Keep first-seen original casing as the display form for that fold key.
        display_by_folded.setdefault(folded, display)

    max_count = max(folded_counts.values())
    tied = sorted(key for key, count in folded_counts.items() if count == max_count)

    tie_fold, _ = _fold_vote(tiebreak_value)
    if tie_fold in tied:
        winner = display_by_folded[tie_fold]
    else:
        winner = display_by_folded[tied[0]]

    if winner.casefold() == NONE_SENTINEL.casefold():
        return None
    return winner


def _field_value(entry: object, field: str) -> str | None:
    raw = getattr(entry, field, None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def compute_cluster_baselines(
    nway: NWayDiffResult,
    tiebreak_label: str = "Claude",
) -> dict[str, BaselineDefaults]:
    """
    Per-cluster adjudicated defaults keyed by display label.

    Presence: majority of models present; on a tie, present iff ``tiebreak_label``
    is in the cluster. Role / identifier_type: majority with Claude (tiebreak
    label) value preferred on ties.
    """
    preferred = tiebreak_label if tiebreak_label in nway.labels else (
        nway.labels[0] if nway.labels else "Claude"
    )
    defaults: dict[str, BaselineDefaults] = {}
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

        roles: list[str | None] = []
        id_types: list[str | None] = []
        tiebreak_role: str | None = None
        tiebreak_id_type: str | None = None
        for label in nway.labels:
            if label not in cluster.membership:
                continue
            entry = cluster.representatives[label]
            role = _field_value(entry, "role")
            id_type = _field_value(entry, "identifier_type")
            roles.append(role)
            id_types.append(id_type)
            if label == tiebreak_label:
                tiebreak_role = role
                tiebreak_id_type = id_type

        defaults[display] = BaselineDefaults(
            present=present,
            role=majority_with_tiebreak(roles, tiebreak_role),
            identifier_type=majority_with_tiebreak(id_types, tiebreak_id_type),
        )
    return defaults
