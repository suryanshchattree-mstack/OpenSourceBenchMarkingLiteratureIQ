"""N-way cluster consensus classification for M1 categorical fields."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from core.compound_matching import NWayDiffResult, canonicalize_name
from core.m1_parsing import QUANTITY_FIELDS, M1CompoundEntry

CONSENSUS_FIELDS = (
    "identifier_type",
    "role",
    "is_section_product",
    *QUANTITY_FIELDS,
)

ConsensusPattern = Literal[
    "unanimous",
    "baseline_majority",
    "majority_vs_baseline",
    "split",
    "single_model",
]

CHART_PATTERNS: tuple[ConsensusPattern, ...] = (
    "unanimous",
    "baseline_majority",
    "majority_vs_baseline",
    "split",
)


@dataclass(frozen=True)
class FieldConsensusRow:
    membership: frozenset[str]
    cluster_identifier: str
    field: str
    values: Mapping[str, Any]
    baseline_value: Any | None
    majority_value: Any | None
    majority_supporters: frozenset[str]
    pattern: ConsensusPattern

    @property
    def not_in_majority(self) -> frozenset[str]:
        """Models with a value that is not the majority (includes baseline when it disagrees)."""
        if self.majority_value is None:
            return frozenset(self.values) - self.majority_supporters
        return frozenset(
            label
            for label, value in self.values.items()
            if value != self.majority_value
        )


def _field_value(entry: M1CompoundEntry, field: str) -> Any | None:
    if field == "identifier_type":
        return entry.identifier_type
    if field == "role":
        return entry.role
    if field == "is_section_product":
        return entry.is_section_product
    if field in QUANTITY_FIELDS:
        return entry.quantity.get(field) is not None
    raise ValueError(f"Unknown consensus field: {field!r}")


def _cluster_identifier(
    representatives: Mapping[str, M1CompoundEntry],
    baseline: str,
) -> str:
    if baseline in representatives:
        return representatives[baseline].identifier
    return min(
        (entry.identifier for entry in representatives.values()),
        key=lambda name: canonicalize_name(name) or name,
    )


def _classify_field(
    *,
    values: Mapping[str, Any],
    baseline: str,
    membership_size: int,
) -> tuple[ConsensusPattern, Any | None, frozenset[str]]:
    """Return (pattern, majority_value, majority_supporters)."""
    if membership_size < 3:
        return "single_model", None, frozenset()

    baseline_value = values.get(baseline)
    non_baseline = {label: value for label, value in values.items() if label != baseline}

    if (
        baseline in values
        and len(values) == membership_size
        and len(set(values.values())) == 1
    ):
        sole = next(iter(values.values()))
        supporters = frozenset(non_baseline)
        return "unanimous", sole, supporters

    if not non_baseline:
        return "split", None, frozenset()

    counts = Counter(non_baseline.values())
    max_count = max(counts.values())
    top_values = [value for value, count in counts.items() if count == max_count]
    majority_value = top_values[0] if len(top_values) == 1 else None
    majority_supporters = frozenset(
        label
        for label, value in non_baseline.items()
        if majority_value is not None and value == majority_value
    )

    baseline_supporters = sum(
        1 for value in non_baseline.values() if value == baseline_value
    )

    if (
        len(top_values) == 1
        and max_count >= 2
        and baseline_value is not None
        and majority_value != baseline_value
        and max_count > baseline_supporters
    ):
        return "majority_vs_baseline", majority_value, majority_supporters

    if (
        baseline_value is not None
        and baseline_supporters >= 1
        and max_count <= baseline_supporters
    ):
        supporters = frozenset(
            label
            for label, value in non_baseline.items()
            if value == baseline_value
        )
        return "baseline_majority", baseline_value, supporters

    return "split", majority_value, majority_supporters


def compute_cluster_consensus(
    nway: NWayDiffResult[M1CompoundEntry],
    baseline: str,
) -> list[FieldConsensusRow]:
    """
    Classify each cluster × categorical field into a consensus pattern.

    Clusters with fewer than 3 present models are tagged ``single_model``.
    Quantity fields use presence (bool), matching pairwise agreement.
    """
    if baseline not in nway.labels:
        raise KeyError(f"Baseline label {baseline!r} not in nway.labels")

    rows: list[FieldConsensusRow] = []
    for cluster in nway.clusters:
        if baseline not in cluster.membership:
            continue
        membership_size = len(cluster.membership)
        cluster_id = _cluster_identifier(cluster.representatives, baseline)

        for field in CONSENSUS_FIELDS:
            values: dict[str, Any] = {}
            for label, entry in cluster.representatives.items():
                value = _field_value(entry, field)
                if value is not None:
                    values[label] = value

            pattern, majority_value, majority_supporters = _classify_field(
                values=values,
                baseline=baseline,
                membership_size=membership_size,
            )
            rows.append(
                FieldConsensusRow(
                    membership=cluster.membership,
                    cluster_identifier=cluster_id,
                    field=field,
                    values=dict(values),
                    baseline_value=values.get(baseline),
                    majority_value=majority_value,
                    majority_supporters=majority_supporters,
                    pattern=pattern,
                )
            )

    rows.sort(
        key=lambda row: (
            canonicalize_name(row.cluster_identifier) or "",
            row.field,
        )
    )
    return rows


def consensus_rows_to_dataframe(
    rows: Sequence[FieldConsensusRow],
    *,
    model_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Flatten consensus rows for Streamlit / tabular display."""
    if model_labels is None:
        labels: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for label in row.values:
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
            for label in row.membership:
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
        model_labels = sorted(labels)

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "cluster_identifier": row.cluster_identifier,
            "field": row.field,
        }
        for label in model_labels:
            value = row.values.get(label)
            record[label] = "" if value is None else value
        record["baseline_value"] = (
            "" if row.baseline_value is None else row.baseline_value
        )
        record["majority_value"] = (
            "" if row.majority_value is None else row.majority_value
        )
        record["majority_supporters"] = ", ".join(sorted(row.majority_supporters))
        record["not_in_majority"] = ", ".join(sorted(row.not_in_majority))
        record["pattern"] = row.pattern
        table_rows.append(record)

    columns = [
        "cluster_identifier",
        "field",
        *list(model_labels),
        "baseline_value",
        "majority_value",
        "majority_supporters",
        "not_in_majority",
        "pattern",
    ]
    if not table_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(table_rows, columns=columns)
