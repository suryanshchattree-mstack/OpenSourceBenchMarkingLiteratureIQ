"""Agreement metrics for matched M1 compounds across models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.compound_matching import NWayDiffResult, canonicalize_name, diff_compounds_nway
from core.m1_parsing import QUANTITY_FIELDS, M1CompoundEntry

FIELD_FILTER_OPTIONS = (
    "Any disagreement",
    "identifier_type",
    "role",
    "is_section_product",
    "aliases",
    "quantity",
    *QUANTITY_FIELDS,
)

ALIAS_DISAGREE_THRESHOLD = 1.0


def _confusion_key(left: str, right: str) -> str:
    a, b = sorted((left, right))
    return f"{a}<->{b}"


def _normalized_alias_set(entry: M1CompoundEntry) -> set[str]:
    names: set[str] = set()
    for alias in entry.aliases:
        if name := canonicalize_name(alias):
            names.add(name)
    return names


def _alias_jaccard(left: M1CompoundEntry, right: M1CompoundEntry) -> float:
    a = _normalized_alias_set(left)
    b = _normalized_alias_set(right)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _quantity_presence(entry: M1CompoundEntry) -> dict[str, bool]:
    return {name: entry.quantity.get(name) is not None for name in QUANTITY_FIELDS}


def _failed_fields(
    *,
    identifier_type_agree: bool,
    role_agree: bool,
    is_section_product_agree: bool,
    alias_jaccard: float,
    quantity_field_agree: Mapping[str, bool],
) -> tuple[str, ...]:
    failed: list[str] = []
    if not identifier_type_agree:
        failed.append("identifier_type")
    if not role_agree:
        failed.append("role")
    if not is_section_product_agree:
        failed.append("is_section_product")
    if alias_jaccard < ALIAS_DISAGREE_THRESHOLD:
        failed.append("aliases")
    for name in QUANTITY_FIELDS:
        if not quantity_field_agree.get(name, True):
            failed.append(name)
    return tuple(failed)


@dataclass
class FieldAgreementStats:
    agree: int = 0
    disagree: int = 0
    confusion_pairs: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return self.agree + self.disagree

    @property
    def rate(self) -> float | None:
        total = self.total
        return self.agree / total if total > 0 else None

    def record(self, left: Any, right: Any) -> bool:
        matched = left == right
        if matched:
            self.agree += 1
        else:
            self.disagree += 1
            self.confusion_pairs[_confusion_key(str(left), str(right))] += 1
        return matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "agree": self.agree,
            "disagree": self.disagree,
            "rate": self.rate,
            "confusion_pairs": dict(sorted(self.confusion_pairs.items())),
        }


@dataclass
class QuantityAgreementStats:
    """Per-field null-vs-populated agreement across quantity keys."""

    field_agree: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in QUANTITY_FIELDS}
    )
    field_disagree: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in QUANTITY_FIELDS}
    )
    comparisons: int = 0

    def record(self, left: M1CompoundEntry, right: M1CompoundEntry) -> dict[str, bool]:
        left_presence = _quantity_presence(left)
        right_presence = _quantity_presence(right)
        per_field: dict[str, bool] = {}
        self.comparisons += 1
        for name in QUANTITY_FIELDS:
            matched = left_presence[name] == right_presence[name]
            per_field[name] = matched
            if matched:
                self.field_agree[name] += 1
            else:
                self.field_disagree[name] += 1
        return per_field

    @property
    def overall_rate(self) -> float | None:
        total_agree = sum(self.field_agree.values())
        total = total_agree + sum(self.field_disagree.values())
        return total_agree / total if total > 0 else None

    def to_dict(self) -> dict[str, Any]:
        per_field = {}
        for name in QUANTITY_FIELDS:
            agree = self.field_agree[name]
            disagree = self.field_disagree[name]
            total = agree + disagree
            per_field[name] = {
                "agree": agree,
                "disagree": disagree,
                "rate": (agree / total) if total > 0 else None,
            }
        return {
            "comparisons": self.comparisons,
            "overall_rate": self.overall_rate,
            "per_field": per_field,
        }


@dataclass
class ModelAgreementSummary:
    label: str
    common: int
    baseline_only: int
    model_only: int
    recall: float | None
    precision: float | None
    identifier_type: FieldAgreementStats
    role: FieldAgreementStats
    is_section_product: FieldAgreementStats
    alias_jaccard_sum: float = 0.0
    alias_jaccard_count: int = 0
    quantity: QuantityAgreementStats = field(default_factory=QuantityAgreementStats)

    @property
    def alias_jaccard_mean(self) -> float | None:
        if self.alias_jaccard_count == 0:
            return None
        return self.alias_jaccard_sum / self.alias_jaccard_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "common": self.common,
            "baseline_only": self.baseline_only,
            "model_only": self.model_only,
            "recall_vs_baseline": self.recall,
            "precision_vs_baseline": self.precision,
            "identifier_type": self.identifier_type.to_dict(),
            "role": self.role.to_dict(),
            "is_section_product": self.is_section_product.to_dict(),
            "alias_jaccard_mean": self.alias_jaccard_mean,
            "quantity": self.quantity.to_dict(),
        }


@dataclass(frozen=True)
class ClusterAgreementRow:
    membership: frozenset[str]
    model_label: str
    baseline_identifier: str
    model_identifier: str
    identifier_type_agree: bool
    baseline_identifier_type: str
    model_identifier_type: str
    role_agree: bool
    baseline_role: str | None
    model_role: str | None
    is_section_product_agree: bool
    baseline_is_section_product: bool
    model_is_section_product: bool
    alias_jaccard: float
    baseline_aliases: tuple[str, ...]
    model_aliases: tuple[str, ...]
    quantity_field_agree: Mapping[str, bool]
    baseline_quantity: Mapping[str, float | None]
    model_quantity: Mapping[str, float | None]
    failed_fields: tuple[str, ...]

    @property
    def has_disagreement(self) -> bool:
        return bool(self.failed_fields)


def filter_cluster_rows(
    rows: Sequence[ClusterAgreementRow],
    *,
    model_labels: Sequence[str] | None = None,
    field_filter: str = "Any disagreement",
    identifier_query: str = "",
    disagreements_only: bool = True,
) -> list[ClusterAgreementRow]:
    """Filter matched cluster rows for the disagreement browser UI."""
    if field_filter not in FIELD_FILTER_OPTIONS:
        raise ValueError(f"Unknown field filter: {field_filter!r}")

    selected_models = set(model_labels) if model_labels is not None else None
    query = identifier_query.strip().lower()
    filtered: list[ClusterAgreementRow] = []

    for row in rows:
        if selected_models is not None and row.model_label not in selected_models:
            continue
        if disagreements_only and not row.has_disagreement:
            continue
        if query:
            haystack = f"{row.baseline_identifier} {row.model_identifier}".lower()
            if query not in haystack:
                continue

        if field_filter == "Any disagreement":
            if disagreements_only and not row.has_disagreement:
                continue
        elif field_filter == "quantity":
            if not any(not row.quantity_field_agree.get(name, True) for name in QUANTITY_FIELDS):
                continue
        elif field_filter == "aliases":
            if "aliases" not in row.failed_fields:
                continue
        elif field_filter in QUANTITY_FIELDS:
            if row.quantity_field_agree.get(field_filter, True):
                continue
        elif field_filter == "identifier_type":
            if row.identifier_type_agree:
                continue
        elif field_filter == "role":
            if row.role_agree:
                continue
        elif field_filter == "is_section_product":
            if row.is_section_product_agree:
                continue
        else:
            raise ValueError(f"Unhandled field filter: {field_filter!r}")

        filtered.append(row)

    return filtered


@dataclass
class M1AgreementResult:
    baseline: str
    nway: NWayDiffResult[M1CompoundEntry]
    summaries: dict[str, ModelAgreementSummary]
    cluster_rows: list[ClusterAgreementRow]

    def summary_rows(self) -> list[dict[str, Any]]:
        return [summary.to_dict() for summary in self.summaries.values()]


def compute_m1_agreement(
    entries_by_label: Mapping[str, list[M1CompoundEntry]],
    baseline: str,
) -> M1AgreementResult:
    """
    Run N-way matching, then score field agreement vs ``baseline`` on shared clusters.

    For every cluster that contains the baseline label, compare the baseline
    representative against each other label present in that cluster.
    """
    if baseline not in entries_by_label:
        raise KeyError(f"Baseline label {baseline!r} not in entries_by_label")
    if len(entries_by_label) < 2:
        raise ValueError("Need at least two labeled M1 compound lists")

    nway = diff_compounds_nway(entries_by_label)
    other_labels = [label for label in nway.labels if label != baseline]

    summaries: dict[str, ModelAgreementSummary] = {}
    for label in other_labels:
        common, baseline_only, model_only, recall, precision = nway.pairwise_metrics(
            baseline, label
        )
        summaries[label] = ModelAgreementSummary(
            label=label,
            common=common,
            baseline_only=baseline_only,
            model_only=model_only,
            recall=recall,
            precision=precision,
            identifier_type=FieldAgreementStats(),
            role=FieldAgreementStats(),
            is_section_product=FieldAgreementStats(),
        )

    cluster_rows: list[ClusterAgreementRow] = []
    for cluster in nway.clusters:
        if baseline not in cluster.membership:
            continue
        baseline_entry = cluster.representatives[baseline]
        for label in sorted(cluster.membership - {baseline}):
            model_entry = cluster.representatives[label]
            summary = summaries[label]

            type_agree = summary.identifier_type.record(
                baseline_entry.identifier_type, model_entry.identifier_type
            )
            role_agree = summary.role.record(baseline_entry.role, model_entry.role)
            product_agree = summary.is_section_product.record(
                baseline_entry.is_section_product, model_entry.is_section_product
            )
            jaccard = _alias_jaccard(baseline_entry, model_entry)
            summary.alias_jaccard_sum += jaccard
            summary.alias_jaccard_count += 1
            quantity_agree = summary.quantity.record(baseline_entry, model_entry)
            failed = _failed_fields(
                identifier_type_agree=type_agree,
                role_agree=role_agree,
                is_section_product_agree=product_agree,
                alias_jaccard=jaccard,
                quantity_field_agree=quantity_agree,
            )

            cluster_rows.append(
                ClusterAgreementRow(
                    membership=cluster.membership,
                    model_label=label,
                    baseline_identifier=baseline_entry.identifier,
                    model_identifier=model_entry.identifier,
                    identifier_type_agree=type_agree,
                    baseline_identifier_type=baseline_entry.identifier_type,
                    model_identifier_type=model_entry.identifier_type,
                    role_agree=role_agree,
                    baseline_role=baseline_entry.role,
                    model_role=model_entry.role,
                    is_section_product_agree=product_agree,
                    baseline_is_section_product=baseline_entry.is_section_product,
                    model_is_section_product=model_entry.is_section_product,
                    alias_jaccard=jaccard,
                    baseline_aliases=baseline_entry.aliases,
                    model_aliases=model_entry.aliases,
                    quantity_field_agree=quantity_agree,
                    baseline_quantity=dict(baseline_entry.quantity),
                    model_quantity=dict(model_entry.quantity),
                    failed_fields=failed,
                )
            )

    cluster_rows.sort(
        key=lambda row: (
            canonicalize_name(row.baseline_identifier) or "",
            row.model_label,
        )
    )
    return M1AgreementResult(
        baseline=baseline,
        nway=nway,
        summaries=summaries,
        cluster_rows=cluster_rows,
    )
