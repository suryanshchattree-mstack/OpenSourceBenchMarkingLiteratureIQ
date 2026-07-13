"""Plotly visualizations for M1 field-agreement summaries."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import plotly.graph_objects as go

from core.m1_agreement import ModelAgreementSummary
from core.m1_consensus import (
    CHART_PATTERNS,
    CONSENSUS_FIELDS,
    FieldConsensusRow,
)

HEATMAP_FIELDS = (
    ("identifier_type", "identifier_type"),
    ("role", "role"),
    ("is_section_product", "is_section_product"),
    ("alias_jaccard", "alias_jaccard"),
    ("quantity_presence", "quantity_presence"),
)

CONSENSUS_PATTERN_COLORS = {
    "unanimous": "#2ca02c",
    "baseline_majority": "#1f77b4",
    "majority_vs_baseline": "#d62728",
    "split": "#ff7f0e",
}


def _rate_for_field(summary: ModelAgreementSummary, field_key: str) -> float | None:
    if field_key == "identifier_type":
        return summary.identifier_type.rate
    if field_key == "role":
        return summary.role.rate
    if field_key == "is_section_product":
        return summary.is_section_product.rate
    if field_key == "alias_jaccard":
        return summary.alias_jaccard_mean
    if field_key == "quantity_presence":
        return summary.quantity.overall_rate
    raise ValueError(f"Unknown heatmap field: {field_key!r}")


def build_agreement_heatmap(
    summaries: Mapping[str, ModelAgreementSummary],
    *,
    title: str = "Field agreement vs baseline",
) -> go.Figure:
    """
    Build a models × fields agreement-rate heatmap.

    Missing rates render as blank cells. Color scale is fixed to [0, 1].
    """
    models = list(summaries.keys())
    field_keys = [key for key, _ in HEATMAP_FIELDS]
    field_labels = [label for _, label in HEATMAP_FIELDS]

    z: list[list[float | None]] = []
    text: list[list[str]] = []
    hover: list[list[str]] = []
    for model in models:
        summary = summaries[model]
        row_z: list[float | None] = []
        row_text: list[str] = []
        row_hover: list[str] = []
        for field_key, field_label in HEATMAP_FIELDS:
            rate = _rate_for_field(summary, field_key)
            row_z.append(rate)
            if rate is None:
                row_text.append("—")
                row_hover.append(
                    f"model={model}<br>field={field_label}<br>rate=n/a"
                )
            else:
                row_text.append(f"{rate:.0%}")
                row_hover.append(
                    f"model={model}<br>field={field_label}<br>rate={rate:.1%}"
                )
        z.append(row_z)
        text.append(row_text)
        hover.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=field_labels,
            y=models,
            text=text,
            texttemplate="%{text}",
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale="RdYlGn",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Agreement", tickformat=".0%"),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=title,
        height=max(220, 80 + 48 * max(len(models), 1)),
        margin=dict(l=120, r=40, t=60, b=80),
        xaxis=dict(title="", side="bottom", tickangle=-25),
        yaxis=dict(title="", autorange="reversed"),
    )
    if not models:
        fig.add_annotation(
            text="No model summaries to plot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    _ = field_keys  # retained for readability / future extensions
    return fig


def build_consensus_chart(
    rows: Sequence[FieldConsensusRow],
    *,
    title: str = "Cluster consensus by field",
) -> go.Figure:
    """
    Stacked bar chart of consensus patterns per field.

    Rows with ``pattern="single_model"`` are excluded (need ≥3 models).
    ``majority_vs_baseline`` uses a distinct red segment as the review signal.
    """
    countable = [row for row in rows if row.pattern in CHART_PATTERNS]
    fields = list(CONSENSUS_FIELDS)
    counts_by_field: dict[str, Counter[str]] = {
        field: Counter() for field in fields
    }
    for row in countable:
        if row.field in counts_by_field:
            counts_by_field[row.field][row.pattern] += 1

    fig = go.Figure()
    for pattern in CHART_PATTERNS:
        fig.add_trace(
            go.Bar(
                name=pattern,
                x=fields,
                y=[counts_by_field[field][pattern] for field in fields],
                marker_color=CONSENSUS_PATTERN_COLORS[pattern],
                hovertemplate=(
                    f"field=%{{x}}<br>pattern={pattern}<br>count=%{{y}}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        barmode="stack",
        height=max(280, 120 + 24 * max(len(fields), 1)),
        margin=dict(l=60, r=40, t=60, b=100),
        xaxis=dict(title="", tickangle=-25),
        yaxis=dict(title="Clusters", rangemode="tozero"),
        legend=dict(title="Pattern", orientation="h", yanchor="bottom", y=1.02),
    )
    if not countable:
        fig.add_annotation(
            text="No multi-model consensus rows to plot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    return fig
