"""Plotly visualizations for M1 field-agreement summaries."""

from __future__ import annotations

from typing import Mapping

import plotly.graph_objects as go

from core.m1_agreement import ModelAgreementSummary

HEATMAP_FIELDS = (
    ("identifier_type", "identifier_type"),
    ("role", "role"),
    ("is_section_product", "is_section_product"),
    ("alias_jaccard", "alias_jaccard"),
    ("quantity_presence", "quantity_presence"),
)


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
