"""Plotly visualizations for multi-run pre-pass comparison."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.flagging import FlagRegion
from core.parsing import Section

TYPE_COLORS: dict[str, str] = {
    "bibliographic": "#636EFA",
    "abstract": "#EF553B",
    "technical_field": "#00CC96",
    "background": "#AB63FA",
    "summary_of_invention": "#FFA15A",
    "beneficial_effects": "#19D3F3",
    "formula_definitions": "#FF6692",
    "description_of_drawings": "#B6E880",
    "experimental_intermediate": "#FF97FF",
    "experimental_example": "#FECB52",
    "comparative_example": "#636EFA",
    "assay_data": "#EF553B",
    "pharmaceutical_compositions": "#00CC96",
    "claims": "#AB63FA",
    "search_report": "#FFA15A",
    "closing_statement": "#19D3F3",
    "other": "#B6E880",
}

RUN_BAR_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]

DEFAULT_COLOR = "#AAAAAA"
FLAG_COLOR = "#FF0000"


def _color_for_type(section_type: str) -> str:
    return TYPE_COLORS.get(section_type, DEFAULT_COLOR)


def _section_bar_traces(
    sections: list[Section],
    row_name: str,
    total_lines: int,
) -> list[go.Bar]:
    traces: list[go.Bar] = []
    for sec in sections:
        start = max(1, sec.start_line)
        end = min(total_lines, sec.end_line)
        if end < start:
            continue
        width = end - start + 1
        traces.append(
            go.Bar(
                x=[width],
                y=[row_name],
                base=[start - 1],
                orientation="h",
                marker_color=_color_for_type(sec.section_type),
                name=sec.section_type,
                hovertemplate=(
                    f"<b>{row_name}</b><br>"
                    f"Label: {sec.section_label}<br>"
                    f"Type: {sec.section_type}<br>"
                    f"Lines: {start}–{end}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    return traces


def build_timeline_figure(
    runs: list[tuple[str, list[Section]]],
    flag_regions: list[FlagRegion],
    total_lines: int,
) -> go.Figure:
    """Stacked horizontal timeline: one row per run plus a disagreement overlay."""
    num_run_rows = len(runs)
    row_heights = [1.0] * num_run_rows + [0.35]
    fig = make_subplots(
        rows=num_run_rows + 1,
        cols=1,
        row_heights=row_heights,
        vertical_spacing=0.06,
    )

    for row_idx, (label, sections) in enumerate(runs, start=1):
        for trace in _section_bar_traces(sections, label, total_lines):
            fig.add_trace(trace, row=row_idx, col=1)

    for region in flag_regions:
        width = region.end_line - region.start_line + 1
        run_details = "<br>".join(
            f"{label}: {region.run_types.get(label)} / {region.run_labels.get(label)}"
            for label in region.run_types
        )
        fig.add_trace(
            go.Bar(
                x=[width],
                y=["Disagreements vs reference"],
                base=[region.start_line - 1],
                orientation="h",
                marker_color=FLAG_COLOR,
                opacity=0.7,
                hovertemplate=(
                    f"Lines {region.start_line}–{region.end_line}<br>"
                    f"Issue: {region.issue_kind}<br>"
                    f"Models: {', '.join(region.disagreeing_models)}<br>"
                    f"{run_details}<br>"
                    f"Worst avg cum sim: {region.avg_cum_sim:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=num_run_rows + 1,
            col=1,
        )

    layout_kwargs = {
        "barmode": "overlay",
        "height": max(280, 120 * (num_run_rows + 1)),
        "margin": dict(l=160, r=20, t=40, b=20),
        "title": "Section Type Timeline",
    }
    for row_idx in range(1, num_run_rows + 2):
        layout_kwargs[f"xaxis{'' if row_idx == 1 else row_idx}"] = dict(
            title="Line number" if row_idx == num_run_rows + 1 else "",
            range=[0, total_lines],
        )
        layout_kwargs[f"yaxis{'' if row_idx == 1 else row_idx}"] = dict(showticklabels=True)

    fig.update_layout(**layout_kwargs)
    return fig


def build_type_histogram(
    type_counts_by_run: dict[str, dict[str, int]],
) -> go.Figure:
    """Grouped bar chart of section counts per type across all runs."""
    all_types = sorted({t for counts in type_counts_by_run.values() for t in counts})
    fig = go.Figure()

    for idx, (label, counts) in enumerate(type_counts_by_run.items()):
        fig.add_trace(
            go.Bar(
                name=label,
                x=all_types,
                y=[counts.get(section_type, 0) for section_type in all_types],
                marker_color=RUN_BAR_COLORS[idx % len(RUN_BAR_COLORS)],
            )
        )

    fig.update_layout(
        barmode="group",
        title="Section Type Distribution",
        xaxis_title="Section type",
        yaxis_title="Section count",
        height=400,
        margin=dict(l=40, r=20, t=60, b=120),
        xaxis_tickangle=-45,
    )
    return fig
