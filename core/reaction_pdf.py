"""PDF export for reactions vs-manual-benchmark comparison."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

import pandas as pd


def build_reactions_pdf_report(
    patent_id: str,
    pipeline_ids_by_label: Mapping[str, str],
    stats_df: pd.DataFrame,
    presence_df: pd.DataFrame,
    reaction_class_df: pd.DataFrame,
    product_df: pd.DataFrame,
) -> bytes:
    """
    Build a values-only PDF: header, ranked metrics, then the three edited grids.

    Returns raw PDF bytes. Requires reportlab at call time.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story: list[Any] = []

    title = patent_id.strip() or "(unknown patent)"
    story.append(Paragraph(f"Reactions vs manual benchmark — {title}", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    if pipeline_ids_by_label:
        pipeline_lines = [
            f"{label}: {pipeline_id}"
            for label, pipeline_id in sorted(pipeline_ids_by_label.items())
        ]
        story.append(
            Paragraph(
                "Pipeline IDs — " + "; ".join(pipeline_lines),
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Ranked model metrics", styles["Heading2"]))
    story.extend(_dataframe_table_flowables(stats_df, styles, empty_message="No stats rows."))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Presence grid", styles["Heading2"]))
    story.extend(_dataframe_table_flowables(presence_df, styles, empty_message="No presence rows."))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Reaction class grid", styles["Heading2"]))
    story.extend(
        _dataframe_table_flowables(
            reaction_class_df, styles, empty_message="No reaction-class rows."
        )
    )
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Product grid", styles["Heading2"]))
    story.extend(_dataframe_table_flowables(product_df, styles, empty_message="No product rows."))

    doc.build(story)
    return buffer.getvalue()


def _dataframe_table_flowables(
    frame: pd.DataFrame,
    styles: Any,
    *,
    empty_message: str,
) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    if frame is None or frame.empty:
        return [Paragraph(empty_message, styles["Normal"])]

    columns = list(frame.columns)
    table_data = [columns]
    for _, row in frame.iterrows():
        table_data.append([_cell_str(row[col]) for col in columns])
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return [table]


def _cell_str(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)
