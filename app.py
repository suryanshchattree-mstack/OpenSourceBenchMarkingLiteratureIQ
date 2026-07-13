"""Unified multi-model benchmark — single-screen Streamlit orchestrator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.blob_client import (
    BlobConfigError,
    FetchResult,
    fetch_markdown,
    fetch_pipeline_artifacts,
    resolve_base_path,
)
from core.blob_paths import BASELINE_PIPELINE_ID
from core.compound_matching import diff_compounds_nway
from core.compound_parsing import parse_compounds_json
from core.compound_report import (
    build_upset_memberships,
    entries_to_dataframe,
    identifier_type_counts,
    nway_clusters_dataframe,
    nway_label_counts_dataframe,
    nway_pairwise_summary_dataframe,
)
from core.embeddings import MODEL_NAMES, compute_vs_reference_similarities, load_models, resolve_device
from core.flagging import build_disagree_mask, collapse_flag_regions
from core.line_arrays import build_line_arrays
from core.m1_agreement import (
    FIELD_FILTER_OPTIONS,
    ClusterAgreementRow,
    compute_m1_agreement,
    filter_cluster_rows,
)
from core.m1_parsing import QUANTITY_FIELDS, parse_m1_json
from core.m1_visuals import build_agreement_heatmap
from core.models import PrepassRun
from core.parsing import parse_prepass_json, total_lines_from_markdown, type_distribution
from core.r1_parsing import parse_r1_json
from core.reaction_matching import compare_reactions
from core.reaction_parsing import parse_reactions_json
from core.reaction_report import (
    false_negatives_to_dataframe,
    false_positives_to_dataframe,
    matched_pairs_to_dataframe as reaction_matched_pairs_to_dataframe,
    summary_to_dataframe as reaction_summary_to_dataframe,
)
from core.scoring import compute_multi_run_scores, resolve_reference_index
from core.upset_viz import render_upset
from core.visuals import build_timeline_figure, build_type_histogram
import matplotlib.pyplot as plt

load_dotenv()

DEFAULT_RUN_LABELS = ["Claude", "DeepSeek"]
DEFAULT_PIPELINE_IDS = [BASELINE_PIPELINE_ID, "section-wise-v1-deepseek-flash"]

FILE_KINDS = ("prepass", "m1", "m2", "r1", "reactions")
FILE_KIND_LABELS = {
    "prepass": "Pre-pass JSON",
    "m1": "M1 JSON",
    "m2": "M2 JSON",
    "r1": "R1 JSON",
    "reactions": "Reactions JSON",
}


@dataclass
class ModelUploads:
    """One labeled model row from the upload matrix."""

    label: str
    files: dict[str, tuple[bytes, str]]  # kind -> (raw bytes, filename)


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _default_label(index: int) -> str:
    if index < len(DEFAULT_RUN_LABELS):
        return DEFAULT_RUN_LABELS[index]
    return f"Run {index + 1}"


def _default_reference_label(labels: list[str]) -> str:
    if "Claude" in labels:
        return "Claude"
    return labels[0]


def _format_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _format_aliases(aliases: tuple[str, ...]) -> str:
    return ", ".join(aliases) if aliases else "—"


def _format_quantity(quantity: dict[str, float | None] | Mapping[str, float | None]) -> str:
    parts = []
    for field in QUANTITY_FIELDS:
        value = quantity.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    return ", ".join(parts) if parts else "—"


def _show_upset(memberships: list[frozenset[str]], *, key: str) -> None:
    """Render an UpSet figure full-width and close it to avoid Matplotlib leaks."""
    from core.upset_viz import _fallback_intersection_figure, membership_category_labels

    fig = None
    try:
        fig = render_upset(memberships)
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    except Exception as exc:  # noqa: BLE001 — Cloud draw/savefig can fail after plot()
        if fig is not None:
            plt.close(fig)
        categories = membership_category_labels(memberships)
        fallback = _fallback_intersection_figure(
            memberships,
            categories,
            reason=str(exc),
            fig_width=10.0,
            fig_height=4.5,
        )
        st.warning(
            "UpSet plot failed on this host; showing intersection bar chart instead. "
            f"Detail: {exc}"
        )
        st.pyplot(fallback, clear_figure=True, use_container_width=True)
        plt.close(fallback)
    else:
        plt.close(fig)
    _ = key  # reserved for future keyed caching / uniqueness


def _input_signature(
    rows: list[ModelUploads],
    baseline: str | None,
    markdown_bytes: bytes | None,
) -> tuple:
    """Hashable signature of uploads/baseline/markdown (excludes UI-only widgets)."""
    row_parts = []
    for row in rows:
        file_parts = tuple(
            sorted(
                (kind, _file_hash(raw), filename)
                for kind, (raw, filename) in row.files.items()
            )
        )
        row_parts.append((row.label, file_parts))
    markdown_hash = _file_hash(markdown_bytes) if markdown_bytes is not None else None
    return (tuple(row_parts), baseline, markdown_hash)


@st.cache_resource(show_spinner="Loading embedding models (first run may download weights)...")
def get_models():
    device = resolve_device()
    models = load_models(device)
    return models, device


@st.cache_data(show_spinner="Computing embeddings and scores...")
def run_line_comparison(
    run_hashes: tuple[str, ...],
    markdown_hash: str,
    run_payloads: tuple[tuple[str, str, bytes], ...],
    markdown_bytes: bytes,
    reference_label: str,
    *,
    parser_kind: str,
):
    """Heavy line-based comparison (pre-pass or R1) — cached by input hashes."""
    _ = (run_hashes, markdown_hash, parser_kind)

    total_lines = total_lines_from_markdown(markdown_bytes)
    runs: list[PrepassRun] = []
    for label, filename, raw in run_payloads:
        source = f"{label} ({filename})"
        if parser_kind == "r1":
            sections = parse_r1_json(raw, source_label=source)
        else:
            sections = parse_prepass_json(raw, source_label=source)
        arrays = build_line_arrays(sections, total_lines)
        runs.append(
            PrepassRun(
                label=label,
                filename=filename,
                sections=sections,
                arrays=arrays,
                type_counts=type_distribution(sections),
            )
        )

    run_labels = [run.label for run in runs]
    reference_index = resolve_reference_index(run_labels, reference_label)
    reference_arrays = runs[reference_index].arrays
    benchmark_arrays = [run.arrays for i, run in enumerate(runs) if i != reference_index]

    models, device = get_models()
    sims_per_benchmark = compute_vs_reference_similarities(
        models,
        reference_arrays,
        benchmark_arrays,
    )
    scores = compute_multi_run_scores(
        [run.arrays for run in runs],
        run_labels,
        reference_index,
        sims_per_benchmark,
    )

    return {
        "runs": runs,
        "scores": scores,
        "total_lines": total_lines,
        "device": device,
        "model_names": MODEL_NAMES,
        "reference_label": reference_label,
    }


def _default_pipeline_id(index: int) -> str:
    if index < len(DEFAULT_PIPELINE_IDS):
        return DEFAULT_PIPELINE_IDS[index]
    return ""


def _format_fetch_status(status: dict[str, FetchResult] | None) -> str:
    if not status:
        return ""
    parts: list[str] = []
    for kind in FILE_KINDS:
        result = status.get(kind)
        label = FILE_KIND_LABELS[kind].replace(" JSON", "")
        if result is None:
            parts.append(f"{label} —")
            continue
        if result.found and result.blob_path:
            short = result.blob_path
            if "/extraction/" in short:
                short = short.split("/extraction/", 1)[1]
                short = f"extraction/{short}"
            elif short.endswith("/reactions.json"):
                short = "reactions.json"
            elif "/markdown.md" in short:
                short = short.rsplit("/", 2)[-2] + "/" + short.rsplit("/", 1)[-1]
            parts.append(f"{label} ✓ `{short}`")
        elif result.error:
            parts.append(f"{label} ✗ {result.error}")
        else:
            parts.append(f"{label} ✗ not found")
    return " · ".join(parts)


def _clear_fetched_row(index: int) -> None:
    for kind in FILE_KINDS:
        st.session_state.pop(f"fetched_{kind}_{index}", None)
    st.session_state.pop(f"fetch_status_{index}", None)


def _base_path_cache() -> dict[str, str]:
    if "blob_base_path_cache" not in st.session_state:
        st.session_state.blob_base_path_cache = {}
    return st.session_state.blob_base_path_cache


def _resolve_base_for_ui(patent_id: str) -> str:
    return resolve_base_path(patent_id, cache=_base_path_cache())


def _collect_model_rows(num_models: int) -> list[ModelUploads]:
    """Build labeled model rows; prefer blob-fetched bytes over uploads."""
    rows: list[ModelUploads] = []
    for index in range(num_models):
        label = st.session_state.get(f"model_label_{index}", _default_label(index)).strip()
        files: dict[str, tuple[bytes, str]] = {}
        for kind in FILE_KINDS:
            fetched = st.session_state.get(f"fetched_{kind}_{index}")
            if fetched is not None:
                files[kind] = fetched
                continue
            uploaded = st.session_state.get(f"{kind}_file_{index}")
            if uploaded is not None:
                files[kind] = (uploaded.getvalue(), uploaded.name)
        if not label:
            if files:
                st.warning(f"Model {index + 1}: provide a label or its files will be skipped.")
            continue
        if files:
            rows.append(ModelUploads(label=label, files=files))
    return rows


def _labels_with_file(rows: list[ModelUploads], kind: str) -> list[str]:
    return [row.label for row in rows if kind in row.files]


def _payloads_for_kind(
    rows: list[ModelUploads],
    kind: str,
) -> list[tuple[str, bytes, str]]:
    """Return (label, raw, filename) for rows that have ``kind`` uploaded."""
    return [
        (row.label, row.files[kind][0], row.files[kind][1])
        for row in rows
        if kind in row.files
    ]


def _render_line_section_results(
    result: dict[str, Any],
    *,
    section_title: str,
    unit_name: str,
    caption: str | None = None,
) -> None:
    """Shared timeline / histogram / flagging UI for pre-pass and R1."""
    scores = result["scores"]
    runs: list[PrepassRun] = result["runs"]
    run_label_list = [run.label for run in runs]
    reference = scores.reference_label

    st.success(
        f"{section_title} complete — {result['total_lines']} lines across {len(runs)} outputs "
        f"vs reference **{reference}**, device: **{result['device']}**"
    )
    if caption:
        st.caption(caption)

    score_rows = []
    for label in scores.benchmark_labels:
        run_scores = scores.per_run[label]
        score_rows.append(
            {
                "model": label,
                f"type agreement vs {reference}": f"{run_scores.score_type_agreement:.1%}",
                f"model 1 sim vs {reference}": round(run_scores.score_model_1, 3),
                f"model 2 sim vs {reference}": round(run_scores.score_model_2, 3),
                f"model 3 sim vs {reference}": round(run_scores.score_model_3, 3),
                f"cumulative sim vs {reference}": round(run_scores.score_cumulative, 3),
            }
        )
    st.subheader(f"Scores vs {reference}")
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
    with st.expander("Embedding model names"):
        for index, model_name in enumerate(result["model_names"], start=1):
            st.write(f"Model {index}: `{model_name}`")

    with st.expander(f"Per-output {unit_name} counts"):
        section_rows = []
        for run in runs:
            section_rows.append(
                {
                    "label": run.label,
                    "role": "reference" if run.label == reference else "benchmark",
                    "file": run.filename,
                    unit_name: len(run.sections),
                }
            )
        st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        threshold = st.slider(
            "Label similarity threshold (for flagging)",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            key=f"threshold_{unit_name}",
            help=f"Flag lines where a benchmark's label similarity vs {reference} falls below this value",
        )

    disagree = build_disagree_mask(scores, threshold)
    flag_regions = collapse_flag_regions(
        disagree,
        [run.arrays for run in runs],
        run_label_list,
        scores,
        threshold,
    )

    ordered_runs = [run for run in runs if run.label == reference]
    ordered_runs.extend(run for run in runs if run.label != reference)
    timeline_fig = build_timeline_figure(
        [(run.label, run.sections) for run in ordered_runs],
        flag_regions,
        result["total_lines"],
    )
    hist_fig = build_type_histogram({run.label: run.type_counts for run in runs})

    with chart_col1:
        st.plotly_chart(timeline_fig, use_container_width=True)
    with chart_col2:
        st.plotly_chart(hist_fig, use_container_width=True)

    st.subheader(f"Flagged regions ({len(flag_regions)})")
    st.caption(
        f"A line is flagged when any benchmark disagrees with **{reference}** on section_type "
        "or its label similarity vs the reference is below the threshold."
    )

    if flag_regions:
        flag_rows = []
        for region in flag_regions:
            row = {
                "start_line": region.start_line,
                "end_line": region.end_line,
                "lines": region.end_line - region.start_line + 1,
                "issue_kind": region.issue_kind,
                "disagreeing_models": ", ".join(region.disagreeing_models),
                "worst_avg_cum_sim": round(region.avg_cum_sim, 4),
            }
            for label in run_label_list:
                row[f"{label} type"] = region.run_types.get(label)
                row[f"{label} label"] = region.run_labels.get(label)
            flag_rows.append(row)
        st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)
    else:
        st.success(f"No disagreements with {reference} found at the current threshold.")


def render_prepass_section(
    rows: list[ModelUploads],
    baseline: str,
    markdown_bytes: bytes | None,
) -> None:
    st.header("Pre-pass")
    labels = _labels_with_file(rows, "prepass")
    if len(labels) < 2:
        st.info(f"Skipped — need ≥ 2 pre-pass uploads, got {len(labels)}.")
        return
    if markdown_bytes is None:
        st.info("Skipped — enriched markdown is required for pre-pass total_lines.")
        return
    if baseline not in labels:
        st.info(
            f"Skipped — baseline **{baseline}** has no pre-pass file. "
            f"Uploads: {', '.join(labels)}."
        )
        return

    payloads = _payloads_for_kind(rows, "prepass")
    run_payloads = tuple((label, filename, raw) for label, raw, filename in payloads)
    try:
        result = run_line_comparison(
            tuple(_file_hash(raw) for _, raw, _ in payloads),
            _file_hash(markdown_bytes),
            run_payloads,
            markdown_bytes,
            baseline,
            parser_kind="prepass",
        )
    except ValueError as error:
        st.error(str(error))
        return

    _render_line_section_results(
        result,
        section_title="Pre-pass",
        unit_name="sections",
    )


def render_r1_section(
    rows: list[ModelUploads],
    baseline: str,
    markdown_bytes: bytes | None,
) -> None:
    st.header("R1 — Step boundaries")
    labels = _labels_with_file(rows, "r1")
    if len(labels) < 2:
        st.info(f"Skipped — need ≥ 2 R1 uploads, got {len(labels)}.")
        return
    if markdown_bytes is None:
        st.info("Skipped — enriched markdown is required for R1 total_lines.")
        return
    if baseline not in labels:
        st.info(
            f"Skipped — baseline **{baseline}** has no R1 file. "
            f"Uploads: {', '.join(labels)}."
        )
        return

    payloads = _payloads_for_kind(rows, "r1")
    run_payloads = tuple((label, filename, raw) for label, raw, filename in payloads)
    try:
        result = run_line_comparison(
            tuple(_file_hash(raw) for _, raw, _ in payloads),
            _file_hash(markdown_bytes),
            run_payloads,
            markdown_bytes,
            baseline,
            parser_kind="r1",
        )
    except ValueError as error:
        st.error(str(error))
        return

    _render_line_section_results(
        result,
        section_title="R1",
        unit_name="steps",
        caption=(
            "Note: `section_type` on R1 steps is inherited from the parent pre-pass section, "
            "not newly generated — type agreement will trend near 100% by construction. "
            "The meaningful signal is step boundary geometry and step_label similarity."
        ),
    )


def _render_m1_disagreement_browser(agreement) -> None:
    """Filterable disagreement-first drill-down for matched M1 clusters."""
    cluster_rows: list[ClusterAgreementRow] = agreement.cluster_rows
    st.subheader("Field disagreement browser")
    if not cluster_rows:
        st.info("No shared clusters with the baseline.")
        return

    model_options = sorted({row.model_label for row in cluster_rows})
    total_disagreements = sum(1 for row in cluster_rows if row.has_disagreement)
    total_agreements = len(cluster_rows) - total_disagreements

    filter_cols = st.columns([1.4, 1.4, 1.6, 1.0])
    with filter_cols[0]:
        selected_models = st.multiselect(
            "Models",
            options=model_options,
            default=model_options,
            key="m1_disagreement_models",
        )
    with filter_cols[1]:
        field_filter = st.selectbox(
            "Field filter",
            options=list(FIELD_FILTER_OPTIONS),
            index=0,
            key="m1_disagreement_field",
        )
    with filter_cols[2]:
        identifier_query = st.text_input(
            "Identifier search",
            value="",
            key="m1_disagreement_query",
            placeholder="Search baseline or model identifier",
        )
    with filter_cols[3]:
        disagreements_only = st.checkbox(
            "Disagreements only",
            value=True,
            key="m1_disagreement_only",
        )

    filtered = filter_cluster_rows(
        cluster_rows,
        model_labels=selected_models,
        field_filter=field_filter,
        identifier_query=identifier_query,
        disagreements_only=disagreements_only,
    )
    hidden = total_agreements if disagreements_only else 0
    st.caption(
        f"Showing {len(filtered)} row(s)"
        + (f"; {hidden} fully agreeing pairs hidden." if disagreements_only else ".")
        + f" Total matched pairs: {len(cluster_rows)} "
        f"({total_disagreements} with ≥1 disagreement)."
    )

    if not filtered:
        st.success("No rows match the current filters.")
        return

    table_rows = []
    for row in filtered:
        table_rows.append(
            {
                "model": row.model_label,
                "baseline_id": row.baseline_identifier,
                "model_id": row.model_identifier,
                "failed_fields": ", ".join(row.failed_fields) if row.failed_fields else "—",
                "alias_jaccard": round(row.alias_jaccard, 3),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("**Pair details**")
    for index, row in enumerate(filtered):
        title = (
            f"{row.model_label}: {row.baseline_identifier} ↔ {row.model_identifier}"
            + (f" [{', '.join(row.failed_fields)}]" if row.failed_fields else " [agree]")
        )
        with st.expander(title, expanded=False):
            detail = pd.DataFrame(
                [
                    {
                        "field": "identifier",
                        "baseline": row.baseline_identifier,
                        "model": row.model_identifier,
                        "agree": row.baseline_identifier == row.model_identifier,
                    },
                    {
                        "field": "identifier_type",
                        "baseline": row.baseline_identifier_type,
                        "model": row.model_identifier_type,
                        "agree": row.identifier_type_agree,
                    },
                    {
                        "field": "role",
                        "baseline": row.baseline_role or "—",
                        "model": row.model_role or "—",
                        "agree": row.role_agree,
                    },
                    {
                        "field": "is_section_product",
                        "baseline": row.baseline_is_section_product,
                        "model": row.model_is_section_product,
                        "agree": row.is_section_product_agree,
                    },
                    {
                        "field": "aliases",
                        "baseline": _format_aliases(row.baseline_aliases),
                        "model": _format_aliases(row.model_aliases),
                        "agree": "aliases" not in row.failed_fields,
                    },
                    {
                        "field": "alias_jaccard",
                        "baseline": round(row.alias_jaccard, 3),
                        "model": round(row.alias_jaccard, 3),
                        "agree": "aliases" not in row.failed_fields,
                    },
                    {
                        "field": "quantity",
                        "baseline": _format_quantity(dict(row.baseline_quantity)),
                        "model": _format_quantity(dict(row.model_quantity)),
                        "agree": all(row.quantity_field_agree.values()),
                    },
                ]
            )
            st.dataframe(detail, use_container_width=True, hide_index=True)
            qty_rows = []
            for field in QUANTITY_FIELDS:
                qty_rows.append(
                    {
                        "quantity_field": field,
                        "baseline": row.baseline_quantity.get(field),
                        "model": row.model_quantity.get(field),
                        "presence_agree": row.quantity_field_agree.get(field, True),
                    }
                )
            st.caption("Quantity presence agreement (null vs populated)")
            st.dataframe(pd.DataFrame(qty_rows), use_container_width=True, hide_index=True)
            _ = index


def render_m1_section(rows: list[ModelUploads], baseline: str) -> None:
    st.header("M1 — Molecule pass 1")
    labels = _labels_with_file(rows, "m1")
    if len(labels) < 2:
        st.info(f"Skipped — need ≥ 2 M1 uploads, got {len(labels)}.")
        return
    if baseline not in labels:
        st.info(
            f"Skipped — baseline **{baseline}** has no M1 file. "
            f"Uploads: {', '.join(labels)}."
        )
        return

    try:
        entries_by_label = {}
        for label, raw, filename in _payloads_for_kind(rows, "m1"):
            entries_by_label[label] = parse_m1_json(
                raw, source_label=f"{label} ({filename})"
            )
        agreement = compute_m1_agreement(entries_by_label, baseline)
    except (ValueError, KeyError) as error:
        st.error(str(error))
        return

    nway = agreement.nway
    st.success(
        f"M1 comparison complete — {len(nway.labels)} models, "
        f"{len(nway.clusters)} compound clusters vs baseline **{baseline}**."
    )
    st.caption(
        "`section_label` is usually deterministic from the document structure — "
        "focus on identifier_type, role, aliases, and quantity field agreement."
    )

    summary_rows = []
    for summary in agreement.summaries.values():
        summary_rows.append(
            {
                "model": summary.label,
                "common": summary.common,
                "baseline_only": summary.baseline_only,
                "model_only": summary.model_only,
                "recall": _format_rate(summary.recall),
                "precision": _format_rate(summary.precision),
                "identifier_type": _format_rate(summary.identifier_type.rate),
                "role": _format_rate(summary.role.rate),
                "is_section_product": _format_rate(summary.is_section_product.rate),
                "alias_jaccard": (
                    "—"
                    if summary.alias_jaccard_mean is None
                    else round(summary.alias_jaccard_mean, 3)
                ),
                "quantity_presence": _format_rate(summary.quantity.overall_rate),
            }
        )
    st.subheader(f"Agreement vs {baseline}")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.subheader("Agreement heatmap")
    st.plotly_chart(
        build_agreement_heatmap(
            agreement.summaries,
            title=f"Field agreement vs {baseline}",
        ),
        use_container_width=True,
    )

    st.subheader("Label counts")
    st.dataframe(nway_label_counts_dataframe(nway), use_container_width=True, hide_index=True)

    memberships = build_upset_memberships(nway)
    st.subheader("UpSet — compound overlap")
    _show_upset(memberships, key="m1_upset")

    st.subheader("Multi-model clusters")
    common_df = nway_clusters_dataframe(nway, min_labels=2)
    if common_df.empty:
        st.warning("No multi-model clusters.")
    else:
        st.dataframe(common_df, use_container_width=True, hide_index=True)

    _render_m1_disagreement_browser(agreement)


def render_m2_section(rows: list[ModelUploads], baseline: str) -> None:
    st.header("M2 — Molecule pass 2")
    labels = _labels_with_file(rows, "m2")
    if len(labels) < 2:
        st.info(f"Skipped — need ≥ 2 M2 uploads, got {len(labels)}.")
        return
    if baseline not in labels:
        st.info(
            f"Skipped — baseline **{baseline}** has no M2 file. "
            f"Uploads: {', '.join(labels)}."
        )
        return

    try:
        entries_by_label = {}
        for label, raw, filename in _payloads_for_kind(rows, "m2"):
            entries_by_label[label] = parse_compounds_json(
                raw, source_label=f"{label} ({filename})"
            )
        nway = diff_compounds_nway(entries_by_label)
    except (ValueError, KeyError) as error:
        st.error(str(error))
        return

    st.success(
        f"M2 comparison complete — {len(nway.labels)} models, "
        f"{len(nway.clusters)} compound clusters vs baseline **{baseline}**."
    )
    st.caption(
        "Compounds match when any normalized identifier or alias overlaps. "
        "No PubChem, RDKit, or embeddings."
    )

    st.subheader(f"Pairwise vs {baseline}")
    pairwise_df = nway_pairwise_summary_dataframe(nway, baseline)
    display_df = pairwise_df.copy()
    if not display_df.empty:
        display_df["recall_vs_baseline"] = display_df["recall_vs_baseline"].map(_format_rate)
        display_df["precision_vs_baseline"] = display_df["precision_vs_baseline"].map(
            _format_rate
        )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Label counts")
    st.dataframe(nway_label_counts_dataframe(nway), use_container_width=True, hide_index=True)

    memberships = build_upset_memberships(nway)
    st.subheader("UpSet — compound overlap")
    _show_upset(memberships, key="m2_upset")

    st.subheader("Multi-model clusters")
    common_df = nway_clusters_dataframe(nway, min_labels=2)
    if common_df.empty:
        st.warning("No multi-model clusters.")
    else:
        st.dataframe(common_df, use_container_width=True, hide_index=True)

    only_cols = st.columns(min(len(labels), 3))
    for index, label in enumerate(labels):
        only_entries = nway.only_entries(label)
        with only_cols[index % len(only_cols)]:
            st.subheader(f"{label} only ({len(only_entries)})")
            if only_entries:
                st.dataframe(
                    entries_to_dataframe(only_entries),
                    use_container_width=True,
                    hide_index=True,
                )
                type_counts = identifier_type_counts(only_entries)
                st.caption(
                    "By identifier_type: "
                    + ", ".join(f"{key}={value}" for key, value in sorted(type_counts.items()))
                )
            else:
                st.success(f"No {label}-only compounds.")


def render_reactions_section(rows: list[ModelUploads], baseline: str) -> None:
    st.header("Reactions")
    labels = _labels_with_file(rows, "reactions")
    if baseline not in labels or len(labels) < 2:
        got = len(labels)
        st.info(
            f"Skipped — need baseline + ≥ 1 other reactions upload "
            f"(baseline present: {baseline in labels}, total uploads: {got})."
        )
        return

    try:
        reactions_by_label = {}
        for label, raw, filename in _payloads_for_kind(rows, "reactions"):
            reactions_by_label[label] = parse_reactions_json(
                raw, source_label=f"{label} ({filename})"
            )
    except ValueError as error:
        st.error(str(error))
        return

    baseline_reactions = reactions_by_label[baseline]
    candidates = [label for label in labels if label != baseline]
    st.success(
        f"Reactions comparison — baseline **{baseline}** "
        f"({len(baseline_reactions)} records) vs {len(candidates)} model(s)."
    )
    st.caption(
        "Pairwise vs baseline (non_synthetic filtered). "
        "Axes: name, SMILES, reactants, procedure cosine, yield, conditions."
    )

    for candidate in candidates:
        st.subheader(f"{candidate} vs {baseline}")
        try:
            report = compare_reactions(
                baseline_reactions,
                reactions_by_label[candidate],
                baseline_label=baseline,
                candidate_label=candidate,
            )
        except Exception as error:
            st.error(f"{candidate}: {error}")
            continue

        metric_cols = st.columns(5)
        metric_cols[0].metric("Precision", _format_rate(report.precision))
        metric_cols[1].metric("Recall", _format_rate(report.recall))
        metric_cols[2].metric("F1", _format_rate(report.f1))
        metric_cols[3].metric("TP", report.true_positives)
        metric_cols[4].metric("FP / FN", f"{report.false_positives} / {report.false_negatives}")

        st.dataframe(
            reaction_summary_to_dataframe(report),
            use_container_width=True,
            hide_index=True,
        )

        matched_df = reaction_matched_pairs_to_dataframe(report)
        with st.expander(f"Matched pairs ({len(matched_df)})"):
            if matched_df.empty:
                st.warning("No content matches above threshold.")
            else:
                st.dataframe(matched_df, use_container_width=True, hide_index=True)

        fp_df = false_positives_to_dataframe(report)
        fn_df = false_negatives_to_dataframe(report)
        fp_col, fn_col = st.columns(2)
        with fp_col:
            st.markdown(f"**False positives ({len(fp_df)})** — {candidate} only")
            if fp_df.empty:
                st.success("None")
            else:
                st.dataframe(fp_df, use_container_width=True, hide_index=True)
        with fn_col:
            st.markdown(f"**False negatives ({len(fn_df)})** — {baseline} only")
            if fn_df.empty:
                st.success("None")
            else:
                st.dataframe(fn_df, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Multi-model Benchmark",
        page_icon="📊",
        layout="wide",
    )
    st.title("Multi-model Benchmark")
    st.caption(
        "Fetch labeled model runs from Azure Blob by Patent ID + Pipeline ID, "
        "or upload files manually. Pick a shared baseline and run all available "
        "benchmarks. Each section skips independently when it lacks enough inputs."
    )

    st.subheader("Patent")
    patent_col, _ = st.columns([2, 3])
    with patent_col:
        st.text_input(
            "Patent ID",
            key="patent_id",
            placeholder="e.g. CN105884573B",
            help="Used with each row's Pipeline ID to fetch artifacts from Azure Blob.",
        )

    st.subheader("Model uploads")
    num_models = st.number_input(
        "Number of models",
        min_value=2,
        max_value=8,
        value=2,
        step=1,
        help="Each row is one model. Fill only the file slots you want to compare.",
    )

    for index in range(num_models):
        st.markdown(f"**Model {index + 1}**")
        label_col, pipeline_col, fetch_col, clear_col = st.columns(
            [1.5, 2.0, 1.0, 1.0],
            vertical_alignment="bottom",
        )
        with label_col:
            st.text_input(
                "Label",
                value=_default_label(index),
                key=f"model_label_{index}",
                placeholder="e.g. Claude, DeepSeek",
            )
        with pipeline_col:
            st.text_input(
                "Pipeline ID",
                value=_default_pipeline_id(index),
                key=f"pipeline_id_{index}",
                placeholder="e.g. section-wise-v1",
            )
        with fetch_col:
            fetch_clicked = st.button(
                "Fetch from blob",
                key=f"fetch_blob_{index}",
                use_container_width=True,
            )
        with clear_col:
            clear_clicked = st.button(
                "Clear fetched",
                key=f"clear_fetched_{index}",
                use_container_width=True,
            )

        if clear_clicked:
            _clear_fetched_row(index)
            st.rerun()

        if fetch_clicked:
            patent_id = st.session_state.get("patent_id", "").strip()
            pipeline_id = st.session_state.get(f"pipeline_id_{index}", "").strip()
            if not patent_id:
                st.error("Enter a Patent ID before fetching.")
            elif not pipeline_id:
                st.error(f"Model {index + 1}: enter a Pipeline ID before fetching.")
            else:
                try:
                    with st.spinner(f"Fetching pipeline `{pipeline_id}` for `{patent_id}`..."):
                        results = fetch_pipeline_artifacts(
                            patent_id,
                            pipeline_id,
                            resolve_base=_resolve_base_for_ui,
                        )
                    for kind, result in results.items():
                        if result.found and result.content is not None and result.filename:
                            st.session_state[f"fetched_{kind}_{index}"] = (
                                result.content,
                                result.filename,
                            )
                        else:
                            st.session_state.pop(f"fetched_{kind}_{index}", None)
                    st.session_state[f"fetch_status_{index}"] = results
                    found_count = sum(1 for r in results.values() if r.found)
                    if found_count:
                        st.success(
                            f"Model {index + 1}: fetched {found_count}/{len(FILE_KINDS)} "
                            f"artifact(s) for `{pipeline_id}`."
                        )
                    else:
                        st.warning(
                            f"Model {index + 1}: no artifacts found for `{patent_id}` / "
                            f"`{pipeline_id}`."
                        )
                except (BlobConfigError, ValueError) as error:
                    st.error(str(error))
                except Exception as error:  # noqa: BLE001
                    st.error(f"Blob fetch failed: {error}")

        status = st.session_state.get(f"fetch_status_{index}")
        status_text = _format_fetch_status(status)
        if status_text:
            st.caption(status_text)

        label_col, *file_cols = st.columns([1.2, 1, 1, 1, 1, 1])
        with label_col:
            st.caption("Manual upload fallback")
        for kind, col in zip(FILE_KINDS, file_cols):
            with col:
                fetched = st.session_state.get(f"fetched_{kind}_{index}")
                label = FILE_KIND_LABELS[kind]
                if fetched is not None:
                    label = f"{label} (fetched)"
                st.file_uploader(
                    label,
                    type=["json"],
                    key=f"{kind}_file_{index}",
                )

    st.subheader("Shared inputs")
    md_label_col, md_file_col, md_fetch_col, md_clear_col = st.columns(
        [1.2, 2.2, 1.0, 1.0],
        vertical_alignment="bottom",
    )
    with md_label_col:
        st.text_input(
            "Markdown label",
            value="Source document",
            key="markdown_label",
        )
    with md_file_col:
        markdown_file = st.file_uploader(
            "Enriched markdown (required for pre-pass + R1)",
            type=["md", "txt", "markdown"],
            key="markdown_file",
        )
    with md_fetch_col:
        fetch_md_clicked = st.button(
            "Fetch markdown",
            key="fetch_markdown",
            use_container_width=True,
        )
    with md_clear_col:
        clear_md_clicked = False
        if st.session_state.get("fetched_markdown") is not None:
            clear_md_clicked = st.button(
                "Clear markdown",
                key="clear_fetched_markdown",
                use_container_width=True,
            )
        if clear_md_clicked:
            st.session_state.pop("fetched_markdown", None)
            st.session_state.pop("fetched_markdown_status", None)
            st.rerun()

    if fetch_md_clicked:
        patent_id = st.session_state.get("patent_id", "").strip()
        if not patent_id:
            st.error("Enter a Patent ID before fetching markdown.")
        else:
            try:
                with st.spinner(f"Fetching markdown for `{patent_id}`..."):
                    md_result = fetch_markdown(
                        patent_id,
                        resolve_base=_resolve_base_for_ui,
                    )
                st.session_state["fetched_markdown_status"] = md_result
                if md_result.found and md_result.content is not None and md_result.filename:
                    st.session_state["fetched_markdown"] = (
                        md_result.content,
                        md_result.filename,
                    )
                    st.success(f"Fetched markdown `{md_result.blob_path}`.")
                else:
                    st.session_state.pop("fetched_markdown", None)
                    detail = md_result.error or "not found"
                    st.warning(f"Markdown {detail}.")
            except (BlobConfigError, ValueError) as error:
                st.error(str(error))
            except Exception as error:  # noqa: BLE001
                st.error(f"Markdown fetch failed: {error}")

    md_status = st.session_state.get("fetched_markdown_status")
    if isinstance(md_status, FetchResult):
        if md_status.found and md_status.blob_path:
            st.caption(f"Markdown ✓ `{md_status.blob_path}`")
        elif md_status.error:
            st.caption(f"Markdown ✗ {md_status.error}")
        elif md_status.blob_path:
            st.caption(f"Markdown ✗ not found at `{md_status.blob_path}`")

    rows = _collect_model_rows(num_models)
    all_labels = [row.label for row in rows]
    duplicate_labels = {label for label in all_labels if all_labels.count(label) > 1}
    if duplicate_labels:
        st.error(
            f"Each model needs a unique label. Duplicates: {', '.join(sorted(duplicate_labels))}"
        )

    baseline: str | None = None
    if all_labels and not duplicate_labels:
        baseline = st.selectbox(
            "Baseline / reference model",
            options=all_labels,
            index=all_labels.index(_default_reference_label(all_labels)),
            help="Used by every benchmark section that has this label among its uploads.",
        )

    can_run = bool(rows) and baseline is not None and not duplicate_labels
    run_clicked = st.button("Run Benchmarks", type="primary", disabled=not can_run)

    fetched_md = st.session_state.get("fetched_markdown")
    if fetched_md is not None:
        markdown_bytes = fetched_md[0]
    else:
        markdown_bytes = markdown_file.getvalue() if markdown_file else None
    signature = _input_signature(rows, baseline, markdown_bytes) if can_run else None

    if "benchmark_active" not in st.session_state:
        st.session_state.benchmark_active = False
    if "benchmark_signature" not in st.session_state:
        st.session_state.benchmark_signature = None

    if run_clicked and can_run:
        st.session_state.benchmark_active = True
        st.session_state.benchmark_signature = signature

    if (
        st.session_state.benchmark_active
        and st.session_state.benchmark_signature is not None
        and signature is not None
        and signature != st.session_state.benchmark_signature
    ):
        st.session_state.benchmark_active = False
        st.warning(
            "Uploads, fetched files, labels, baseline, or markdown changed since the last run. "
            "Click **Run Benchmarks** again."
        )

    if not st.session_state.benchmark_active:
        st.info(
            "Fetch or upload at least one file per model you want to compare, choose a baseline, "
            "then click **Run Benchmarks**. Sections with fewer than two relevant files "
            "are skipped automatically. Threshold sliders keep results visible after a run."
        )
        return

    assert baseline is not None

    render_prepass_section(rows, baseline, markdown_bytes)
    st.divider()
    render_m1_section(rows, baseline)
    st.divider()
    render_m2_section(rows, baseline)
    st.divider()
    render_r1_section(rows, baseline, markdown_bytes)
    st.divider()
    render_reactions_section(rows, baseline)


if __name__ == "__main__":
    main()
