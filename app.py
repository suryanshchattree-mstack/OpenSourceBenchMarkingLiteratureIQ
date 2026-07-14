"""Unified multi-model benchmark — single-screen Streamlit orchestrator."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from core.blob_client import (
    BlobConfigError,
    FetchResult,
    fetch_markdown,
    fetch_pipeline_artifacts,
    resolve_base_path,
)
from core.compound_baseline import (
    NONE_SENTINEL,
    compute_cluster_baselines,
    field_value_options,
)
from core.compound_colors import color_for_value, css_cell
from core.compound_grid import COMPOUND_GRID_FIELDS, drop_compound_rows
from core.compound_matching import diff_compounds_nway, format_match_tier
from core.compound_parsing import parse_compounds_json
from core.compound_pdf import build_compounds_pdf_report
from core.compound_report import build_upset_memberships, cluster_display_label, clusters_sorted_by_consensus
from core.compound_stats import (
    compute_field_accuracy,
    compute_presence_stats,
    filter_by_presence_baseline,
    rank_models,
    seed_view_dataframe,
)
from core.embeddings import MODEL_NAMES, compute_vs_reference_similarities, load_models, resolve_device
from core.flagging import build_disagree_mask, collapse_flag_regions
from core.line_arrays import build_line_arrays
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

DEFAULT_RUN_LABELS = [
    "Claude",
    "DeepSeekFlash",
    "GLM",
    "DeepSeekPro",
    "Kimi",
    "MiniMax",
]

FILE_KINDS = ("prepass", "compounds", "r1", "reactions")
FILE_KIND_LABELS = {
    "prepass": "Pre-pass JSON",
    "compounds": "Compounds JSON",
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



def _show_upset(memberships: list[frozenset[str]], *, key: str) -> None:
    """Render intersection viz full-width without crashing the page on Cloud."""
    from core.upset_viz import (
        _fallback_intersection_figure,
        figure_to_png_bytes,
        membership_category_labels,
    )

    def _display_fig(fig) -> None:
        # Prefer PNG + st.image: st.pyplot has segfaulted on Cloud Python 3.14.
        try:
            st.image(figure_to_png_bytes(fig), use_container_width=True)
        except TypeError:
            st.image(figure_to_png_bytes(fig), width="stretch")

    fig = None
    try:
        fig = render_upset(memberships)
        _display_fig(fig)
    except Exception as exc:  # noqa: BLE001 — Cloud draw/savefig can fail after plot()
        if fig is not None:
            plt.close(fig)
            fig = None
        categories = membership_category_labels(memberships)
        fallback = None
        try:
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
            _display_fig(fallback)
        except Exception as fallback_exc:  # noqa: BLE001
            st.warning(
                "Could not render intersection chart; showing counts as a table instead. "
                f"Detail: {exc} / {fallback_exc}"
            )
            from collections import Counter

            counts = Counter(
                " ∩ ".join(sorted(m)) if m else "(empty)"
                for m in memberships
                if m
            )
            st.dataframe(
                [{"intersection": label, "clusters": n} for label, n in counts.most_common()],
                use_container_width=True,
                hide_index=True,
            )
        finally:
            if fallback is not None:
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
    """Default blob pipeline id for model row ``index``: section-wise-v1, v2, …"""
    return f"section-wise-v{index + 1}"


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



def _compounds_editor_signature(rows: list[ModelUploads], patent_id: str) -> str:
    """Stable key fragment so data_editor edits reset when inputs change."""
    payloads = _payloads_for_kind(rows, "compounds")
    parts = [
        patent_id.strip(),
        *(f"{label}:{_file_hash(raw)}" for label, raw, _ in sorted(payloads, key=lambda item: item[0])),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return digest


def _panel_height(row_count: int) -> int:
    return 35 * (row_count + 1) + 3


def _is_bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return True
    type_name = type(value).__name__
    return type_name in {"bool_", "bool8"}


def _style_presence_values(df: pd.DataFrame, model_labels: list[str]):
    """Value-stable Presence colors: True/present=green, False/absent=red."""

    value_cols = ["Baseline", *model_labels]
    present_style = css_cell("#c6f6c6")
    absent_style = css_cell("#f8c6c6")

    def _style_row(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        for col in value_cols:
            if col not in row.index:
                continue
            idx = int(row.index.get_loc(col))
            value = row[col]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if _is_bool_like(value):
                styles[idx] = present_style if bool(value) else absent_style
        return styles

    return df.style.apply(_style_row, axis=1)


def _style_categorical_values(df: pd.DataFrame, model_labels: list[str]):
    """Color every model column and Baseline by the cell's own value (stable pastel)."""

    value_cols = ["Baseline", *model_labels]

    def _style_row(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        for col in value_cols:
            if col not in row.index:
                continue
            idx = int(row.index.get_loc(col))
            styles[idx] = css_cell(color_for_value(row[col]))
        return styles

    return df.style.apply(_style_row, axis=1)


def _editor_seed_key(field: str, signature: str) -> str:
    return f"compounds_{field}_seed_{signature}"


def _editor_widget_key(field: str, signature: str) -> str:
    return f"compounds_{field}_editor_{signature}"


def _editor_current_key(field: str, signature: str) -> str:
    return f"compounds_{field}_current_{signature}"


def _delete_compounds(signature: str, names: list[str]) -> None:
    """Remove matching Compound rows from seed slots and clear widget edit-state."""
    for field in COMPOUND_GRID_FIELDS:
        seed_key = _editor_seed_key(field, signature)
        if seed_key not in st.session_state:
            continue
        st.session_state[seed_key] = drop_compound_rows(
            st.session_state[seed_key], names
        )
        st.session_state.pop(_editor_widget_key(field, signature), None)
        st.session_state.pop(_editor_current_key(field, signature), None)


def _render_compound_delete_controls(signature: str) -> None:
    """Explicit multi-select delete that keeps the three compound grids aligned."""
    presence_df = st.session_state.get(_editor_current_key("presence", signature))
    if not isinstance(presence_df, pd.DataFrame):
        presence_df = st.session_state.get(_editor_seed_key("presence", signature))
    if not isinstance(presence_df, pd.DataFrame) or presence_df.empty:
        return
    options = [
        str(value).strip()
        for value in presence_df.get("Compound", pd.Series(dtype=str)).tolist()
        if value is not None and str(value).strip()
    ]
    if not options:
        return
    st.markdown("#### Delete compounds")
    selected = st.multiselect(
        "Rows to delete",
        options=options,
        key=f"compounds_delete_select_{signature}",
        help="Removes the selected compounds from Presence, Role, and Identifier-type.",
    )
    if st.button("Delete selected rows", key=f"compounds_delete_btn_{signature}"):
        if selected:
            _delete_compounds(signature, selected)
            st.rerun()
        else:
            st.warning("Select at least one compound to delete.")


def _seed_editor_df(
    *,
    field: str,
    signature: str,
    clusters: list,
    model_labels: list[str],
    defaults: dict,
    preferred: str,
) -> pd.DataFrame:
    seed_key = _editor_seed_key(field, signature)
    if seed_key not in st.session_state:
        st.session_state[seed_key] = seed_view_dataframe(
            clusters,
            model_labels,
            field,
            defaults=defaults,
            preferred_label=preferred,
        )
    return st.session_state[seed_key]


def _render_editable_view(
    *,
    title: str,
    field: str,
    signature: str,
    clusters: list,
    model_labels: list[str],
    defaults: dict,
    preferred: str,
    select_options: list[str] | None = None,
) -> pd.DataFrame:
    st.subheader(title)
    seed_key = _editor_seed_key(field, signature)
    widget_key = _editor_widget_key(field, signature)
    current_key = _editor_current_key(field, signature)
    _seed_editor_df(
        field=field,
        signature=signature,
        clusters=clusters,
        model_labels=model_labels,
        defaults=defaults,
        preferred=preferred,
    )

    column_config: dict[str, Any] = {
        "Compound": st.column_config.TextColumn("Compound", required=True),
        "Match tier": st.column_config.TextColumn("Match tier"),
    }
    if field == "presence":
        column_config["Baseline"] = st.column_config.CheckboxColumn("Baseline")
        for label in model_labels:
            column_config[label] = st.column_config.CheckboxColumn(label)
    else:
        options = select_options or [NONE_SENTINEL]
        column_config["Baseline"] = st.column_config.SelectboxColumn(
            "Baseline", options=options
        )
        for label in model_labels:
            column_config[label] = st.column_config.SelectboxColumn(
                label, options=options
            )

    left_col, right_col = st.columns([1, 1])
    with right_col:
        # Pass stable seed as data=; widget key= holds in-flight edits (do not
        # feed the previous return value back into data=).
        edited_df = st.data_editor(
            st.session_state[seed_key],
            key=widget_key,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            height=_panel_height(max(len(st.session_state[seed_key]), 1)),
        )
        st.session_state[current_key] = edited_df
        st.caption(
            "Use **+** to add a row; use **Delete selected rows** below to remove "
            "compounds from all three grids."
        )
    with left_col:
        styler = (
            _style_presence_values(edited_df, model_labels)
            if field == "presence"
            else _style_categorical_values(edited_df, model_labels)
        )
        st.dataframe(
            styler,
            use_container_width=True,
            hide_index=True,
            height=_panel_height(max(len(edited_df), 1)),
        )
    return edited_df


def _render_compound_inspector_sidebar(
    *,
    signature: str,
    presence_df: pd.DataFrame,
    clusters: list,
    preferred: str,
) -> None:
    cluster_by_display = {
        cluster_display_label(cluster, preferred): cluster for cluster in clusters
    }
    compounds = [
        str(value).strip()
        for value in presence_df.get("Compound", pd.Series(dtype=str)).tolist()
        if value is not None and str(value).strip()
    ]
    with st.sidebar:
        st.subheader("Raw compound inspector")
        if not compounds:
            st.caption("No compounds to inspect.")
            return
        selected = st.selectbox(
            "Inspect a compound",
            options=compounds,
            key=f"compound_inspect_{signature}",
        )
        selected_cluster = cluster_by_display.get(selected)
        if selected_cluster is None:
            st.caption("No raw data (manually added)")
            return
        st.caption(f"Matched via: `{format_match_tier(selected_cluster.match_tier)}`")
        for label in sorted(selected_cluster.membership):
            entry = selected_cluster.representatives[label]
            with st.expander(f"{label} — {entry.identifier}"):
                st.json(dict(entry.raw) if entry.raw else {"identifier": entry.identifier})


def render_compounds_section(rows: list[ModelUploads], baseline: str) -> None:
    st.header("Compounds")
    labels = _labels_with_file(rows, "compounds")
    if len(labels) < 2:
        st.info(f"Skipped — need ≥ 2 compounds uploads, got {len(labels)}.")
        return

    try:
        entries_by_label = {}
        for label, raw, filename in _payloads_for_kind(rows, "compounds"):
            entries_by_label[label] = parse_compounds_json(
                raw, source_label=f"{label} ({filename})"
            )
        nway = diff_compounds_nway(entries_by_label)
    except (ValueError, KeyError) as error:
        st.error(str(error))
        return

    preferred = baseline if baseline in nway.labels else (
        "Claude" if "Claude" in nway.labels else nway.labels[0]
    )
    defaults = compute_cluster_baselines(nway, tiebreak_label="Claude")
    clusters = clusters_sorted_by_consensus(nway)
    model_labels = list(nway.labels)
    signature = _compounds_editor_signature(rows, st.session_state.get("patent_id", ""))

    st.success(
        f"Compounds comparison complete — {len(nway.labels)} models, "
        f"{len(nway.clusters)} compound clusters "
        f"(adjudicated baseline: majority + Claude tiebreak)."
    )
    st.caption(
        "Matching waterfall: InChIKey → SMILES → molecular formula → normalized name/alias. "
        "Edit any model column or Baseline; use **+** to add rows and **Delete selected rows** "
        "to remove compounds from all three grids. Stats/PDF use the grids."
    )

    # Seed presence early so the sidebar inspector can list compounds (incl. added rows).
    presence_seed = _seed_editor_df(
        field="presence",
        signature=signature,
        clusters=clusters,
        model_labels=model_labels,
        defaults=defaults,
        preferred=preferred,
    )
    presence_for_sidebar = st.session_state.get(
        _editor_current_key("presence", signature), presence_seed
    )
    _render_compound_inspector_sidebar(
        signature=signature,
        presence_df=presence_for_sidebar,
        clusters=clusters,
        preferred=preferred,
    )

    role_options = field_value_options(entries_by_label, "role")
    id_type_options = field_value_options(entries_by_label, "identifier_type")

    edited_presence = _render_editable_view(
        title="Presence",
        field="presence",
        signature=signature,
        clusters=clusters,
        model_labels=model_labels,
        defaults=defaults,
        preferred=preferred,
    )

    edited_role = _render_editable_view(
        title="Role",
        field="role",
        signature=signature,
        clusters=clusters,
        model_labels=model_labels,
        defaults=defaults,
        preferred=preferred,
        select_options=role_options,
    )

    edited_id_type = _render_editable_view(
        title="Identifier type",
        field="identifier_type",
        signature=signature,
        clusters=clusters,
        model_labels=model_labels,
        defaults=defaults,
        preferred=preferred,
        select_options=id_type_options,
    )

    _render_compound_delete_controls(signature)
    # Re-read current caches after editors (and possible delete+rerun path).
    edited_presence = st.session_state[_editor_current_key("presence", signature)]
    edited_role = st.session_state[_editor_current_key("role", signature)]
    edited_id_type = st.session_state[_editor_current_key("identifier_type", signature)]

    memberships = build_upset_memberships(nway)
    st.subheader("UpSet — compound overlap")
    _show_upset(memberships, key="compounds_upset")

    st.subheader("Compare vs manual benchmark")
    weight_cols = st.columns(3)
    with weight_cols[0]:
        w_presence = st.number_input(
            "Presence F1 weight",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key=f"compounds_weight_presence_{signature}",
        )
    with weight_cols[1]:
        w_role = st.number_input(
            "Role accuracy weight",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key=f"compounds_weight_role_{signature}",
        )
    with weight_cols[2]:
        w_id_type = st.number_input(
            "Identifier-type accuracy weight",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key=f"compounds_weight_id_type_{signature}",
        )
    weights = {
        "presence_f1": float(w_presence),
        "role_accuracy": float(w_role),
        "identifier_type_accuracy": float(w_id_type),
    }

    presence_stats = compute_presence_stats(edited_presence, model_labels)
    role_scoped = filter_by_presence_baseline(edited_role, edited_presence)
    id_type_scoped = filter_by_presence_baseline(edited_id_type, edited_presence)
    role_acc = compute_field_accuracy(role_scoped, model_labels).rename(
        columns={"accuracy": "role_accuracy"}
    )
    id_acc = compute_field_accuracy(id_type_scoped, model_labels).rename(
        columns={"accuracy": "identifier_type_accuracy"}
    )
    stats_df = presence_stats.merge(role_acc, on="model", how="outer").merge(
        id_acc, on="model", how="outer"
    )
    stats_df = rank_models(stats_df, weights=weights)

    display_stats = stats_df.copy()
    for col in [
        "composite_score",
        "presence_precision",
        "presence_recall",
        "presence_f1",
        "role_accuracy",
        "identifier_type_accuracy",
    ]:
        if col not in display_stats.columns:
            continue
        display_stats[col] = display_stats[col].map(
            lambda v: "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.1%}"
        )
    st.dataframe(display_stats, use_container_width=True, hide_index=True)

    chart_df = stats_df.melt(
        id_vars=["model"],
        value_vars=["presence_f1", "role_accuracy", "identifier_type_accuracy"],
        var_name="metric",
        value_name="score",
    )
    chart_df = chart_df.dropna(subset=["score"])
    if not chart_df.empty:
        bar_fig = px.bar(
            chart_df,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            title="F1 / accuracy vs edited baseline (ranked)",
            range_y=[0, 1],
            category_orders={"model": list(stats_df["model"])},
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    pipeline_ids_by_label: dict[str, str] = {}
    num_models_state = st.session_state.get("num_models")
    try:
        num_models_int = int(num_models_state) if num_models_state is not None else 0
    except (TypeError, ValueError):
        num_models_int = 0
    for index in range(num_models_int):
        label = st.session_state.get(f"model_label_{index}", "").strip()
        pipeline_id = st.session_state.get(f"pipeline_id_{index}", "").strip()
        if label in model_labels and pipeline_id:
            pipeline_ids_by_label[label] = pipeline_id

    patent_id = st.session_state.get("patent_id", "").strip() or "compounds"
    try:
        pdf_bytes = build_compounds_pdf_report(
            patent_id,
            pipeline_ids_by_label,
            stats_df,
            edited_presence,
            edited_role,
            edited_id_type,
        )
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"{patent_id}-compounds-benchmark.pdf",
            mime="application/pdf",
            key=f"compounds_pdf_{signature}",
        )
    except Exception as error:  # noqa: BLE001
        st.warning(f"PDF export unavailable: {error}")



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


def _apply_fetch_results(index: int, results: dict[str, FetchResult]) -> int:
    """Store fetch results for a model row; return count of found artifacts."""
    for kind, result in results.items():
        if result.found and result.content is not None and result.filename:
            st.session_state[f"fetched_{kind}_{index}"] = (
                result.content,
                result.filename,
            )
        else:
            st.session_state.pop(f"fetched_{kind}_{index}", None)
    st.session_state[f"fetch_status_{index}"] = results
    return sum(1 for result in results.values() if result.found)


def _fetch_one_row(
    index: int, patent_id: str, pipeline_id: str
) -> tuple[int, dict[str, FetchResult] | BaseException]:
    try:
        results = fetch_pipeline_artifacts(
            patent_id,
            pipeline_id,
            resolve_base=_resolve_base_for_ui,
        )
        return index, results
    except Exception as error:  # noqa: BLE001
        return index, error


@st.fragment
def _render_model_row(index: int) -> None:
    """One model row; fragment-scoped so fetch/clear only reruns this row."""
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
            placeholder="e.g. Claude, DeepSeekFlash",
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
                found_count = _apply_fetch_results(index, results)
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
        value=6,
        step=1,
        key="num_models",
        help="Each row is one model. Fill only the file slots you want to compare.",
    )

    fetch_all_clicked = st.button("Fetch all", type="secondary")
    if fetch_all_clicked:
        patent_id = st.session_state.get("patent_id", "").strip()
        if not patent_id:
            st.error("Enter a Patent ID before fetching.")
        else:
            jobs: list[tuple[int, str]] = []
            for index in range(int(num_models)):
                pipeline_id = st.session_state.get(f"pipeline_id_{index}", "").strip()
                if pipeline_id:
                    jobs.append((index, pipeline_id))
            if not jobs:
                st.warning("No pipeline IDs configured — nothing to fetch.")
            else:
                with st.spinner(f"Fetching {len(jobs)} model row(s) in parallel..."):
                    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as executor:
                        futures = [
                            executor.submit(_fetch_one_row, index, patent_id, pipeline_id)
                            for index, pipeline_id in jobs
                        ]
                        for future in as_completed(futures):
                            index, outcome = future.result()
                            if isinstance(outcome, BaseException):
                                st.error(f"Model {index + 1}: {outcome}")
                                continue
                            found_count = _apply_fetch_results(index, outcome)
                            pipeline_id = st.session_state.get(
                                f"pipeline_id_{index}", ""
                            ).strip()
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

    for index in range(int(num_models)):
        _render_model_row(index)

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
    render_compounds_section(rows, baseline)
    st.divider()
    render_r1_section(rows, baseline, markdown_bytes)
    st.divider()
    render_reactions_section(rows, baseline)


if __name__ == "__main__":
    main()
