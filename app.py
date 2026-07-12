"""Pre-pass benchmark comparison — Streamlit entrypoint."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.embeddings import MODEL_NAMES, compute_vs_reference_similarities, load_models, resolve_device
from core.flagging import build_disagree_mask, collapse_flag_regions
from core.models import PrepassRun, build_prepass_run
from core.parsing import total_lines_from_markdown
from core.scoring import compute_multi_run_scores, resolve_reference_index
from core.visuals import build_timeline_figure, build_type_histogram

DEFAULT_RUN_LABELS = ["Claude", "DeepSeek"]


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@st.cache_resource(show_spinner="Loading embedding models (first run may download weights)...")
def get_models():
    device = resolve_device()
    models = load_models(device)
    return models, device


@st.cache_data(show_spinner="Computing embeddings and scores...")
def run_comparison(
    run_hashes: tuple[str, ...],
    markdown_hash: str,
    run_payloads: tuple[tuple[str, str, bytes], ...],
    markdown_bytes: bytes,
    markdown_label: str,
    reference_label: str,
):
    """Heavy comparison pass — cached by input file hashes."""
    _ = (run_hashes, markdown_hash)

    total_lines = total_lines_from_markdown(markdown_bytes)
    runs: list[PrepassRun] = [
        build_prepass_run(label, filename, raw, total_lines)
        for label, filename, raw in run_payloads
    ]
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
        "markdown_label": markdown_label,
        "reference_label": reference_label,
    }


def _default_label(index: int) -> str:
    if index < len(DEFAULT_RUN_LABELS):
        return DEFAULT_RUN_LABELS[index]
    return f"Run {index + 1}"


def _collect_run_inputs(num_runs: int) -> list[tuple[str, bytes, str]]:
    """Return (label, bytes, filename) for each uploaded run with a non-empty label."""
    collected: list[tuple[str, bytes, str]] = []
    for index in range(num_runs):
        label = st.session_state.get(f"run_label_{index}", _default_label(index)).strip()
        uploaded = st.session_state.get(f"run_file_{index}")
        if uploaded is None:
            continue
        if not label:
            st.warning(f"Run {index + 1}: provide a label or the file will be skipped.")
            continue
        collected.append((label, uploaded.getvalue(), uploaded.name))
    return collected


def _default_reference_label(labels: list[str]) -> str:
    if "Claude" in labels:
        return "Claude"
    return labels[0]


def main():
    st.set_page_config(
        page_title="Pre-pass Benchmark",
        page_icon="📊",
        layout="wide",
    )
    st.title("Pre-pass Benchmark Comparison")
    st.caption(
        "Compare multiple pre-pass JSON outputs against a reference (typically Claude). "
        "Each benchmark model is scored on type agreement and label similarity vs the reference. "
        "Use the sidebar to open **M2 Compound Diff** for molecule pass 2 comparison."
    )

    st.subheader("Pre-pass outputs")
    num_runs = st.number_input(
        "Number of outputs to compare",
        min_value=2,
        max_value=8,
        value=2,
        step=1,
        help="Upload at least two labeled pre-pass JSON files.",
    )

    for index in range(num_runs):
        label_col, file_col = st.columns([1, 2])
        with label_col:
            st.text_input(
                f"Label for output {index + 1}",
                value=_default_label(index),
                key=f"run_label_{index}",
                placeholder="e.g. Claude, DeepSeek, GPT-4o",
            )
        with file_col:
            st.file_uploader(
                f"Pre-pass JSON for output {index + 1}",
                type=["json"],
                key=f"run_file_{index}",
            )

    st.subheader("Source document")
    md_label_col, md_file_col = st.columns([1, 2])
    with md_label_col:
        markdown_label = st.text_input(
            "Markdown label",
            value="Source document",
            key="markdown_label",
        )
    with md_file_col:
        markdown_file = st.file_uploader(
            "Enriched markdown",
            type=["md", "txt", "markdown"],
            key="markdown_file",
        )

    run_inputs = _collect_run_inputs(num_runs)
    labels = [label for label, _, _ in run_inputs]
    duplicate_labels = {label for label in labels if labels.count(label) > 1}
    if duplicate_labels:
        st.error(f"Each output needs a unique label. Duplicates: {', '.join(sorted(duplicate_labels))}")

    reference_label = None
    if len(labels) >= 2 and not duplicate_labels:
        reference_label = st.selectbox(
            "Reference output (baseline for comparison)",
            options=labels,
            index=labels.index(_default_reference_label(labels)),
            help="All other outputs are compared against this reference, usually Claude.",
        )

    run_clicked = st.button(
        "Run Comparison",
        type="primary",
        disabled=not (len(run_inputs) >= 2 and markdown_file and reference_label and not duplicate_labels),
    )

    if not run_clicked:
        st.info(
            "Upload at least two labeled pre-pass JSON files plus enriched markdown, "
            "choose a reference output, then click **Run Comparison**."
        )
        return

    markdown_bytes = markdown_file.getvalue()
    run_payloads = tuple((label, filename, raw) for label, raw, filename in run_inputs)
    run_hashes = tuple(_file_hash(raw) for _, _, raw in run_payloads)

    try:
        result = run_comparison(
            run_hashes,
            _file_hash(markdown_bytes),
            run_payloads,
            markdown_bytes,
            markdown_label.strip() or "Source document",
            reference_label,
        )
    except ValueError as e:
        st.error(str(e))
        return

    scores = result["scores"]
    runs: list[PrepassRun] = result["runs"]
    run_label_list = [run.label for run in runs]
    reference = scores.reference_label

    st.success(
        f"Comparison complete — {result['total_lines']} lines across {len(runs)} outputs "
        f"vs reference **{reference}**, device: **{result['device']}**"
    )

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

    with st.expander("Per-output section counts"):
        section_rows = []
        for run in runs:
            section_rows.append(
                {
                    "label": run.label,
                    "role": "reference" if run.label == reference else "benchmark",
                    "file": run.filename,
                    "sections": len(run.sections),
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


if __name__ == "__main__":
    main()
