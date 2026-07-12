"""M2 compound diff — Streamlit page."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from core.compound_matching import diff_compounds
from core.compound_parsing import parse_compounds_json
from core.compound_report import (
    build_diff_json_payload,
    entries_to_dataframe,
    identifier_type_counts,
    matched_pairs_to_dataframe,
)

DEFAULT_CLAUDE_LABEL = "Claude"
DEFAULT_BENCHMARK_LABEL = "DeepSeek"


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@st.cache_data(show_spinner="Comparing compounds...")
def run_compound_diff(
    claude_hash: str,
    benchmark_hash: str,
    claude_bytes: bytes,
    benchmark_bytes: bytes,
    claude_filename: str,
    benchmark_filename: str,
    claude_label: str,
    benchmark_label: str,
):
    _ = (claude_hash, benchmark_hash)
    claude_compounds = parse_compounds_json(
        claude_bytes,
        source_label=f"{claude_label} ({claude_filename})",
    )
    benchmark_compounds = parse_compounds_json(
        benchmark_bytes,
        source_label=f"{benchmark_label} ({benchmark_filename})",
    )
    result = diff_compounds(claude_compounds, benchmark_compounds)
    return result


st.set_page_config(
    page_title="M2 Compound Diff",
    page_icon="🧪",
    layout="wide",
)

st.title("M2 Compound Diff")
st.caption(
    "Deterministically compare two molecule pass 2 (M2) JSON outputs. "
    "Compounds match when any normalized identifier or alias overlaps. "
    "No PubChem, RDKit, or embeddings — unmatched entries are candidates for a future structure-matching pass."
)

st.subheader("Inputs")

col1, col2 = st.columns(2)
with col1:
    claude_label = st.text_input(
        "Baseline label",
        value=DEFAULT_CLAUDE_LABEL,
        key="m2_claude_label",
        help="Usually Claude — treated as the reference side.",
    )
    claude_file = st.file_uploader(
        "Baseline M2 JSON",
        type=["json"],
        key="m2_claude_file",
        help="Flat JSON array of M2 compound objects (concatenate all sections for the patent).",
    )
with col2:
    benchmark_label = st.text_input(
        "Benchmark label",
        value=DEFAULT_BENCHMARK_LABEL,
        key="m2_benchmark_label",
        help="Open-source or alternate model output.",
    )
    benchmark_file = st.file_uploader(
        "Benchmark M2 JSON",
        type=["json"],
        key="m2_benchmark_file",
    )

run_clicked = st.button(
    "Run Compound Diff",
    type="primary",
    disabled=not (claude_file and benchmark_file and claude_label.strip() and benchmark_label.strip()),
)

if not run_clicked:
    st.info(
        "Upload two M2 JSON files (baseline + benchmark), then click **Run Compound Diff**. "
        "Each file should be a JSON array of compound objects with an `identifier` field."
    )
    with st.expander("Expected M2 JSON shape"):
        st.code(
            """[
  {
    "identifier": "N,N-diisopropylethylamine",
    "identifier_type": "iupac",
    "aliases": ["DIPEA"],
    "resolved": true,
    "unresolved_reference": false,
    "section_label": "Example 1",
    "role": "base"
  }
]""",
            language="json",
        )
    st.stop()

claude_label = claude_label.strip()
benchmark_label = benchmark_label.strip()
claude_bytes = claude_file.getvalue()
benchmark_bytes = benchmark_file.getvalue()

try:
    result = run_compound_diff(
        _file_hash(claude_bytes),
        _file_hash(benchmark_bytes),
        claude_bytes,
        benchmark_bytes,
        claude_file.name,
        benchmark_file.name,
        claude_label,
        benchmark_label,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

recall = (
    result.common / result.deduped_claude_count
    if result.deduped_claude_count > 0
    else 0.0
)
precision = (
    result.common / result.deduped_benchmark_count
    if result.deduped_benchmark_count > 0
    else 0.0
)

st.success(
    f"Diff complete — {result.raw_claude_count} raw {claude_label} entries, "
    f"{result.raw_benchmark_count} raw {benchmark_label} entries "
    f"({result.deduped_claude_count} / {result.deduped_benchmark_count} deduped unique)"
)

st.subheader("Summary")
metric_cols = st.columns(5)
metric_cols[0].metric("Common (matched)", result.common)
metric_cols[1].metric(f"{claude_label} only", result.claude_only)
metric_cols[2].metric(f"{benchmark_label} only", result.benchmark_only)
metric_cols[3].metric(f"Recall vs {claude_label}", f"{recall:.1%}")
metric_cols[4].metric(f"Precision vs {claude_label}", f"{precision:.1%}")

summary_df = pd.DataFrame(
    [
        {
            "side": claude_label,
            "raw_entries": result.raw_claude_count,
            "deduped_unique": result.deduped_claude_count,
        },
        {
            "side": benchmark_label,
            "raw_entries": result.raw_benchmark_count,
            "deduped_unique": result.deduped_benchmark_count,
        },
    ]
)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

json_payload = build_diff_json_payload(result, claude_label, benchmark_label)
st.download_button(
    label="Download JSON report",
    data=json.dumps(json_payload, indent=2),
    file_name="m2-compound-diff.json",
    mime="application/json",
)

st.divider()

st.subheader(f"Matched pairs ({result.common})")
st.caption("Compounds linked by overlapping normalized identifier or alias.")
if result.matched_pairs:
    st.dataframe(
        matched_pairs_to_dataframe(result.matched_pairs, claude_label, benchmark_label),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("No matched pairs — every compound differs by name/alias under deterministic rules.")

only_col1, only_col2 = st.columns(2)

with only_col1:
    st.subheader(f"{claude_label} only ({result.claude_only})")
    if result.claude_only_entries:
        st.dataframe(
            entries_to_dataframe(result.claude_only_entries),
            use_container_width=True,
            hide_index=True,
        )
        type_counts = identifier_type_counts(result.claude_only_entries)
        st.caption(
            "By identifier_type: "
            + ", ".join(f"{key}={value}" for key, value in sorted(type_counts.items()))
        )
    else:
        st.success(f"Every {claude_label} compound matched something in {benchmark_label}.")

with only_col2:
    st.subheader(f"{benchmark_label} only ({result.benchmark_only})")
    if result.benchmark_only_entries:
        st.dataframe(
            entries_to_dataframe(result.benchmark_only_entries),
            use_container_width=True,
            hide_index=True,
        )
        type_counts = identifier_type_counts(result.benchmark_only_entries)
        st.caption(
            "By identifier_type: "
            + ", ".join(f"{key}={value}" for key, value in sorted(type_counts.items()))
        )
    else:
        st.success(f"Every {benchmark_label} compound matched something in {claude_label}.")
