# Pre-pass Benchmark Streamlit App

Standalone tool to compare two pre-pass JSON outputs (e.g. Claude baseline vs. an open-source LLM) line-by-line, with visual timeline, type-distribution histogram, embedding-based label similarity, and disagreement flagging.

## Requirements

- Python 3.10+
- macOS Apple Silicon (MPS) recommended — falls back to CPU if MPS unavailable
- ~4 GB free RAM for loading three embedding models simultaneously

## Setup

```bash
cd /Users/suryansh.c/Desktop/MstackRepos/prepass-benchmark

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

First run will download three Hugging Face models (~3 GB total, cached locally afterward):

- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-large-v2`
- `all-mpnet-base-v2`

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Use the **sidebar** to switch between:

| Page | Purpose |
|------|---------|
| **app** (home) | Pre-pass section discovery comparison |
| **M2 Compound Diff** | Deterministic molecule pass 2 compound set diff |

### Pre-pass page

Upload three files via the UI:

| File | Description |
|------|-------------|
| Baseline pre-pass JSON | Claude output, pipeline `section-wise-v1` |
| Benchmark pre-pass JSON | Open-source model output, e.g. `section-wise-v1-deepseek-flash` |
| Enriched markdown | Source document; used to derive total line count |

### Pre-pass JSON format

Top-level array of groups, each with a `sections` array:

```json
[
  {
    "group_index": 0,
    "section_type": "experimental_example",
    "group_token_estimate": 1600,
    "sections": [
      {
        "section_index": 0,
        "section_label": "Example 1",
        "section_type": "experimental_example",
        "start_line": 1,
        "end_line": 120,
        "estimated_tokens": 1600
      }
    ]
  }
]
```

## Outputs

### Five scores

1. **Type agreement** — fraction of lines where both models assign the same `section_type`
2. **Model 1 similarity** — mean per-line cosine similarity (`BAAI/bge-large-en-v1.5`)
3. **Model 2 similarity** — mean per-line cosine similarity (`intfloat/e5-large-v2`)
4. **Model 3 similarity** — mean per-line cosine similarity (`all-mpnet-base-v2`)
5. **Cumulative similarity** — mean of the per-line average across all three models

### Visuals

- **Timeline** — two colored bars (baseline + benchmark) segmented by `section_type`, plus a disagreement overlay row
- **Histogram** — side-by-side section count per `section_type`

### Flagging

A line is flagged when:

- `section_type` differs between baseline and benchmark, **OR**
- cumulative label similarity (average of 3 models) is below the threshold slider (default 0.75)

Flagged regions are collapsed into contiguous ranges and shown in a table. No resolution/verdict workflow — highlight only.

### M2 Compound Diff page

Upload two M2 (molecule pass 2) JSON files — baseline (Claude) and benchmark (e.g. DeepSeek). Each file should be a flat JSON array of compound objects concatenated across all sections for one patent.

Headless CLI equivalent:

```bash
python compound_compare_cli.py \
  --claude Claude:path/to/claude-m2.json \
  --benchmark DeepSeek:path/to/deepseek-m2.json
```

Outputs: summary counts (`common`, baseline-only, benchmark-only), matched pairs table, unmatched tables with `identifier_type` breakdown, and a downloadable JSON report. No embeddings or network calls.

## Project structure

```
prepass-benchmark/
├── app.py                      # Pre-pass Streamlit home page
├── compound_compare_cli.py     # M2 diff CLI
├── compare_cli.py              # Pre-pass CLI
├── pages/
│   └── 2_M2_Compound_Diff.py   # M2 diff Streamlit page
├── requirements.txt
├── README.md
└── core/
    ├── parsing.py              # Pre-pass JSON + markdown parsing
    ├── line_arrays.py          # per-line type/label painting
    ├── embeddings.py           # 3 models, MPS, cosine similarity
    ├── scoring.py              # 5 summary scores
    ├── flagging.py             # disagreement regions
    ├── visuals.py              # Plotly charts
    ├── compound_parsing.py     # M2 JSON parsing
    ├── compound_matching.py    # deterministic compound diff
    └── compound_report.py      # shared CLI/UI report formatting
```

## Notes

- This app is fully standalone — no connection to `literatureiq-engine`, Azure, or any database.
- Adjusting the threshold slider re-runs only the cheap flagging step; embeddings are cached.
- Scores weight each line equally; longer sections naturally contribute more to averages.
