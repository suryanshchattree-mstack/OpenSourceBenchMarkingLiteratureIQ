# Multi-model Benchmark Streamlit App

Single-screen tool to compare N labeled model runs across four optional output kinds (pre-pass, compounds, R1 step boundaries, reactions). Fetch artifacts from Azure Blob by Patent ID + Pipeline ID, or upload files manually, pick one shared baseline, click **Run Benchmarks** — each section runs independently and skips with a notice when it lacks enough inputs.

## Requirements

- Python 3.10+ (for Streamlit Community Cloud, set **Python 3.12** in Advanced settings — `runtime.txt` is ignored by Cloud and the platform may default to 3.14)
- macOS Apple Silicon (MPS) recommended — falls back to CPU if MPS unavailable
- ~4 GB free RAM for loading three embedding models (used by pre-pass and R1)
- `rdkit` (reactions SMILES axis) and `upsetplot` (compounds overlap plots; falls back to a bar chart on Python 3.13+)
- Optional: Azure Storage connection string for blob fetch (manual upload still works without it)

## Setup

```bash
cd /Users/suryansh.c/Desktop/MstackRepos/prepass-benchmark

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Azure Blob (optional)

Copy `.env.example` to `.env` and set:

```bash
AZURE_STORAGE_CONNECTION_STRING=...   # full connection string (or SAS-backed equivalent)
AZURE_STORAGE_CONTAINER=datalake-raw-store
```

Without `.env`, the app stays upload-only. With it, each model row can **Fetch from blob** using the shared Patent ID and that row's Pipeline ID.

First pre-pass/R1 run downloads three Hugging Face models (~3 GB total, cached locally afterward):

- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-large-v2`
- `all-mpnet-base-v2`

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501 — one page, no sidebar multi-page nav.

### Blob fetch workflow

1. Enter a **Patent ID** (e.g. `CN105884573B`).
2. Per model row: set **Label**, **Pipeline ID** (defaults: `section-wise-v1`, `section-wise-v2`, …), click **Fetch from blob**.
3. Optionally click **Fetch markdown** for the shared enriched markdown.
4. Status captions show which kinds were found (and blob path) vs missing.
5. **Clear fetched** (or **Clear markdown**) drops session-fetched bytes and falls back to manual uploaders.
6. Fetched bytes take priority over uploads for the same slot when running benchmarks.

Blob layout (mirrors literatureiq-engine):

```
literature/patents/{countryCode}/{hashBucket}/{uuid5}/
  enriched/en/markdown.md          # preferred; falls back to en/markdown.md
  compounds.json                   # section-wise-v1 only (no fallback for other pipelines)
  reactions.json                   # section-wise-v1 only (no fallback for other pipelines)
  extraction/{pipelineId}/
    pre-pass-{timestamp}.json      # latest by prefix
    reaction-pass-1-consolidated.json
    compounds.json                 # non-baseline pipelines only
    reactions.json                 # non-baseline pipelines only
```

Root `compounds.json` / `reactions.json` are used only for `section-wise-v1`. Other pipeline IDs resolve solely under `extraction/{pipelineId}/` — no root or `persistent-store` fallback.

### Upload matrix

| Slot | Used by |
|------|---------|
| Pre-pass JSON | Pre-pass section timeline / scores |
| Compounds JSON | Presence matrix, UpSet, role/identifier_type matrices, confusion matrices |
| R1 JSON | Step-boundary comparison (same visuals as pre-pass) |
| Reactions JSON | Pairwise reaction panels vs baseline |
| Enriched markdown | Shared `total_lines` for pre-pass + R1 |

Fill only the slots you need. A section needs ≥ 2 uploads of that kind (reactions: baseline + ≥ 1 other). Missing kinds are skipped; the rest still run.

### Pre-pass / R1 JSON

Pre-pass: top-level array of groups with `sections[]`. R1: flat or double-encoded step lists mapped onto the same `Section` geometry (`start_line` / `end_line` are document-global). R1 `section_type` is inherited from pre-pass, so type agreement is less meaningful than boundary geometry.

### Compounds

Deterministic identifier/alias union-find across N models on persistent-store `compounds.json`. Views:

1. **Presence matrix** — clusters × models, present/absent, sorted by consensus count
2. **UpSet** — same cluster memberships as overlap plot
3. **Role matrix** — multi-model clusters only; cell = `role`; disagreements first
4. **Identifier-type matrix** — same skeleton for `identifier_type`
5. **Confusion matrices** — aggregate baseline×other co-occurrence for role and identifier_type

### Reactions

One pairwise panel per non-baseline model (name / SMILES / reactants / procedure cosine / yield / conditions). `non_synthetic` records are filtered before scoring.

## Headless CLIs (still available)

```bash
# Pre-pass
python compare_cli.py ...

# Compounds 2-way
python compound_compare_cli.py \
  --claude Claude:path/to/claude-compounds.json \
  --benchmark DeepSeek:path/to/deepseek-compounds.json
```

## Project structure

```
prepass-benchmark/
├── app.py                      # Unified Streamlit screen
├── compound_compare_cli.py     # Compounds diff CLI
├── compare_cli.py              # Pre-pass CLI
├── requirements.txt
├── .env.example                # Azure connection string + container
├── README.md
└── core/
    ├── blob_paths.py           # Patent → blob path builders (no network)
    ├── blob_client.py          # Azure fetch orchestration
    ├── parsing.py              # Pre-pass JSON + markdown
    ├── r1_parsing.py           # R1 → Section adapter
    ├── line_arrays.py          # per-line type/label painting
    ├── embeddings.py           # 3 models, MPS, cosine similarity
    ├── scoring.py / flagging.py / visuals.py
    ├── compound_parsing.py / compound_matching.py / compound_report.py
    ├── compound_visuals.py     # Presence / role / id-type / confusion heatmaps
    ├── upset_viz.py
    └── reaction_parsing.py / reaction_matching.py / reaction_report.py
```

## Notes

- Blob fetch is optional — without `.env` / credentials, use manual uploads as before.
- If murmur3 hash-bucket disagrees with production, the client brute-forces all 128 buckets for that country code once per patent (cached in session).
- Pre-pass/R1 threshold slider re-runs only flagging; embeddings are cached.
- Scores weight each line equally; longer sections contribute more to averages.
