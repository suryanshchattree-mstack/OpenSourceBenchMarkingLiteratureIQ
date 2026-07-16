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
| Reactions JSON | N-way presence / class / product grids, UpSet, ranking, PDF; pairwise under expander |
| Enriched markdown | Shared `total_lines` for pre-pass + R1 |

Fill only the slots you need. A section needs ≥ 2 uploads of that kind (reactions: baseline + ≥ 1 other). Missing kinds are skipped; the rest still run.

### Pre-pass / R1 JSON

Pre-pass: top-level array of groups with `sections[]`. R1: flat or double-encoded step lists mapped onto the same `Section` geometry (`start_line` / `end_line` are document-global). R1 `section_type` is inherited from pre-pass, so type agreement is less meaningful than boundary geometry.

### Compounds

Deterministic identifier/alias union-find across N models on persistent-store `compounds.json`. Views:

1. **Raw compound inspector** (sidebar) — match tier + per-model raw JSON
2. **Presence** editable grid — clusters × models + adjudicated Baseline
3. **Role** / **Identifier type** editable grids — categorical cells; synced delete
4. **UpSet** — compound overlap from cluster memberships
5. **Compare vs manual benchmark** — presence F1 + field accuracy, weighted rank, bar chart, PDF

Matching waterfall: InChIKey → SMILES → molecular formula → normalized name/alias. Baseline: majority presence + Claude tiebreak.

### Reactions

N-way clustering on enriched `reactions.json` via a **reaction-then-procedure** waterfall (same view stack as compounds). `non_synthetic` filtered.

1. **Raw reaction inspector** (sidebar)
2. **Presence** / **Reaction class** / **Product** editable grids (Product cell = `product_name`, else short SMILES)
3. Synced **Delete selected rows** across the three grids
4. **UpSet — reaction overlap**
5. **Compare vs manual benchmark** — presence F1 + class/product accuracy, weights, ranked table, bar chart, PDF + reaction-groups JSON
6. Collapsed **Pairwise diagnostics** expander — legacy 6-axis matcher (name / SMILES / reactants / procedure / yield / conditions) vs baseline

Matching waterfall: reaction-vector cosine ≥ 0.95, then procedure-vector ≥ 0.95 for leftovers (greedy cross-model edges, at most one reaction per model per cluster; defaults). Changing τ rebuilds presence/class/product grids and ranking from the new clusters (edits for that upload+τ combo are discarded; enrichment stays cached on upload hashes). Missing `reaction_vector`s are filled at compare time via existing `canonical_rxn` / SMILES → RDKit assemble → live **PubChem then OPSIN** name→SMILES (cached per run, Markush skipped) → **rxnfp** (pip package bundled `bert_ft` weights — install with `pip install --no-deps rxnfp`). Procedure embed text prefers `procedure_summary` then `procedure_text` (SciBERT).

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
    ├── scibert.py / procedure_vectors.py  # SciBERT + ensure procedure vectors
    ├── rxnfp_embed.py / reaction_vectors.py / smiles_resolve.py  # rxnfp + SMILES ladder
    ├── scoring.py / flagging.py / visuals.py
    ├── compound_parsing.py / compound_matching.py / compound_report.py
    ├── compound_baseline.py / compound_stats.py / compound_grid.py / compound_pdf.py
    ├── upset_viz.py / compound_colors.py
    ├── reaction_parsing.py / reaction_matching.py / reaction_nway.py
    ├── reaction_baseline.py / reaction_stats.py / reaction_grid.py
    ├── reaction_report.py / reaction_pdf.py
```

## Notes

- Blob fetch is optional — without `.env` / credentials, use manual uploads as before.
- If murmur3 hash-bucket disagrees with production, the client brute-forces all 128 buckets for that country code once per patent (cached in session).
- Pre-pass/R1 threshold slider re-runs only flagging; embeddings are cached.
- Reaction enrichment (PubChem/rxnfp/SciBERT) is cached on upload hashes; τ sliders re-cluster only.
- Scores weight each line equally; longer sections contribute more to averages.
