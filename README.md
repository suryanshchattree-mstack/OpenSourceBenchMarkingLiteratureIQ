# Multi-model Benchmark Streamlit App

Single-screen tool to compare N labeled model runs across five optional output kinds (pre-pass, M1, M2, R1 step boundaries, reactions). Upload a matrix of files, pick one shared baseline, click **Run Benchmarks** — each section runs independently and skips with a notice when it lacks enough uploads.

## Requirements

- Python 3.10+
- macOS Apple Silicon (MPS) recommended — falls back to CPU if MPS unavailable
- ~4 GB free RAM for loading three embedding models (used by pre-pass and R1)
- `rdkit` (reactions SMILES axis) and `upsetplot` (M1/M2 overlap plots)

## Setup

```bash
cd /Users/suryansh.c/Desktop/MstackRepos/prepass-benchmark

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

First pre-pass/R1 run downloads three Hugging Face models (~3 GB total, cached locally afterward):

- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-large-v2`
- `all-mpnet-base-v2`

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501 — one page, no sidebar multi-page nav.

### Upload matrix

| Slot | Used by |
|------|---------|
| Pre-pass JSON | Pre-pass section timeline / scores |
| M1 JSON | Molecule pass 1 N-way match + field agreement |
| M2 JSON | Molecule pass 2 N-way match + UpSet |
| R1 JSON | Step-boundary comparison (same visuals as pre-pass) |
| Reactions JSON | Pairwise reaction panels vs baseline |
| Enriched markdown | Shared `total_lines` for pre-pass + R1 |

Fill only the slots you need. A section needs ≥ 2 uploads of that kind (reactions: baseline + ≥ 1 other). Missing kinds are skipped; the rest still run.

### Pre-pass / R1 JSON

Pre-pass: top-level array of groups with `sections[]`. R1: flat or double-encoded step lists mapped onto the same `Section` geometry (`start_line` / `end_line` are document-global). R1 `section_type` is inherited from pre-pass, so type agreement is less meaningful than boundary geometry.

### M1 / M2

Deterministic identifier/alias union-find across N models, UpSet overlap plot, and pairwise recall/precision vs baseline. M1 adds field-agreement metrics (identifier_type, role, aliases, quantity presence).

### Reactions

One pairwise panel per non-baseline model (name / SMILES / reactants / procedure cosine / yield / conditions). `non_synthetic` records are filtered before scoring.

## Headless CLIs (still available)

```bash
# Pre-pass
python compare_cli.py ...

# M2 2-way
python compound_compare_cli.py \
  --claude Claude:path/to/claude-m2.json \
  --benchmark DeepSeek:path/to/deepseek-m2.json
```

## Project structure

```
prepass-benchmark/
├── app.py                      # Unified Streamlit screen
├── compound_compare_cli.py     # M2 diff CLI
├── compare_cli.py              # Pre-pass CLI
├── requirements.txt
├── README.md
└── core/
    ├── parsing.py              # Pre-pass JSON + markdown
    ├── r1_parsing.py           # R1 → Section adapter
    ├── line_arrays.py          # per-line type/label painting
    ├── embeddings.py           # 3 models, MPS, cosine similarity
    ├── scoring.py / flagging.py / visuals.py
    ├── m1_parsing.py / m1_agreement.py
    ├── compound_parsing.py / compound_matching.py / compound_report.py
    ├── upset_viz.py
    └── reaction_parsing.py / reaction_matching.py / reaction_report.py
```

## Notes

- Fully standalone — no connection to `literatureiq-engine`, Azure, or any database.
- Pre-pass/R1 threshold slider re-runs only flagging; embeddings are cached.
- Scores weight each line equally; longer sections contribute more to averages.
