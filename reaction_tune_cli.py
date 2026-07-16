#!/usr/bin/env python3
"""Label-free threshold-sweep diagnostics for the tiered reaction matcher.

There is no gold alignment to optimize against, so this tool does *not* score
correctness. Instead it shows how the clustering **responds** to each threshold,
so you can pick a config by inspection and have it generalize across patents:

* cluster counts and multi-model agreement at each threshold,
* the tier that merged each cluster (provenance / compound Jaccard / combined),
* signal coverage (how many reactions even carry lines / SMILES / vectors),
* stability — where a knob has a flat plateau (robust) vs a cliff (fragile).

Usage::

    python reaction_tune_cli.py \
        --run Claude:CLAUDE/reactions.json \
        --run GPT:GPT/reactions.json \
        --r1 Claude:CLAUDE/r1.json \
        --r1 GPT:GPT/r1.json

By default names are NOT resolved over the network (fast, deterministic). Pass
``--online`` to resolve missing names via PubChem/OPSIN exactly as the app does.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.reaction_line_join import join_reactions_to_r1
from core.reaction_nway import diff_reactions_nway, prepare_reaction_entries
from core.reaction_parsing import ReactionEntry, parse_reactions_json
from core.reaction_scoring import (
    TIER_COMBINED,
    TIER_COMPOUND_JACCARD,
    TIER_ORDER,
    TIER_PROVENANCE,
    MatchConfig,
)
from core.r1_parsing import parse_r1_step_dicts


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL:path/to/file.json")
    label, path_str = value.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Label cannot be empty")
    path = Path(path_str)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return label, path


def _offline_resolve(_name: str) -> None:
    """Resolve function that never hits the network (existing SMILES only)."""
    return None


def _load_reactions(runs: list[tuple[str, Path]], *, online: bool) -> dict[str, list[ReactionEntry]]:
    entries_by_label: dict[str, list[ReactionEntry]] = {}
    for label, path in runs:
        entries_by_label[label] = parse_reactions_json(
            path.read_bytes(), source_label=f"{label} ({path.name})"
        )
    resolve_fn = None if online else _offline_resolve
    return prepare_reaction_entries(entries_by_label, resolve_fn=resolve_fn)


def _join_r1(
    reactions_by_label: dict[str, list[ReactionEntry]],
    r1_runs: list[tuple[str, Path]],
) -> dict[str, list[ReactionEntry]]:
    if not r1_runs:
        return reactions_by_label
    r1_steps_by_label: dict[str, list] = {}
    for label, path in r1_runs:
        try:
            r1_steps_by_label[label] = parse_r1_step_dicts(
                path.read_bytes(), source_label=f"{label} R1 ({path.name})"
            )
        except (ValueError, KeyError) as error:
            print(f"  ! R1 for {label} skipped: {error}", file=sys.stderr)
    return join_reactions_to_r1(reactions_by_label, r1_steps_by_label)


def _coverage(reactions_by_label: dict[str, list[ReactionEntry]]) -> None:
    print("=== SIGNAL COVERAGE (per model) ===")
    header = f"  {'model':16s} {'rxns':>5s} {'lines':>6s} {'prodSMILES':>11s} {'cmpdSet':>8s} {'procVec':>8s} {'rxnVec':>7s}"
    print(header)
    for label, entries in reactions_by_label.items():
        n = len(entries)
        if n == 0:
            print(f"  {label:16s} {0:5d}   (empty)")
            continue
        lines = sum(1 for e in entries if e.start_line is not None and e.end_line is not None)
        prod = sum(1 for e in entries if e.product_smiles)
        cset = sum(1 for e in entries if e.compound_smiles)
        pvec = sum(1 for e in entries if e.procedure_vector)
        rvec = sum(1 for e in entries if e.reaction_vector)

        def pct(x: int) -> str:
            return f"{x:3d}/{n:<3d}"

        print(
            f"  {label:16s} {n:5d} {pct(lines):>6s} {pct(prod):>11s} "
            f"{pct(cset):>8s} {pct(pvec):>8s} {pct(rvec):>7s}"
        )
    print(
        "  (columns: reactions, with R1 line span, with product SMILES, with "
        "non-empty compound set, with procedure vector, with reaction vector)\n"
    )


def _diagnostics(reactions_by_label, config: MatchConfig) -> dict:
    nway = diff_reactions_nway(reactions_by_label, skip_ensure=True, config=config)
    n_models = len(nway.labels)
    majority = ceil(n_models / 2) if n_models else 0
    multimodel = [c for c in nway.clusters if len(c.membership) > 1]
    tier_counts: Counter[str] = Counter(
        c.match_tier for c in multimodel if c.match_tier is not None
    )
    full_consensus = sum(1 for c in nway.clusters if len(c.membership) == n_models)
    majority_consensus = sum(1 for c in nway.clusters if len(c.membership) >= majority)
    sizes = [len(c.membership) for c in multimodel]
    return {
        "clusters": len(nway.clusters),
        "multimodel": len(multimodel),
        "singletons": len(nway.clusters) - len(multimodel),
        "full_consensus": full_consensus,
        "majority_consensus": majority_consensus,
        "mean_multimodel_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "tier_counts": tier_counts,
    }


def _print_summary(reactions_by_label, config: MatchConfig) -> None:
    diag = _diagnostics(reactions_by_label, config)
    print("=== BASELINE (current default config) ===")
    print(
        f"  clusters={diag['clusters']}  multi-model={diag['multimodel']}  "
        f"singletons={diag['singletons']}  full-consensus={diag['full_consensus']}  "
        f"majority-consensus={diag['majority_consensus']}  "
        f"mean multi-model size={diag['mean_multimodel_size']:.2f}"
    )
    tier_str = "  ".join(
        f"{tier}={diag['tier_counts'].get(tier, 0)}" for tier in TIER_ORDER
    )
    print(f"  merges by tier: {tier_str}\n")


def _sweep(reactions_by_label, base: MatchConfig, knob: str, values: list[float]) -> None:
    label = {
        "tau_provenance": "Provenance τ",
        "tau_jaccard": "Compound-Jaccard τ",
        "tau_combined": "Combined τ",
    }[knob]
    print(f"=== SWEEP: {label} (other knobs at default) ===")
    print(f"  {'τ':>5s} {'clusters':>9s} {'multi':>6s} {'full':>5s} {'maj':>4s}   merges-by-tier")
    prev = None
    for value in values:
        config = replace(base, **{knob: value})
        diag = _diagnostics(reactions_by_label, config)
        tiers = " ".join(
            f"{tier[:4]}={diag['tier_counts'].get(tier, 0)}" for tier in TIER_ORDER
        )
        delta = "" if prev is None else f"  (Δclusters {diag['clusters'] - prev:+d})"
        print(
            f"  {value:5.2f} {diag['clusters']:9d} {diag['multimodel']:6d} "
            f"{diag['full_consensus']:5d} {diag['majority_consensus']:4d}   {tiers}{delta}"
        )
        prev = diag["clusters"]
    print("  Flat stretches = robust thresholds; big Δ = a cliff to avoid sitting on.\n")


def _frange(start: float, stop: float, step: float) -> list[float]:
    out: list[float] = []
    value = start
    while value <= stop + 1e-9:
        out.append(round(value, 4))
        value += step
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_labeled_path,
        required=True,
        help="LABEL:path/to/reactions.json (repeat, minimum 2)",
    )
    parser.add_argument(
        "--r1",
        action="append",
        type=_parse_labeled_path,
        default=[],
        help="LABEL:path/to/r1.json (repeat) — enables the provenance tier",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Resolve missing names via PubChem/OPSIN (network; matches the app).",
    )
    args = parser.parse_args()

    if len(args.run) < 2:
        parser.error("Provide at least two --run arguments")
    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        parser.error("Each --run label must be unique")

    print(f"Loading {len(args.run)} models "
          f"({'online name resolution' if args.online else 'offline / existing SMILES only'})...\n")
    reactions_by_label = _load_reactions(args.run, online=args.online)
    reactions_by_label = _join_r1(reactions_by_label, args.r1)

    _coverage(reactions_by_label)

    base = MatchConfig()
    _print_summary(reactions_by_label, base)

    _sweep(reactions_by_label, base, "tau_provenance", _frange(0.0, 1.0, 0.1))
    _sweep(reactions_by_label, base, "tau_jaccard", _frange(0.5, 1.0, 0.05))
    _sweep(reactions_by_label, base, "tau_combined", _frange(0.4, 1.0, 0.1))

    print(
        "No gold labels → these are response curves, not accuracy. Pick thresholds on "
        "flat plateaus, then spot-check a few merged clusters in the Streamlit inspector."
    )


if __name__ == "__main__":
    main()
