"""Tests for compound-set Jaccard N-way reaction clustering."""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Sequence

from core.procedure_vectors import ensure_procedure_vectors, procedure_text_for_embed
from core.reaction_nway import (
    DEFAULT_COMPOUND_JACCARD_TAU,
    MATCH_WATERFALL,
    compound_smiles_set,
    diff_reactions_nway,
    interval_jaccard,
    prepare_reaction_entries,
    product_name_key,
    product_smiles_key,
    rxn_smiles_key,
    smiles_set_jaccard,
)
from core.reaction_parsing import ReactionEntry
from core.reaction_product_enrich import ensure_product_canonical_smiles
from core.reaction_report import build_reaction_groups_json
from core.reaction_scoring import MatchConfig
from core.reaction_vectors import ensure_reaction_vectors
from core.smiles_resolve import looks_like_markush, resolve_name_to_smiles


def _rxn(
    *,
    product_name: str | None = None,
    product_smiles: str | None = None,
    reactant_names: tuple[str, ...] = (),
    reactant_smiles: tuple[str, ...] = (),
    reaction_class: str | None = "substitution",
    non_synthetic: bool = False,
    section_label: str | None = "Ex 1",
    step_label: str | None = "Step 1",
    reaction_id: str | None = None,
    canonical_rxn: str | None = None,
    procedure_text: str | None = None,
    procedure_vector: tuple[float, ...] | None = None,
    reaction_vector: tuple[float, ...] | None = None,
    product_yield_pct: float | None = None,
    step_index: int | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    line_join: str | None = None,
    compound_smiles: frozenset[str] | None = None,
    raw: dict | None = None,
) -> ReactionEntry:
    return ReactionEntry(
        product_name=product_name,
        product_smiles=product_smiles,
        reactant_names=reactant_names,
        reactant_smiles=reactant_smiles,
        product_yield_pct=product_yield_pct,
        procedure_text=procedure_text,
        procedure_vector=procedure_vector,
        temperature_c=None,
        room_temperature=None,
        time_h=None,
        atmosphere=None,
        reaction_class=reaction_class,
        non_synthetic=non_synthetic,
        section_label=section_label,
        step_label=step_label,
        reaction_id=reaction_id,
        canonical_rxn=canonical_rxn,
        reaction_vector=reaction_vector,
        step_index=step_index,
        start_line=start_line,
        end_line=end_line,
        line_join=line_join,
        compound_smiles=(
            frozenset() if compound_smiles is None else frozenset(compound_smiles)
        ),
        raw=raw or {},
    )


def _lined(
    *,
    start_line: int,
    end_line: int,
    reaction_vector: tuple[float, ...] | None = None,
    product_name: str | None = None,
    product_smiles: str | None = None,
    section_label: str | None = "Ex 1",
    step_label: str | None = "Step 1",
    line_join: str | None = "exact_index",
    compound_smiles: frozenset[str] | None = None,
    raw: dict | None = None,
    **kwargs,
) -> ReactionEntry:
    return _rxn(
        product_name=product_name,
        product_smiles=product_smiles,
        section_label=section_label,
        step_label=step_label,
        reaction_vector=reaction_vector,
        start_line=start_line,
        end_line=end_line,
        line_join=line_join,
        compound_smiles=compound_smiles,
        raw=raw,
        **kwargs,
    )


def _fake_procedure_embed(texts: Sequence[str]) -> list[tuple[float, ...]]:
    known = {
        "cyanation procedure": (1.0, 0.0, 0.0),
        "methoxylation procedure": (0.0, 1.0, 0.0),
        "hydrolysis procedure": (0.0, 0.0, 1.0),
    }
    out: list[tuple[float, ...]] = []
    for text in texts:
        if text in known:
            out.append(known[text])
        else:
            h = abs(hash(text)) % 1000
            out.append((float(h), float(1000 - h), 0.0))
    return out


def _fake_reaction_embed(smiles_list: Sequence[str]) -> list[tuple[float, ...] | None]:
    """Map known reaction SMILES to orthogonal / identical unit vectors."""
    known = {
        "CCO>>CC": (1.0, 0.0, 0.0),
        "CC>>CCO": (0.0, 1.0, 0.0),
        "c1ccccc1.CCO>>CCO": (0.0, 0.0, 1.0),
    }
    out: list[tuple[float, ...] | None] = []
    for smiles in smiles_list:
        if smiles in known:
            out.append(known[smiles])
        else:
            h = abs(hash(smiles)) % 1000
            out.append((float(h), 0.0, float(1000 - h)))
    return out


def _fake_resolve(name: str) -> str | None:
    table = {
        "ethanol": "CCO",
        "acetic acid": "CC(=O)O",
        "benzene": "c1ccccc1",
        "ethane": "CC",
        "2,5-dichlorobenzonitrile": "N#Cc1cc(Cl)ccc1Cl",
        "methanol": "CO",
        "water": "O",
        "cucl": "Cl[Cu]",
    }
    return table.get(name.strip().lower())


class ReactionNwayKeysTest(unittest.TestCase):
    def test_rxn_smiles_key_orders_reactants(self) -> None:
        entry = _rxn(
            product_smiles="CCO",
            reactant_smiles=("c1ccccc1", "CCO"),
        )
        key = rxn_smiles_key(entry)
        self.assertIsNotNone(key)
        assert key is not None
        self.assertTrue(key.startswith("CCO<<"))
        self.assertIn("|", key)

    def test_skip_tiers_when_fields_missing(self) -> None:
        name_only = _rxn(product_name="tert-Butanol")
        self.assertIsNone(rxn_smiles_key(name_only))
        self.assertIsNone(product_smiles_key(name_only))
        self.assertEqual(product_name_key(name_only), "tert-butanol")


class IntervalJaccardTest(unittest.TestCase):
    def test_full_overlap(self) -> None:
        self.assertEqual(interval_jaccard(10, 20, 10, 20), 1.0)

    def test_partial_overlap(self) -> None:
        # [1,10] ∩ [6,15] = 5; union = 15 → 1/3
        self.assertAlmostEqual(interval_jaccard(1, 10, 6, 15), 5 / 15)

    def test_no_overlap(self) -> None:
        self.assertEqual(interval_jaccard(1, 5, 10, 20), 0.0)


class SmilesSetJaccardTest(unittest.TestCase):
    def test_identical_sets(self) -> None:
        a = frozenset({"CCO", "CC"})
        self.assertEqual(smiles_set_jaccard(a, a), 1.0)

    def test_empty_is_zero(self) -> None:
        self.assertEqual(smiles_set_jaccard(frozenset(), frozenset({"CCO"})), 0.0)
        self.assertEqual(smiles_set_jaccard(frozenset({"CCO"}), frozenset()), 0.0)

    def test_partial(self) -> None:
        left = frozenset({"CCO", "CC", "c1ccccc1"})
        right = frozenset({"CCO", "CC", "CO"})
        # |∩|=2, |∪|=4 → 0.5
        self.assertAlmostEqual(smiles_set_jaccard(left, right), 0.5)


class SmilesResolveTest(unittest.TestCase):
    def test_markush_skipped(self) -> None:
        self.assertTrue(looks_like_markush("Hal"))
        self.assertTrue(looks_like_markush("compound of formula (I)"))
        self.assertTrue(looks_like_markush("formula II"))
        self.assertFalse(looks_like_markush("ethanol"))

    def test_resolve_uses_pubchem_then_opsin_injectable(self) -> None:
        calls: list[str] = []

        def pubchem(name: str) -> str | None:
            calls.append(f"pubchem:{name}")
            return None

        def opsin(name: str) -> str | None:
            calls.append(f"opsin:{name}")
            return "CCO"

        cache: dict[str, str | None] = {}
        result = resolve_name_to_smiles(
            "ethanol",
            cache=cache,
            pubchem_fn=pubchem,
            opsin_fn=opsin,
        )
        self.assertEqual(result, "CCO")
        self.assertEqual(calls, ["pubchem:ethanol", "opsin:ethanol"])
        self.assertEqual(cache["ethanol"], "CCO")
        calls.clear()
        self.assertEqual(
            resolve_name_to_smiles(
                "ethanol", cache=cache, pubchem_fn=pubchem, opsin_fn=opsin
            ),
            "CCO",
        )
        self.assertEqual(calls, [])


class EnsureProductCanonicalSmilesTest(unittest.TestCase):
    def test_canonicalizes_existing_product_smiles(self) -> None:
        entry = _rxn(product_smiles="C(O)C")  # ethanol, different atom order
        filled = ensure_product_canonical_smiles([entry], resolve_fn=_fake_resolve)
        self.assertEqual(filled[0].product_smiles, "CCO")
        self.assertIn("CCO", filled[0].compound_smiles)

    def test_resolves_product_name(self) -> None:
        entry = _rxn(product_name="ethanol")
        filled = ensure_product_canonical_smiles([entry], resolve_fn=_fake_resolve)
        self.assertEqual(filled[0].product_smiles, "CCO")
        self.assertEqual(filled[0].compound_smiles, frozenset({"CCO"}))

    def test_falls_back_to_is_product_compound(self) -> None:
        entry = _rxn(
            raw={
                "compounds": [
                    {"identifier": "CuCN", "is_product": False},
                    {
                        "identifier": "2,5-dichlorobenzonitrile",
                        "is_product": True,
                    },
                ]
            }
        )
        filled = ensure_product_canonical_smiles([entry], resolve_fn=_fake_resolve)
        self.assertEqual(filled[0].product_smiles, "N#Cc1cc(Cl)ccc1Cl")

    def test_compound_smiles_preferred_over_identifier(self) -> None:
        entry = _rxn(
            raw={
                "compounds": [
                    {
                        "identifier": "ethanol",
                        "smiles": "C(O)C",
                        "is_product": True,
                    }
                ]
            }
        )
        filled = ensure_product_canonical_smiles([entry], resolve_fn=_fake_resolve)
        self.assertEqual(filled[0].product_smiles, "CCO")

    def test_role_filtered_set_includes_reactants_excludes_solvent(self) -> None:
        entry = _rxn(
            product_name="ethanol",
            reactant_names=("ethane",),
            raw={
                "compounds": [
                    {"identifier": "ethanol", "role": "product", "is_product": True},
                    {"identifier": "ethane", "role": "reactant"},
                    {"identifier": "CuCl", "role": "catalyst"},
                    {"identifier": "methanol", "role": "solvent"},
                    {"identifier": "water", "role": "drying_agent"},
                ]
            },
        )
        filled = ensure_product_canonical_smiles([entry], resolve_fn=_fake_resolve)
        smiles = filled[0].compound_smiles
        self.assertIn("CCO", smiles)
        self.assertIn("CC", smiles)
        # catalyst included
        self.assertTrue(any("Cu" in s or "Cl" in s for s in smiles))
        # solvent / drying_agent excluded
        self.assertNotIn("CO", smiles)
        self.assertNotIn("O", smiles)

    def test_markush_name_left_unresolved(self) -> None:
        entry = _rxn(product_name="compound of formula (I)")
        filled = ensure_product_canonical_smiles(
            [entry],
            resolve_fn=_fake_resolve,
            cache={},
        )
        self.assertIsNone(filled[0].product_smiles)
        self.assertEqual(filled[0].compound_smiles, frozenset())

    def test_shared_cache_across_entries(self) -> None:
        calls: list[str] = []

        def resolve(name: str) -> str | None:
            calls.append(name)
            return _fake_resolve(name)

        cache: dict[str, str | None] = {}
        entries = [
            _rxn(product_name="ethanol", section_label="Ex 1"),
            _rxn(product_name="ethanol", section_label="Ex 2"),
        ]
        ensure_product_canonical_smiles(entries, resolve_fn=resolve, cache=cache)
        self.assertEqual(calls, ["ethanol"])
        self.assertEqual(cache["ethanol"], "CCO")


class EnsureProcedureVectorsTest(unittest.TestCase):
    def test_prefers_summary_over_text(self) -> None:
        entry = _rxn(
            procedure_text="long verbose procedure body",
            raw={"procedure_summary": "cyanation procedure"},
        )
        self.assertEqual(procedure_text_for_embed(entry), "cyanation procedure")
        filled = ensure_procedure_vectors([entry], embed_fn=_fake_procedure_embed)
        self.assertEqual(filled[0].procedure_vector, (1.0, 0.0, 0.0))

    def test_falls_back_to_procedure_text(self) -> None:
        entry = _rxn(procedure_text="cyanation procedure")
        self.assertEqual(procedure_text_for_embed(entry), "cyanation procedure")

    def test_prefers_existing_vector(self) -> None:
        entry = _rxn(
            procedure_text="cyanation procedure",
            procedure_vector=(0.5, 0.5, 0.0),
        )
        filled = ensure_procedure_vectors([entry], embed_fn=_fake_procedure_embed)
        self.assertEqual(filled[0].procedure_vector, (0.5, 0.5, 0.0))


class EnsureReactionVectorsTest(unittest.TestCase):
    def test_existing_vector_untouched(self) -> None:
        entry = _rxn(reaction_vector=(0.1, 0.2, 0.3), canonical_rxn="CC>>CCO")
        filled = ensure_reaction_vectors(
            [entry], embed_fn=_fake_reaction_embed, resolve_fn=_fake_resolve
        )
        self.assertEqual(filled[0].reaction_vector, (0.1, 0.2, 0.3))

    def test_names_only_mocked_resolve_builds_rxn(self) -> None:
        entry = _rxn(
            product_name="ethanol",
            reactant_names=("ethane",),
        )
        filled = ensure_reaction_vectors(
            [entry], embed_fn=_fake_reaction_embed, resolve_fn=_fake_resolve
        )
        self.assertEqual(filled[0].canonical_rxn, "CC>>CCO")
        self.assertEqual(filled[0].reaction_vector, (0.0, 1.0, 0.0))
        self.assertEqual(filled[0].product_smiles, "CCO")
        self.assertEqual(filled[0].reactant_smiles, ("CC",))


class DiffReactionsNwayTest(unittest.TestCase):
    def test_same_sets_jaccard_one_merge(self) -> None:
        rxn_a = _rxn(product_smiles="CCO", product_name="A")
        rxn_b = _rxn(product_smiles="C(O)C", product_name="B")  # same ethanol
        result = diff_reactions_nway(
            {"Claude": [rxn_a], "GPT": [rxn_b]},
            skip_ensure=False,
            resolve_fn=_fake_resolve,
        )
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].membership, frozenset({"Claude", "GPT"}))
        # Both products canonicalize to CCO; with no reactants the compound
        # set is just {CCO} on both sides → compound-Jaccard tier wins.
        self.assertEqual(result.clusters[0].match_tier, "compound_jaccard")

    def test_same_resolved_name_merge(self) -> None:
        left = _rxn(product_name="ethanol")
        right = _rxn(product_name="ethanol", section_label="Ex 9")
        result = diff_reactions_nway(
            {"Claude": [left], "GPT": [right]},
            resolve_fn=_fake_resolve,
        )
        self.assertEqual(len(result.clusters), 1)
        # Names resolve to the same product SMILES → same compound set → Jaccard tier.
        self.assertEqual(result.clusters[0].match_tier, "compound_jaccard")
        for entry in result.clusters[0].representatives.values():
            self.assertEqual(entry.product_smiles, "CCO")
            self.assertEqual(compound_smiles_set(entry), frozenset({"CCO"}))

    def test_tau_gates_partial_overlap(self) -> None:
        # |∩|=4, |∪|=6 → Jaccard ≈ 0.667 — merges at 0.60, splits at 0.85
        left = _rxn(
            compound_smiles=frozenset({"CCO", "CC", "c1ccccc1", "CC(=O)O", "Cl[Cu]"})
        )
        right = _rxn(
            compound_smiles=frozenset({"CCO", "CC", "c1ccccc1", "CC(=O)O", "CO"})
        )
        self.assertAlmostEqual(
            smiles_set_jaccard(
                compound_smiles_set(left), compound_smiles_set(right)
            ),
            4 / 6,
        )
        merge = diff_reactions_nway(
            {"Claude": [left], "GPT": [right]},
            skip_ensure=True,
            tau_jaccard=0.60,
        )
        self.assertEqual(len(merge.clusters), 1)
        split = diff_reactions_nway(
            {"Claude": [left], "GPT": [right]},
            skip_ensure=True,
            tau_jaccard=0.85,
        )
        self.assertEqual(len(split.clusters), 2)

    def test_solvent_only_difference_still_merges(self) -> None:
        """Solvent excluded from sets → cores match → merge at default τ."""
        left = _rxn(
            product_name="ethanol",
            reactant_names=("ethane",),
            raw={
                "compounds": [
                    {"identifier": "ethanol", "role": "product", "is_product": True},
                    {"identifier": "ethane", "role": "reactant"},
                    {"identifier": "methanol", "role": "solvent"},
                ]
            },
        )
        right = _rxn(
            product_name="ethanol",
            reactant_names=("ethane",),
            section_label="Ex 2",
            raw={
                "compounds": [
                    {"identifier": "ethanol", "role": "product", "is_product": True},
                    {"identifier": "ethane", "role": "reactant"},
                    {"identifier": "water", "role": "solvent"},
                ]
            },
        )
        result = diff_reactions_nway(
            {"Claude": [left], "GPT": [right]},
            resolve_fn=_fake_resolve,
            tau_jaccard=DEFAULT_COMPOUND_JACCARD_TAU,
        )
        self.assertEqual(len(result.clusters), 1)
        # Same product (ethanol→CCO) and same reactant (ethane); solvent excluded
        # from the set on both sides → identical compound sets → Jaccard tier.
        self.assertEqual(result.clusters[0].match_tier, "compound_jaccard")

    def test_different_smiles_no_merge(self) -> None:
        left = _rxn(
            product_smiles="CCO",
            compound_smiles=frozenset({"CCO"}),
        )
        right = _rxn(
            product_smiles="c1ccccc1",
            compound_smiles=frozenset({"c1ccccc1"}),
        )
        result = diff_reactions_nway(
            {"Claude": [left], "GPT": [right]},
            skip_ensure=True,
        )
        self.assertEqual(len(result.clusters), 2)

    def test_lines_merge_via_provenance_without_sets(self) -> None:
        """Same R1 line span merges via provenance even with empty compound sets.

        This is the naming-independent path: no product SMILES, no compound set,
        differing product names — provenance alone groups them.
        """
        left = _lined(start_line=10, end_line=20, product_name="A")
        right = _lined(start_line=10, end_line=20, product_name="B")
        result = diff_reactions_nway(
            {"Claude": [left], "GPT": [right]},
            skip_ensure=True,
        )
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].match_tier, "provenance")

    def test_provenance_disabled_keeps_singletons(self) -> None:
        """With provenance off and no chemistry signal, same lines stay split."""
        left = _lined(start_line=10, end_line=20, product_name="A")
        right = _lined(start_line=10, end_line=20, product_name="B")
        result = diff_reactions_nway(
            {"Claude": [left], "GPT": [right]},
            config=MatchConfig(enable_provenance=False),
            skip_ensure=True,
        )
        self.assertEqual(len(result.clusters), 2)

    def test_markush_names_merge_via_provenance(self) -> None:
        """Generic / Markush names with matching R1 lines merge via provenance.

        The trihalobenzene case: the generic name never resolves to a SMILES, so
        compound sets are empty and Jaccard cannot match — but the shared source
        line span does.
        """
        left = _lined(
            start_line=10,
            end_line=20,
            product_name="compound of formula (I)",
            section_label="Ex 1",
        )
        right = _lined(
            start_line=10,
            end_line=20,
            product_name="compound of formula (I)",
            section_label="Ex 1",
        )
        result = diff_reactions_nway({"Claude": [left], "GPT": [right]})
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].match_tier, "provenance")

    def test_empty_sets_remain_singletons(self) -> None:
        result = diff_reactions_nway(
            {
                "Claude": [_rxn(product_name="unknown-xyz-not-in-table")],
                "GPT": [_rxn(product_name="also-unknown-abc")],
            },
            resolve_fn=_fake_resolve,
        )
        self.assertEqual(len(result.clusters), 2)

    def test_one_per_model_for_same_set(self) -> None:
        set_cco = frozenset({"CCO"})
        result = diff_reactions_nway(
            {
                "Claude": [
                    _rxn(product_smiles="CCO", product_name="a", compound_smiles=set_cco),
                    _rxn(
                        product_smiles="CCO",
                        product_name="b",
                        section_label="Ex 2",
                        compound_smiles=set_cco,
                    ),
                ],
                "GPT": [
                    _rxn(
                        product_smiles="CCO",
                        product_name="c",
                        compound_smiles=set_cco,
                    ),
                ],
            },
            skip_ensure=True,
        )
        self.assertEqual(result.raw_counts["Claude"], 2)
        claude_clusters = [c for c in result.clusters if "Claude" in c.membership]
        self.assertEqual(len(claude_clusters), 2)
        three_way = [
            c for c in result.clusters if c.membership == frozenset({"Claude", "GPT"})
        ]
        self.assertEqual(len(three_way), 1)
        # Identical compound sets ({CCO}) → compound-Jaccard tier.
        self.assertEqual(three_way[0].match_tier, "compound_jaccard")

    def test_filters_non_synthetic(self) -> None:
        set_cco = frozenset({"CCO"})
        keep = _rxn(
            product_smiles="CCO",
            product_name="product-a",
            non_synthetic=False,
            compound_smiles=set_cco,
        )
        drop = _rxn(
            product_smiles="CCO",
            product_name="product-a",
            non_synthetic=True,
            section_label="Ex junk",
            compound_smiles=set_cco,
        )
        result = diff_reactions_nway(
            {
                "Claude": [keep, drop],
                "GPT": [
                    _rxn(
                        product_smiles="CCO",
                        product_name="product-a",
                        section_label="Ex 9",
                        compound_smiles=set_cco,
                    )
                ],
            },
            skip_ensure=True,
        )
        self.assertEqual(result.raw_counts["Claude"], 1)
        self.assertEqual(len(result.clusters), 1)
        self.assertEqual(result.clusters[0].membership, frozenset({"Claude", "GPT"}))
        # Identical compound sets ({CCO}) → compound-Jaccard tier.
        self.assertEqual(result.clusters[0].match_tier, "compound_jaccard")

    def test_three_way_same_set(self) -> None:
        result = diff_reactions_nway(
            {
                "Claude": [_rxn(compound_smiles=frozenset({"CCO"}))],
                "GPT": [_rxn(compound_smiles=frozenset({"CCO"}))],
                "GLM": [
                    _rxn(
                        compound_smiles=frozenset({"c1ccccc1"}),
                        section_label="Ex Z",
                    )
                ],
            },
            skip_ensure=True,
        )
        memberships = {c.membership for c in result.clusters}
        self.assertIn(frozenset({"Claude", "GPT"}), memberships)
        self.assertIn(frozenset({"GLM"}), memberships)

    def test_match_waterfall(self) -> None:
        self.assertEqual(
            MATCH_WATERFALL,
            ("provenance", "compound_jaccard", "combined"),
        )
        self.assertEqual(DEFAULT_COMPOUND_JACCARD_TAU, 0.85)

    def test_single_linkage_transitively_merges_via_chained_edges(self) -> None:
        """A-B and B-C each qualify, A-C does not — single-linkage still merges all three.

        This is a known, accepted trade-off, not an oversight. A stricter
        complete-linkage variant (require a new member to qualify against
        EVERY existing group member, not just one) was implemented and
        measured against a real 6-model patent export: it fragmented many
        genuinely-identical multi-model reactions into singletons (real
        per-model extractions are heterogeneous enough — different SMILES
        resolved, different line-boundary drift — that requiring universal
        pairwise agreement rejects far more correct merges than incorrect
        ones) while still failing to prevent the specific false merge it
        targeted. It cost far more recall than it bought in precision, so
        single-linkage (any one qualifying edge merges two components) is
        what ships.
        """
        config = MatchConfig(
            tau_provenance=0.15,
            enable_compound_jaccard=False,
            enable_combined=False,
        )
        a = _lined(start_line=0, end_line=10, product_name="A")  # len 11
        b = _lined(start_line=5, end_line=15, product_name="B")  # len 11
        c = _lined(start_line=12, end_line=25, product_name="C")  # len 14

        # A-B: overlap [5,10]=6, union=16 -> J=0.375 (qualifies, stronger)
        self.assertAlmostEqual(interval_jaccard(0, 10, 5, 15), 6 / 16)
        # B-C: overlap [12,15]=4, union=21 -> J=0.190 (qualifies, weaker)
        self.assertAlmostEqual(interval_jaccard(5, 15, 12, 25), 4 / 21)
        # A-C: no overlap at all (10 < 12) -> 0.0, never qualifies directly
        self.assertEqual(interval_jaccard(0, 10, 12, 25), 0.0)

        result = diff_reactions_nway(
            {"M1": [a], "M2": [b], "M3": [c]},
            skip_ensure=True,
            config=config,
        )
        memberships = {cluster.membership for cluster in result.clusters}
        # A and C end up in the same cluster via the A-B / B-C chain, despite
        # never qualifying directly against each other.
        self.assertIn(frozenset({"M1", "M2", "M3"}), memberships)

    def test_prepare_reaction_entries_enriches(self) -> None:
        keep = _rxn(
            product_name="ethanol",
            reactant_names=("ethane",),
            procedure_text="cyanation procedure",
        )
        drop = _rxn(
            product_name="junk",
            non_synthetic=True,
            procedure_text="ignored",
        )
        prepared = prepare_reaction_entries(
            {"Claude": [keep, drop]},
            resolve_fn=_fake_resolve,
        )
        self.assertEqual(len(prepared["Claude"]), 1)
        entry = prepared["Claude"][0]
        self.assertEqual(entry.product_smiles, "CCO")
        self.assertEqual(entry.compound_smiles, frozenset({"CCO", "CC"}))
        self.assertIsNone(entry.canonical_rxn)
        self.assertIsNone(entry.reaction_vector)
        self.assertIsNone(entry.procedure_vector)


class ReactionGroupsJsonTest(unittest.TestCase):
    def test_build_reaction_groups_json_includes_smiles_and_lines(self) -> None:
        with_raw = replace(
            _lined(
                start_line=10,
                end_line=25,
                product_name="ethanol",
                product_smiles="CCO",
                step_index=0,
                line_join="exact_index",
                compound_smiles=frozenset({"CCO"}),
            ),
            raw={"product_name": "ethanol", "custom_field": 42},
        )
        without_raw = _lined(
            start_line=12,
            end_line=30,
            product_name="ethanol",
            product_smiles="CCO",
            step_index=0,
            line_join="exact_label",
            compound_smiles=frozenset({"CCO"}),
        )
        result = diff_reactions_nway(
            {"Claude": [with_raw], "GPT": [without_raw]},
            skip_ensure=True,
        )
        payload = build_reaction_groups_json(
            result,
            patent_id="WO123",
            baseline_label="Claude",
        )
        self.assertEqual(payload["patent_id"], "WO123")
        self.assertEqual(
            payload["match_waterfall"],
            ["provenance", "compound_jaccard", "combined"],
        )
        self.assertEqual(payload["cluster_count"], 1)
        cluster = payload["clusters"][0]
        # Line spans 10-25 / 12-30 overlap ≥ τ_provenance (0.50) → provenance tier.
        self.assertEqual(cluster["match_tier"], "provenance")
        self.assertEqual(cluster["product_smiles"], "CCO")
        self.assertEqual(cluster["compound_smiles"], ["CCO"])
        self.assertNotIn("product_smiles_conflict", cluster)
        self.assertEqual(cluster["line_span"], {"start_line": 10, "end_line": 30})
        self.assertEqual(cluster["models"]["Claude"]["custom_field"], 42)
        self.assertEqual(cluster["models"]["Claude"]["product_smiles"], "CCO")
        self.assertEqual(cluster["models"]["Claude"]["compound_smiles"], ["CCO"])
        self.assertEqual(cluster["models"]["Claude"]["start_line"], 10)
        self.assertEqual(cluster["models"]["Claude"]["end_line"], 25)
        self.assertEqual(cluster["models"]["Claude"]["line_join"], "exact_index")
        self.assertEqual(cluster["models"]["GPT"]["product_name"], "ethanol")
        self.assertEqual(cluster["models"]["GPT"]["start_line"], 12)
        self.assertEqual(cluster["models"]["GPT"]["end_line"], 30)
        self.assertNotIn("custom_field", cluster["models"]["GPT"])

    def test_different_products_two_clusters_no_conflict_flag(self) -> None:
        left = _rxn(
            product_smiles="CCO",
            compound_smiles=frozenset({"CCO"}),
        )
        right = _rxn(
            product_smiles="c1ccccc1",
            compound_smiles=frozenset({"c1ccccc1"}),
        )
        result = diff_reactions_nway(
            {"Claude": [left], "GPT": [right]},
            skip_ensure=True,
        )
        payload = build_reaction_groups_json(
            result, patent_id="WO1", baseline_label="Claude"
        )
        self.assertEqual(len(payload["clusters"]), 2)
        for cluster in payload["clusters"]:
            self.assertNotIn("product_smiles_conflict", cluster)
            self.assertIn("compound_smiles", cluster)

    def test_singleton_exports_compound_smiles(self) -> None:
        only = _rxn(
            product_smiles="CCO",
            compound_smiles=frozenset({"CCO"}),
        )
        result = diff_reactions_nway({"Claude": [only], "GPT": []}, skip_ensure=True)
        payload = build_reaction_groups_json(
            result, patent_id="WO1", baseline_label="Claude"
        )
        self.assertEqual(len(payload["clusters"]), 1)
        self.assertEqual(payload["clusters"][0]["product_smiles"], "CCO")
        self.assertEqual(payload["clusters"][0]["compound_smiles"], ["CCO"])
        self.assertNotIn("product_smiles_conflict", payload["clusters"][0])


class ReactionParseExtendTest(unittest.TestCase):
    def test_parse_reaction_id_canonical_rxn_and_vector(self) -> None:
        from core.reaction_parsing import parse_reactions_json

        raw = """[
          {
            "reaction_id": "rxn-42",
            "canonical_rxn": "CC>>CCO",
            "reaction_vector": [0.1, 0.2, 0.3],
            "product_name": "ethanol",
            "product_smiles": "CCO",
            "reactant_names": ["acetic acid"],
            "reactant_smiles": ["CC(=O)O"],
            "section_label": "Ex 1",
            "step_label": "Step 1",
            "step_index": 2
          }
        ]"""
        entries = parse_reactions_json(raw)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].reaction_id, "rxn-42")
        self.assertEqual(entries[0].canonical_rxn, "CC>>CCO")
        self.assertEqual(entries[0].reaction_vector, (0.1, 0.2, 0.3))
        self.assertEqual(entries[0].step_index, 2)
        self.assertIsNone(entries[0].start_line)
        self.assertIsNone(entries[0].end_line)
        self.assertEqual(entries[0].compound_smiles, frozenset())
        self.assertIn("reaction_id", entries[0].raw)


if __name__ == "__main__":
    unittest.main()
