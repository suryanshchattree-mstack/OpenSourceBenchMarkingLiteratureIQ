"""Tests for the tiered multi-signal reaction pair scorer."""

from __future__ import annotations

import unittest

from core.reaction_parsing import ReactionEntry
from core.reaction_scoring import (
    TIER_COMBINED,
    TIER_COMPOUND_JACCARD,
    TIER_PROVENANCE,
    MatchConfig,
    combined_score,
    cosine_similarity,
    interval_jaccard,
    pair_match,
    product_name_similarity,
    provenance_overlap,
)


def _rxn(
    *,
    product_name: str | None = None,
    product_smiles: str | None = None,
    reactant_names: tuple[str, ...] = (),
    reactant_smiles: tuple[str, ...] = (),
    procedure_vector: tuple[float, ...] | None = None,
    reaction_vector: tuple[float, ...] | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    compound_smiles: frozenset[str] | None = None,
) -> ReactionEntry:
    return ReactionEntry(
        product_name=product_name,
        product_smiles=product_smiles,
        reactant_names=reactant_names,
        reactant_smiles=reactant_smiles,
        product_yield_pct=None,
        procedure_text=None,
        procedure_vector=procedure_vector,
        temperature_c=None,
        room_temperature=None,
        time_h=None,
        atmosphere=None,
        reaction_class=None,
        non_synthetic=False,
        section_label=None,
        step_label=None,
        reaction_id=None,
        canonical_rxn=None,
        reaction_vector=reaction_vector,
        step_index=None,
        start_line=start_line,
        end_line=end_line,
        line_join=None,
        compound_smiles=frozenset() if compound_smiles is None else frozenset(compound_smiles),
        raw={},
    )


class CosineTest(unittest.TestCase):
    def test_identical_vectors_is_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)), 1.0)

    def test_orthogonal_is_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (0.0, 1.0)), 0.0)

    def test_missing_or_mismatched_is_none(self) -> None:
        self.assertIsNone(cosine_similarity(None, (1.0,)))
        self.assertIsNone(cosine_similarity((1.0,), None))
        self.assertIsNone(cosine_similarity((), (1.0,)))
        self.assertIsNone(cosine_similarity((1.0, 2.0), (1.0,)))

    def test_zero_norm_is_none(self) -> None:
        self.assertIsNone(cosine_similarity((0.0, 0.0), (1.0, 1.0)))


class ProvenanceTest(unittest.TestCase):
    def test_none_when_any_span_missing(self) -> None:
        self.assertIsNone(provenance_overlap(_rxn(start_line=1), _rxn(start_line=1, end_line=5)))
        self.assertIsNone(provenance_overlap(_rxn(), _rxn()))

    def test_full_overlap(self) -> None:
        a = _rxn(start_line=10, end_line=20)
        b = _rxn(start_line=10, end_line=20)
        self.assertEqual(provenance_overlap(a, b), 1.0)

    def test_partial_overlap(self) -> None:
        a = _rxn(start_line=1, end_line=10)
        b = _rxn(start_line=6, end_line=15)
        self.assertAlmostEqual(provenance_overlap(a, b), 5 / 15)
        self.assertAlmostEqual(interval_jaccard(1, 10, 6, 15), 5 / 15)


class ProductNameSimilarityTest(unittest.TestCase):
    def test_name_similarity(self) -> None:
        self.assertEqual(
            product_name_similarity(_rxn(product_name="Ethanol"), _rxn(product_name="ethanol")),
            1.0,
        )
        self.assertEqual(
            product_name_similarity(_rxn(product_name="ethanol"), _rxn(product_name="methanol")),
            0.0,
        )
        self.assertIsNone(product_name_similarity(_rxn(product_name="ethanol"), _rxn()))


class CombinedScoreTest(unittest.TestCase):
    def test_renormalizes_over_available_signals(self) -> None:
        # Only procedure vectors present → combined == the (clamped) procedure cosine.
        a = _rxn(procedure_vector=(1.0, 0.0, 0.0))
        b = _rxn(procedure_vector=(1.0, 0.0, 0.0))
        self.assertAlmostEqual(combined_score(a, b, MatchConfig()), 1.0)

    def test_no_signal_is_zero(self) -> None:
        self.assertEqual(combined_score(_rxn(), _rxn(), MatchConfig()), 0.0)

    def test_blend_of_two_signals(self) -> None:
        # Product names match (1.0) but reaction vectors orthogonal (0.0).
        cfg = MatchConfig(w_product_name=0.30, w_reaction=0.15)
        a = _rxn(product_name="x", reaction_vector=(1.0, 0.0))
        b = _rxn(product_name="x", reaction_vector=(0.0, 1.0))
        # (0.30*1.0 + 0.15*0.0) / (0.30 + 0.15) = 0.666...
        self.assertAlmostEqual(combined_score(a, b, cfg), 0.30 / 0.45)


class PairMatchTierOrderTest(unittest.TestCase):
    def test_provenance_wins_over_compound_jaccard(self) -> None:
        a = _rxn(compound_smiles=frozenset({"CCO"}), start_line=10, end_line=20)
        b = _rxn(compound_smiles=frozenset({"CCO"}), start_line=10, end_line=20)
        tier, score = pair_match(a, b, MatchConfig())
        self.assertEqual(tier, TIER_PROVENANCE)
        self.assertEqual(score, 1.0)

    def test_compound_jaccard_when_no_provenance(self) -> None:
        a = _rxn(compound_smiles=frozenset({"CCO", "CC"}))
        b = _rxn(compound_smiles=frozenset({"CCO", "CC"}))
        tier, score = pair_match(a, b, MatchConfig())
        self.assertEqual(tier, TIER_COMPOUND_JACCARD)
        self.assertEqual(score, 1.0)

    def test_compound_jaccard_rejects_convergent_product_with_disjoint_reactants(
        self,
    ) -> None:
        """Same final product, different routes (disjoint reactants) must NOT merge.

        A process patent commonly describes several distinct reactions that all
        end at the same target compound. Matching on product SMILES alone would
        wrongly merge them; the compound-Jaccard tier looks at the WHOLE
        reaction (product + reactants), so disjoint reactant sets correctly
        drag the Jaccard score below threshold even though the product matches.
        """
        # Route A: nitrile hydrolysis -> dicamba. Route B: direct methylation -> dicamba.
        a = _rxn(compound_smiles=frozenset({"COc1ccccc1C#N", "COc1ccccc1C(=O)O"}))
        b = _rxn(compound_smiles=frozenset({"Oc1ccccc1C(=O)O", "CCl", "COc1ccccc1C(=O)O"}))
        tier, _ = pair_match(a, b, MatchConfig())
        self.assertIsNone(tier)

    def test_combined_fallback_catches_generic_names(self) -> None:
        # No provenance, no compound sets — but matching procedure text
        # embeddings pull the pair together via the combined tier.
        a = _rxn(procedure_vector=(1.0, 0.0, 0.0))
        b = _rxn(procedure_vector=(1.0, 0.0, 0.0))
        tier, _ = pair_match(a, b, MatchConfig(tau_combined=0.7))
        self.assertEqual(tier, TIER_COMBINED)

    def test_no_match_returns_none(self) -> None:
        a = _rxn(compound_smiles=frozenset({"CCO"}))
        b = _rxn(compound_smiles=frozenset({"c1ccccc1"}))
        tier, _ = pair_match(a, b, MatchConfig())
        self.assertIsNone(tier)

    def test_disabled_tiers_are_skipped(self) -> None:
        a = _rxn(compound_smiles=frozenset({"CCO", "CC"}), start_line=10, end_line=20)
        b = _rxn(compound_smiles=frozenset({"CCO", "CC"}), start_line=10, end_line=20)
        # Provenance off → falls through to compound-Jaccard tier.
        tier, _ = pair_match(a, b, MatchConfig(enable_provenance=False))
        self.assertEqual(tier, TIER_COMPOUND_JACCARD)
        # All three off → no match, even though the compound sets are identical
        # (combined_score also folds in compound Jaccard as one of its signals).
        tier, _ = pair_match(
            a,
            b,
            MatchConfig(
                enable_provenance=False,
                enable_compound_jaccard=False,
                enable_combined=False,
            ),
        )
        self.assertIsNone(tier)


if __name__ == "__main__":
    unittest.main()
