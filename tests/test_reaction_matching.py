"""Tests for reaction comparator axis scoring and skip/match behavior."""

from __future__ import annotations

import unittest

from core.reaction_matching import (
    TP_THRESHOLD,
    W_NAME,
    W_PROCEDURE,
    W_SMILES,
    canonicalize_smiles,
    compare_reactions,
    compose_weighted_sum,
    score_conditions,
    score_name,
    score_procedure,
    score_product_smiles,
    score_reactants,
    score_yield,
    strip_atom_maps,
)
from core.reaction_parsing import ReactionEntry


def _rxn(
    *,
    product_name: str | None = "product",
    product_smiles: str | None = None,
    reactant_names: tuple[str, ...] = (),
    reactant_smiles: tuple[str, ...] = (),
    product_yield_pct: float | None = None,
    procedure_vector: tuple[float, ...] | None = None,
    temperature_c: float | None = None,
    room_temperature: bool | None = None,
    time_h: float | None = None,
    atmosphere: str | None = None,
    reaction_class: str | None = None,
    non_synthetic: bool = False,
    section_label: str | None = "Ex 1",
    step_label: str | None = "Step 1",
) -> ReactionEntry:
    return ReactionEntry(
        product_name=product_name,
        product_smiles=product_smiles,
        reactant_names=reactant_names,
        reactant_smiles=reactant_smiles,
        product_yield_pct=product_yield_pct,
        procedure_text=None,
        procedure_vector=procedure_vector,
        temperature_c=temperature_c,
        room_temperature=room_temperature,
        time_h=time_h,
        atmosphere=atmosphere,
        reaction_class=reaction_class,
        non_synthetic=non_synthetic,
        section_label=section_label,
        step_label=step_label,
    )


class AxisSkipMatchTest(unittest.TestCase):
    def test_name_exact_and_skip_when_missing(self) -> None:
        a = _rxn(product_name="tert-Butanol")
        b = _rxn(product_name="tert - butanol")
        self.assertEqual(score_name(a, b), 1.0)
        self.assertIsNone(score_name(_rxn(product_name=None), b))
        self.assertIsNone(score_name(a, _rxn(product_name=None)))

    def test_smiles_match_after_canonicalize_and_atom_map_strip(self) -> None:
        left = _rxn(product_smiles="CCO")
        right = _rxn(product_smiles="C(O)C")  # ethanol, different atom order
        self.assertEqual(score_product_smiles(left, right), 1.0)

        mapped = _rxn(product_smiles="[CH3:1][CH2:2][OH:3]")
        self.assertEqual(strip_atom_maps(mapped.product_smiles), "[CH3][CH2][OH]")
        self.assertEqual(score_product_smiles(left, mapped), 1.0)

        mismatch = _rxn(product_smiles="c1ccccc1")
        self.assertEqual(score_product_smiles(left, mismatch), 0.0)
        self.assertIsNone(score_product_smiles(left, _rxn(product_smiles=None)))

    def test_reactants_jaccard_smiles_and_name_fallback(self) -> None:
        base = _rxn(reactant_smiles=("CCO", "c1ccccc1"))
        same = _rxn(reactant_smiles=("OCC", "c1ccccc1"))
        self.assertEqual(score_reactants(base, same), 1.0)

        partial = _rxn(reactant_smiles=("CCO",))
        self.assertAlmostEqual(score_reactants(base, partial), 0.5)

        # No SMILES on one side → fall back to names.
        by_name = _rxn(reactant_names=("ethanol", "benzene"))
        other_names = _rxn(reactant_names=("Ethanol", "benzene"))
        self.assertEqual(score_reactants(by_name, other_names), 1.0)

        self.assertIsNone(
            score_reactants(
                _rxn(reactant_names=(), reactant_smiles=()),
                _rxn(reactant_names=(), reactant_smiles=()),
            )
        )

    def test_procedure_cosine_and_skip_without_vector(self) -> None:
        vec_a = (1.0, 0.0, 0.0)
        vec_b = (1.0, 0.0, 0.0)
        self.assertAlmostEqual(
            score_procedure(_rxn(procedure_vector=vec_a), _rxn(procedure_vector=vec_b)),
            1.0,
        )
        self.assertIsNone(
            score_procedure(_rxn(procedure_vector=vec_a), _rxn(procedure_vector=None))
        )
        self.assertIsNone(
            score_procedure(_rxn(procedure_vector=None), _rxn(procedure_vector=None))
        )

    def test_yield_buckets(self) -> None:
        base = _rxn(product_yield_pct=50.0)
        self.assertEqual(score_yield(base, _rxn(product_yield_pct=51.0)), 1.0)  # <=2
        self.assertEqual(score_yield(base, _rxn(product_yield_pct=54.0)), 0.75)  # <=5
        self.assertEqual(score_yield(base, _rxn(product_yield_pct=58.0)), 0.40)  # <=10
        self.assertEqual(score_yield(base, _rxn(product_yield_pct=70.0)), 0.0)
        self.assertIsNone(score_yield(base, _rxn(product_yield_pct=None)))

    def test_conditions_match_and_skip(self) -> None:
        base = _rxn(temperature_c=80.0, time_h=2.0, atmosphere="argon")
        close = _rxn(temperature_c=82.0, time_h=2.1, atmosphere="Ar")
        score = score_conditions(base, close)
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.9)

        self.assertIsNone(
            score_conditions(
                _rxn(temperature_c=None, time_h=None, atmosphere=None),
                _rxn(temperature_c=None, time_h=None, atmosphere=None),
            )
        )

    def test_compose_weighted_sum_renormalizes_skipped_axes(self) -> None:
        # Only name + smiles present → weights renormalize over 0.40 + 0.15.
        composite = compose_weighted_sum(1.0, 1.0, None, None, None, None)
        self.assertAlmostEqual(composite, 1.0)

        # Name perfect, procedure zero, others skipped.
        composite2 = compose_weighted_sum(1.0, None, None, 0.0, None, None)
        expected = (W_NAME * 1.0 + W_PROCEDURE * 0.0) / (W_NAME + W_PROCEDURE)
        self.assertAlmostEqual(composite2, expected)

        self.assertIsNone(compose_weighted_sum(None, None, None, None, None, None))

    def test_smiles_weight_constant(self) -> None:
        self.assertAlmostEqual(W_SMILES, 0.15)


class CompareReactionsIntegrationTest(unittest.TestCase):
    def test_perfect_pair_is_true_positive(self) -> None:
        baseline = [
            _rxn(
                product_name="ethanol",
                product_smiles="CCO",
                reactant_smiles=("C", "O"),
                product_yield_pct=80.0,
                temperature_c=25.0,
                room_temperature=True,
                time_h=1.0,
                atmosphere="air",
                procedure_vector=(1.0, 0.0),
            )
        ]
        candidate = [
            _rxn(
                product_name="Ethanol",
                product_smiles="OCC",
                reactant_smiles=("C", "O"),
                product_yield_pct=81.0,
                temperature_c=25.0,
                room_temperature=True,
                time_h=1.0,
                atmosphere="air",
                procedure_vector=(1.0, 0.0),
            )
        ]
        report = compare_reactions(baseline, candidate, baseline_label="claude", candidate_label="gpt")
        self.assertEqual(report.true_positives, 1)
        self.assertEqual(report.false_positives, 0)
        self.assertEqual(report.false_negatives, 0)
        self.assertGreaterEqual(report.match_details[0].composite_score or 0.0, TP_THRESHOLD)

    def test_non_synthetic_filtered_before_scoring(self) -> None:
        baseline = [
            _rxn(product_name="real", product_smiles="CCO", non_synthetic=False),
            _rxn(product_name="skip-me", product_smiles="c1ccccc1", non_synthetic=True),
        ]
        candidate = [
            _rxn(product_name="real", product_smiles="CCO", non_synthetic=False),
        ]
        report = compare_reactions(baseline, candidate)
        self.assertEqual(report.non_synthetic_skipped_baseline, 1)
        self.assertEqual(report.baseline_reaction_count, 1)
        self.assertEqual(report.true_positives, 1)

    def test_unrelated_pair_is_fp_and_fn(self) -> None:
        baseline = [_rxn(product_name="alpha-compound", product_smiles="CCO")]
        candidate = [_rxn(product_name="totally-different-molecule", product_smiles="c1ccccc1")]
        report = compare_reactions(baseline, candidate)
        self.assertEqual(report.true_positives, 0)
        self.assertEqual(report.false_positives, 1)
        self.assertEqual(report.false_negatives, 1)

    def test_canonicalize_smiles_helper(self) -> None:
        self.assertEqual(canonicalize_smiles("CCO"), canonicalize_smiles("OCC"))
        self.assertIsNone(canonicalize_smiles(None))
        self.assertIsNone(canonicalize_smiles("not-a-smiles!!!"))


if __name__ == "__main__":
    unittest.main()
