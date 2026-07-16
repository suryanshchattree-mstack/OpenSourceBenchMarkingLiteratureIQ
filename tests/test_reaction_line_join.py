"""Tests for R1 → reaction line joining."""

from __future__ import annotations

import unittest
from dataclasses import replace

from core.reaction_line_join import join_reactions_to_r1, normalize_step_label
from core.reaction_parsing import ReactionEntry


def _rxn(
    *,
    section_label: str | None = "Ex 1",
    step_label: str | None = "Step 1",
    step_index: int | None = 0,
    product_name: str | None = "product",
) -> ReactionEntry:
    return ReactionEntry(
        product_name=product_name,
        product_smiles=None,
        reactant_names=(),
        reactant_smiles=(),
        product_yield_pct=None,
        procedure_text=None,
        procedure_vector=None,
        temperature_c=None,
        room_temperature=None,
        time_h=None,
        atmosphere=None,
        reaction_class=None,
        non_synthetic=False,
        section_label=section_label,
        step_label=step_label,
        step_index=step_index,
    )


def _r1_step(
    *,
    section_label: str = "Ex 1",
    step_index: int = 0,
    step_label: str = "Step 1",
    start_line: int = 10,
    end_line: int = 20,
) -> dict:
    return {
        "section_label": section_label,
        "step_index": step_index,
        "step_label": step_label,
        "start_line": start_line,
        "end_line": end_line,
    }


class NormalizeStepLabelTest(unittest.TestCase):
    def test_casefold_underscore_whitespace(self) -> None:
        self.assertEqual(normalize_step_label("Step_A"), "step a")
        self.assertEqual(normalize_step_label("  STEP   A  "), "step a")


class JoinReactionsToR1Test(unittest.TestCase):
    def test_exact_index_join(self) -> None:
        entries = {"Claude": [_rxn(step_index=1, step_label="Other")]}
        r1 = {
            "Claude": [
                _r1_step(step_index=0, start_line=1, end_line=5),
                _r1_step(step_index=1, step_label="Step B", start_line=50, end_line=60),
            ]
        }
        joined = join_reactions_to_r1(entries, r1)
        entry = joined["Claude"][0]
        self.assertEqual(entry.start_line, 50)
        self.assertEqual(entry.end_line, 60)
        self.assertEqual(entry.line_join, "exact_index")

    def test_label_fallback_when_index_misses(self) -> None:
        entries = {"Claude": [_rxn(step_index=99, step_label="Step_A")]}
        r1 = {
            "Claude": [
                _r1_step(step_index=0, step_label="step a", start_line=100, end_line=110),
            ]
        }
        joined = join_reactions_to_r1(entries, r1)
        entry = joined["Claude"][0]
        self.assertEqual(entry.start_line, 100)
        self.assertEqual(entry.end_line, 110)
        self.assertEqual(entry.line_join, "exact_label")

    def test_missing_r1_leaves_null_lines(self) -> None:
        entries = {"Claude": [_rxn()], "GPT": [_rxn()]}
        r1 = {"Claude": [_r1_step()]}
        joined = join_reactions_to_r1(entries, r1)
        self.assertEqual(joined["Claude"][0].line_join, "exact_index")
        self.assertIsNone(joined["GPT"][0].start_line)
        self.assertIsNone(joined["GPT"][0].end_line)
        self.assertIsNone(joined["GPT"][0].line_join)

    def test_replace_returns_new_frozen_entry(self) -> None:
        original = _rxn()
        joined = join_reactions_to_r1(
            {"Claude": [original]},
            {"Claude": [_r1_step(start_line=7, end_line=9)]},
        )
        self.assertIsNone(original.start_line)
        self.assertEqual(joined["Claude"][0].start_line, 7)
        # Still a ReactionEntry; line fields set via dataclasses.replace
        self.assertIsInstance(joined["Claude"][0], ReactionEntry)
        annotated = replace(original, start_line=1, end_line=2, line_join="exact_index")
        self.assertEqual(annotated.line_join, "exact_index")


if __name__ == "__main__":
    unittest.main()
