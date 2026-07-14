"""Tests for stable categorical colors and cross-grid compound delete."""

from __future__ import annotations

import unittest

import pandas as pd

from core.compound_baseline import NONE_SENTINEL
from core.compound_colors import ABSENT_GRAY, CATEGORICAL_PALETTE, color_for_value, css_cell
from core.compound_grid import compound_key, drop_compound_rows, drop_compounds_from_frames


class CompoundKeyTest(unittest.TestCase):
    def test_strip_casefold(self) -> None:
        self.assertEqual(compound_key("  Ethanol "), "ethanol")
        self.assertEqual(compound_key("ACETONE"), "acetone")

    def test_none_and_nan(self) -> None:
        self.assertEqual(compound_key(None), "")
        self.assertEqual(compound_key(float("nan")), "")


class ColorForValueTest(unittest.TestCase):
    def test_same_value_same_color(self) -> None:
        self.assertEqual(color_for_value("reagent"), color_for_value("Reagent"))
        self.assertEqual(color_for_value("iupac"), color_for_value("IUPAC"))

    def test_none_and_blank_are_gray(self) -> None:
        self.assertEqual(color_for_value(NONE_SENTINEL), ABSENT_GRAY)
        self.assertEqual(color_for_value("(None)"), ABSENT_GRAY)
        self.assertEqual(color_for_value(""), ABSENT_GRAY)
        self.assertEqual(color_for_value("   "), ABSENT_GRAY)
        self.assertEqual(color_for_value(None), ABSENT_GRAY)
        self.assertEqual(color_for_value(float("nan")), ABSENT_GRAY)

    def test_different_values_can_differ(self) -> None:
        colors = {color_for_value(v) for v in ("reagent", "solvent", "catalyst", "product")}
        self.assertGreaterEqual(len(colors), 2)

    def test_color_is_from_palette(self) -> None:
        color = color_for_value("reagent")
        self.assertIn(color, CATEGORICAL_PALETTE)

    def test_css_cell_includes_black_text(self) -> None:
        self.assertEqual(
            css_cell("#c6f6d5"),
            "background-color: #c6f6d5; color: #000000",
        )


class DropCompoundRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.presence = pd.DataFrame(
            {
                "Compound": ["ethanol", "acetone", "water"],
                "Match tier": ["name", "—", "name"],
                "Claude": [True, False, True],
                "Baseline": [True, False, True],
            }
        )
        self.role = pd.DataFrame(
            {
                "Compound": ["ethanol", "acetone", "water"],
                "Match tier": ["name", "—", "name"],
                "Claude": ["reagent", "solvent", "solvent"],
                "Baseline": ["reagent", "solvent", "solvent"],
            }
        )
        self.id_type = pd.DataFrame(
            {
                "Compound": ["ethanol", "acetone", "water"],
                "Match tier": ["name", "—", "name"],
                "Claude": ["iupac", "iupac", "common"],
                "Baseline": ["iupac", "iupac", "common"],
            }
        )

    def test_drops_named_rows_casefold(self) -> None:
        out = drop_compound_rows(self.presence, ["Acetone"])
        self.assertEqual(list(out["Compound"]), ["ethanol", "water"])

    def test_unrelated_rows_remain(self) -> None:
        out = drop_compound_rows(self.role, ["ethanol"])
        self.assertEqual(list(out["Compound"]), ["acetone", "water"])
        self.assertEqual(out.loc[0, "Claude"], "solvent")

    def test_empty_names_noop(self) -> None:
        out = drop_compound_rows(self.presence, [])
        self.assertEqual(list(out["Compound"]), ["ethanol", "acetone", "water"])

    def test_sync_across_three_frames(self) -> None:
        frames = drop_compounds_from_frames(
            {
                "presence": self.presence,
                "role": self.role,
                "identifier_type": self.id_type,
            },
            ["acetone", "WATER"],
        )
        for key, frame in frames.items():
            with self.subTest(key=key):
                self.assertEqual(list(frame["Compound"]), ["ethanol"])


if __name__ == "__main__":
    unittest.main()
