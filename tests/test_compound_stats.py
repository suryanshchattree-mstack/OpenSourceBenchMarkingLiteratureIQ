"""Tests for editable-grid seeding, stats-from-edited, and model ranking."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from core.compound_baseline import NONE_SENTINEL, BaselineDefaults
from core.compound_matching import NWayCluster
from core.compound_parsing import CompoundEntry
from core.compound_stats import (
    compute_field_accuracy,
    compute_presence_stats,
    filter_by_presence_baseline,
    rank_models,
    seed_view_dataframe,
)


def _entry(
    identifier: str,
    *,
    role: str | None = "reagent",
    identifier_type: str = "iupac",
) -> CompoundEntry:
    return CompoundEntry(
        identifier=identifier,
        identifier_type=identifier_type,
        aliases=(),
        resolved=True,
        unresolved_reference=False,
        section_label="Example 1",
        role=role,
    )


class SeedViewDataframeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clusters = [
            NWayCluster(
                membership=frozenset({"Claude", "GPT"}),
                representatives={
                    "Claude": _entry("ethanol", role="reagent"),
                    "GPT": _entry("EtOH", role="solvent"),
                },
                match_tier="name",
            ),
            NWayCluster(
                membership=frozenset({"GPT"}),
                representatives={"GPT": _entry("acetone", role="solvent")},
                match_tier=None,
            ),
        ]
        self.model_labels = ["Claude", "GPT"]
        self.defaults = {
            "ethanol": BaselineDefaults(True, "reagent", "iupac"),
            "acetone": BaselineDefaults(False, "solvent", "iupac"),
        }

    def test_presence_bool_columns_and_absent_defaults(self) -> None:
        df = seed_view_dataframe(
            self.clusters,
            self.model_labels,
            "presence",
            defaults=self.defaults,
            preferred_label="Claude",
        )
        self.assertEqual(list(df.columns), ["Compound", "Match tier", "Claude", "GPT", "Baseline"])
        ethanol = df.loc[df["Compound"] == "ethanol"].iloc[0]
        self.assertTrue(bool(ethanol["Claude"]))
        self.assertTrue(bool(ethanol["GPT"]))
        self.assertTrue(bool(ethanol["Baseline"]))
        acetone = df.loc[df["Compound"] == "acetone"].iloc[0]
        self.assertFalse(bool(acetone["Claude"]))
        self.assertTrue(bool(acetone["GPT"]))
        self.assertFalse(bool(acetone["Baseline"]))
        self.assertEqual(acetone["Match tier"], "—")

    def test_role_uses_none_sentinel_for_absent(self) -> None:
        df = seed_view_dataframe(
            self.clusters,
            self.model_labels,
            "role",
            defaults=self.defaults,
            preferred_label="Claude",
        )
        acetone = df.loc[df["Compound"] == "acetone"].iloc[0]
        self.assertEqual(acetone["Claude"], NONE_SENTINEL)
        self.assertEqual(acetone["GPT"], "solvent")
        self.assertEqual(acetone["Baseline"], "solvent")
        ethanol = df.loc[df["Compound"] == "ethanol"].iloc[0]
        self.assertEqual(ethanol["Claude"], "reagent")
        self.assertEqual(ethanol["GPT"], "solvent")
        self.assertEqual(ethanol["Baseline"], "reagent")


class ComputePresenceStatsTest(unittest.TestCase):
    def test_precision_recall_f1_including_added_rows(self) -> None:
        presence_df = pd.DataFrame(
            [
                {"Compound": "a", "Match tier": "Name", "Claude": True, "GPT": True, "Baseline": True},
                {"Compound": "b", "Match tier": "—", "Claude": True, "GPT": False, "Baseline": False},
                {"Compound": "added", "Match tier": "", "Claude": False, "GPT": True, "Baseline": True},
            ]
        )
        stats = compute_presence_stats(presence_df, ["Claude", "GPT"]).set_index("model")
        # Claude: TP=a, FP=b, FN=added → P=1/2, R=1/2, F1=0.5
        self.assertAlmostEqual(stats.loc["Claude", "presence_precision"], 0.5)
        self.assertAlmostEqual(stats.loc["Claude", "presence_recall"], 0.5)
        self.assertAlmostEqual(stats.loc["Claude", "presence_f1"], 0.5)
        # GPT: TP=a+added, FP=0, FN=0 → P=1, R=1, F1=1
        self.assertAlmostEqual(stats.loc["GPT", "presence_precision"], 1.0)
        self.assertAlmostEqual(stats.loc["GPT", "presence_recall"], 1.0)
        self.assertAlmostEqual(stats.loc["GPT", "presence_f1"], 1.0)


class ComputeFieldAccuracyTest(unittest.TestCase):
    def test_excludes_none_baseline_and_counts_model_none_as_miss(self) -> None:
        role_df = pd.DataFrame(
            [
                {
                    "Compound": "a",
                    "Match tier": "Name",
                    "Claude": "solvent",
                    "GPT": "reagent",
                    "Baseline": "solvent",
                },
                {
                    "Compound": "b",
                    "Match tier": "Name",
                    "Claude": NONE_SENTINEL,
                    "GPT": "solvent",
                    "Baseline": "solvent",
                },
                {
                    "Compound": "c",
                    "Match tier": "—",
                    "Claude": "reagent",
                    "GPT": "reagent",
                    "Baseline": NONE_SENTINEL,
                },
            ]
        )
        acc = compute_field_accuracy(role_df, ["Claude", "GPT"]).set_index("model")
        # Claude: rows a,b comparable; a match, b miss → 1/2
        self.assertAlmostEqual(acc.loc["Claude", "accuracy"], 0.5)
        # GPT: a mismatch, b match → 1/2
        self.assertAlmostEqual(acc.loc["GPT", "accuracy"], 0.5)


class RankModelsTest(unittest.TestCase):
    def test_composite_nanmean_rank_and_tiebreak(self) -> None:
        stats_df = pd.DataFrame(
            [
                {
                    "model": "A",
                    "presence_f1": 0.9,
                    "role_accuracy": 0.5,
                    "identifier_type_accuracy": 0.5,
                },
                {
                    "model": "B",
                    "presence_f1": 0.8,
                    "role_accuracy": 0.8,
                    "identifier_type_accuracy": 0.8,
                },
                {
                    "model": "C",
                    "presence_f1": 1.0,
                    "role_accuracy": None,
                    "identifier_type_accuracy": None,
                },
            ]
        )
        ranked = rank_models(stats_df)
        self.assertEqual(list(ranked.columns[:2]), ["Rank", "composite_score"])
        # Defaults: F1=1.0, role=0.25, id_type=0.50 (denom 1.75)
        # C composite = nanmean(1.0) = 1.0 → rank 1
        # B composite = (0.8 + 0.2 + 0.4) / 1.75 = 0.8 → rank 2
        # A composite = (0.9 + 0.125 + 0.25) / 1.75 ≈ 0.729 → rank 3
        self.assertEqual(list(ranked["model"]), ["C", "B", "A"])
        self.assertEqual(list(ranked["Rank"]), [1, 2, 3])
        self.assertTrue(math.isclose(ranked.iloc[0]["composite_score"], 1.0))

    def test_tie_break_by_presence_f1_then_role(self) -> None:
        stats_df = pd.DataFrame(
            [
                {
                    "model": "low_f1",
                    "presence_f1": 0.5,
                    "role_accuracy": 1.0,
                    "identifier_type_accuracy": 0.5,
                },
                {
                    "model": "high_f1",
                    "presence_f1": 0.9,
                    "role_accuracy": 0.6,
                    "identifier_type_accuracy": 0.5,
                },
            ]
        )
        # Equal weights → both composite = 2/3; high_f1 wins on presence_f1 tie-break
        equal = {
            "presence_f1": 1.0,
            "role_accuracy": 1.0,
            "identifier_type_accuracy": 1.0,
        }
        ranked = rank_models(stats_df, weights=equal)
        self.assertEqual(list(ranked["model"]), ["high_f1", "low_f1"])
        self.assertEqual(list(ranked["Rank"]), [1, 2])

    def test_weighted_composite_presence_only(self) -> None:
        stats_df = pd.DataFrame(
            [
                {
                    "model": "high_f1",
                    "presence_f1": 1.0,
                    "role_accuracy": 0.0,
                    "identifier_type_accuracy": 0.0,
                },
                {
                    "model": "high_role",
                    "presence_f1": 0.0,
                    "role_accuracy": 1.0,
                    "identifier_type_accuracy": 1.0,
                },
            ]
        )
        ranked = rank_models(
            stats_df,
            weights={
                "presence_f1": 1.0,
                "role_accuracy": 0.0,
                "identifier_type_accuracy": 0.0,
            },
        )
        self.assertEqual(list(ranked["model"]), ["high_f1", "high_role"])
        self.assertTrue(math.isclose(ranked.iloc[0]["composite_score"], 1.0))
        self.assertTrue(math.isclose(ranked.iloc[1]["composite_score"], 0.0))

    def test_all_zero_weights_fall_back_to_equal(self) -> None:
        stats_df = pd.DataFrame(
            [
                {
                    "model": "A",
                    "presence_f1": 0.9,
                    "role_accuracy": 0.3,
                    "identifier_type_accuracy": 0.3,
                },
                {
                    "model": "B",
                    "presence_f1": 0.5,
                    "role_accuracy": 0.5,
                    "identifier_type_accuracy": 0.5,
                },
            ]
        )
        zero = {
            "presence_f1": 0.0,
            "role_accuracy": 0.0,
            "identifier_type_accuracy": 0.0,
        }
        equal = {
            "presence_f1": 1.0,
            "role_accuracy": 1.0,
            "identifier_type_accuracy": 1.0,
        }
        ranked_zero = rank_models(stats_df, weights=zero)
        ranked_equal = rank_models(stats_df, weights=equal)
        self.assertEqual(list(ranked_zero["model"]), list(ranked_equal["model"]))
        for left, right in zip(ranked_zero["composite_score"], ranked_equal["composite_score"]):
            self.assertTrue(math.isclose(float(left), float(right)))

    def test_default_weights_prefer_f1_over_role(self) -> None:
        stats_df = pd.DataFrame(
            [
                {
                    "model": "high_f1",
                    "presence_f1": 1.0,
                    "role_accuracy": 0.0,
                    "identifier_type_accuracy": 0.0,
                },
                {
                    "model": "high_role",
                    "presence_f1": 0.0,
                    "role_accuracy": 1.0,
                    "identifier_type_accuracy": 1.0,
                },
            ]
        )
        # Defaults 1.0 / 0.25 / 0.50 → high_f1 = 1.0/1.75, high_role = 0.75/1.75
        ranked = rank_models(stats_df)
        self.assertEqual(list(ranked["model"]), ["high_f1", "high_role"])
        self.assertTrue(
            math.isclose(ranked.iloc[0]["composite_score"], 1.0 / 1.75)
        )

class FilterByPresenceBaselineTest(unittest.TestCase):
    def test_keeps_only_presence_true_matching_casefold(self) -> None:
        presence = pd.DataFrame(
            [
                {"Compound": "Ethanol", "Baseline": True},
                {"Compound": "acetone", "Baseline": False},
                {"Compound": "water", "Baseline": True},
            ]
        )
        role = pd.DataFrame(
            [
                {"Compound": "ethanol", "Claude": "reagent", "Baseline": "reagent"},
                {"Compound": "Acetone", "Claude": "solvent", "Baseline": "solvent"},
                {"Compound": "water", "Claude": "solvent", "Baseline": "solvent"},
                {"Compound": "orphan", "Claude": "catalyst", "Baseline": "catalyst"},
            ]
        )
        scoped = filter_by_presence_baseline(role, presence)
        self.assertEqual(list(scoped["Compound"]), ["ethanol", "water"])

    def test_orphan_not_in_presence_excluded(self) -> None:
        presence = pd.DataFrame([{"Compound": "a", "Baseline": True}])
        role = pd.DataFrame(
            [
                {"Compound": "a", "Baseline": "reagent"},
                {"Compound": "only_in_role", "Baseline": "solvent"},
            ]
        )
        scoped = filter_by_presence_baseline(role, presence)
        self.assertEqual(list(scoped["Compound"]), ["a"])

    def test_empty_presence_excludes_all(self) -> None:
        role = pd.DataFrame([{"Compound": "a", "Baseline": "reagent"}])
        presence = pd.DataFrame(columns=["Compound", "Baseline"])
        scoped = filter_by_presence_baseline(role, presence)
        self.assertTrue(scoped.empty)


if __name__ == "__main__":
    unittest.main()
