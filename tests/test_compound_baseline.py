"""Tests for majority-with-Claude-tiebreak compound baselines."""

from __future__ import annotations

import unittest

from core.compound_baseline import (
    NONE_SENTINEL,
    BaselineDefaults,
    compute_cluster_baselines,
    field_value_options,
    majority_with_tiebreak,
)
from core.compound_matching import NWayCluster, NWayDiffResult
from core.compound_parsing import CompoundEntry


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


class MajorityWithTiebreakTest(unittest.TestCase):
    def test_unanimous(self) -> None:
        self.assertEqual(
            majority_with_tiebreak(["solvent", "Solvent", "SOLVENT"], "reagent"),
            "solvent",
        )

    def test_split_prefers_tiebreak_when_tied(self) -> None:
        # solvent vs reagent tied 1-1; Claude said solvent
        self.assertEqual(
            majority_with_tiebreak(["solvent", "reagent"], "solvent"),
            "solvent",
        )

    def test_split_without_tiebreak_uses_alphabetical(self) -> None:
        # tied solvent/reagent; tiebreak is catalyst (not among tied) → alphabetical
        self.assertEqual(
            majority_with_tiebreak(["solvent", "reagent"], "catalyst"),
            "reagent",
        )

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(majority_with_tiebreak([None, None], "solvent"))

    def test_none_majority_wins_over_single_value(self) -> None:
        # 2 empties + 1 reagent → (none) wins
        self.assertIsNone(
            majority_with_tiebreak([None, None, "reagent"], "solvent")
        )

    def test_value_majority_beats_single_none(self) -> None:
        self.assertEqual(
            majority_with_tiebreak([None, "reagent", "reagent"], "solvent"),
            "reagent",
        )

    def test_blank_tiebreak_counts_as_none(self) -> None:
        # tied (none) vs reagent; Claude blank → (none) wins
        self.assertIsNone(
            majority_with_tiebreak([None, "reagent"], None)
        )


class ComputeClusterBaselinesTest(unittest.TestCase):
    def test_unanimous_presence_and_role(self) -> None:
        nway = NWayDiffResult(
            labels=("Claude", "GPT", "GLM"),
            clusters=[
                NWayCluster(
                    membership=frozenset({"Claude", "GPT", "GLM"}),
                    representatives={
                        "Claude": _entry("ethanol", role="reagent"),
                        "GPT": _entry("EtOH", role="reagent"),
                        "GLM": _entry("ethanol", role="reagent"),
                    },
                )
            ],
            raw_counts={"Claude": 1, "GPT": 1, "GLM": 1},
            deduped_counts={"Claude": 1, "GPT": 1, "GLM": 1},
        )
        defaults = compute_cluster_baselines(nway, tiebreak_label="Claude")
        self.assertEqual(defaults["ethanol"], BaselineDefaults(True, "reagent", "iupac"))

    def test_presence_tie_with_claude_present(self) -> None:
        # 2 models total, 1 present → tie; Claude present → present True
        nway = NWayDiffResult(
            labels=("Claude", "GPT"),
            clusters=[
                NWayCluster(
                    membership=frozenset({"Claude"}),
                    representatives={"Claude": _entry("acetone", role="solvent")},
                )
            ],
            raw_counts={"Claude": 1, "GPT": 0},
            deduped_counts={"Claude": 1, "GPT": 0},
        )
        defaults = compute_cluster_baselines(nway, tiebreak_label="Claude")
        self.assertTrue(defaults["acetone"].present)

    def test_presence_tie_without_claude(self) -> None:
        nway = NWayDiffResult(
            labels=("Claude", "GPT"),
            clusters=[
                NWayCluster(
                    membership=frozenset({"GPT"}),
                    representatives={"GPT": _entry("acetone", role="solvent")},
                )
            ],
            raw_counts={"Claude": 0, "GPT": 1},
            deduped_counts={"Claude": 0, "GPT": 1},
        )
        defaults = compute_cluster_baselines(nway, tiebreak_label="Claude")
        self.assertFalse(defaults["acetone"].present)

    def test_role_split_claude_tiebreak(self) -> None:
        nway = NWayDiffResult(
            labels=("Claude", "GPT"),
            clusters=[
                NWayCluster(
                    membership=frozenset({"Claude", "GPT"}),
                    representatives={
                        "Claude": _entry("methanol", role="solvent"),
                        "GPT": _entry("MeOH", role="reagent"),
                    },
                )
            ],
            raw_counts={"Claude": 1, "GPT": 1},
            deduped_counts={"Claude": 1, "GPT": 1},
        )
        defaults = compute_cluster_baselines(nway, tiebreak_label="Claude")
        self.assertEqual(defaults["methanol"].role, "solvent")

    def test_role_none_majority_among_extractors(self) -> None:
        # All three extracted; 2 empty roles + 1 reagent → Baseline role is None
        nway = NWayDiffResult(
            labels=("Claude", "GPT", "GLM"),
            clusters=[
                NWayCluster(
                    membership=frozenset({"Claude", "GPT", "GLM"}),
                    representatives={
                        "Claude": _entry("water", role=None),
                        "GPT": _entry("H2O", role=None),
                        "GLM": _entry("water", role="reagent"),
                    },
                )
            ],
            raw_counts={"Claude": 1, "GPT": 1, "GLM": 1},
            deduped_counts={"Claude": 1, "GPT": 1, "GLM": 1},
        )
        defaults = compute_cluster_baselines(nway, tiebreak_label="Claude")
        self.assertIsNone(defaults["water"].role)
        self.assertTrue(defaults["water"].present)

class FieldValueOptionsTest(unittest.TestCase):
    def test_casefold_dedup_preserves_first_seen_and_prefixes_none(self) -> None:
        entries_by_label = {
            "Claude": [
                _entry("a", role="solvent"),
                _entry("b", role="Reagent"),
            ],
            "GPT": [
                _entry("c", role="SOLVENT"),
                _entry("d", role="catalyst"),
            ],
        }
        options = field_value_options(entries_by_label, "role")
        self.assertEqual(options[0], NONE_SENTINEL)
        # first-seen casing kept; case-fold dedup drops later SOLVENT
        self.assertEqual(options[1:], ["catalyst", "Reagent", "solvent"])


if __name__ == "__main__":
    unittest.main()
