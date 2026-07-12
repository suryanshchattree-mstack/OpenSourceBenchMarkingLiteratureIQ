"""Smoke tests for UpSet figure sizing helpers."""

from __future__ import annotations

import unittest

from core.upset_viz import estimate_upset_size, membership_category_labels, render_upset


class UpsetVizTest(unittest.TestCase):
    def test_estimate_size_grows_with_intersections(self) -> None:
        small = [
            frozenset({"Claude", "A"}),
            frozenset({"Claude"}),
        ]
        large = [
            frozenset({f"M{i}", "Claude"}) for i in range(8)
        ] + [frozenset({f"M{i}"}) for i in range(8)]
        sw, sh, se = estimate_upset_size(small)
        lw, lh, le = estimate_upset_size(large)
        self.assertLessEqual(sw, lw)
        self.assertLessEqual(se, 36.0)
        self.assertLessEqual(le, se)
        self.assertGreaterEqual(sh, 4.0)
        self.assertLessEqual(lh, 10.0)

    def test_membership_category_labels_sorted(self) -> None:
        labels = membership_category_labels(
            [frozenset({"B", "A"}), frozenset({"Claude", "A"})]
        )
        self.assertEqual(labels, ["A", "B", "Claude"])

    def test_render_upset_empty_and_nonempty(self) -> None:
        empty = render_upset([])
        self.assertIsNotNone(empty)
        fig = render_upset(
            [
                frozenset({"Claude", "DeepSeek"}),
                frozenset({"Claude"}),
                frozenset({"DeepSeek", "GLM"}),
            ]
        )
        self.assertIsNotNone(fig)
        width, height = fig.get_size_inches()
        self.assertGreaterEqual(width, 7.0)
        self.assertGreaterEqual(height, 4.0)


if __name__ == "__main__":
    unittest.main()
