"""Tests for M1 agreement heatmap construction."""

from __future__ import annotations

import unittest

from core.m1_agreement import compute_m1_agreement
from core.m1_parsing import QUANTITY_FIELDS, M1CompoundEntry
from core.m1_visuals import build_agreement_heatmap


class M1VisualsTest(unittest.TestCase):
    def _entry(self, identifier: str, **kwargs) -> M1CompoundEntry:
        qty = {field: None for field in QUANTITY_FIELDS}
        qty.update(kwargs.pop("quantity", {}) or {})
        return M1CompoundEntry(
            identifier=identifier,
            identifier_type=kwargs.get("identifier_type", "iupac"),
            aliases=kwargs.get("aliases", ()),
            role=kwargs.get("role", "reagent"),
            is_section_product=kwargs.get("is_section_product", False),
            commercially_available=False,
            quantity=qty,
            ms_mz=None,
            notes=None,
            section_label="Example 1",
        )

    def test_heatmap_has_expected_axes(self) -> None:
        result = compute_m1_agreement(
            {
                "claude": [self._entry("methanol", role="solvent")],
                "gpt": [self._entry("methanol", role="reagent")],
            },
            baseline="claude",
        )
        fig = build_agreement_heatmap(result.summaries)
        heatmap = fig.data[0]
        self.assertEqual(list(heatmap.y), ["gpt"])
        self.assertIn("role", list(heatmap.x))
        self.assertEqual(heatmap.zmin, 0.0)
        self.assertEqual(heatmap.zmax, 1.0)


if __name__ == "__main__":
    unittest.main()
