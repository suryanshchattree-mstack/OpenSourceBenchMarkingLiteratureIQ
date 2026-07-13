"""Tests for app.py helpers that do not require a live Streamlit session."""

from __future__ import annotations

import unittest


class DefaultPipelineIdTest(unittest.TestCase):
    def test_section_wise_v_sequence(self) -> None:
        from app import _default_pipeline_id

        self.assertEqual(_default_pipeline_id(0), "section-wise-v1")
        self.assertEqual(_default_pipeline_id(1), "section-wise-v2")
        self.assertEqual(_default_pipeline_id(2), "section-wise-v3")
        self.assertEqual(_default_pipeline_id(7), "section-wise-v8")


if __name__ == "__main__":
    unittest.main()
