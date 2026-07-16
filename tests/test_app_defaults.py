"""Tests for app.py helpers that do not require a live Streamlit session."""

from __future__ import annotations

import inspect
import unittest


class DefaultPipelineIdTest(unittest.TestCase):
    def test_section_wise_v_sequence(self) -> None:
        from app import _default_pipeline_id

        self.assertEqual(_default_pipeline_id(0), "section-wise-v1")
        self.assertEqual(_default_pipeline_id(1), "section-wise-v2")
        self.assertEqual(_default_pipeline_id(2), "section-wise-v3")
        self.assertEqual(_default_pipeline_id(7), "section-wise-v8")


class ReactionSignatureTest(unittest.TestCase):
    def test_editor_signature_tracks_match_config(self) -> None:
        from app import (
            ModelUploads,
            _reactions_editor_signature,
            _reactions_upload_signature,
        )
        from core.reaction_scoring import MatchConfig

        rows = [ModelUploads(label="Claude", files={})]
        upload_a = _reactions_upload_signature(rows, "WO1")
        upload_b = _reactions_upload_signature(rows, "WO1")
        self.assertEqual(upload_a, upload_b)

        default_config = MatchConfig()
        editor_default = _reactions_editor_signature(rows, "WO1", config=default_config)
        self.assertTrue(editor_default.startswith(upload_a))
        # Same uploads + same config → stable signature.
        self.assertEqual(
            editor_default,
            _reactions_editor_signature(rows, "WO1", config=MatchConfig()),
        )

        # Changing any matcher knob changes the signature (grids rebuild).
        editor_low_jaccard = _reactions_editor_signature(
            rows, "WO1", config=MatchConfig(tau_jaccard=0.60)
        )
        self.assertNotEqual(editor_default, editor_low_jaccard)

        editor_prov_off = _reactions_editor_signature(
            rows, "WO1", config=MatchConfig(enable_provenance=False)
        )
        self.assertNotEqual(editor_default, editor_prov_off)

        # The vector-enrichment toggle is part of the signature too.
        editor_enriched = _reactions_editor_signature(
            rows, "WO1", config=default_config, enrich_vectors=True
        )
        self.assertNotEqual(editor_default, editor_enriched)

        params = inspect.signature(_reactions_editor_signature).parameters
        self.assertIn("config", params)
        self.assertIn("enrich_vectors", params)

    def test_upload_signature_includes_r1(self) -> None:
        from app import ModelUploads, _reactions_upload_signature

        rxn = (b'[{"product_name":"a"}]', "r.json")
        r1 = (b'[{"step_index":0,"start_line":1,"end_line":2}]', "r1.json")
        base = ModelUploads(label="Claude", files={"reactions": rxn})
        with_r1 = ModelUploads(label="Claude", files={"reactions": rxn, "r1": r1})
        sig_without = _reactions_upload_signature([base], "WO1")
        sig_with = _reactions_upload_signature([with_r1], "WO1")
        self.assertNotEqual(sig_without, sig_with)


class DefaultRunLabelsTest(unittest.TestCase):
    def test_six_standard_labels(self) -> None:
        from app import DEFAULT_RUN_LABELS, _default_label

        self.assertEqual(
            DEFAULT_RUN_LABELS,
            ["Claude", "DeepSeekFlash", "GLM", "DeepSeekPro", "Kimi", "MiniMax"],
        )
        self.assertEqual(_default_label(0), "Claude")
        self.assertEqual(_default_label(5), "MiniMax")
        self.assertEqual(_default_label(6), "Run 7")


if __name__ == "__main__":
    unittest.main()
