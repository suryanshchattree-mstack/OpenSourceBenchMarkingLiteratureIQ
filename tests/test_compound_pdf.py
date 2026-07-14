"""Smoke test for compounds PDF export."""

from __future__ import annotations

import re
import zlib
import unittest

import pandas as pd

from core.compound_pdf import build_compounds_pdf_report


def _pdf_plaintext(pdf_bytes: bytes) -> str:
    """Best-effort decode of ASCII85+Flate content streams for assertions."""
    import base64

    chunks: list[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", pdf_bytes, flags=re.DOTALL):
        raw = match.group(1).strip()
        try:
            inflated = zlib.decompress(base64.a85decode(raw, adobe=True))
            chunks.append(inflated.decode("latin-1", errors="ignore"))
        except (zlib.error, ValueError):
            try:
                chunks.append(zlib.decompress(raw).decode("latin-1", errors="ignore"))
            except zlib.error:
                chunks.append(raw.decode("latin-1", errors="ignore"))
    return "\n".join(chunks)


class CompoundPdfTest(unittest.TestCase):
    def test_pdf_starts_with_magic_and_embeds_grids(self) -> None:
        stats_df = pd.DataFrame(
            [
                {
                    "Rank": 1,
                    "composite_score": 0.9,
                    "model": "Claude",
                    "presence_precision": 1.0,
                    "presence_recall": 0.9,
                    "presence_f1": 0.95,
                    "role_accuracy": 0.8,
                    "identifier_type_accuracy": 0.7,
                },
                {
                    "Rank": 2,
                    "composite_score": 0.4,
                    "model": "GLM",
                    "presence_precision": 0.5,
                    "presence_recall": 0.5,
                    "presence_f1": 0.5,
                    "role_accuracy": 0.4,
                    "identifier_type_accuracy": 0.3,
                },
            ]
        )
        presence_df = pd.DataFrame(
            [
                {
                    "Compound": "ethanol",
                    "Match tier": "Name",
                    "Claude": True,
                    "GLM": False,
                    "Baseline": True,
                }
            ]
        )
        role_df = pd.DataFrame(
            [
                {
                    "Compound": "ethanol",
                    "Match tier": "Name",
                    "Claude": "reagent",
                    "GLM": "(none)",
                    "Baseline": "reagent",
                }
            ]
        )
        id_type_df = pd.DataFrame(
            [
                {
                    "Compound": "ethanol",
                    "Match tier": "Name",
                    "Claude": "iupac",
                    "GLM": "abbreviation",
                    "Baseline": "iupac",
                }
            ]
        )
        pdf_bytes = build_compounds_pdf_report(
            "WO2015086698A1",
            {"Claude": "section-wise-v1", "GLM": "section-wise-v2"},
            stats_df,
            presence_df,
            role_df,
            id_type_df,
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 500)
        text = _pdf_plaintext(pdf_bytes)
        self.assertIn("ethanol", text)
        self.assertIn("reagent", text)
        self.assertIn("iupac", text)
        self.assertIn("Ranked model metrics", text)
        self.assertIn("Presence grid", text)
        self.assertIn("Role grid", text)
        self.assertIn("Identifier-type grid", text)


if __name__ == "__main__":
    unittest.main()
