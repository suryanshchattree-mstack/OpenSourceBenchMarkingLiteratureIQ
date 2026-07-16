"""Tests for R1 step-boundary JSON → Section parsing."""

from __future__ import annotations

import json
import unittest

from core.parsing import Section
from core.r1_parsing import parse_r1_json, parse_r1_step_dicts


def _step(
    step_index: int,
    step_label: str,
    *,
    start_line: int,
    end_line: int,
    section_type: str = "experimental",
) -> dict:
    return {
        "step_index": step_index,
        "step_label": step_label,
        "section_type": section_type,
        "start_line": start_line,
        "end_line": end_line,
    }


class R1ParsingTest(unittest.TestCase):
    def test_flat_list_of_steps(self) -> None:
        payload = [
            _step(0, "Step A", start_line=10, end_line=20),
            _step(1, "Step B", start_line=21, end_line=40, section_type="procedure"),
        ]
        sections = parse_r1_json(json.dumps(payload), source_label="flat")
        self.assertEqual(len(sections), 2)
        self.assertIsInstance(sections[0], Section)
        self.assertEqual(sections[0].section_index, 0)
        self.assertEqual(sections[0].section_label, "Step A")
        self.assertEqual(sections[0].section_type, "experimental")
        self.assertEqual(sections[0].start_line, 10)
        self.assertEqual(sections[0].end_line, 20)
        self.assertEqual(sections[1].section_label, "Step B")
        self.assertEqual(sections[1].section_type, "procedure")
        self.assertEqual(sections[1].start_line, 21)

    def test_double_encoded_consolidated_shape(self) -> None:
        """Outer array = one entry per pre-pass section; each entry a JSON string of steps."""
        section0_steps = [
            _step(0, "Dissolve", start_line=166, end_line=186),
            _step(1, "Heat", start_line=187, end_line=210),
        ]
        section1_steps: list[dict] = []
        section2_steps = [
            _step(0, "Workup", start_line=300, end_line=320),
        ]
        payload = [
            json.dumps(section0_steps),
            json.dumps(section1_steps),
            json.dumps(section2_steps),
        ]
        sections = parse_r1_json(json.dumps(payload), source_label="double-encoded")
        self.assertEqual(len(sections), 3)
        # Document-global line numbers preserved (not section-relative).
        self.assertEqual(sections[0].start_line, 166)
        self.assertEqual(sections[0].end_line, 186)
        self.assertEqual(sections[0].section_label, "Dissolve")
        self.assertEqual(sections[1].start_line, 187)
        self.assertEqual(sections[2].start_line, 300)
        self.assertEqual(sections[2].section_label, "Workup")
        # Sorted by document order even when step_index resets per section.
        self.assertEqual(
            [s.section_index for s in sections],
            [0, 1, 0],
        )

    def test_already_decoded_consolidated_lists(self) -> None:
        payload = [
            [_step(0, "First", start_line=1, end_line=5)],
            [],
            [_step(0, "Third", start_line=50, end_line=60)],
        ]
        sections = parse_r1_json(json.dumps(payload), source_label="decoded-lists")
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].section_label, "First")
        self.assertEqual(sections[1].section_label, "Third")
        self.assertEqual(sections[1].start_line, 50)

    def test_empty_payload_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_r1_json("[]", source_label="empty")
        with self.assertRaises(ValueError):
            parse_r1_json("", source_label="blank")

    def test_parse_r1_step_dicts_keeps_parent_section_label(self) -> None:
        payload = [
            {
                "step_index": 0,
                "step_label": "Step A",
                "section_label": "Example 1",
                "start_line": 10,
                "end_line": 20,
            }
        ]
        steps = parse_r1_step_dicts(json.dumps(payload), source_label="flat")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["section_label"], "Example 1")
        self.assertEqual(steps[0]["step_label"], "Step A")
        # Section adapter still maps step_label → section_label for pre-pass UI
        sections = parse_r1_json(json.dumps(payload), source_label="flat")
        self.assertEqual(sections[0].section_label, "Step A")


if __name__ == "__main__":
    unittest.main()
