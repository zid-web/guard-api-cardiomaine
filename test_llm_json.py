import json
import unittest

from llm_json import parse_llm_json, repair_common_json_issues, _close_truncated_json


class TestLlmJson(unittest.TestCase):
    def test_plain_json(self):
        data = parse_llm_json('{"rows": [], "warnings": []}')
        self.assertEqual(data["rows"], [])

    def test_markdown_fence(self):
        data = parse_llm_json('```json\n{"a": 1}\n```')
        self.assertEqual(data["a"], 1)

    def test_trailing_comma(self):
        data = parse_llm_json('{"a": 1, "b": [2, 3,],}')
        self.assertEqual(data["b"], [2, 3])

    def test_prefix_suffix_noise(self):
        data = parse_llm_json('Voici le résultat:\n{"ok": true}\nMerci')
        self.assertTrue(data["ok"])

    def test_truncated_closes(self):
        broken = '{"rows": [{"row_label": "Garde Nuit", "cells": [{"day_name": "LUNDI"'
        closed = _close_truncated_json(broken)
        self.assertIsNotNone(closed)
        data = json.loads(closed)
        self.assertEqual(data["rows"][0]["row_label"], "Garde Nuit")

    def test_truncated_parse_llm_json(self):
        # Simule la troncature mid-value (cas fréquent avec max_tokens)
        broken = (
            '{"week_label": "SEMAINE 30", "dates_by_day": {}, "rows": ['
            '{"row_label": "Garde Nuit", "matched_row_key": "Garde Nuit", '
            '"cells": [{"day_name": "LUNDI", "doctors": ["W"], "raw_text": "W", '
            '"confidence": "high"}, {"day_name": "MARDI", "doctors": ["P"], '
            '"raw_text": "P/'
        )
        data = parse_llm_json(broken)
        self.assertEqual(data["week_label"], "SEMAINE 30")
        self.assertEqual(data["rows"][0]["cells"][0]["doctors"], ["W"])

    def test_python_literals(self):
        repaired = repair_common_json_issues('{"a": True, "b": None,}')
        self.assertEqual(json.loads(repaired), {"a": True, "b": None})


if __name__ == "__main__":
    unittest.main()
