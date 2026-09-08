"""Offline regression tests for paired revision estimates and corruption gates."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import revision_paired_analysis as analysis


class RevisionTests(unittest.TestCase):
    def test_cluster_is_item_not_model_or_response(self):
        cells = [("item1", 1, 1, 0, 1), ("item2", 0, 1, 0, 1)]
        original = analysis.cluster_estimate(cells, 500, 9)
        repeated = analysis.cluster_estimate(cells*30, 500, 9)
        self.assertEqual(repeated["clusters"], 2)
        for field in ("difference_pp", "ci95_low_pp", "ci95_high_pp"):
            self.assertEqual(original[field], repeated[field])

    def test_empty_is_not_zero(self):
        result = analysis.cluster_estimate([], 100)
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("difference_pp", result)

    def test_invalid_denominator(self):
        with self.assertRaises(ValueError):
            analysis.cluster_estimate([("i", 1, 0, 0, 1)], 100)

    def test_paired_difference_sign(self):
        result = analysis.cluster_estimate([("i", 0, 1, 1, 1)], 100)
        self.assertEqual(result["difference_pp"], -100)
        self.assertEqual(result["ci95_high_pp"], -100)

    def test_duplicate_keys_fail(self):
        with self.assertRaises(ValueError):
            analysis.unique([{"id": 1}, {"id": 1}], lambda r: r["id"])

    def test_corrupt_prefix_never_returned(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(analysis, "source_name", lambda p: p.name):
            path = Path(tmp)/"bad.gz"
            path.write_bytes(gzip.compress(b'{"id":1}\nnot-json\n'))
            log = []
            self.assertIsNone(analysis.strict_read(path, log))
            self.assertEqual(log[0]["accepted_rows"], 0)
            self.assertEqual(log[0]["readable_prefix_rows"], 1)

    def test_gzip_footer_must_be_valid(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(analysis, "source_name", lambda p: p.name):
            path = Path(tmp)/"truncated.gz"
            path.write_bytes(gzip.compress(b'{"id":1}\n')[:-4])
            log = []
            self.assertIsNone(analysis.strict_read(path, log))
            self.assertEqual(log[0]["status"], "rejected")

    def test_count_gate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(analysis, "source_name", lambda p: p.name):
            path = Path(tmp)/"good.gz"
            path.write_bytes(gzip.compress(b'{"id":1}\n'))
            self.assertIsNone(analysis.strict_read(path, [], 2))
            self.assertEqual(len(analysis.strict_read(path, [], 1)), 1)

    def test_pair_excludes_invalid_and_mismatched_initials(self):
        a = [dict(item_id="i", valid=True, initial_answer="A", event=1),
             dict(item_id="j", valid=False, initial_answer="A", event=0)]
        b = [dict(item_id="i", valid=True, initial_answer="B", event=1),
             dict(item_id="j", valid=True, initial_answer="A", event=1)]
        result = analysis.paired_contrast(a, b, lambda r: r["item_id"], 100, 0)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["exclusions"]["initial_answer_mismatch"], 1)
        self.assertEqual(result["exclusions"]["ineligible_or_unparsed_pair"], 1)

    def test_any_wrong_is_not_tracked_wrong(self):
        item = dict(correct_answer="A", choices={"A": "a", "B": "b", "C": "c"},
                    verifiability="GT", injected_wrong_answer_state="wrong",
                    tracked_answer_states={"wrong": {"native_label": "B"}})
        row = dict(initial_answer="A", final_answer="C", initial_correct=True,
                   correct_answer="A", eligible=True, truth_departure=True)
        result = analysis.normalize_trial(row, "GT", "static", item)
        self.assertEqual(result["event"], 1)
        self.assertEqual(result["tracked_event"], 0)
        row["final_answer"] = None
        result = analysis.normalize_trial(row, "GT", "static", item)
        self.assertFalse(result["valid"])
        self.assertTrue(result["legacy_eligible"])
        self.assertEqual(result["reason"], "invalid_final")

    def test_sub_neutral_bias_and_pair_metrics(self):
        variants = {"neutral": {}, "value_relevant_A": {}, "value_relevant_B": {}}
        panels = {"NGT": {"i": {"context_variants": variants}}}
        rows = [dict(branch="NGT", model="m", item_id="i", variant=v,
                     cue_type="value_relevant" if v != "neutral" else None,
                     direction=v[-1] if v != "neutral" else None, answer="A") for v in variants]
        with patch.object(analysis, "MAIN_MODELS", ["m"]), patch.object(analysis, "CUES", ("value_relevant",)):
            result = analysis.context_analysis(rows, panels, 100, 1)
        rates = {r["metric"]: r for r in result["rates"] if r["branch"] == "SUB" and r["cue"] == "all" and r["model"] == "all"}
        self.assertEqual(rates["both"]["rate_a_pct"], 0)
        self.assertEqual(rates["marginal"]["rate_a_pct"], 50)
        self.assertEqual(result["neutral_A"][0]["rate_a_pct"], 100)

    def test_exact_rank_test_matches_existing_helper_with_ties(self):
        x, y = [1, 2, 3, 4], [1, 1, 3, 2]
        expected = analysis.aggregate.spearman_exact_permutation(x, y)
        rho, p, n = analysis.exact_rank_test(x, y)
        self.assertAlmostEqual(rho, expected[0])
        self.assertAlmostEqual(p, expected[1])
        self.assertEqual(n, 24)

    def test_sub_missing_direction_is_not_a_failed_conformity_event(self):
        variants = {"neutral": {}, "value_relevant_A": {}, "value_relevant_B": {}}
        panels = {"NGT": {"i": {"context_variants": variants}}}
        rows = [dict(branch="NGT", model="m", item_id="i", variant=v,
                     cue_type="value_relevant" if v != "neutral" else None,
                     direction=v[-1] if v != "neutral" else None,
                     answer=None if v.endswith("_A") else "B") for v in variants]
        with patch.object(analysis, "MAIN_MODELS", ["m"]), patch.object(analysis, "CUES", ("value_relevant",)):
            result = analysis.context_analysis(rows, panels, 100, 1)
        rates = {r["metric"]: r for r in result["rates"] if r["branch"] == "SUB" and r["cue"] == "all" and r["model"] == "all"}
        self.assertEqual(rates["both"]["status"], "unavailable")
        self.assertEqual(rates["marginal_all_valid_directed"]["denom_a"], 1)
        self.assertEqual(rates["marginal_all_valid_directed"]["rate_a_pct"], 100)

    def test_neutral_missing_does_not_disqualify_valid_AB_pair(self):
        variants = {"neutral": {}, "value_relevant_A": {}, "value_relevant_B": {}}
        panels = {"NGT": {"i": {"context_variants": variants}}}
        rows = [dict(branch="NGT", model="m", item_id="i", variant=v,
                     cue_type="value_relevant" if v != "neutral" else None,
                     direction=v[-1] if v != "neutral" else None,
                     answer=v[-1] if v != "neutral" else None) for v in variants]
        with patch.object(analysis, "MAIN_MODELS", ["m"]), patch.object(analysis, "CUES", ("value_relevant",)):
            result = analysis.context_analysis(rows, panels, 100, 1)
        rates = {r["metric"]: r for r in result["rates"] if r["branch"] == "SUB" and r["cue"] == "all" and r["model"] == "all"}
        self.assertEqual(rates["both"]["rate_a_pct"], 100)
        self.assertEqual(rates["neutral_matched_marginal_lift"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
