"""Offline checks for result reporting without manuscript drafting advice."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rebuttal_artificial_analysis_index as capability
import rebuttal_trace_findings as traces
import report_results as reports
import revision_paired_analysis as paired


class ReportProseTests(unittest.TestCase):
    def test_paired_report_retains_estimates_and_metric_definitions(self):
        rates = [
            dict(model="all", branch="SUB", cue=cue, metric=metric,
                 rate_a_pct=value, ci95_low_pp=12.5, ci95_high_pp=37.5, denom_a=8)
            for cue in ("all",) + paired.CUES
            for metric, value in (("both", 25.0), ("marginal", 62.5))
        ]
        result = dict(cohorts={"fixture": dict(
            context_status="accepted", trigger_status="unavailable",
            context=dict(rates=rates))}, capability=dict(rows=[dict(
                metric="fixture_metric", exclude_Command_R=True,
                n_models=8, rho=-0.5, exact_p_two_sided=0.125)]))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            paired.write_report(out, result)
            text = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("|all|25.000% [12.500, 37.500]|62.500%|8|", text)
        self.assertIn("|fixture_metric|True|8|-0.5000|0.125000|", text)
        for definition in ("P(y_A=A and y_B=B)", "initial_correct", "SUB is eligible",
                           "base-item cluster bootstrap", "unadjusted 95%", "provenance.json"):
            self.assertIn(definition, text)
        for instruction in ("Manuscript corrections", "Intro 23.8%", "after original restoration",
                            "Replace response-wise", "Do not retain claims"):
            self.assertNotIn(instruction, text)

    def test_trace_report_does_not_infer_rankings_or_effects(self):
        for outcome, impression in ((10.0, 90.0), (90.0, 10.0), (0.0, 0.0)):
            with self.subTest(outcome=outcome, impression=impression):
                context = [dict(branch="SUB", cue_type=cue, event_n=int(value),
                                denominator=100, rate_pct=value)
                           for cue, value in (("outcome_relevant", outcome),
                                              ("impression_relevant", impression))]
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "summary.md"
                    traces.write_summary(path, inventory=[dict(
                        corpus="fixture", branch="SUB", mode="adaptive", records=100,
                        truncated=False)], full_events=[], markers=[], orientations=[],
                        temporal_rounds=[], context_events=context, small_generators=[])
                    text = path.read_text(encoding="utf-8")
                for row in context:
                    self.assertIn(traces.fmt_count_rate(row), text)
                self.assertIn("100 records; truncated=False", text)
                self.assertIn("not available", text)
                self.assertIn("trace_marker_summary.csv", text)
                for claim in ("strongest", "weakest", "more often", "sufficient",
                              "cleaner", "Suggested Rebuttal Framing", "Findings To Use"):
                    self.assertNotIn(claim, text)

    def test_capability_report_keeps_correlations_without_fixed_conclusion(self):
        for rho in (-1.0, 0.0, 1.0):
            with self.subTest(rho=rho), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                scores = [dict(model="fixture-model", primary_score=42.0,
                               primary_aa_name="Fixture", primary_match_note="")]
                correlations = [dict(policy="primary_strict_exact_n8",
                                     metric="context_obj_truth_departure",
                                     spearman_rho=rho, n=8)]
                with patch.object(capability, "REPORT_DIR", out):
                    capability.write_summary(scores, correlations)
                text = (out / "artificial_analysis_index_summary.md").read_text(encoding="utf-8")
                self.assertIn(f"Spearman rho={rho}, n=8", text)
                self.assertIn("fixture-model: 42 (Fixture)", text)
                self.assertIn("Command-R=2", text)
                self.assertNotIn("Interpretation:", text)
                self.assertNotIn("Capability alone is insufficient", text)

    def test_context_report_keeps_results_commands_completeness_and_previews(self):
        summary = {"models": {"fixture/model": {
            "gt": dict(correct_to_incorrect_rate=0.25, truth_preservation_rate=0.75,
                       answer_change_rate=0.5),
            "ngt": {"all": dict(user_answer_agreement=dict(lift=0.125),
                                answer_change_rate=0.375),
                    "paired_directionality": dict(answer_change_by_user_direction_rate=0.625),
                    "by_cue": {"outcome_relevant": dict(user_answer_agreement=dict(lift=0.125))}},
        }}}
        records = [dict(run_type="context", model="fixture/model", branch="GT",
                        item_id="complete-item", response_text="A", answer="A",
                        confidence=4, response_metadata={}),
                   dict(run_type="context", model="fixture/model", branch="NGT",
                        item_id="failed-item", response_error="fixture timeout")]
        command = "python Experimental/run_context.py --help"
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(reports, "args", SimpleNamespace(run_id="fixture"), create=True), \
                    patch.object(reports, "load_context_result_path", return_value=None), \
                    patch.object(reports, "audit_summary_lines", return_value=[]), \
                    patch.object(reports, "png_bar_chart", return_value=True) as plot:
                reports.write_report(out, None, summary, records,
                                     include_context=True, commands=[command])
            text = (out / "report.md").read_text(encoding="utf-8")
            self.assertEqual(plot.call_count, 5)
            for call in plot.call_args_list:
                path = call.args[0]
                self.assertIn(f"![{path.stem}]({path.name})", text)
        for value in ("25.0%", "75.0%", "12.5%", "37.5%", "62.5%", "4.00/5"):
            self.assertIn(value, text)
        for retained in (command, "Planned keys: 2", "Completed keys: 1 (50.0%)",
                         "Residual incomplete keys: 1 (50.0%)", "failed-item",
                         "fixture timeout", "completed rows only", "## Figure Previews"):
            self.assertIn(retained, text)
        self.assertNotIn("Paper Results-Section Writing Plan", text)
        self.assertNotIn("## Figure Plan", text)

    def test_empty_context_report_keeps_missing_result_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(reports, "args", SimpleNamespace(run_id="fixture"), create=True), \
                    patch.object(reports, "load_context_result_path", return_value=None), \
                    patch.object(reports, "audit_summary_lines", return_value=[]):
                reports.write_report(out, None, {}, [], include_context=True)
            text = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("Planned keys: 0", text)
        self.assertIn("no completed model estimates", text)
        self.assertIn("No context figures were generated.", text)
        self.assertIn("run_context_eval_and_report.ps1", text)
        self.assertNotIn("Writing Plan", text)


if __name__ == "__main__":
    unittest.main()
