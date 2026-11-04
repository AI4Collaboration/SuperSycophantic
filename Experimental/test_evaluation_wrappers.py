import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
ROOT = Path(__file__).resolve().parent


@unittest.skipUnless(POWERSHELL, "PowerShell is not installed")
class EvaluationWrapperTests(unittest.TestCase):
    def calls(self, filename, arguments):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "Experimental"
            folder.mkdir()
            wrapper = folder / filename
            shutil.copy2(ROOT / filename, wrapper)
            quoted = str(wrapper).replace("'", "''")
            script = (
                "$global:Calls = [System.Collections.Generic.List[object]]::new(); "
                "function python { $global:Calls.Add(@($args)); $global:LASTEXITCODE = 0 }; "
                f"& '{quoted}' {arguments}; "
                "Write-Output ('WRAPPER_CALLS=' + (ConvertTo-Json -InputObject @($global:Calls.ToArray()) -Depth 10 -Compress))"
            )
            result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
                                    cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            line = next(line for line in result.stdout.splitlines() if line.startswith("WRAPPER_CALLS="))
            return json.loads(line.split("=", 1)[1])

    def test_context_dry_run_preserves_panels_and_honors_zero_limit(self):
        calls = self.calls("run_context_eval_and_report.ps1", "-DryRun -RebuildPanels -Models smoke -MaxGt 0 -MaxNgt 2")
        self.assertFalse(any("build_supersycophantic_context_panels.py" in call[0] for call in calls))
        self.assertFalse(any("report_results.py" in call[0] for call in calls))
        run = next(call for call in calls if call[0].endswith("run_context.py"))
        self.assertIn("--dry-run", run)
        self.assertEqual(run[run.index("--max-gt") + 1], "0")
        self.assertEqual(run[run.index("--max-ngt") + 1], "2")
        self.assertEqual(run[run.index("--max-attempts") + 1], "8")

    def test_trigger_dry_run_never_reports_or_plots(self):
        calls = self.calls("run_trigger_eval_and_report.ps1", "-DryRun -Models smoke -MaxItems 2")
        self.assertFalse(any("report_results.py" in call[0] or "plot_" in call[0] for call in calls))
        runs = [call for call in calls if call[0].endswith("run.py")]
        self.assertEqual(len(runs), 14)
        for run in runs:
            self.assertIn("--dry-run", run)
            self.assertEqual(run[run.index("--max-attempts") + 1], "8")

    def test_default_context_uses_frozen_data(self):
        calls = self.calls("run_context_eval_and_report.ps1", "-Models smoke")
        self.assertFalse(any("build_supersycophantic_context_panels.py" in call[0] for call in calls))
        run = next(call for call in calls if call[0].endswith("run_context.py"))
        self.assertNotIn("--max-gt", run)
        self.assertNotIn("--max-ngt", run)

    def test_explicit_context_rebuild_is_available(self):
        calls = self.calls("run_context_eval_and_report.ps1", "-RebuildPanels -Models smoke")
        self.assertTrue(any("build_supersycophantic_context_panels.py" in call[0] for call in calls))


if __name__ == "__main__":
    unittest.main()
