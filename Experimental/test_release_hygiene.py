import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from audit_release import inspect_file, inspect_revision, private_path
from run import load_dotenv


class ReleaseHygieneTests(unittest.TestCase):
    def test_private_and_generated_paths_are_rejected(self):
        for name in [".env", ".env.production", "nested/.env", "key.pem", "id_ed25519",
                     "settings.local.json", "Experimental/results/run.jsonl.gz",
                     "Experimental/IAA/llm_judge_run/raw.json", "main.tex", "sections/intro.tex",
                     "images/Figure1.png", "label_studio/label_studio_export_1.json",
                     "Experimental/IAA/Human-LLM-Judge-IAA-100.json",
                     "label_studio/label_studio_iaa_results.json",
                     "label_studio/label_studio_iaa_summary.md",
                     "logs/run.txt", "nested/trajectories/run.json", "traces/run.csv",
                     "run.jsonl.gz", "run.log.1", "run_stdout.txt", "run_stderr.txt",
                     "run_trajectory.json", "run_transcripts.json", "run_responses.jsonl",
                     "trajectory.json", "trajectories_model.jsonl", "transcript.json",
                     "responses.jsonl"]:
            with self.subTest(name=name):
                self.assertTrue(private_path(name))

    def test_public_panels_code_and_template_are_allowed(self):
        for name in [".env.example", "Experimental/run.py", "LICENSE",
                     "Experimental/data/rebuttal_positive_control/positive_update_gt_hard40.jsonl",
                     "Experimental/figures/trigger_confidence_trajectory.py",
                     "Experimental/revision_trace_audit.py", "images/logos/openai.png"]:
            with self.subTest(name=name):
                self.assertFalse(private_path(name))

    def test_credentials_are_reported_without_values(self):
        secrets = ["sk-" + "x" * 36, "ghp_" + "x" * 36, "AKIA" + "X" * 16,
                   "xoxb-" + "1" * 32, "-----BEGIN " + "PRIVATE KEY-----"]
        for secret in secrets:
            with self.subTest(kind=secret[:4]):
                findings = inspect_file("example.py", ("\n" + secret).encode())
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0].line, 2)
                self.assertNotIn(secret, str(findings))

    def test_private_urls_and_home_paths_are_rejected(self):
        values = ["https://user:" + "credential@example.test",
                  "https://git@" + "git.overleaf.com/project",
                  "C:" + "/" + "Users/example/private", "/home/" + "example/private"]
        for value in values:
            self.assertTrue(inspect_file("README.md", value.encode()))

    def test_binary_assets_are_not_decoded_as_text(self):
        self.assertEqual(inspect_file("images/logos/model.png", b"\x89PNG\0\xff"), [])

    def test_renaming_responses_does_not_make_them_release_data(self):
        for record in [{"messages": [{"role": "assistant", "content": "answer"}]},
                       {"turns": [{"assistant": "answer"}]},
                       {"initial_answer": "A", "final_answer": "B"},
                       {"raw_response": {"choices": []}}]:
            for suffix in ["json", "jsonl"]:
                with self.subTest(record=record, suffix=suffix):
                    findings = inspect_file(f"Experimental/data/example.{suffix}",
                                            json.dumps(record).encode())
                    self.assertIn("model response or conversation output", [f.reason for f in findings])
        panel = {"question": "Question", "choices": {"A": "yes", "B": "no"},
                 "correct_answer": "A", "initial_prompt": "Choose one"}
        self.assertEqual(inspect_file("Experimental/data/example.json", json.dumps(panel).encode()), [])

    def test_malformed_release_json_fails(self):
        self.assertEqual(inspect_file("data.json", b"{")[0].reason,
                         "invalid structured release data")

    def test_revision_checks_git_blobs_not_working_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            path = root / "example.py"
            path.write_text("sk-" + "x" * 36, encoding="utf-8")
            subprocess.run(["git", "add", "example.py"], cwd=root, check=True)
            tree = subprocess.check_output(["git", "write-tree"], cwd=root).decode().strip()
            path.write_text("safe local content", encoding="utf-8")
            names, findings = inspect_revision(root, tree)
            self.assertEqual(names, ["example.py"])
            self.assertEqual([f.reason for f in findings], ["provider credential"])

    def test_generated_outputs_are_ignored(self):
        root = Path(__file__).resolve().parents[1]
        paths = ["run.log", "run.log.1", "nested/logs/run.txt", "run_stdout.txt",
                 "trajectories/model.json", "nested/transcripts/run.jsonl",
                 "trajectory.json", "responses.jsonl", "run.jsonl.gz",
                 "Experimental/IAA/Human-LLM-Judge-IAA-100.json",
                 "label_studio/label_studio_iaa_results.json",
                 "label_studio/label_studio_iaa_summary.md"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=target, check=True)
            (target / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
            result = subprocess.run(
                ["git", "check-ignore", "--no-index", "-z", "--stdin"],
                cwd=target, input=("\0".join(paths) + "\0").encode(),
                capture_output=True, check=True,
            )
            self.assertEqual(set(result.stdout.decode().split("\0")) - {""}, set(paths))

    def test_revision_rejects_generated_paths_even_with_safe_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "trajectories").mkdir()
            (root / "trajectories/run.json").write_text("{}", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            tree = subprocess.check_output(["git", "write-tree"], cwd=root).decode().strip()
            _, findings = inspect_revision(root, tree)
            self.assertEqual([f.reason for f in findings], ["private or generated artifact"])

    def test_dotenv_does_not_override_explicit_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('OPENROUTER_API_KEY="local-example"\nOPENROUTER_ONLY=1\n', encoding="utf-8")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "environment-example"}, clear=True):
                load_dotenv(path)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "environment-example")
                self.assertEqual(os.environ["OPENROUTER_ONLY"], "1")


if __name__ == "__main__":
    unittest.main()
