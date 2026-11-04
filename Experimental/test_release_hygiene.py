import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_release import inspect_file, private_path
from run import load_dotenv


class ReleaseHygieneTests(unittest.TestCase):
    def test_private_and_generated_paths_are_rejected(self):
        for name in [".env", ".env.production", "nested/.env", "key.pem", "id_ed25519",
                     "settings.local.json", "Experimental/results/run.jsonl.gz",
                     "Experimental/IAA/llm_judge_run/raw.json", "main.tex", "sections/intro.tex",
                     "images/Figure1.png", "label_studio/label_studio_export_1.json"]:
            with self.subTest(name=name):
                self.assertTrue(private_path(name))

    def test_public_panels_code_and_template_are_allowed(self):
        for name in [".env.example", "Experimental/run.py", "LICENSE",
                     "Experimental/data/rebuttal_positive_control/positive_update_gt_hard40.jsonl",
                     "Experimental/IAA/Human-LLM-Judge-IAA-100.json", "images/logos/openai.png"]:
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
