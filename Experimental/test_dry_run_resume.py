"""Offline regression tests for planning records and live-run admission."""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import run


MODEL = "openai/gpt-5.4-mini"
RUNNERS = {
    "first-turn": run.run_first_turn_async,
    "eval": run.run_eval_async,
    "temporal": run.run_temporal_async,
}
LOADERS = (
    (run.load_completed_first_turn_trials, run.first_turn_key),
    (run.load_completed_trials, run.trial_key),
    (run.load_completed_temporal_trials, run.temporal_key),
)


def saved_record(item_id, **fields):
    return {
        "item_id": item_id,
        "model": MODEL,
        "trigger": "authority",
        "tone": "moderate",
        "tone_sequence": ["mild", "moderate", "strong"],
        "trigger_prompt_variant": "static",
        "first_response_text": "Live response",
        "initial_answer": "A",
        "initial_confidence": 4,
        "initial_correct": True,
        "truth_departure": True,
        **fields,
    }


class DryRunResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        client = patch.object(run, "OpenRouterAsyncClient")
        chat = patch.object(run, "async_openrouter_chat", new_callable=AsyncMock,
                            side_effect=AssertionError("Model calls are forbidden"))
        public_get = patch.object(run.urllib.request, "urlopen",
                                  side_effect=AssertionError("Network calls are forbidden"))
        self.client = client.start()
        self.chat = chat.start()
        self.public_get = public_get.start()
        for mock_patch in (client, chat, public_get):
            self.addCleanup(mock_patch.stop)
        self.addCleanup(self.chat.assert_not_called)
        self.addCleanup(self.public_get.assert_not_called)

    def write_rows(self, path, rows):
        with run.open_text(path, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path.read_bytes()

    def arguments(self, command, output, dry_run=False):
        argv = ["run.py", command, "--models", MODEL, "--output", str(output)]
        if command != "first-turn":
            argv.extend(["--triggers", "authority"])
        if dry_run:
            argv.append("--dry-run")
        with patch.object(sys, "argv", argv):
            return run.parse_args()

    def execute(self, args):
        with redirect_stdout(io.StringIO()):
            return asyncio.run(RUNNERS[args.command](args, self.root))

    def test_completion_loaders_ignore_plans_and_preserve_live_rows(self):
        legacy = saved_record("legacy")
        live = saved_record("live", dry_run=False)
        plan = saved_record("planned", dry_run=True)
        for suffix in (".jsonl", ".jsonl.gz"):
            path = self.root / ("records" + suffix)
            self.write_rows(path, [plan, legacy, live])
            for loader, key in LOADERS:
                with self.subTest(suffix=suffix, loader=loader.__name__):
                    self.assertEqual(loader(path), {key(legacy), key(live)})

    def test_internal_caches_and_switch_counts_ignore_plans(self):
        plan = saved_record("shared", dry_run=True, first_response_text="Plan", initial_answer="B")
        live = saved_record("shared")
        plan_only = saved_record("planned", dry_run=True)
        for suffix in (".jsonl", ".jsonl.gz"):
            path = self.root / ("cache" + suffix)
            self.write_rows(path, [plan, plan_only, live])
            for loader, key in (
                (run.load_existing_initials_and_switches, run.trial_key),
                (run.load_existing_temporal_initials_and_switches, run.temporal_key),
            ):
                with self.subTest(suffix=suffix, loader=loader.__name__):
                    cache, switches = loader(path, {MODEL: {key(live), key(plan_only)}})
                    self.assertEqual(set(cache), {("shared", MODEL)})
                    self.assertEqual(cache[("shared", MODEL)]["first_text"], "Live response")
                    self.assertEqual(cache[("shared", MODEL)]["initial_answer"], "A")
                    self.assertEqual(switches, {MODEL: 1})

    def test_external_initial_cache_ignores_plans_for_each_text_field(self):
        for suffix in (".jsonl", ".jsonl.gz"):
            for text_field in ("first_response_text", "initial_response_text", "response_text"):
                with self.subTest(suffix=suffix, text_field=text_field):
                    live = saved_record("shared", run_type="context", variant="value_relevant_A")
                    del live["first_response_text"]
                    live[text_field] = "Live response"
                    plan = {**live, "dry_run": True, text_field: "Plan", "initial_answer": "B"}
                    plan_only = {**plan, "item_id": "planned"}
                    path = self.root / ("external" + suffix)
                    self.write_rows(path, [plan, plan_only, live])
                    cache = run.load_initial_cache_from_jsonl(path)
                    key = ("shared__value_relevant_A", MODEL)
                    self.assertEqual(set(cache), {key})
                    self.assertEqual(cache[key]["first_text"], "Live response")
                    self.assertEqual(cache[key]["initial_answer"], "A")

    def test_first_turn_correct_count_ignores_plans(self):
        rows = [saved_record("planned", dry_run=True), saved_record("live")]
        for suffix in (".jsonl", ".jsonl.gz"):
            with self.subTest(suffix=suffix):
                path = self.root / ("first" + suffix)
                self.write_rows(path, rows)
                planned = {MODEL: {run.first_turn_key(row) for row in rows}}
                self.assertEqual(run.load_existing_first_turn_correct(path, planned), {MODEL: 1})

    def test_output_admission_allows_fresh_empty_and_same_mode_files(self):
        fresh = self.root / "missing" / "fresh.jsonl.gz"
        for dry_run in (False, True):
            run.validate_output_mode(fresh, dry_run)
        self.assertFalse(fresh.parent.exists())
        for suffix in (".jsonl", ".jsonl.gz"):
            path = self.root / ("compatible" + suffix)
            for rows, dry_run in (
                ([], False), ([], True),
                ([saved_record("legacy"), saved_record("live", dry_run=False)], False),
                ([saved_record("planned", dry_run=True)], True),
            ):
                with self.subTest(suffix=suffix, dry_run=dry_run, rows=len(rows)):
                    before = self.write_rows(path, rows)
                    run.validate_output_mode(path, dry_run)
                    self.assertEqual(path.read_bytes(), before)

    def test_output_admission_rejects_mixed_files_without_rewriting(self):
        rows = [saved_record("live"), saved_record("planned", dry_run=True)]
        for suffix in (".jsonl", ".jsonl.gz"):
            path = self.root / ("mixed" + suffix)
            before = self.write_rows(path, rows)
            for dry_run in (False, True):
                with self.subTest(suffix=suffix, dry_run=dry_run):
                    with self.assertRaisesRegex(SystemExit, "--output.*fresh output path") as error:
                        run.validate_output_mode(path, dry_run)
                    self.assertIn(str(path), str(error.exception))
                    self.assertEqual(path.read_bytes(), before)

    def test_output_admission_checks_plain_gzip_fallback(self):
        for saved_suffix, requested_suffix in ((".jsonl", ".jsonl.gz"), (".jsonl.gz", ".jsonl")):
            with self.subTest(saved_suffix=saved_suffix):
                stem = "fallback_" + ("plain" if saved_suffix == ".jsonl" else "gzip")
                saved = self.root / (stem + saved_suffix)
                requested = self.root / (stem + requested_suffix)
                before = self.write_rows(saved, [saved_record("planned", dry_run=True)])
                with self.assertRaisesRegex(SystemExit, "dry-run planning records"):
                    run.validate_output_mode(requested, False)
                self.assertEqual(saved.read_bytes(), before)
                self.assertFalse(requested.exists())

    def test_all_runners_reject_opposite_mode_before_credentials_or_writes(self):
        for command in RUNNERS:
            for suffix in (".jsonl", ".jsonl.gz"):
                for dry_run in (False, True):
                    with self.subTest(command=command, suffix=suffix, dry_run=dry_run):
                        path = self.root / (command + suffix)
                        before = self.write_rows(path, [saved_record("opposite", dry_run=not dry_run)])
                        files = set(self.root.iterdir())
                        args = self.arguments(command, path, dry_run)
                        with patch.object(run, "openrouter_api_key_for_run") as credentials, \
                             patch.object(run, "read_jsonl") as read_items:
                            with self.assertRaisesRegex(SystemExit, "--output.*fresh output path"):
                                self.execute(args)
                            credentials.assert_not_called()
                            read_items.assert_not_called()
                        self.client.assert_not_called()
                        self.assertEqual(path.read_bytes(), before)
                        self.assertEqual(set(self.root.iterdir()), files)

    def test_locked_admission_releases_lock_on_conflict(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                path = self.root / "locked.jsonl.gz"
                before = self.write_rows(path, [saved_record("opposite", dry_run=not dry_run)])
                with patch.object(run.atexit, "register") as register:
                    with self.assertRaisesRegex(SystemExit, "fresh output path"):
                        run.acquire_output_lock(path, dry_run=dry_run)
                    register.assert_not_called()
                self.assertFalse(Path(str(path) + ".lock").exists())
                self.assertEqual(path.read_bytes(), before)

    def test_all_runners_recheck_mode_under_lock(self):
        for command in RUNNERS:
            with self.subTest(command=command):
                path = self.root / (command + "_race.jsonl.gz")
                snapshots = []

                def create_planning_output(*args):
                    snapshots.append(self.write_rows(path, [saved_record("planned", dry_run=True)]))
                    return "unused-test-key"

                with patch.object(run, "openrouter_api_key_for_run", side_effect=create_planning_output), \
                     patch.object(run, "read_jsonl", return_value=[]):
                    with self.assertRaisesRegex(SystemExit, "dry-run planning records"):
                        self.execute(self.arguments(command, path))
                self.client.assert_not_called()
                self.assertEqual(path.read_bytes(), snapshots[0])
                self.assertFalse(Path(str(path) + ".lock").exists())

    def test_fresh_dry_runs_work_but_cannot_resume_as_live(self):
        item = {"id": "fixture", "question": "What is 1 + 1?",
                "choices": {"A": "2", "B": "3"}, "correct_answer": "A"}
        for command, (loader, _) in zip(RUNNERS, LOADERS):
            with self.subTest(command=command):
                path = self.root / (command + "_plan.jsonl.gz")
                args = self.arguments(command, path, dry_run=True)
                try:
                    with patch.object(run, "read_jsonl", return_value=[item]), \
                         patch.object(run, "openrouter_api_key_for_run", return_value=""):
                        self.assertEqual(self.execute(args), 0)
                finally:
                    run.release_output_lock(Path(str(path) + ".lock"))
                with run.open_text(path, "rt") as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
                self.assertEqual(len(rows), 1)
                self.assertIs(rows[0]["dry_run"], True)
                self.assertEqual(loader(path), set())
                before = path.read_bytes()
                args.dry_run = False
                with patch.object(run, "openrouter_api_key_for_run") as credentials:
                    with self.assertRaisesRegex(SystemExit, "dry-run planning records"):
                        self.execute(args)
                    credentials.assert_not_called()
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
