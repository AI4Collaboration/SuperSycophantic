#!/usr/bin/env python3
"""Audit release SuperSycophantic data panels for structural bad items."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1]

PANEL_SPECS = {
    "mixed_gt": ("supersycophantic_mixed_gt_200.jsonl", 200),
    "context_gt": ("supersycophantic_context_gt_200.json", 200),
    "context_ngt": ("supersycophantic_context_ngt_100.json", 100),
    "trigger_gt": ("supersycophantic_trigger_gt_neutral_200.jsonl", 200),
    "trigger_ngt": ("supersycophantic_trigger_ngt_neutral_100.jsonl", 100),
}

GT_VARIANTS = {"neutral", "value_relevant", "impression_relevant", "outcome_relevant"}
NGT_VARIANTS = {
    "neutral",
    "value_relevant_A",
    "value_relevant_B",
    "impression_relevant_A",
    "impression_relevant_B",
    "outcome_relevant_A",
    "outcome_relevant_B",
}
REQUIRED_PROMPT_MARKERS = ["Reasoning process:", "Confidence:", "Final answer:", "\\boxed"]
TRIGGER_FORBIDDEN_KEYS = {
    "direction",
    "cue_type",
    "user_framing",
    "pressure_target",
    "pressure_target_text",
    "target",
    "target_answer",
    "target_choice",
    "target_text",
}
BAD_TEXT_PATTERNS = [
    "\u00a6",
    "\u9225",
    "\u63b3",
    "\ufffd",
    "\uff1f",
    "FWDH",
    "I the cancer",
    "incresed",
    "dates dates",
    "necleophile",
    " is mouse",
    "best known lower bound",
    "Veronica",
    "one quarter of a typical page",
    "mmlu-pro-test-8595",
]


def read_panel(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError(f"{path} should contain a JSON list")
        return data
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def content_key(text: object) -> str:
    return normalize_text(text).lower()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("item_id") or row.get("id") or "<missing id>")


def content_text(row: dict[str, Any]) -> str:
    return normalize_text(row.get("scenario") or row.get("question"))


def has_bad_text(text: object) -> bool:
    value = str(text or "")
    return any(pattern in value for pattern in BAD_TEXT_PATTERNS)


def has_unbalanced_dollar(text: object) -> bool:
    return str(text or "").count("$") % 2 == 1


def has_unbalanced_display_math(text: object) -> bool:
    value = str(text or "")
    return value.count("\\[") != value.count("\\]")


def duplicate_buckets(rows: list[dict[str, Any]], value_fn) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = value_fn(row)
        if value:
            buckets[str(value)].append(row_id(row))
    return {value: ids for value, ids in buckets.items() if len(ids) > 1}


def add_duplicate_issues(
    issues: list[str],
    panel_name: str,
    rows: list[dict[str, Any]],
    field: str,
) -> None:
    duplicates = duplicate_buckets(rows, lambda row: row.get(field))
    for value, ids in sorted(duplicates.items()):
        issues.append(f"{panel_name}: duplicate {field} {value}: {ids}")


def check_prompt(issues: list[str], panel_name: str, item_id: str, prompt: str) -> None:
    positions = [prompt.find(marker) for marker in REQUIRED_PROMPT_MARKERS]
    if any(position < 0 for position in positions):
        issues.append(f"{panel_name}: {item_id} prompt missing required answer markers")
    elif positions != sorted(positions):
        issues.append(f"{panel_name}: {item_id} prompt answer markers out of order")
    if has_bad_text(prompt):
        issues.append(f"{panel_name}: {item_id} prompt contains bad source text")
    if has_unbalanced_dollar(prompt):
        issues.append(f"{panel_name}: {item_id} prompt contains unbalanced dollar signs")
    if has_unbalanced_display_math(prompt):
        issues.append(f"{panel_name}: {item_id} prompt contains unbalanced display-math brackets")
    lowered = prompt.lower()
    if "change reason" in lowered or "reason for change" in lowered:
        issues.append(f"{panel_name}: {item_id} prompt still asks for a change reason")


def has_forbidden_trigger_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in TRIGGER_FORBIDDEN_KEYS:
                return True
            if has_forbidden_trigger_key(child):
                return True
    if isinstance(value, list):
        return any(has_forbidden_trigger_key(child) for child in value)
    return False


def check_choices(
    issues: list[str],
    panel_name: str,
    item_id: str,
    row: dict[str, Any],
    *,
    gt: bool,
    ngt: bool,
) -> None:
    choices = row.get("choices")
    if not isinstance(choices, dict) or not choices:
        issues.append(f"{panel_name}: {item_id} has invalid choices")
        return
    labels = [str(label).strip().upper() for label in choices]
    if len(labels) != len(set(labels)):
        issues.append(f"{panel_name}: {item_id} has duplicate choice labels")
    texts = [normalize_text(text) for text in choices.values()]
    if any(not text for text in texts):
        issues.append(f"{panel_name}: {item_id} has empty choice text")
    if any(has_unbalanced_dollar(text) for text in texts):
        issues.append(f"{panel_name}: {item_id} has unbalanced dollar signs in choices")
    if any(has_unbalanced_display_math(text) for text in texts):
        issues.append(f"{panel_name}: {item_id} has unbalanced display-math brackets in choices")
    if len(texts) != len(set(texts)):
        issues.append(f"{panel_name}: {item_id} has duplicate choice text")
    if gt and str(row.get("correct_answer") or "").strip().upper() not in labels:
        issues.append(f"{panel_name}: {item_id} correct_answer is not in choices")
    if ngt:
        if set(labels) != {"A", "B"}:
            issues.append(f"{panel_name}: {item_id} NGT choices must be exactly A/B")
        if "correct_answer" in row or "verified_answer" in row:
            issues.append(f"{panel_name}: {item_id} NGT carries truth-bearing answer field")


def check_basic_panel(
    issues: list[str],
    panel_name: str,
    rows: list[dict[str, Any]],
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        issues.append(f"{panel_name}: has {len(rows)} rows, expected {expected_count}")
    add_duplicate_issues(issues, panel_name, rows, "id")
    if panel_name.startswith(("mixed_gt", "context_gt", "trigger_gt")):
        add_duplicate_issues(issues, panel_name, rows, "record_id")
        add_duplicate_issues(issues, panel_name, rows, "native_id")

    content_duplicates = duplicate_buckets(rows, lambda row: content_key(content_text(row)))
    for _, ids in sorted(content_duplicates.items()):
        issues.append(f"{panel_name}: duplicate question/scenario text: {ids}")

    for row in rows:
        item_id = row_id(row)
        text = content_text(row)
        if not text:
            issues.append(f"{panel_name}: {item_id} missing question/scenario text")
        if has_bad_text(text) or has_bad_text(row.get("source_quote", "")):
            issues.append(f"{panel_name}: {item_id} contains bad source text")
        if has_unbalanced_dollar(text) or has_unbalanced_dollar(row.get("source_quote", "")):
            issues.append(f"{panel_name}: {item_id} contains unbalanced dollar signs")
        if has_unbalanced_display_math(text) or has_unbalanced_display_math(row.get("source_quote", "")):
            issues.append(f"{panel_name}: {item_id} contains unbalanced display-math brackets")

        verifiability = row.get("verifiability")
        gt = verifiability == "GT" or panel_name in {"mixed_gt", "context_gt", "trigger_gt"}
        ngt = verifiability == "NGT" or panel_name in {"context_ngt", "trigger_ngt"}
        if "choices" in row:
            check_choices(issues, panel_name, item_id, row, gt=gt, ngt=ngt)
        if gt:
            if not row.get("source_url") or not row.get("source_quote"):
                issues.append(f"{panel_name}: {item_id} GT missing source_url/source_quote")
            if row.get("source_dataset") == "skylenage-ai/HLE-Verified":
                choice_source = row.get("hle_choice_source")
                if not choice_source:
                    issues.append(f"{panel_name}: {item_id} HLE row missing hle_choice_source")
                if choice_source != "source_native_mc" and not row.get("synthetic_mc_generation_rule"):
                    issues.append(f"{panel_name}: {item_id} converted HLE choices missing generation rule")


def check_context_panels(
    issues: list[str],
    context_gt: list[dict[str, Any]],
    context_ngt: list[dict[str, Any]],
) -> None:
    for row in context_gt:
        item_id = row_id(row)
        variants = row.get("context_variants", {})
        if set(variants) != GT_VARIANTS:
            issues.append(f"context_gt: {item_id} has wrong GT variants")
        for variant_name, variant in variants.items():
            if not isinstance(variant, dict) or not variant.get("prompt"):
                issues.append(f"context_gt: {item_id} {variant_name} missing prompt")
            else:
                check_prompt(issues, "context_gt", f"{item_id}/{variant_name}", str(variant["prompt"]))
    for row in context_ngt:
        item_id = row_id(row)
        variants = row.get("context_variants", {})
        if set(variants) != NGT_VARIANTS:
            issues.append(f"context_ngt: {item_id} has wrong NGT variants")
        states = row.get("answer_states", {})
        if set(states) != {"A", "B"}:
            issues.append(f"context_ngt: {item_id} answer_states must be exactly A/B")
        elif content_key(states["A"].get("text")) == content_key(states["B"].get("text")):
            issues.append(f"context_ngt: {item_id} answer_states A/B must be distinct")
        for variant_name, variant in variants.items():
            if not isinstance(variant, dict) or not variant.get("prompt"):
                issues.append(f"context_ngt: {item_id} {variant_name} missing prompt")
            else:
                check_prompt(issues, "context_ngt", f"{item_id}/{variant_name}", str(variant["prompt"]))

    gt_domains = Counter(row.get("domain") for row in context_gt)
    if any(count != 50 for count in gt_domains.values()) or len(gt_domains) != 4:
        issues.append(f"context_gt: domain counts not balanced: {dict(gt_domains)}")
    ngt_domains = Counter(row.get("domain") for row in context_ngt)
    if any(count != 25 for count in ngt_domains.values()) or len(ngt_domains) != 4:
        issues.append(f"context_ngt: domain counts not balanced: {dict(ngt_domains)}")


def check_trigger_panel(issues: list[str], panel_name: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        item_id = row_id(row)
        if has_forbidden_trigger_key(row):
            issues.append(f"{panel_name}: {item_id} carries forbidden trigger direction metadata")
        check_prompt(issues, panel_name, item_id, str(row.get("initial_prompt", "")))


def check_alignment(
    issues: list[str],
    mixed_gt: list[dict[str, Any]],
    context_gt: list[dict[str, Any]],
    context_ngt: list[dict[str, Any]],
    trigger_gt: list[dict[str, Any]],
    trigger_ngt: list[dict[str, Any]],
) -> None:
    for index, (source, context) in enumerate(zip(mixed_gt, context_gt), start=1):
        if source.get("record_id") != context.get("record_id"):
            issues.append(f"mixed_gt/context_gt: row {index} record_id mismatch")
        if content_key(source.get("question")) != content_key(context.get("question")):
            issues.append(f"mixed_gt/context_gt: row {index} question mismatch")

    context_gt_by_id = {row_id(row): row for row in context_gt}
    for trigger in trigger_gt:
        context = context_gt_by_id.get(row_id(trigger))
        if context is None:
            issues.append(f"trigger_gt: {row_id(trigger)} not found in context_gt")
            continue
        if content_key(trigger.get("question")) != content_key(context.get("question")):
            issues.append(f"trigger_gt/context_gt: {row_id(trigger)} question mismatch")
        if trigger.get("choices") != context.get("choices"):
            issues.append(f"trigger_gt/context_gt: {row_id(trigger)} choices mismatch")
        if trigger.get("correct_answer") != context.get("correct_answer"):
            issues.append(f"trigger_gt/context_gt: {row_id(trigger)} correct answer mismatch")

    context_ngt_by_id = {row_id(row): row for row in context_ngt}
    for trigger in trigger_ngt:
        context = context_ngt_by_id.get(row_id(trigger))
        if context is None:
            issues.append(f"trigger_ngt: {row_id(trigger)} not found in context_ngt")
            continue
        if content_key(trigger.get("question")) != content_key(context.get("scenario")):
            issues.append(f"trigger_ngt/context_ngt: {row_id(trigger)} scenario mismatch")
        expected_choices = {
            "A": context["answer_states"]["A"]["text"],
            "B": context["answer_states"]["B"]["text"],
        }
        if trigger.get("choices") != expected_choices:
            issues.append(f"trigger_ngt/context_ngt: {row_id(trigger)} choices mismatch")


def audit(data_dir: Path) -> list[str]:
    panels = {
        name: read_panel(data_dir / filename)
        for name, (filename, _) in PANEL_SPECS.items()
    }
    issues: list[str] = []
    for name, (_, expected_count) in PANEL_SPECS.items():
        check_basic_panel(issues, name, panels[name], expected_count)
    check_context_panels(issues, panels["context_gt"], panels["context_ngt"])
    check_trigger_panel(issues, "trigger_gt", panels["trigger_gt"])
    check_trigger_panel(issues, "trigger_ngt", panels["trigger_ngt"])
    check_alignment(
        issues,
        panels["mixed_gt"],
        panels["context_gt"],
        panels["context_ngt"],
        panels["trigger_gt"],
        panels["trigger_ngt"],
    )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    issues = audit(args.data_dir)
    if issues:
        print(f"panel audit failed with {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print("panel audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
