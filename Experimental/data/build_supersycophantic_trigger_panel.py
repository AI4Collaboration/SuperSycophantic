#!/usr/bin/env python3
"""Build trigger-run JSONL panels from the current SuperSycophantic context JSON.

The context data keep GT and NGT in rich JSON objects. The trigger runner uses a
flat JSONL format with one first-turn prompt per line and one internally
recorded pressure target. This script is the canonical bridge between those
schemas for neutral-context trigger runs. Biased first-turn prompts are
context-only controls and are not valid trigger inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent


def read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def other_side(side: str) -> str:
    return "B" if side == "A" else "A"


def state(item: dict[str, Any], side: str) -> dict[str, Any]:
    value = item.get("answer_states", {}).get(side)
    if not isinstance(value, dict):
        raise ValueError(f"{item.get('item_id')} missing answer state {side}")
    return value


def state_text(item: dict[str, Any], side: str) -> str:
    text = state(item, side).get("text")
    if not text:
        raise ValueError(f"{item.get('item_id')} missing answer-state text for {side}")
    return str(text)


def state_native_label(item: dict[str, Any], side: str) -> str | None:
    label = state(item, side).get("native_label")
    if label is None:
        return None
    label = str(label).strip()
    return label or None


def native_choices(item: dict[str, Any]) -> dict[str, str] | None:
    choices = item.get("native_choices")
    if not isinstance(choices, dict) or not choices:
        return None
    return {str(label): str(text) for label, text in choices.items()}


def variant_prompt(item: dict[str, Any], variant_name: str) -> str:
    variant = item.get("context_variants", {}).get(variant_name, {})
    if isinstance(variant, dict) and variant.get("prompt"):
        return str(variant["prompt"])
    if variant_name == "neutral" and item.get("neutral_prompt"):
        return str(item["neutral_prompt"])
    raise ValueError(f"{item.get('item_id')} missing prompt for context variant {variant_name}")


def variant_metadata(item: dict[str, Any], variant_name: str) -> dict[str, Any]:
    if variant_name != "neutral":
        raise ValueError("trigger panels must use neutral first-turn context only")
    variant = item.get("context_variants", {}).get(variant_name, {})
    if not isinstance(variant, dict):
        variant = {}
    return {
        "context_variant": variant_name,
        "context_condition": "neutral",
        "cue_type": variant.get("cue_type"),
        "direction": variant.get("direction"),
        "user_framing": variant.get("user_framing"),
        "initial_prompt": variant_prompt(item, variant_name),
    }


def base_metadata(item: dict[str, Any], variant_name: str) -> dict[str, Any]:
    base_id = item.get("item_id") or item.get("id")
    row_id = str(base_id) if variant_name == "neutral" else f"{base_id}__{variant_name}"
    row = {
        "id": row_id,
        "source": base_id,
        "verifiability": item.get("verifiability"),
        "domain": item.get("domain"),
        "base_decision_id": item.get("base_decision_id"),
        **variant_metadata(item, variant_name),
    }
    if item.get("verifiability") == "GT":
        row.update(
            {
                "source_dataset": item.get("source"),
                "source_file": item.get("source_file"),
                "source_url": item.get("source_url"),
                "source_quote": item.get("source_quote"),
                "source_cache_file": item.get("source_cache_file"),
                "record_id": item.get("record_id"),
                "native_id": item.get("native_id"),
            }
        )
    return row


def gt_trigger_item(item: dict[str, Any], variant_name: str) -> dict[str, Any]:
    correct_side = str(item.get("correct_answer_state") or "").upper()
    if correct_side not in {"A", "B"}:
        for label, state_value in item.get("answer_states", {}).items():
            if isinstance(state_value, dict) and state_value.get("role") == "verified_answer":
                correct_side = str(label).upper()
                break
    if correct_side not in {"A", "B"}:
        raise ValueError(f"{item.get('item_id')} missing tracked GT correct answer state")
    wrong_side = other_side(correct_side)
    injected = item.get("context_variants", {}).get("injected_wrong_answer", {})
    if isinstance(injected, dict) and injected.get("direction") in {"A", "B"}:
        wrong_side = str(injected["direction"]).upper()

    row = base_metadata(item, variant_name)
    row.update(
        {
            "question": item["question"],
            "answer_mode": item.get("answer_mode", "multiple_choice"),
            "pressure_target_source": "gt_injected_wrong_answer",
            "correct_answer_state": correct_side,
            "pressure_target_answer_state": wrong_side,
        }
    )

    if row["answer_mode"] == "exact":
        verified = item.get("verified_answer", {})
        correct = verified.get("text") if isinstance(verified, dict) else item.get("correct_answer_text")
        pressure = injected.get("injected_belief_answer_text") if isinstance(injected, dict) else None
        if not correct or not pressure:
            raise ValueError(f"{item.get('item_id')} missing exact GT correct or pressure answer")
        row.update(
            {
                "choices": None,
                "correct_answer": str(correct),
                "pressure_target": str(pressure),
            }
        )
        return row

    choices = native_choices(item)
    correct = item.get("correct_answer") or state_native_label(item, correct_side) or correct_side
    pressure = state_native_label(item, wrong_side) or wrong_side
    if not choices:
        choices = {
            "A": state_text(item, "A"),
            "B": state_text(item, "B"),
        }
    if correct not in choices:
        correct = correct_side
    if pressure not in choices:
        pressure = wrong_side
    row.update(
        {
            "choices": choices,
            "correct_answer": correct,
            "pressure_target": pressure,
        }
    )
    return row


def ngt_trigger_item(item: dict[str, Any], index: int, variant_name: str) -> dict[str, Any]:
    if variant_name == "neutral":
        pressure = "A" if index % 2 == 0 else "B"
        pressure_source = "ngt_counterbalanced_answer_state"
    else:
        variant = item.get("context_variants", {}).get(variant_name, {})
        pressure = str(variant.get("direction", "")).upper() if isinstance(variant, dict) else ""
        if pressure not in {"A", "B"}:
            raise ValueError(f"{item.get('item_id')} {variant_name} missing A/B direction")
        pressure_source = "ngt_context_variant_direction"
    row = base_metadata(item, variant_name)
    row.update(
        {
            "question": item.get("scenario", ""),
            "answer_mode": item.get("answer_mode", "tracked_ab"),
            "choices": {
                "A": state_text(item, "A"),
                "B": state_text(item, "B"),
            },
            "pressure_target": pressure,
            "pressure_target_source": pressure_source,
            "correct_answer_state": None,
            "pressure_target_answer_state": pressure,
        }
    )
    return row


def selected_variants(item: dict[str, Any], context_condition: str) -> list[str]:
    if context_condition == "neutral":
        return ["neutral"]
    raise ValueError("trigger panels must use neutral first-turn context only")


def validate(rows: list[dict[str, Any]]) -> None:
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("trigger panel has duplicate ids")
    for row in rows:
        if not row.get("id") or not row.get("question") or not row.get("initial_prompt"):
            raise ValueError(f"{row.get('id')} missing id, question, or initial_prompt")
        if not row.get("source"):
            raise ValueError(f"{row.get('id')} missing source item id")
        prompt = str(row.get("initial_prompt"))
        if "Confidence:" not in prompt or "\\boxed" not in prompt:
            raise ValueError(f"{row.get('id')} initial prompt missing confidence or boxed answer format")
        if row.get("context_condition") != "neutral":
            raise ValueError(f"{row.get('id')} trigger row is not neutral-context")
        if row.get("verifiability") == "GT":
            if not str(row.get("source_file", "")).startswith(("http://", "https://")):
                raise ValueError(f"{row.get('id')} GT source_file must be a URL")
            if not row.get("source_url") or not row.get("source_quote"):
                raise ValueError(f"{row.get('id')} GT row missing source_url or source_quote")
        if row.get("verifiability") == "NGT":
            for field in ["source_file", "source_url", "source_quote", "correct_answer", "verified_answer"]:
                if field in row:
                    raise ValueError(f"{row.get('id')} NGT row should not carry {field}")
        if row.get("answer_mode") == "exact":
            if not row.get("correct_answer") or not row.get("pressure_target"):
                raise ValueError(f"{row.get('id')} missing exact answer fields")
        else:
            choices = row.get("choices")
            if not isinstance(choices, dict) or row.get("pressure_target") not in choices:
                raise ValueError(f"{row.get('id')} has invalid choices or pressure target")
            if row.get("verifiability") == "GT" and row.get("correct_answer") not in choices:
                raise ValueError(f"{row.get('id')} has invalid GT correct answer")
            if row.get("verifiability") == "NGT" and "correct_answer" in row:
                raise ValueError(f"{row.get('id')} NGT row should not carry correct_answer")


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not args.ngt_only:
        gt_items = read_json(DATA_DIR / args.gt_input)
        for item in gt_items:
            rows.extend(
                gt_trigger_item(item, variant_name)
                for variant_name in selected_variants(item, args.context_condition)
            )
    if not args.gt_only:
        ngt_items = read_json(DATA_DIR / args.ngt_input)
        for idx, item in enumerate(ngt_items):
            rows.extend(
                ngt_trigger_item(item, idx, variant_name)
                for variant_name in selected_variants(item, args.context_condition)
            )
    validate(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-input", default="supersycophantic_gt_200.json")
    parser.add_argument("--ngt-input", default="supersycophantic_ngt_100.json")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--context-condition",
        choices=["neutral"],
        default="neutral",
        help="Trigger panels use neutral first-turn context only.",
    )
    parser.add_argument("--gt-only", action="store_true")
    parser.add_argument("--ngt-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gt_only and args.ngt_only:
        raise SystemExit("--gt-only and --ngt-only cannot be combined")
    rows = build(args)
    if args.output is None:
        branch = "_gt" if args.gt_only else "_ngt" if args.ngt_only else ""
        args.output = f"supersycophantic_trigger{branch}_{args.context_condition}_{len(rows)}.jsonl"
    output = DATA_DIR / args.output
    write_jsonl(output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("verifiability"))
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {len(rows)} rows to {output} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
