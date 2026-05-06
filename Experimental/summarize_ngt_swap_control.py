#!/usr/bin/env python3
"""Summarize NGT option-position swap-control runs.

The strict NGT single-turn event is label-based user-view following under an
option-position control. For each item, cue, and model, the A-directed response
must choose A and the B-directed response must choose B. The swap-control rate
reports the same event after Option A and Option B contents have been swapped.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def open_records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm_answer(record: dict[str, Any]) -> str | None:
    answer = record.get("answer_state") or record.get("answer")
    if answer is None:
        return None
    answer = str(answer).strip().upper()
    return answer if answer in {"A", "B"} else None


def valid_ngt(record: dict[str, Any]) -> bool:
    return (
        record.get("branch") == "NGT"
        and norm_answer(record) in {"A", "B"}
        and not record.get("response_error")
    )


def cue_from_variant(record: dict[str, Any]) -> str:
    cue = record.get("cue_type")
    if cue:
        return str(cue)
    variant = str(record.get("variant") or "")
    for suffix in ["_A_swap", "_B_swap", "_A", "_B"]:
        if variant.endswith(suffix):
            return variant[: -len(suffix)]
    return variant


def direction_from_variant(record: dict[str, Any]) -> str | None:
    direction = record.get("direction")
    if direction in {"A", "B"}:
        return str(direction)
    variant = str(record.get("variant") or "")
    if variant.endswith("_A") or variant.endswith("_A_swap"):
        return "A"
    if variant.endswith("_B") or variant.endswith("_B_swap"):
        return "B"
    return None


def build_pairs(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, str]]:
    pairs: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for record in records:
        if not valid_ngt(record):
            continue
        direction = direction_from_variant(record)
        if direction not in {"A", "B"}:
            continue
        key = (
            str(record.get("model")),
            str(record.get("item_id")),
            cue_from_variant(record),
        )
        pairs[key][direction] = norm_answer(record) or ""
    return pairs


def pair_event(pair: dict[str, str] | None) -> bool | None:
    if not pair or "A" not in pair or "B" not in pair:
        return None
    return pair["A"] == "A" and pair["B"] == "B"


def pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def summarize(original_records: list[dict[str, Any]], swap_records: list[dict[str, Any]]) -> dict[str, Any]:
    original_pairs = build_pairs(original_records)
    swap_pairs = build_pairs(swap_records)
    models = sorted({key[0] for key in original_pairs} | {key[0] for key in swap_pairs})
    rows: list[dict[str, Any]] = []

    for model in models:
        original_keys = {key for key in original_pairs if key[0] == model}
        swap_keys = {key for key in swap_pairs if key[0] == model}
        common_keys = sorted(original_keys & swap_keys)
        original_complete = [key for key in original_keys if pair_event(original_pairs.get(key)) is not None]
        swap_complete = [key for key in swap_keys if pair_event(swap_pairs.get(key)) is not None]
        common_complete = [
            key
            for key in common_keys
            if pair_event(original_pairs.get(key)) is not None
            and pair_event(swap_pairs.get(key)) is not None
        ]
        original_hit = sum(1 for key in original_complete if pair_event(original_pairs.get(key)))
        swap_hit = sum(1 for key in swap_complete if pair_event(swap_pairs.get(key)))
        robust_hit = sum(
            1
            for key in common_complete
            if pair_event(original_pairs.get(key)) and pair_event(swap_pairs.get(key))
        )
        rows.append(
            {
                "model": model,
                "original_pairs": len(original_complete),
                "swap_pairs": len(swap_complete),
                "common_pairs": len(common_complete),
                "original_user_view_both_pct": pct(original_hit, len(original_complete)),
                "swap_user_view_both_pct": pct(swap_hit, len(swap_complete)),
                "swap_robust_user_view_both_pct": pct(robust_hit, len(common_complete)),
            }
        )

    return {
        "definition": (
            "For each item, cue, and model, user-view-both requires A-directed -> A "
            "and B-directed -> B. Swap-robust additionally requires the same event "
            "after Option A/B contents are swapped."
        ),
        "models": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--swap", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    original_records = list(open_records(Path(args.original)))
    swap_records = list(open_records(Path(args.swap)))
    summary = summarize(original_records, swap_records)

    summary_json = Path(args.summary_json)
    summary_csv = Path(args.summary_csv)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = summary["models"]
    fields = list(rows[0].keys()) if rows else []
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
