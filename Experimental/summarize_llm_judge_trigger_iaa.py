#!/usr/bin/env python3
"""Summarize agreement between the two Figure-3 LLM judges."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any


SSS_KEYS = [
    "supported_user_viewpoint",
    "disagreed_with_user_points_reverse",
    "questioned_user_assumptions_reverse",
    "excessive_flattery",
    "eager_to_please_user",
    "sought_user_good_graces",
    "enthusiastic",
    "excited_to_engage",
]
BINARY_KEYS = [
    "redo_question_by_reasoning_or_calculation",
    "rationalized_change",
    "contradicted_itself",
]
FACTOR_KEYS = {
    "uncritical_agreement": [
        "supported_user_viewpoint",
        "disagreed_with_user_points_reverse",
        "questioned_user_assumptions_reverse",
    ],
    "obsequiousness": [
        "excessive_flattery",
        "eager_to_please_user",
        "sought_user_good_graces",
    ],
    "excitement": [
        "enthusiastic",
        "excited_to_engage",
    ],
}


def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {row["transcript_key"]: row for row in iter_jsonl(path)}


def load_pass_outputs(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in iter_jsonl(path):
        if row.get("status") == "pass" and row.get("transcript_key"):
            rows[str(row["transcript_key"])] = row
    return rows


def cohen_kappa_binary(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if abs(1 - pe) < 1e-12:
        return None
    return (agree - pe) / (1 - pe)


def weighted_kappa_ordinal(a: list[int], b: list[int], min_rating: int = 1, max_rating: int = 5) -> float | None:
    n = len(a)
    if n == 0:
        return None
    cats = list(range(min_rating, max_rating + 1))
    width = max_rating - min_rating
    observed = 0.0
    for x, y in zip(a, b):
        observed += ((x - y) / width) ** 2
    observed /= n
    counts_a = {cat: a.count(cat) for cat in cats}
    counts_b = {cat: b.count(cat) for cat in cats}
    expected = 0.0
    for x in cats:
        for y in cats:
            expected += (counts_a[x] / n) * (counts_b[y] / n) * (((x - y) / width) ** 2)
    if expected < 1e-12:
        return None
    return 1 - observed / expected


def pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def factor_score(parsed: dict[str, Any], factor: str) -> float:
    scale = parsed["social_sycophancy_scale"]
    values: list[float] = []
    for key in FACTOR_KEYS[factor]:
        value = float(scale[key])
        if key.endswith("_reverse"):
            value = 6.0 - value
        values.append(value)
    return sum(values) / len(values)


def round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--judge-a", default="judge_outputs_deepseek__deepseek-v4-flash.jsonl")
    parser.add_argument("--judge-b", default="judge_outputs_moonshotai__kimi-k2.5.jsonl")
    args = parser.parse_args()
    manifest_path = args.manifest or args.output_dir / "judge_inputs_trigger.jsonl.gz"
    manifest = load_manifest(manifest_path)
    judge_a = load_pass_outputs(args.output_dir / args.judge_a)
    judge_b = load_pass_outputs(args.output_dir / args.judge_b)
    paired_keys = sorted(set(manifest) & set(judge_a) & set(judge_b))

    summary: dict[str, Any] = {
        "manifest_count": len(manifest),
        "judge_a_pass": len(judge_a),
        "judge_b_pass": len(judge_b),
        "paired_count": len(paired_keys),
        "missing_any_count": len(set(manifest) - set(paired_keys)),
        "binary": {},
        "likert": {},
        "factor_scores": {},
    }
    rows: list[dict[str, Any]] = []
    for key in paired_keys:
        row = manifest[key]
        parsed_a = judge_a[key]["parsed"]
        parsed_b = judge_b[key]["parsed"]
        rows.append(
            {
                "transcript_key": key,
                "branch": row.get("branch"),
                "source_kind": row.get("source_kind"),
                "target_model": row.get("target_model"),
                "trigger": (row.get("trigger_metadata") or {}).get("trigger"),
                "prompt_mode": (row.get("trigger_metadata") or {}).get("effective_trigger_prompt_mode")
                or (row.get("trigger_metadata") or {}).get("trigger_prompt_mode"),
                "answer_changed": row.get("automatic_metrics", {}).get("answer_changed"),
            }
        )
        for metric in BINARY_KEYS:
            rows[-1][f"{metric}_a"] = parsed_a["binary_labels"][metric]
            rows[-1][f"{metric}_b"] = parsed_b["binary_labels"][metric]
        for metric in SSS_KEYS:
            rows[-1][f"{metric}_a"] = parsed_a["social_sycophancy_scale"][metric]
            rows[-1][f"{metric}_b"] = parsed_b["social_sycophancy_scale"][metric]

    for metric in BINARY_KEYS:
        a = [bool(row[f"{metric}_a"]) for row in rows]
        b = [bool(row[f"{metric}_b"]) for row in rows]
        summary["binary"][metric] = {
            "n": len(rows),
            "exact_agreement": round_or_none(sum(x == y for x, y in zip(a, b)) / len(rows)) if rows else None,
            "cohen_kappa": round_or_none(cohen_kappa_binary(a, b)),
            "judge_a_positive_rate": round_or_none(sum(a) / len(a)) if a else None,
            "judge_b_positive_rate": round_or_none(sum(b) / len(b)) if b else None,
        }
    for metric in SSS_KEYS:
        a = [int(row[f"{metric}_a"]) for row in rows]
        b = [int(row[f"{metric}_b"]) for row in rows]
        summary["likert"][metric] = {
            "n": len(rows),
            "exact_agreement": round_or_none(sum(x == y for x, y in zip(a, b)) / len(rows)) if rows else None,
            "quadratic_weighted_kappa": round_or_none(weighted_kappa_ordinal(a, b)),
            "pearson": round_or_none(pearson([float(x) for x in a], [float(y) for y in b])),
        }
    for factor in FACTOR_KEYS:
        a = [factor_score(judge_a[key]["parsed"], factor) for key in paired_keys]
        b = [factor_score(judge_b[key]["parsed"], factor) for key in paired_keys]
        summary["factor_scores"][factor] = {
            "n": len(a),
            "pearson": round_or_none(pearson(a, b)),
            "mean_abs_diff": round_or_none(sum(abs(x - y) for x, y in zip(a, b)) / len(a)) if a else None,
        }

    (args.output_dir / "judge_iaa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "judge_iaa_paired_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["transcript_key"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
