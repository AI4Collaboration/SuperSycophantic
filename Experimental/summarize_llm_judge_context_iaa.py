#!/usr/bin/env python3
"""Summarize agreement between the two context LLM judges."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

from summarize_llm_judge_trigger_iaa import (
    BINARY_KEYS,
    FACTOR_KEYS,
    SSS_KEYS,
    cohen_kappa_binary,
    factor_score,
    pearson,
    round_or_none,
    weighted_kappa_ordinal,
)


def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {row["context_key"]: row for row in iter_jsonl(path)}


def load_pass_outputs(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in iter_jsonl(path):
        if row.get("status") == "pass" and row.get("context_key"):
            rows[str(row["context_key"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--judge-a", default="judge_outputs_deepseek__deepseek-v4-flash.jsonl")
    parser.add_argument("--judge-b", default="judge_outputs_moonshotai__kimi-k2.5.jsonl")
    args = parser.parse_args()
    manifest_path = args.manifest or args.output_dir / "judge_inputs_context.jsonl.gz"
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
        out = {
            "context_key": key,
            "branch": row.get("branch"),
            "variant": row.get("variant"),
            "cue_type": row.get("cue_type"),
            "target_model": row.get("target_model"),
            "truth_status": row.get("automatic_metrics", {}).get("truth_status"),
        }
        for metric in BINARY_KEYS:
            out[f"{metric}_a"] = parsed_a["binary_labels"][metric]
            out[f"{metric}_b"] = parsed_b["binary_labels"][metric]
        for metric in SSS_KEYS:
            out[f"{metric}_a"] = parsed_a["social_sycophancy_scale"][metric]
            out[f"{metric}_b"] = parsed_b["social_sycophancy_scale"][metric]
        rows.append(out)

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
        fieldnames = list(rows[0]) if rows else ["context_key"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
