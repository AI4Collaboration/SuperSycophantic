#!/usr/bin/env python3
"""Summarize adaptive outcome rates after the frontier-model follow-up audit."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    REPO_DIR
    / "Experimental"
    / "rebuttal_adaptive_audit"
    / "adaptive_followup_audit_openai_gpt-5.4_n3780.jsonl"
)
DEFAULT_OUT = REPO_DIR / "Experimental" / "rebuttal_adaptive_audit"


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def rate(values: list[bool]) -> dict[str, Any]:
    n = len(values)
    successes = sum(values)
    lo, hi = wilson(successes, n)
    return {
        "n": n,
        "successes": successes,
        "rate": successes / n if n else None,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def read_audit_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_outcomes(audit_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    needed: dict[str, set[int]] = defaultdict(set)
    for row in audit_rows:
        needed[row["source_file"]].add(int(row["row_index"]))

    outcomes: dict[tuple[str, int], dict[str, Any]] = {}
    for source_file, row_indices in needed.items():
        path = REPO_DIR / source_file
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if index not in row_indices:
                    continue
                row = json.loads(line)
                outcomes[(source_file, index)] = {
                    "single_trigger_answer_switch": row.get("single_trigger_answer_switch"),
                    "single_trigger_truth_departure": row.get("single_trigger_truth_departure"),
                }
    return outcomes


def summarize(audit_rows: list[dict[str, Any]], outcomes: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for branch in ["OBJ", "SUB"]:
        branch_rows = [row for row in audit_rows if row.get("branch") == branch]
        summary[branch] = {}
        outcome_key = "single_trigger_truth_departure" if branch == "OBJ" else "single_trigger_answer_switch"
        for label, subset in [
            ("all_audited", branch_rows),
            ("audit_pass", [row for row in branch_rows if row.get("audit_pass")]),
            ("audit_flagged", [row for row in branch_rows if not row.get("audit_pass")]),
        ]:
            values: list[bool] = []
            for row in subset:
                outcome = outcomes.get((row["source_file"], int(row["row_index"])), {})
                value = outcome.get(outcome_key)
                if value is not None:
                    values.append(bool(value))
            summary[branch][label] = rate(values)
    return summary


def pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{100 * value:.2f}%"


def write_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adaptive_followup_audit_behavior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["# Adaptive follow-up audit behavior summary", ""]
    for branch in ["OBJ", "SUB"]:
        metric = "truth departure" if branch == "OBJ" else "switching"
        lines.append(f"## {branch} {metric}")
        for label, stats in summary[branch].items():
            lines.append(
                f"- {label}: {stats['successes']}/{stats['n']} "
                f"({pct(stats['rate'])}, 95% CI {pct(stats['ci95_low'])}-{pct(stats['ci95_high'])})"
            )
        lines.append("")
    (out_dir / "adaptive_followup_audit_behavior_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = read_audit_rows(args.audit)
    outcomes = load_outcomes(rows)
    summary = summarize(rows, outcomes)
    write_outputs(summary, args.out_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
