#!/usr/bin/env python3
"""Summarize adaptive-trigger family rankings by generator model."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Experimental" / "rebuttal_robustness_results" / "adaptive_generator_main_8x8.jsonl.gz"
DEFAULT_OUT_DIR = ROOT / "Experimental" / "reports" / "rebuttal"


def iter_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: values[key], reverse=True)
    return {key: index + 1 for index, key in enumerate(ordered)}


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if n < 2:
        return math.nan
    ar = ranks({key: a[key] for key in keys})
    br = ranks({key: b[key] for key in keys})
    d2 = sum((ar[key] - br[key]) ** 2 for key in keys)
    return 1 - (6 * d2) / (n * (n * n - 1))


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        branch = str(row.get("branch") or "")
        generator = str(row.get("generator_model") or "")
        family = str(row.get("family") or "")
        if branch and generator and family:
            grouped[(branch, generator, family)].append(row)

    family_rows: list[dict[str, Any]] = []
    rates: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (branch, generator, family), subset in sorted(grouped.items()):
        metric = "single_trigger_answer_switch" if branch == "SUB" else "single_trigger_truth_departure"
        values = [bool(row.get(metric)) for row in subset if row.get(metric) is not None]
        successes = sum(values)
        n = len(values)
        lo, hi = wilson(successes, n)
        rate = successes / n if n else math.nan
        rates[(branch, generator)][family] = rate
        family_rows.append(
            {
                "branch": branch,
                "generator_model": generator,
                "family": family,
                "records": len(subset),
                "denominator": n,
                "successes": successes,
                "rate_pct": round(rate * 100, 2) if n else "",
                "ci95_low_pct": round(lo * 100, 2) if n else "",
                "ci95_high_pct": round(hi * 100, 2) if n else "",
            }
        )

    rank_rows: list[dict[str, Any]] = []
    for branch in sorted({key[0] for key in rates}):
        generators = sorted(generator for b, generator in rates if b == branch)
        for i, left in enumerate(generators):
            for right in generators[i + 1 :]:
                rank_rows.append(
                    {
                        "branch": branch,
                        "generator_a": left,
                        "generator_b": right,
                        "family_rank_spearman": round(spearman(rates[(branch, left)], rates[(branch, right)]), 3),
                    }
                )
    return family_rows, rank_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, family_rows: list[dict[str, Any]], rank_rows: list[dict[str, Any]]) -> None:
    lines = ["# Adaptive family ranking by generator", ""]
    lines.append("## Family rates")
    for branch in ["SUB", "OBJ"]:
        lines.append(f"### {branch}")
        generators = sorted({row["generator_model"] for row in family_rows if row["branch"] == branch})
        for generator in generators:
            subset = [row for row in family_rows if row["branch"] == branch and row["generator_model"] == generator]
            subset.sort(key=lambda row: float(row["rate_pct"]), reverse=True)
            top = subset[0]
            bottom = subset[-1]
            lines.append(
                f"- {generator}: highest {top['family']}={top['rate_pct']}%; "
                f"lowest {bottom['family']}={bottom['rate_pct']}%."
            )
    lines.append("")
    lines.append("## Rank correlations")
    for row in rank_rows:
        lines.append(
            f"- {row['branch']}: {row['generator_a']} vs {row['generator_b']} "
            f"Spearman={row['family_rank_spearman']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    rows = list(iter_rows(args.input))
    family_rows, rank_rows = summarize(rows)
    write_csv(args.out_dir / "adaptive_family_by_generator.csv", family_rows)
    write_csv(args.out_dir / "adaptive_family_rank_correlations.csv", rank_rows)
    write_markdown(args.out_dir / "adaptive_family_ranking_summary.md", family_rows, rank_rows)
    print(f"rows={len(rows)} family_rows={len(family_rows)} rank_rows={len(rank_rows)}", flush=True)


if __name__ == "__main__":
    main()
