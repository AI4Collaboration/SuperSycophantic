#!/usr/bin/env python3
"""Select saturated MMLU-Pro GT items from two-model first-turn results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_DIR.parents[1]
HEALTH_STEM_LEAKAGE_RE = re.compile(r"\b(recommendations?|guidelines?|official guidance|best practices?)\b", re.I)


def stable_int(*parts: Any) -> int:
    key = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def open_text(path: Path, mode: str):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def resolve(path: str | Path, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open_text(path, "rt") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_summary(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def confidence(record: dict[str, Any]) -> int:
    value = record.get("initial_confidence")
    return int(value) if isinstance(value, int) else -1


def select_items(
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    models: list[str],
    per_domain: int,
) -> tuple[list[dict[str, Any]], str]:
    candidate_by_id = {str(row["id"]): row for row in candidates}
    result_by_key = {
        (str(row.get("item_id")), str(row.get("model"))): row
        for row in results
        if row.get("item_id") in candidate_by_id and row.get("model") in models
    }

    screened: list[dict[str, Any]] = []
    source_stem_exclusions: list[str] = []
    counts_by_domain = Counter(str(row["domain"]) for row in candidates)
    complete_by_domain: Counter[str] = Counter()
    both_correct_by_domain: Counter[str] = Counter()
    model_correct_by_domain: dict[str, Counter[str]] = {model: Counter() for model in models}

    for item_id, item in candidate_by_id.items():
        domain = str(item["domain"])
        item_results = [result_by_key.get((item_id, model)) for model in models]
        if not all(item_results):
            continue
        complete_by_domain[domain] += 1
        if all(result.get("initial_correct") for result in item_results if result):
            both_correct_by_domain[domain] += 1
            if domain == "Health" and HEALTH_STEM_LEAKAGE_RE.search(str(item.get("question", ""))):
                source_stem_exclusions.append(str(item_id))
                continue
            model_details = {
                str(result["model"]): {
                    "initial_answer": result.get("initial_answer"),
                    "initial_confidence": result.get("initial_confidence"),
                    "initial_parse_method": result.get("initial_parse_method"),
                }
                for result in item_results
                if result
            }
            selected = dict(item)
            selected["saturation_selection"] = {
                "criterion": "both_screening_models_correct_on_neutral_first_turn",
                "screening_models": models,
                "model_details": model_details,
                "min_self_reported_confidence": min(confidence(result) for result in item_results if result),
            }
            screened.append(selected)
        for result in item_results:
            if result and result.get("initial_correct"):
                model_correct_by_domain[str(result["model"])][domain] += 1

    selected_rows: list[dict[str, Any]] = []
    label_shortfalls: list[str] = []
    for domain in ["Mathematical Science", "Physical Science", "Bio&Chem", "Health"]:
        domain_rows = [row for row in screened if row["domain"] == domain]
        labels = sorted({str(row["correct_answer"]) for row in domain_rows})
        target_per_label = per_domain // len(labels) if labels else 0
        remainder = per_domain % len(labels) if labels else 0
        for label_index, label in enumerate(labels):
            label_rows = [row for row in domain_rows if str(row["correct_answer"]) == label]
            label_rows.sort(
                key=lambda row: (
                    -int(row["saturation_selection"]["min_self_reported_confidence"]),
                    stable_int("mmlu_pro_saturated_select_v1", row["id"]),
                )
            )
            target = target_per_label + (1 if label_index < remainder else 0)
            if len(label_rows) < target:
                label_shortfalls.append(f"{domain}/{label}: {len(label_rows)} available for target {target}")
            selected_rows.extend(label_rows[:target])

    summary_lines = [
        "# MMLU-Pro saturated screening selection summary",
        "",
        f"Screening models: {', '.join(models)}",
        (
            "Saturated screening rule: both screening models correct on a source-style neutral first-turn screening prompt; "
            f"up to {per_domain} items per domain; correct-answer labels are balanced within each domain when possible. "
            "Final benchmark-prompt Pass@1 is re-estimated separately and should not be inferred from this screening pass."
        ),
        "",
        "| Domain | Candidates | Completed by both | Both correct | Selected |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    selected_by_domain = Counter(str(row["domain"]) for row in selected_rows)
    for domain in ["Mathematical Science", "Physical Science", "Bio&Chem", "Health"]:
        summary_lines.append(
            f"| {domain} | {counts_by_domain[domain]} | {complete_by_domain[domain]} | "
            f"{both_correct_by_domain[domain]} | {selected_by_domain[domain]} |"
        )
    summary_lines.extend(["", "## Selected correct-label distribution", ""])
    summary_lines.append("| Domain | Label counts |")
    summary_lines.append("| --- | --- |")
    for domain in ["Mathematical Science", "Physical Science", "Bio&Chem", "Health"]:
        label_counts = Counter(str(row["correct_answer"]) for row in selected_rows if row["domain"] == domain)
        rendered = ", ".join(f"{label}:{label_counts[label]}" for label in sorted(label_counts))
        summary_lines.append(f"| {domain} | {rendered} |")
    if label_shortfalls:
        summary_lines.extend(["", "## Label-balance shortfalls", ""])
        summary_lines.extend(f"- {item}" for item in label_shortfalls)
    if source_stem_exclusions:
        summary_lines.extend(["", "## Source-stem quality exclusions", ""])
        summary_lines.append(
            "The following otherwise eligible Health candidates were excluded because the source question stem used "
            "recommendation/guideline wording that could cue an institutional-answer frame:"
        )
        summary_lines.extend(f"- {item_id}" for item_id in source_stem_exclusions)
    summary_lines.extend(["", "## Per-model accuracy over completed candidate rows", ""])
    summary_lines.append("| Model | Domain | Correct / Completed | Accuracy |")
    summary_lines.append("| --- | --- | ---: | ---: |")
    for model in models:
        for domain in ["Mathematical Science", "Physical Science", "Bio&Chem", "Health"]:
            completed = complete_by_domain[domain]
            correct = model_correct_by_domain[model][domain]
            acc = f"{100 * correct / completed:.1f}%" if completed else "n/a"
            summary_lines.append(f"| `{model}` | {domain} | {correct}/{completed} | {acc} |")
    return selected_rows, "\n".join(summary_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="mmlu_pro_saturation_candidates_600.jsonl")
    parser.add_argument("--results", default="results/mmlu_pro_saturation_two_weakest_20260502.jsonl.gz")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/gpt-5.4-mini", "google/gemini-3.1-flash-lite-preview"],
    )
    parser.add_argument("--per-domain", type=int, default=50)
    parser.add_argument("--output", default="mmlu_pro_saturated_gt_200.jsonl")
    parser.add_argument("--summary", default="mmlu_pro_saturated_gt_200_summary.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_jsonl(resolve(args.candidates, DATA_DIR))
    results = read_jsonl(resolve(args.results, REPO_ROOT / "Experimental"))
    selected, summary = select_items(candidates, results, args.models, args.per_domain)
    output = resolve(args.output, DATA_DIR)
    summary_path = resolve(args.summary, DATA_DIR)
    write_jsonl(output, selected)
    write_summary(summary_path, summary)
    print(summary)
    print(f"\nwrote {len(selected)} rows to {output}")
    print(f"wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
