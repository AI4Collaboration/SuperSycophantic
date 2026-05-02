#!/usr/bin/env python3
"""Build an MMLU-Pro candidate screen for saturated GT panel selection.

The output is a neutral first-turn JSONL that can be passed directly to
Experimental/run.py first-turn. It intentionally uses only MMLU-Pro categories
that map onto the current GT domains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATA_DIR = Path(__file__).resolve().parents[1]
MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
SOURCE_URL = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"

DOMAIN_BY_CATEGORY = {
    "math": "Mathematical Science",
    "physics": "Physical Science",
    "biology": "Bio&Chem",
    "chemistry": "Bio&Chem",
    "health": "Health",
}

PREFERRED_SOURCES = {
    "math": {
        "ori_mmlu-high_school_mathematics",
        "ori_mmlu-high_school_statistics",
        "ori_mmlu-abstract_algebra",
        "ori_mmlu-college_mathematics",
    },
    "physics": {
        "ori_mmlu-conceptual_physics",
        "ori_mmlu-high_school_physics",
        "ori_mmlu-astronomy",
        "ori_mmlu-college_physics",
    },
    "biology": {
        "ori_mmlu-high_school_biology",
        "ori_mmlu-college_biology",
    },
    "chemistry": {
        "ori_mmlu-high_school_chemistry",
        "ori_mmlu-college_chemistry",
    },
    "health": {
        "ori_mmlu-professional_medicine",
        "ori_mmlu-nutrition",
        "ori_mmlu-human_aging",
        "ori_mmlu-anatomy",
        "ori_mmlu-clinical_knowledge",
        "ori_mmlu-medical_genetics",
        "ori_mmlu-college_medicine",
        "ori_mmlu-virology",
    },
}


def stable_int(*parts: Any) -> int:
    key = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def labels_for_options(options: list[str]) -> list[str]:
    if len(options) > len(string.ascii_uppercase):
        raise ValueError(f"too many options: {len(options)}")
    return list(string.ascii_uppercase[: len(options)])


def correct_label(row: dict[str, Any], labels: list[str]) -> str:
    answer_index = int(row["answer_index"])
    if answer_index < 0 or answer_index >= len(labels):
        raise ValueError(f"invalid answer_index {answer_index}")
    label = labels[answer_index]
    answer = str(row.get("answer", "")).strip().upper()
    if answer and answer != label:
        raise ValueError(f"answer label mismatch for question_id={row['question_id']}: {answer} != {label}")
    return label


def source_allowed(row: dict[str, Any], source_tier: str) -> bool:
    if source_tier == "all":
        return True
    if source_tier != "mmlu_original_moderate":
        raise ValueError(f"unknown source tier: {source_tier}")
    category = str(row["category"])
    return str(row["src"]) in PREFERRED_SOURCES.get(category, set())


def normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    category = str(row["category"])
    domain = DOMAIN_BY_CATEGORY[category]
    labels = labels_for_options(list(row["options"]))
    choices = {label: str(option).strip() for label, option in zip(labels, row["options"])}
    correct = correct_label(row, labels)
    item_id = f"MMLU_PRO-{domain.upper().replace('&', 'AND').replace(' ', '_')}-{index:04d}"
    return {
        "id": item_id,
        "source": f"mmlu_pro:{row['question_id']}",
        "verifiability": "GT",
        "domain": domain,
        "source_dataset": MMLU_PRO_DATASET,
        "source_file": SOURCE_URL,
        "source_url": SOURCE_URL,
        "record_id": str(row["question_id"]),
        "native_id": f"mmlu-pro-test-{row['question_id']}",
        "mmlu_pro_category": category,
        "mmlu_pro_src": row["src"],
        "answer_mode": "multiple_choice",
        "question": str(row["question"]).strip(),
        "choices": choices,
        "correct_answer": correct,
        "source_quote": f"Question: {str(row['question']).strip()} Correct answer ({correct}): {choices[correct]}",
        "gt_panel_role": "saturated_main_candidate",
        "difficulty_status": "mmlu_pro_saturation_screen_candidate",
    }


def target_count_for_stratum(domain: str, category: str, per_domain: int) -> int:
    if domain == "Bio&Chem":
        return per_domain // 2 + (1 if category == "biology" and per_domain % 2 else 0)
    return per_domain


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = load_dataset(MMLU_PRO_DATASET, split=args.split)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        category = str(row["category"])
        if category not in DOMAIN_BY_CATEGORY:
            continue
        if not source_allowed(row, args.source_tier):
            continue
        domain = DOMAIN_BY_CATEGORY[category]
        buckets[(domain, category)].append(dict(row))

    rows: list[dict[str, Any]] = []
    index_by_domain: dict[str, int] = defaultdict(int)
    for domain in ["Mathematical Science", "Physical Science", "Bio&Chem", "Health"]:
        categories = [category for (candidate_domain, category) in buckets if candidate_domain == domain]
        if not categories:
            raise ValueError(f"no MMLU-Pro rows available for {domain}")
        for category in sorted(categories):
            candidates = sorted(
                buckets[(domain, category)],
                key=lambda row: stable_int(args.seed, domain, category, row["question_id"]),
            )
            target = target_count_for_stratum(domain, category, args.per_domain)
            if len(candidates) < target:
                raise ValueError(
                    f"{domain}/{category} has only {len(candidates)} rows for target {target}; "
                    "use --source-tier all or lower --per-domain"
                )
            for row in candidates[:target]:
                index_by_domain[domain] += 1
                rows.append(normalize_row(row, index_by_domain[domain]))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--per-domain", type=int, default=150)
    parser.add_argument("--source-tier", default="mmlu_original_moderate", choices=["mmlu_original_moderate", "all"])
    parser.add_argument("--seed", default="mmlu_pro_saturation_v1")
    parser.add_argument("--output", default="mmlu_pro_saturation_candidates_600.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build(args)
    output = Path(args.output)
    if not output.is_absolute():
        output = DATA_DIR / output
    write_jsonl(output, rows)
    counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["domain"]] += 1
        source_counts[f"{row['domain']}::{row['mmlu_pro_src']}"] += 1
    print(f"wrote {len(rows)} rows to {output}")
    for domain in sorted(counts):
        print(f"{domain}: {counts[domain]}")
    for key in sorted(source_counts):
        print(f"{key}: {source_counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
