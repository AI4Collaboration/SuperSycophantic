#!/usr/bin/env python3
"""Prepare a GPQA JSONL panel for the single-turn trigger screen."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://github.com/idavidrein/gpqa/raw/main/dataset.zip"
DEFAULT_PASSWORD = "deserted-untie-orchid"
SPLIT_TO_FILE = {
    "main": "dataset/gpqa_main.csv",
    "diamond": "dataset/gpqa_diamond.csv",
    "experts": "dataset/gpqa_experts.csv",
    "extended": "dataset/gpqa_extended.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--split", default="diamond", choices=sorted(SPLIT_TO_FILE))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--zip", type=Path, default=None, help="Optional local official GPQA dataset.zip.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Official GPQA dataset.zip URL.")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Password from the official GPQA README.")
    return parser.parse_args()


def get_zip_path(args: argparse.Namespace) -> Path:
    if args.zip is not None:
        return args.zip
    cache_dir = Path(tempfile.gettempdir()) / "supersycophantic_gpqa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "dataset.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(args.url, zip_path)
    return zip_path


def read_split(zip_path: Path, split: str, password: str) -> list[dict[str, str]]:
    member = SPLIT_TO_FILE[split]
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(member, pwd=password.encode("utf-8"))
    text = raw.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def make_item(row: dict[str, str], split: str, index: int, seed: int) -> dict[str, Any]:
    choices = [
        {"text": row["Correct Answer"].strip(), "is_correct": True},
        {"text": row["Incorrect Answer 1"].strip(), "is_correct": False},
        {"text": row["Incorrect Answer 2"].strip(), "is_correct": False},
        {"text": row["Incorrect Answer 3"].strip(), "is_correct": False},
    ]

    rng = random.Random(seed + index)
    rng.shuffle(choices)
    labels = ["A", "B", "C", "D"]
    labelled_choices = {label: choice["text"] for label, choice in zip(labels, choices)}
    correct_label = next(label for label, choice in zip(labels, choices) if choice["is_correct"])
    pressure_label = next(label for label, choice in zip(labels, choices) if not choice["is_correct"])

    return {
        "id": f"gpqa-{split}-{index + 1:04d}",
        "source": f"gpqa_{split}",
        "record_id": row.get("Record ID", ""),
        "high_level_domain": row.get("High-level domain", ""),
        "subdomain": row.get("Subdomain", ""),
        "question": row["Question"].strip(),
        "choices": labelled_choices,
        "correct_answer": correct_label,
        "correct_answer_text": labelled_choices[correct_label],
        "pressure_target": pressure_label,
        "pressure_target_text": labelled_choices[pressure_label],
    }


def main() -> int:
    args = parse_args()
    rows = read_split(get_zip_path(args), args.split, args.password)
    if args.max_items is not None:
        rows = rows[: args.max_items]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps(make_item(row, args.split, index, args.seed), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
