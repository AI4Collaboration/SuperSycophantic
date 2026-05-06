"""Run OpenAI-only GT neutral pass@k sampling.

This is a diagnostic script, separate from the canonical pass@1 eval runners.
It samples each GT base item multiple times under neutral context and estimates
pass@k from the same sample set.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import run as trigger_run


DEFAULT_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
]
DEFAULT_K_VALUES = [1, 3, 5, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="Experimental/data/supersycophantic_context_gt_200.json",
        help="GT context JSON file. Neutral context prompts are used.",
    )
    parser.add_argument("--output", default=None, help="Output JSONL.GZ path.")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--samples", type=int, default=10, help="Number of samples per item/model.")
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=120)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument(
        "--rerun-invalid",
        action="store_true",
        help="When resuming, rerun records whose latest parse_method is unparsed/request_error.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def default_paths(base_dir: Path) -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"openai_passk_gt_neutral_n10_{stamp}"
    out_dir = base_dir / "Experimental" / "results" / "passk"
    return (
        out_dir / f"{stem}.jsonl.gz",
        out_dir / f"{stem}_summary.json",
        out_dir / f"{stem}_summary.csv",
    )


def read_gt_items(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if max_items is not None:
        items = items[: max(0, max_items)]
    return items


def planned_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return str(record["item_id"]), str(record["model"]), int(record["sample_index"])


def has_valid_parse(record: dict[str, Any]) -> bool:
    return bool(record.get("answer")) and record.get("parse_method") not in {
        None,
        "unparsed",
        "request_error",
    }


def completed_keys(path: Path, rerun_invalid: bool = False) -> set[tuple[str, str, int]]:
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return set()
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(key in record for key in ["item_id", "model", "sample_index"]):
                latest[planned_key(record)] = record
    if not rerun_invalid:
        return set(latest)
    return {key for key, record in latest.items() if has_valid_parse(record)}


def iter_tasks(
    items: list[dict[str, Any]],
    models: list[str],
    samples: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for item in items:
        neutral = (item.get("context_variants") or {}).get("neutral") or {}
        prompt = neutral.get("prompt")
        if not prompt:
            raise ValueError(f"Missing neutral prompt for {item.get('id')}")
        for model in models:
            for sample_index in range(samples):
                tasks.append(
                    {
                        "item": item,
                        "model": model,
                        "sample_index": sample_index,
                        "prompt": prompt,
                    }
                )
    return tasks


def make_base_record(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    item = task["item"]
    return {
        "run_type": "openai_passk_gt_neutral",
        "item_id": item.get("id"),
        "domain": item.get("domain"),
        "source": item.get("source"),
        "model": task["model"],
        "sample_index": task["sample_index"],
        "samples_per_item": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "correct_answer": item.get("correct_answer"),
        "prompt_variant": "neutral",
    }


def pass_at_k(n: int, c: int, k: int) -> float | None:
    if n <= 0 or k <= 0 or n < k:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize_records(records: list[dict[str, Any]], k_values: list[int], samples: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        model = str(record.get("model"))
        by_model[model].append(record)
        grouped[(model, str(record.get("item_id")))].append(record)

    summary_rows = []
    for model, model_records in sorted(by_model.items()):
        item_scores: dict[int, list[float]] = {k: [] for k in k_values}
        first_sample_correct = []
        any_correct = []
        valid_group_count = 0
        complete_group_count = 0
        correct_total = 0
        valid_total = 0
        error_total = 0
        parse_invalid_total = 0

        for (group_model, _item_id), rows in grouped.items():
            if group_model != model:
                continue
            rows = sorted(rows, key=lambda row: int(row.get("sample_index", 0)))
            errors = [row for row in rows if row.get("parse_method") == "request_error"]
            invalid = [row for row in rows if row.get("parse_method") in {None, "unparsed"}]
            # For pass@k, invalid or errored generations are failed samples, not exclusions.
            n = samples
            c = sum(1 for row in rows if row.get("correct") is True)
            valid_total += len(rows) - len(errors) - len(invalid)
            correct_total += c
            error_total += len(errors)
            parse_invalid_total += len(invalid)
            valid_group_count += 1
            any_correct.append(c > 0)
            first = next((row for row in rows if int(row.get("sample_index", -1)) == 0), None)
            first_sample_correct.append(bool(first and first.get("correct") is True))
            for k in k_values:
                value = pass_at_k(n, c, k)
                if value is not None:
                    item_scores[k].append(value)
            if len(rows) >= samples:
                complete_group_count += 1

        expected_samples = valid_group_count * samples
        row = {
            "model": model,
            "records": len(model_records),
            "expected_samples": expected_samples,
            "valid_samples": valid_total,
            "correct_samples": correct_total,
            "request_errors": error_total,
            "parse_invalid": parse_invalid_total,
            "valid_item_groups": valid_group_count,
            "complete_item_groups": complete_group_count,
            "sample_accuracy": correct_total / expected_samples if expected_samples else None,
            "first_sample_accuracy": (
                sum(first_sample_correct) / len(first_sample_correct)
                if first_sample_correct
                else None
            ),
            "oracle_at_n": sum(any_correct) / len(any_correct) if any_correct else None,
        }
        for k in k_values:
            values = item_scores[k]
            row[f"pass@{k}"] = sum(values) / len(values) if values else None
            row[f"pass@{k}_groups"] = len(values)
        summary_rows.append(row)

    return {
        "k_values": k_values,
        "models": summary_rows,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if all(key in record for key in ["item_id", "model", "sample_index"]):
                    latest[planned_key(record)] = record
    return list(latest.values())


async def run(args: argparse.Namespace) -> int:
    base_dir = repo_root()
    trigger_run.load_dotenv(base_dir / ".env")

    default_output, default_summary_json, default_summary_csv = default_paths(base_dir)
    output_path = resolve_path(base_dir, args.output) if args.output else default_output
    summary_json_path = resolve_path(base_dir, args.summary_json) if args.summary_json else default_summary_json
    summary_csv_path = resolve_path(base_dir, args.summary_csv) if args.summary_csv else default_summary_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    items = read_gt_items(resolve_path(base_dir, args.input), args.max_items)
    models = trigger_run.resolve_models(args.models)
    tasks = iter_tasks(items, models, args.samples)
    done = completed_keys(output_path, args.rerun_invalid)
    remaining = [task for task in tasks if planned_key(make_base_record(task, args)) not in done]
    planned = len(tasks)
    print(
        f"planned={planned} remaining={len(remaining)} items={len(items)} "
        f"models={len(models)} samples={args.samples} output={output_path}",
        flush=True,
    )
    if args.dry_run:
        return 0

    api_key = trigger_run.openrouter_api_key_for_run(models, args.dry_run)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()
    completed = len(done)
    correct_by_model = defaultdict(int)
    done_by_model = defaultdict(int)
    extra_payload = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.top_p is not None:
        extra_payload["top_p"] = args.top_p

    async with trigger_run.OpenRouterAsyncClient(api_key, args.concurrency):
        with gzip.open(output_path, "at", encoding="utf-8") as out:

            async def write(record: dict[str, Any]) -> None:
                nonlocal completed
                async with write_lock:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    completed += 1
                    model = str(record.get("model"))
                    done_by_model[model] += 1
                    if record.get("correct") is True:
                        correct_by_model[model] += 1
                    if completed % 50 == 0 or completed == planned:
                        parts = []
                        for model in models:
                            done_model = done_by_model[model]
                            acc = correct_by_model[model] / done_model if done_model else 0.0
                            parts.append(f"{model}: {done_model} acc={acc:.3f}")
                        print(f"completed={completed}/{planned} | " + " | ".join(parts), flush=True)

            async def run_one(task: dict[str, Any]) -> None:
                item = task["item"]
                record = make_base_record(task, args)
                try:
                    response = await trigger_run.async_openrouter_chat(
                        api_key,
                        task["model"],
                        [
                            {"role": "system", "content": trigger_run.SYSTEM_PROMPT},
                            {"role": "user", "content": task["prompt"]},
                        ],
                        semaphore,
                        request_timeout=args.request_timeout,
                        max_attempts=args.max_attempts,
                        extra_payload=extra_payload,
                    )
                    text = trigger_run.response_text(response)
                    answer, confidence, parse_method = trigger_run.extract_item_answer(text, item)
                    correct = (
                        trigger_run.answers_equal(item, answer, item.get("correct_answer"))
                        if answer
                        else False
                    )
                    record.update(
                        {
                            "answer": answer,
                            "confidence": confidence,
                            "parse_method": parse_method,
                            "correct": correct,
                            "response_text": text,
                            "response_metadata": trigger_run.response_metadata(response),
                            "usage": response.get("usage"),
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "answer": None,
                            "confidence": None,
                            "parse_method": "request_error",
                            "correct": False,
                            "response_text": "",
                            "response_metadata": None,
                            "usage": None,
                            "response_error": str(exc),
                        }
                    )
                await write(record)

            await asyncio.gather(*(run_one(task) for task in remaining))

    records = load_records(output_path)
    summary = summarize_records(records, sorted(set(args.k_values)), args.samples)
    summary.update(
        {
            "output": str(output_path),
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "input": str(resolve_path(base_dir, args.input)),
            "samples": args.samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "request_timeout": args.request_timeout,
            "max_attempts": args.max_attempts,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fieldnames = list(summary["models"][0].keys()) if summary["models"] else []
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["models"])
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
