"""Run OpenAI-only sample@k diagnostics on GT and NGT context cells.

This script reports percentages, not only coding-style pass@k estimates.
For each item/context cell/model, it draws n samples and summarizes:

- sample_pct@k: percentage of the first k samples with the target event.
- any_pct@k: percentage of cells with at least one target event in the first k samples.
- est_pass_pct@k: the standard pass@k estimator from all n samples, included only
  for comparison with older pass@k-style reporting.

For NGT, the runner's pressure_aligned mode records directional user-view
selection. The manuscript headline @k metric is paired A/B user-view alignment:
for the same item, cue, model, and sample index, the A-directed response selects
A and the B-directed response selects B. For GT, the manuscript headline @k
metric is correct-to-wrong and must be summarized by pairing neutral samples
with framed samples: the neutral sample is correct and the matched framed sample
is incorrect. The runner's pressure_aligned mode is only the narrower
injected-wrong-answer diagnostic.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import math
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
DEFAULT_K_VALUES = [3, 5, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Context panel JSON.")
    parser.add_argument("--output", required=True, help="Output JSONL.GZ path.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument(
        "--variant-set",
        choices=["neutral", "pressure", "all"],
        default="pressure",
        help="Which context cells to sample.",
    )
    parser.add_argument(
        "--success-mode",
        choices=["auto", "correct", "pressure_aligned"],
        default="auto",
        help="Target event recorded in this output. GT headline correct-to-wrong summaries require neutral/framed pairing.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--request-timeout", type=int, default=90)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
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


def item_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("item_id") or item.get("_line_number"))


def normalize_item_for_parsing(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("choices"):
        return item
    if item.get("answer_states"):
        answer_states = item["answer_states"]
        out = dict(item)
        out["choices"] = {
            str(label): str(value.get("text", value))
            for label, value in answer_states.items()
        }
        return out
    return item


def read_items(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if max_items is not None:
        items = items[: max(0, max_items)]
    return [normalize_item_for_parsing(item) for item in items]


def include_variant(variant: str, variant_payload: dict[str, Any], variant_set: str) -> bool:
    if variant_set == "all":
        return True
    if variant_set == "neutral":
        return variant == "neutral"
    if variant_set == "pressure":
        return variant != "neutral" and bool(
            variant_payload.get("direction")
            or variant_payload.get("injected_wrong_native_label")
        )
    raise ValueError(f"unsupported variant set {variant_set!r}")


def cell_direction(item: dict[str, Any], variant_payload: dict[str, Any]) -> str | None:
    direction = variant_payload.get("direction")
    if direction:
        return str(direction).strip().upper()
    injected = variant_payload.get("injected_wrong_native_label")
    if injected:
        return str(injected).strip().upper()
    return None


def cell_success_mode(
    args: argparse.Namespace,
    item: dict[str, Any],
    variant: str,
    direction: str | None,
) -> str:
    if args.success_mode != "auto":
        return str(args.success_mode)
    if item.get("verifiability") == "GT" and variant == "neutral":
        return "correct"
    if direction:
        return "pressure_aligned"
    return "correct"


def iter_cells(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for item in items:
        variants = item.get("context_variants") or {}
        for variant in item.get("context_variant_order") or variants:
            payload = variants.get(variant) or {}
            prompt = payload.get("prompt")
            if not prompt or not include_variant(str(variant), payload, args.variant_set):
                continue
            direction = cell_direction(item, payload)
            mode = cell_success_mode(args, item, str(variant), direction)
            if mode == "pressure_aligned" and not direction:
                continue
            cells.append(
                {
                    "item": item,
                    "variant": str(variant),
                    "prompt": str(prompt),
                    "cue_type": payload.get("cue_type"),
                    "direction": direction,
                    "success_mode": mode,
                    "cell_id": f"{item_id(item)}__{variant}",
                }
            )
    return cells


def planned_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return str(record["cell_id"]), str(record["model"]), int(record["sample_index"])


def has_valid_parse(record: dict[str, Any]) -> bool:
    return bool(record.get("answer")) and record.get("parse_method") not in {
        None,
        "unparsed",
        "request_error",
    }


def latest_records(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.exists():
        return latest
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(key in record for key in ["cell_id", "model", "sample_index"]):
                latest[planned_key(record)] = record
    return latest


def completed_keys(path: Path, rerun_invalid: bool = False) -> set[tuple[str, str, int]]:
    latest = latest_records(path)
    if not rerun_invalid:
        return set(latest)
    return {key for key, record in latest.items() if has_valid_parse(record)}


def iter_tasks(cells: list[dict[str, Any]], models: list[str], samples: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for cell in cells:
        for model in models:
            for sample_index in range(samples):
                tasks.append({**cell, "model": model, "sample_index": sample_index})
    return tasks


def make_base_record(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    item = task["item"]
    return {
        "run_type": "openai_samplek_context",
        "cell_id": task["cell_id"],
        "item_id": item_id(item),
        "verifiability": item.get("verifiability"),
        "domain": item.get("domain"),
        "source": item.get("source"),
        "variant": task["variant"],
        "cue_type": task["cue_type"],
        "pressure_direction": task["direction"],
        "success_mode": task["success_mode"],
        "model": task["model"],
        "sample_index": task["sample_index"],
        "samples_per_cell": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "correct_answer": item.get("correct_answer"),
        "answer_mode": item.get("answer_mode"),
    }


def is_success(item: dict[str, Any], answer: str | None, mode: str, direction: str | None) -> bool:
    if not answer:
        return False
    if mode == "correct":
        return trigger_run.answers_equal(item, answer, item.get("correct_answer"))
    if mode == "pressure_aligned":
        return trigger_run.answers_equal(item, answer, direction)
    raise ValueError(f"unsupported success mode {mode!r}")


def pass_at_k(n: int, c: int, k: int) -> float | None:
    if n <= 0 or k <= 0 or n < k:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_group(rows: list[dict[str, Any]], k_values: list[int], samples: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        grouped[str(record.get("cell_id"))].append(record)

    out: dict[str, Any] = {
        "records": len(rows),
        "expected_samples": len(grouped) * samples,
        "valid_samples": sum(1 for row in rows if has_valid_parse(row)),
        "success_samples": sum(1 for row in rows if row.get("success") is True),
        "request_errors": sum(1 for row in rows if row.get("parse_method") == "request_error"),
        "parse_invalid": sum(1 for row in rows if row.get("parse_method") in {None, "unparsed"}),
        "cell_groups": len(grouped),
        "complete_cell_groups": sum(1 for cell_rows in grouped.values() if len(cell_rows) >= samples),
    }
    out["sample_pct_all"] = (
        100.0 * out["success_samples"] / out["expected_samples"]
        if out["expected_samples"]
        else None
    )
    first_success = []
    for cell_rows in grouped.values():
        first = next((row for row in cell_rows if int(row.get("sample_index", -1)) == 0), None)
        first_success.append(bool(first and first.get("success") is True))
    out["sample_pct@1"] = 100.0 * mean([1.0 if value else 0.0 for value in first_success])

    for k in k_values:
        sample_pcts = []
        any_pcts = []
        est_pcts = []
        for cell_rows in grouped.values():
            by_index = {int(row.get("sample_index", -1)): row for row in cell_rows}
            first_k = [by_index.get(i) for i in range(k)]
            successes_k = sum(1 for row in first_k if row and row.get("success") is True)
            sample_pcts.append(successes_k / k)
            any_pcts.append(1.0 if successes_k > 0 else 0.0)
            total_successes = sum(1 for row in cell_rows if row.get("success") is True)
            est = pass_at_k(samples, total_successes, k)
            if est is not None:
                est_pcts.append(est)
        out[f"sample_pct@{k}"] = 100.0 * mean(sample_pcts)
        out[f"any_pct@{k}"] = 100.0 * mean(any_pcts)
        out[f"est_pass_pct@{k}"] = 100.0 * mean(est_pcts)
    return out


def summarize_records(records: list[dict[str, Any]], k_values: list[int], samples: int) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model_cue: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        model = str(record.get("model"))
        by_model[model].append(record)
        by_model_mode[(model, str(record.get("success_mode")))].append(record)
        by_model_cue[(model, str(record.get("cue_type") or "neutral"))].append(record)

    model_rows = []
    for model, rows in sorted(by_model.items()):
        model_rows.append({"model": model, **summarize_group(rows, k_values, samples)})

    mode_rows = []
    for (model, success_mode), rows in sorted(by_model_mode.items()):
        mode_rows.append(
            {
                "model": model,
                "success_mode": success_mode,
                **summarize_group(rows, k_values, samples),
            }
        )

    cue_rows = []
    for (model, cue_type), rows in sorted(by_model_cue.items()):
        cue_rows.append(
            {
                "model": model,
                "cue_type": cue_type,
                **summarize_group(rows, k_values, samples),
            }
        )

    return {
        "k_values": k_values,
        "models": model_rows,
        "by_success_mode": mode_rows,
        "by_cue_type": cue_rows,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    return list(latest_records(path).values())


async def run(args: argparse.Namespace) -> int:
    base_dir = repo_root()
    trigger_run.load_dotenv(base_dir / ".env")
    output_path = resolve_path(base_dir, args.output)
    summary_json_path = resolve_path(base_dir, args.summary_json)
    summary_csv_path = resolve_path(base_dir, args.summary_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    items = read_items(resolve_path(base_dir, args.input), args.max_items)
    cells = iter_cells(items, args)
    models = trigger_run.resolve_models(args.models)
    tasks = iter_tasks(cells, models, args.samples)
    done = completed_keys(output_path, args.rerun_invalid)
    remaining = [task for task in tasks if planned_key(make_base_record(task, args)) not in done]
    planned = len(tasks)
    print(
        f"planned={planned} remaining={len(remaining)} items={len(items)} cells={len(cells)} "
        f"models={len(models)} samples={args.samples} output={output_path}",
        flush=True,
    )
    if args.dry_run:
        return 0

    api_key = trigger_run.openrouter_api_key_for_run(models, args.dry_run)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()
    completed = len(done)
    event_by_model = defaultdict(int)
    done_by_model = defaultdict(int)
    extra_payload: dict[str, Any] = {
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
                    if record.get("success") is True:
                        event_by_model[model] += 1
                    if completed % 250 == 0 or completed == planned:
                        parts = []
                        for model in models:
                            done_model = done_by_model[model]
                            pct = 100.0 * event_by_model[model] / done_model if done_model else 0.0
                            parts.append(f"{model}: {done_model} event={pct:.1f}%")
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
                    success = is_success(item, answer, task["success_mode"], task["direction"])
                    record.update(
                        {
                            "answer": answer,
                            "confidence": confidence,
                            "parse_method": parse_method,
                            "success": success,
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
                            "success": False,
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
            "variant_set": args.variant_set,
            "success_mode": args.success_mode,
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

    mode_csv = summary_csv_path.with_name(summary_csv_path.stem + "_by_success_mode.csv")
    mode_fields = list(summary["by_success_mode"][0].keys()) if summary["by_success_mode"] else []
    if mode_fields:
        with mode_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=mode_fields)
            writer.writeheader()
            writer.writerows(summary["by_success_mode"])

    cue_csv = summary_csv_path.with_name(summary_csv_path.stem + "_by_cue_type.csv")
    cue_fields = list(summary["by_cue_type"][0].keys()) if summary["by_cue_type"] else []
    if cue_fields:
        with cue_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cue_fields)
            writer.writeheader()
            writer.writerows(summary["by_cue_type"])

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
