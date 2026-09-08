"""Compute raw-record statistical tests for OpenAI sample@k diagnostics.

This script intentionally reads the raw ``Experimental/results/samplek`` JSONL
files rather than aggregate summaries. It evaluates the release-facing paired
events:

- OBJ: matched neutral/framed samples where neutral is correct and framed is
  incorrect.
- SUB: matched A/B framing pairs where the A-directed sample selects A and the
  B-directed sample selects B.

For each event stream it reports cluster bootstrap confidence intervals and
paired sign-flip tests over matched item/cue cells.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--gt-input", type=Path, default=root / "Experimental/results/samplek/gt.jsonl.gz")
    parser.add_argument("--ngt-input", type=Path, default=root / "Experimental/results/samplek/ngt.jsonl.gz")
    parser.add_argument("--out-dir", type=Path, default=root / "Experimental/reports/samplek_statistics")
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260507)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_bh_q_values(rows: list[dict[str, Any]], p_key: str = "p_value_primary") -> None:
    indexed = [
        (idx, float(row[p_key]))
        for idx, row in enumerate(rows)
        if row.get(p_key) not in {"", None}
    ]
    if not indexed:
        return
    ordered = sorted(indexed, key=lambda pair: pair[1])
    m = len(ordered)
    q_by_idx: dict[int, float] = {}
    running = 1.0
    for rank_from_end, (idx, p_value) in enumerate(reversed(ordered), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        q_by_idx[idx] = min(1.0, running)
    for idx, row in enumerate(rows):
        row["q_value_bh_within_table"] = q_by_idx.get(idx, "")


def read_latest_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    lines = 0
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            lines += 1
            record = json.loads(line)
            key = (str(record["cell_id"]), str(record["model"]), int(record["sample_index"]))
            latest[key] = record
    return list(latest.values()), lines


def label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def parsed_valid(record: dict[str, Any]) -> bool:
    return bool(record.get("answer")) and record.get("parse_method") not in {
        None,
        "unparsed",
        "request_error",
    }


def answer_is(record: dict[str, Any], expected: str | None) -> bool:
    return label(record.get("answer")) == label(expected)


def validate_records(records: list[dict[str, Any]], path: Path, raw_lines: int) -> dict[str, Any]:
    invalid = [record for record in records if not parsed_valid(record)]
    if invalid:
        raise SystemExit(f"{path} has {len(invalid)} invalid latest records; rerun/clean before statistics.")
    return {
        "file": str(path),
        "raw_lines": raw_lines,
        "unique_records": len(records),
        "duplicate_lines_removed_before_analysis": raw_lines - len(records),
        "invalid_latest_records": len(invalid),
    }


def build_obj_events(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[bool]]:
    neutral: dict[tuple[str, str, int], dict[str, Any]] = {}
    framed: list[dict[str, Any]] = []
    for record in records:
        variant = str(record.get("variant"))
        sample = int(record.get("sample_index"))
        key = (str(record.get("model")), str(record.get("item_id")), sample)
        if variant == "neutral":
            neutral[key] = record
        else:
            framed.append(record)

    events: dict[tuple[str, str, str], dict[int, bool]] = defaultdict(dict)
    missing_neutral = 0
    for frame in framed:
        model = str(frame.get("model"))
        item = str(frame.get("item_id"))
        sample = int(frame.get("sample_index"))
        cue = str(frame.get("cue_type"))
        neutral_record = neutral.get((model, item, sample))
        if neutral_record is None:
            missing_neutral += 1
            continue
        correct = label(frame.get("correct_answer"))
        neutral_correct = answer_is(neutral_record, correct)
        framed_wrong = not answer_is(frame, correct)
        events[(model, item, cue)][sample] = neutral_correct and framed_wrong

    if missing_neutral:
        raise SystemExit(f"OBJ statistics missing {missing_neutral} matched neutral records.")
    return finalize_events(events, "OBJ")


def build_sub_events(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[bool]]:
    directional: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for record in records:
        direction = label(record.get("pressure_direction"))
        if direction not in {"A", "B"}:
            continue
        key = (
            str(record.get("model")),
            str(record.get("item_id")),
            str(record.get("cue_type")),
            int(record.get("sample_index")),
            direction,
        )
        directional[key] = record

    events: dict[tuple[str, str, str], dict[int, bool]] = defaultdict(dict)
    missing_pair = 0
    for model, item, cue, sample, direction in list(directional):
        if direction != "A":
            continue
        a_record = directional[(model, item, cue, sample, "A")]
        b_record = directional.get((model, item, cue, sample, "B"))
        if b_record is None:
            missing_pair += 1
            continue
        events[(model, item, cue)][sample] = answer_is(a_record, "A") and answer_is(b_record, "B")

    if missing_pair:
        raise SystemExit(f"SUB statistics missing {missing_pair} matched B-direction records.")
    return finalize_events(events, "SUB")


def finalize_events(
    event_by_sample: dict[tuple[str, str, str], dict[int, bool]],
    branch: str,
) -> dict[tuple[str, str, str], list[bool]]:
    finalized = {}
    incomplete = []
    for key, by_sample in event_by_sample.items():
        samples = sorted(by_sample)
        expected = list(range(max(samples) + 1)) if samples else []
        if samples != expected:
            incomplete.append((key, samples))
            continue
        finalized[key] = [bool(by_sample[index]) for index in expected]
    if incomplete:
        raise SystemExit(f"{branch} statistics found {len(incomplete)} incomplete sample cells.")
    return finalized


def sample_value(events: list[bool], k: int) -> float:
    first_k = events[:k]
    if len(first_k) != k:
        raise ValueError(f"cell has {len(first_k)} samples, cannot compute k={k}")
    return sum(1 for event in first_k if event) / k


def any_value(events: list[bool], k: int) -> float:
    first_k = events[:k]
    if len(first_k) != k:
        raise ValueError(f"cell has {len(first_k)} samples, cannot compute k={k}")
    return 1.0 if any(first_k) else 0.0


def values_for_metric(unit_events: dict[tuple[str, str, str], list[bool]], metric: str, k: int) -> dict[tuple[str, str, str], float]:
    fn = sample_value if metric == "sample_pct" else any_value
    return {key: fn(events, k) for key, events in unit_events.items()}


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def bootstrap_ci(values: list[float], iterations: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    boot = []
    n = len(values)
    for _ in range(iterations):
        boot.append(mean(values[rng.randrange(n)] for _ in range(n)))
    return percentile(boot, 0.025), percentile(boot, 0.975)


def paired_diff_ci_and_p(
    diffs: list[float],
    iterations: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    if not diffs:
        return 0.0, 0.0, 1.0
    low, high = bootstrap_ci(diffs, iterations, rng)
    observed = abs(mean(diffs))
    if observed == 0:
        return low, high, 1.0
    extreme = 0
    n = len(diffs)
    for _ in range(iterations):
        flipped_mean = mean(diff * (1 if rng.random() < 0.5 else -1) for diff in diffs)
        if abs(flipped_mean) >= observed:
            extreme += 1
    return low, high, (extreme + 1) / (iterations + 1)


def logsumexp(log_values: list[float]) -> float:
    if not log_values:
        return float("-inf")
    max_value = max(log_values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in log_values))


def binomial_logpmf(k: int, n: int, p: float = 0.5) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def exact_two_sided_binomial_p(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    tail = min(successes, trials - successes)
    log_terms = [binomial_logpmf(k, trials, 0.5) for k in range(tail + 1)]
    return min(1.0, 2.0 * math.exp(logsumexp(log_terms)))


def exact_mcnemar_p(values_a: list[float], values_b: list[float]) -> tuple[int, int, float]:
    a_only = 0
    b_only = 0
    for left, right in zip(values_a, values_b):
        left_bool = bool(round(left))
        right_bool = bool(round(right))
        if left_bool and not right_bool:
            a_only += 1
        elif right_bool and not left_bool:
            b_only += 1
    return a_only, b_only, exact_two_sided_binomial_p(min(a_only, b_only), a_only + b_only)


def model_unit_key(unit_key: tuple[str, str, str]) -> tuple[str, str]:
    _model, item, cue = unit_key
    return item, cue


def write_model_metric_rows(
    branch: str,
    unit_events: dict[tuple[str, str, str], list[bool]],
    k_values: list[int],
    iterations: int,
    rng: random.Random,
    out_dir: Path,
) -> None:
    rows = []
    by_model: dict[str, dict[tuple[str, str, str], list[bool]]] = defaultdict(dict)
    for key, events in unit_events.items():
        model = key[0]
        by_model[model][key] = events

    for model in DEFAULT_MODELS:
        model_events = by_model.get(model, {})
        if not model_events:
            continue
        for metric in ["sample_pct", "any_pct"]:
            for k in k_values:
                values = list(values_for_metric(model_events, metric, k).values())
                low, high = bootstrap_ci(values, iterations, rng)
                rows.append(
                    {
                        "branch": branch,
                        "model": model,
                        "metric": f"{metric}@{k}",
                        "matched_units": len(values),
                        "rate_pct": 100.0 * mean(values),
                        "ci95_low_pct": 100.0 * low,
                        "ci95_high_pct": 100.0 * high,
                        "ci_method": "cluster_bootstrap_over_item_cue_units",
                    }
                )
    write_csv(out_dir / f"{branch.lower()}_samplek_metric_ci.csv", rows)


def write_model_contrast_rows(
    branch: str,
    unit_events: dict[tuple[str, str, str], list[bool]],
    k_values: list[int],
    iterations: int,
    rng: random.Random,
    out_dir: Path,
) -> None:
    rows = []
    by_model_metric: dict[tuple[str, str, int], dict[tuple[str, str], float]] = {}
    for metric in ["sample_pct", "any_pct"]:
        for k in k_values:
            values = values_for_metric(unit_events, metric, k)
            for model in DEFAULT_MODELS:
                model_values = {
                    model_unit_key(key): value
                    for key, value in values.items()
                    if key[0] == model
                }
                by_model_metric[(model, metric, k)] = model_values

    for model_a, model_b in combinations(DEFAULT_MODELS, 2):
        for metric in ["sample_pct", "any_pct"]:
            for k in k_values:
                a_values = by_model_metric[(model_a, metric, k)]
                b_values = by_model_metric[(model_b, metric, k)]
                matched = sorted(set(a_values) & set(b_values))
                left = [a_values[key] for key in matched]
                right = [b_values[key] for key in matched]
                diffs = [b_value - a_value for a_value, b_value in zip(left, right)]
                low, high, p_value = paired_diff_ci_and_p(diffs, iterations, rng)
                a_only = ""
                b_only = ""
                mcnemar_p: float | str = ""
                primary_p = p_value
                primary_test = "paired_sign_flip_over_matched_item_cue_units"
                if metric == "any_pct":
                    a_only, b_only, mcnemar_p = exact_mcnemar_p(left, right)
                    primary_p = float(mcnemar_p)
                    primary_test = "exact_mcnemar_on_matched_binary_any_events"
                rows.append(
                    {
                        "branch": branch,
                        "model_a": model_a,
                        "model_b": model_b,
                        "contrast": f"{model_b} minus {model_a}",
                        "metric": f"{metric}@{k}",
                        "matched_units": len(matched),
                        "rate_a_pct": 100.0 * mean(a_values[key] for key in matched),
                        "rate_b_pct": 100.0 * mean(b_values[key] for key in matched),
                        "diff_pp": 100.0 * mean(diffs),
                        "ci95_low_pp": 100.0 * low,
                        "ci95_high_pp": 100.0 * high,
                        "p_value_sign_flip": p_value,
                        "mcnemar_a_only": a_only,
                        "mcnemar_b_only": b_only,
                        "p_value_exact_mcnemar": mcnemar_p,
                        "p_value_primary": primary_p,
                        "ci_method": "paired_cluster_bootstrap_over_matched_item_cue_units",
                        "test_method": primary_test,
                    }
                )
    add_bh_q_values(rows)
    write_csv(out_dir / f"{branch.lower()}_samplek_model_contrasts.csv", rows)


def write_k_contrast_rows(
    branch: str,
    unit_events: dict[tuple[str, str, str], list[bool]],
    k_values: list[int],
    iterations: int,
    rng: random.Random,
    out_dir: Path,
) -> None:
    rows = []
    target_k = max(k_values)
    comparison_ks = [k for k in k_values if k != target_k]
    for model in DEFAULT_MODELS:
        model_events = {key: events for key, events in unit_events.items() if key[0] == model}
        if not model_events:
            continue
        for metric in ["sample_pct", "any_pct"]:
            target_values = values_for_metric(model_events, metric, target_k)
            for base_k in comparison_ks:
                base_values = values_for_metric(model_events, metric, base_k)
                matched = sorted(set(base_values) & set(target_values))
                left = [base_values[key] for key in matched]
                right = [target_values[key] for key in matched]
                diffs = [target - base for base, target in zip(left, right)]
                low, high, p_value = paired_diff_ci_and_p(diffs, iterations, rng)
                base_only = ""
                target_only = ""
                mcnemar_p: float | str = ""
                primary_p = p_value
                primary_test = "paired_sign_flip_over_matched_item_cue_units"
                if metric == "any_pct":
                    base_only, target_only, mcnemar_p = exact_mcnemar_p(left, right)
                    primary_p = float(mcnemar_p)
                    primary_test = "exact_mcnemar_on_matched_binary_any_events"
                rows.append(
                    {
                        "branch": branch,
                        "model": model,
                        "contrast": f"{metric}@{target_k} minus {metric}@{base_k}",
                        "matched_units": len(matched),
                        "rate_base_pct": 100.0 * mean(base_values[key] for key in matched),
                        "rate_target_pct": 100.0 * mean(target_values[key] for key in matched),
                        "diff_pp": 100.0 * mean(diffs),
                        "ci95_low_pp": 100.0 * low,
                        "ci95_high_pp": 100.0 * high,
                        "p_value_sign_flip": p_value,
                        "mcnemar_base_only": base_only,
                        "mcnemar_target_only": target_only,
                        "p_value_exact_mcnemar": mcnemar_p,
                        "p_value_primary": primary_p,
                        "ci_method": "paired_cluster_bootstrap_over_matched_item_cue_units",
                        "test_method": primary_test,
                    }
                )
    add_bh_q_values(rows)
    write_csv(out_dir / f"{branch.lower()}_samplek_k_contrasts.csv", rows)


def write_cue_contrast_rows(
    branch: str,
    unit_events: dict[tuple[str, str, str], list[bool]],
    k_values: list[int],
    iterations: int,
    rng: random.Random,
    out_dir: Path,
) -> None:
    rows = []
    cues = sorted({key[2] for key in unit_events})
    for model in DEFAULT_MODELS:
        model_events = {key: events for key, events in unit_events.items() if key[0] == model}
        if not model_events:
            continue
        for metric in ["sample_pct", "any_pct"]:
            for k in k_values:
                values = values_for_metric(model_events, metric, k)
                by_cue = {
                    cue: {
                        item: value
                        for (_model, item, unit_cue), value in values.items()
                        if unit_cue == cue
                    }
                    for cue in cues
                }
                for cue_a, cue_b in combinations(cues, 2):
                    a_values = by_cue.get(cue_a, {})
                    b_values = by_cue.get(cue_b, {})
                    matched = sorted(set(a_values) & set(b_values))
                    left = [a_values[key] for key in matched]
                    right = [b_values[key] for key in matched]
                    diffs = [b_value - a_value for a_value, b_value in zip(left, right)]
                    low, high, p_value = paired_diff_ci_and_p(diffs, iterations, rng)
                    a_only = ""
                    b_only = ""
                    mcnemar_p: float | str = ""
                    primary_p = p_value
                    primary_test = "paired_sign_flip_over_matched_items"
                    if metric == "any_pct":
                        a_only, b_only, mcnemar_p = exact_mcnemar_p(left, right)
                        primary_p = float(mcnemar_p)
                        primary_test = "exact_mcnemar_on_matched_binary_any_events"
                    rows.append(
                        {
                            "branch": branch,
                            "model": model,
                            "cue_a": cue_a,
                            "cue_b": cue_b,
                            "contrast": f"{cue_b} minus {cue_a}",
                            "metric": f"{metric}@{k}",
                            "matched_items": len(matched),
                            "rate_a_pct": 100.0 * mean(left),
                            "rate_b_pct": 100.0 * mean(right),
                            "diff_pp": 100.0 * mean(diffs),
                            "ci95_low_pp": 100.0 * low,
                            "ci95_high_pp": 100.0 * high,
                            "p_value_sign_flip": p_value,
                            "mcnemar_a_only": a_only,
                            "mcnemar_b_only": b_only,
                            "p_value_exact_mcnemar": mcnemar_p,
                            "p_value_primary": primary_p,
                            "ci_method": "paired_cluster_bootstrap_over_matched_items",
                            "test_method": primary_test,
                        }
                    )
    add_bh_q_values(rows)
    write_csv(out_dir / f"{branch.lower()}_samplek_cue_contrasts.csv", rows)


def model_order_slope(values: list[float]) -> float:
    xs = list(range(len(values)))
    x_mean = mean(xs)
    y_mean = mean(values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom


def write_model_order_trend_rows(
    branch: str,
    unit_events: dict[tuple[str, str, str], list[bool]],
    k_values: list[int],
    iterations: int,
    rng: random.Random,
    out_dir: Path,
) -> None:
    rows = []
    model_order = DEFAULT_MODELS
    for metric in ["sample_pct", "any_pct"]:
        for k in k_values:
            values = values_for_metric(unit_events, metric, k)
            by_unit: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
            for (model, item, cue), value in values.items():
                by_unit[(item, cue)][model] = value
            slopes = []
            for model_values in by_unit.values():
                if all(model in model_values for model in model_order):
                    slopes.append(model_order_slope([model_values[model] for model in model_order]))
            low, high, p_value = paired_diff_ci_and_p(slopes, iterations, rng)
            rows.append(
                {
                    "branch": branch,
                    "model_order": " -> ".join(model_order),
                    "metric": f"{metric}@{k}",
                    "matched_units": len(slopes),
                    "slope_pp_per_step": 100.0 * mean(slopes),
                    "ci95_low_pp_per_step": 100.0 * low,
                    "ci95_high_pp_per_step": 100.0 * high,
                    "p_value_sign_flip": p_value,
                    "p_value_primary": p_value,
                    "q_value_bh_within_table": "",
                    "ci_method": "cluster_bootstrap_over_matched_item_cue_units",
                    "test_method": "paired_sign_flip_on_per_unit_ordered_model_slope",
                    "interpretation": "positive means the outcome rate rises from GPT-5.4 to mini to nano",
                }
            )
    add_bh_q_values(rows)
    write_csv(out_dir / f"{branch.lower()}_samplek_model_order_trend.csv", rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    gt_records, gt_raw_lines = read_latest_records(args.gt_input)
    ngt_records, ngt_raw_lines = read_latest_records(args.ngt_input)
    validation_rows = [
        validate_records(gt_records, args.gt_input, gt_raw_lines),
        validate_records(ngt_records, args.ngt_input, ngt_raw_lines),
    ]

    obj_events = build_obj_events(gt_records)
    sub_events = build_sub_events(ngt_records)

    for branch, events in [("OBJ", obj_events), ("SUB", sub_events)]:
        write_model_metric_rows(branch, events, args.k_values, args.bootstrap_iterations, rng, args.out_dir)
        write_model_contrast_rows(branch, events, args.k_values, args.bootstrap_iterations, rng, args.out_dir)
        write_k_contrast_rows(branch, events, args.k_values, args.bootstrap_iterations, rng, args.out_dir)
        write_cue_contrast_rows(branch, events, args.k_values, args.bootstrap_iterations, rng, args.out_dir)
        write_model_order_trend_rows(branch, events, args.k_values, args.bootstrap_iterations, rng, args.out_dir)

    manifest = [
        {
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "k_values": " ".join(str(k) for k in args.k_values),
            "obj_event": "neutral_correct_and_framed_incorrect",
            "sub_event": "a_directed_selects_a_and_b_directed_selects_b",
            "obj_matched_units_per_model": len(obj_events) // len(DEFAULT_MODELS),
            "sub_matched_units_per_model": len(sub_events) // len(DEFAULT_MODELS),
            "included_tests": (
                "cluster bootstrap CIs; paired sign-flip tests for fractional sample_pct; "
                "exact McNemar tests for binary any_pct contrasts; Benjamini-Hochberg q-values within each contrast table"
            ),
            "excluded_tests": (
                "no OBJ-vs-SUB significance test because the branches measure different constructs over different item pools"
            ),
        }
    ]
    write_csv(args.out_dir / "samplek_statistical_test_manifest.csv", manifest)
    write_csv(args.out_dir / "samplek_raw_validation.csv", validation_rows)
    print(args.out_dir)


if __name__ == "__main__":
    main()
