"""Compute statistical summaries for release trigger analyses.

This script reads ignored raw result files under ``Experimental/results`` and
writes CSV artifacts for appendix/statistical reporting. It deliberately fails
when the official run files are missing, because using stale pilot files would
make the reported uncertainty invalid.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import plot_trigger_figures as trigger_plot


EXPECTED_BRANCH_ITEMS = {"gt": 200, "ngt": 100}
EXPECTED_MODEL_COUNT = len(trigger_plot.MODELS)
EXPECTED_TRIGGERS = 8
EXPECTED_TONES = 3
EXPECTED_TEMPORAL_SEQUENCES = 14
EXPECTED_REQUIRED_SUFFIXES = [
    "{branch}_first_turn.jsonl.gz",
    "{branch}_trigger_static.jsonl.gz",
    "{branch}_trigger_adaptive.jsonl.gz",
    "{branch}_trigger_temporal_static.jsonl.gz",
    "{branch}_trigger_temporal_adaptive.jsonl.gz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", type=Path, default=root / "Experimental/results")
    parser.add_argument("--out-dir", type=Path, default=root / "Experimental/reports/statistics")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument(
        "--samplek-summary-json",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Optional sample@k/pass@k summary JSON files. These produce "
            "descriptive Wilson intervals from aggregate summaries only; use "
            "raw sample@k records for inferential claims."
        ),
    )
    return parser.parse_args()


def pct(value: float) -> float:
    return 100.0 * value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def required_trigger_files(results_dir: Path, run_id: str) -> list[Path]:
    paths = []
    for branch in ["gt", "ngt"]:
        for suffix in EXPECTED_REQUIRED_SUFFIXES:
            paths.append(results_dir / f"{run_id}_{suffix.format(branch=branch)}")
    return paths


def count_jsonl_gz(path: Path) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def expected_count_for_file(path: Path) -> int:
    name = path.name
    branch = "gt" if "_gt_" in name else "ngt"
    items = EXPECTED_BRANCH_ITEMS[branch]
    if "_first_turn" in name:
        return EXPECTED_MODEL_COUNT * items
    if "_trigger_temporal_" in name:
        return EXPECTED_MODEL_COUNT * items * EXPECTED_TEMPORAL_SEQUENCES
    if "_trigger_" in name:
        return EXPECTED_MODEL_COUNT * items * EXPECTED_TRIGGERS * EXPECTED_TONES
    raise ValueError(f"Cannot infer expected count for {path}")


def validate_official_trigger_run(results_dir: Path, run_id: str) -> list[dict]:
    required = required_trigger_files(results_dir, run_id)
    missing = [path for path in required if not path.exists()]
    if missing:
        candidates = sorted(path.name for path in results_dir.glob("*.jsonl*"))
        details = "\n".join(f"  - {path}" for path in missing)
        candidate_text = "\n".join(f"  - {name}" for name in candidates[:40]) or "  <none>"
        raise SystemExit(
            "Missing official trigger result files; refusing to compute statistics from stale/pilot data.\n"
            f"Expected run_id={run_id!r} under {results_dir}.\n"
            f"Missing files:\n{details}\n"
            f"Visible candidate result files:\n{candidate_text}"
        )

    rows = []
    mismatches = []
    for path in required:
        observed = count_jsonl_gz(path)
        expected = expected_count_for_file(path)
        ok = observed == expected
        rows.append(
            {
                "file": path.name,
                "observed_rows": observed,
                "expected_rows": expected,
                "status": "ok" if ok else "mismatch",
            }
        )
        if not ok:
            mismatches.append(f"  - {path.name}: observed {observed}, expected {expected}")
    if mismatches:
        raise SystemExit(
            "Official run files are present but row counts do not match the release grid.\n"
            + "\n".join(mismatches)
        )
    return rows


def missing_value(value) -> bool:
    return value is None or value == "" or value == []


def validate_records(records: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for source, source_records in records.items():
        seen = set()
        duplicates = 0
        models = set()
        malformed = 0
        for record in source_records:
            models.add(record.get("model"))
            if source == "single":
                key = (
                    record.get("_branch"),
                    record.get("_mode"),
                    record.get("model"),
                    record.get("item_id"),
                    record.get("trigger"),
                    record.get("tone"),
                )
                required = ["_branch", "_mode", "model", "item_id", "trigger", "tone"]
            else:
                key = (
                    record.get("_branch"),
                    record.get("_mode"),
                    record.get("model"),
                    record.get("item_id"),
                    tuple(record.get("trigger_sequence") or []),
                    tuple(record.get("tone_sequence") or []),
                )
                required = ["_branch", "_mode", "model", "item_id", "trigger_sequence", "tone_sequence"]
            if any(missing_value(record.get(field)) for field in required):
                malformed += 1
            if key in seen:
                duplicates += 1
            seen.add(key)
        missing_models = sorted(set(trigger_plot.MODELS) - models)
        extra_models = sorted(models - set(trigger_plot.MODELS))
        rows.append(
            {
                "source": source,
                "records": len(source_records),
                "unique_analysis_keys": len(seen),
                "duplicate_analysis_keys": duplicates,
                "malformed_rows": malformed,
                "model_count": len(models),
                "missing_models": ";".join(missing_models),
                "extra_models": ";".join(extra_models),
                "status": "ok" if not duplicates and not malformed and not missing_models and not extra_models else "fail",
            }
        )
    failures = [row for row in rows if row["status"] != "ok"]
    if failures:
        raise SystemExit(
            "Trigger raw records failed uniqueness/schema validation:\n"
            + "\n".join(json.dumps(row, ensure_ascii=False) for row in failures)
        )
    return rows


def write_rate_ci(tables: dict[str, list[dict]], out_dir: Path) -> None:
    rows = []
    for table_name, table_rows in tables.items():
        if table_name == "confidence_trajectory":
            continue
        for row in table_rows:
            denom = int(row.get("denom") or 0)
            events = int(row.get("events") or 0)
            if denom <= 0:
                continue
            low, high = trigger_plot.wilson_interval(events, denom)
            rows.append(
                {
                    "table": table_name,
                    "branch": row.get("branch", ""),
                    "model": row.get("model", ""),
                    "mode": row.get("mode", ""),
                    "tone": row.get("tone", ""),
                    "stage": row.get("stage", ""),
                    "source": row.get("source", ""),
                    "events": events,
                    "denom": denom,
                    "rate_pct": pct(row["rate"]),
                    "ci95_low_pct": pct(low),
                    "ci95_high_pct": pct(high),
                    "ci_method": "wilson_score_interval",
                }
            )
    write_csv(out_dir / "trigger_rate_wilson_ci.csv", rows)


def confidence_groups() -> list[tuple[str, str, object]]:
    return [
        ("OBJ_vs_SUB", "OBJ", lambda row: row["branch"] == "GT"),
        ("OBJ_vs_SUB", "SUB", lambda row: row["branch"] == "NGT"),
        (
            "stable_vs_sycophantic",
            "stable",
            lambda row: (row["branch"] == "GT" and row["category"] == "preserved")
            or (row["branch"] == "NGT" and row["category"] == "held"),
        ),
        (
            "stable_vs_sycophantic",
            "sycophantic",
            lambda row: (row["branch"] == "GT" and row["category"] == "departed")
            or (row["branch"] == "NGT" and row["category"] == "switched"),
        ),
    ]


def write_confidence_ci(tables: dict[str, list[dict]], out_dir: Path) -> None:
    confidence_rows = tables["confidence_trajectory"]
    rows = []
    for panel, group, predicate in confidence_groups():
        for turn in range(4):
            mean, low, high = trigger_plot.combine_confidence_rows(confidence_rows, predicate, turn)
            n = sum(row.get("n", 0) for row in confidence_rows if row["turn"] == turn and predicate(row))
            rows.append(
                {
                    "panel": panel,
                    "group": group,
                    "turn": turn,
                    "n": n,
                    "mean_confidence": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci_method": "normal_interval_from_trial_confidence_scores",
                }
            )
    write_csv(out_dir / "trigger_confidence_ci.csv", rows)


def event_for_record(record: dict, temporal: bool = False) -> int:
    branch = record["_branch"]
    return int(trigger_plot.metric(record, branch, temporal))


def eligible_single(record: dict) -> bool:
    return trigger_plot.cialdini_single(record) and trigger_plot.denom_ok(record, record["_branch"])


def eligible_temporal(record: dict, stage: str) -> bool:
    return trigger_plot.temporal_stage(record) == stage and trigger_plot.denom_ok(record, record["_branch"])


def paired_bootstrap_diff(
    pairs: list[tuple[int, int]], iterations: int, seed: int
) -> tuple[float, float, float]:
    if not pairs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(pairs)
    observed = sum(a - b for a, b in pairs) / n
    samples = []
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            total += a - b
        samples.append(total / n)
    samples.sort()
    low = samples[int(0.025 * (iterations - 1))]
    high = samples[int(0.975 * (iterations - 1))]
    return observed, low, high


def mcnemar_exact_p(events_a: int, events_b: int) -> float:
    discordant = events_a + events_b
    if discordant == 0:
        return 1.0
    tail = min(events_a, events_b)
    cdf = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * cdf)


def single_event_map(records: list[dict], filters: dict) -> dict[tuple, int]:
    out = {}
    for record in records:
        if any(record.get(key) != value for key, value in filters.items()):
            continue
        if not eligible_single(record):
            continue
        key = (
            record["_branch"],
            record["model"],
            record["item_id"],
            record.get("trigger"),
            record.get("tone"),
        )
        out[key] = event_for_record(record)
    return out


def tone_event_map(records: list[dict], filters: dict) -> dict[tuple, int]:
    out = {}
    for record in records:
        if any(record.get(key) != value for key, value in filters.items()):
            continue
        if not eligible_single(record):
            continue
        key = (
            record["_branch"],
            record["_mode"],
            record["model"],
            record["item_id"],
            record.get("trigger"),
        )
        out[key] = event_for_record(record)
    return out


def write_paired_single_contrasts(records: dict[str, list[dict]], out_dir: Path, iterations: int, seed: int) -> None:
    single = records["single"]
    rows = []
    contrast_specs = []
    for branch in ["GT", "NGT"]:
        contrast_specs.append(
            (
                branch,
                "adaptive_minus_static",
                single_event_map(single, {"_branch": branch, "_mode": "adaptive"}),
                single_event_map(single, {"_branch": branch, "_mode": "static"}),
                "matched by model/item/trigger/tone",
            )
        )
        for mode in ["static", "adaptive"]:
            mild = tone_event_map(single, {"_branch": branch, "_mode": mode, "tone": "mild"})
            moderate = tone_event_map(single, {"_branch": branch, "_mode": mode, "tone": "moderate"})
            strong = tone_event_map(single, {"_branch": branch, "_mode": mode, "tone": "strong"})
            contrast_specs.extend(
                [
                    (branch, f"{mode}_moderate_minus_mild", moderate, mild, "matched by model/item/trigger"),
                    (branch, f"{mode}_strong_minus_moderate", strong, moderate, "matched by model/item/trigger"),
                ]
            )

    for idx, (branch, contrast, a_map, b_map, matched_on) in enumerate(contrast_specs):
        keys = sorted(set(a_map) & set(b_map))
        pairs = [(a_map[key], b_map[key]) for key in keys]
        diff, low, high = paired_bootstrap_diff(pairs, iterations, seed + idx)
        a_only = sum(1 for a, b in pairs if a and not b)
        b_only = sum(1 for a, b in pairs if b and not a)
        rows.append(
            {
                "branch": branch,
                "contrast": contrast,
                "n_pairs": len(pairs),
                "rate_a_pct": pct(sum(a for a, _ in pairs) / len(pairs)) if pairs else 0.0,
                "rate_b_pct": pct(sum(b for _, b in pairs) / len(pairs)) if pairs else 0.0,
                "paired_diff_pp": pct(diff),
                "ci95_low_pp": pct(low),
                "ci95_high_pp": pct(high),
                "mcnemar_a_only": a_only,
                "mcnemar_b_only": b_only,
                "mcnemar_exact_p": mcnemar_exact_p(a_only, b_only),
                "ci_method": "paired_bootstrap_over_matched_cells",
                "test_method": "exact_mcnemar_on_discordant_pairs",
                "matched_on": matched_on,
            }
        )
    write_csv(out_dir / "trigger_paired_single_contrasts.csv", rows)


def cluster_stage_stats(records: dict[str, list[dict]], branch: str, mode: str, stage: str) -> dict[tuple, tuple[int, int]]:
    if stage == "single":
        source = records["single"]
        selected = [
            record
            for record in source
            if record["_branch"] == branch and record["_mode"] == mode and eligible_single(record)
        ]
        temporal = False
    else:
        source = records["temporal"]
        selected = [
            record
            for record in source
            if record["_branch"] == branch and record["_mode"] == mode and eligible_temporal(record, stage)
        ]
        temporal = True

    stats: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for record in selected:
        key = (record["_branch"], record["model"], record["item_id"])
        stats[key][0] += event_for_record(record, temporal)
        stats[key][1] += 1
    return {key: (value[0], value[1]) for key, value in stats.items()}


def rate_from_cluster_sample(keys: list[tuple], stats: dict[tuple, tuple[int, int]]) -> float:
    events = sum(stats.get(key, (0, 0))[0] for key in keys)
    denom = sum(stats.get(key, (0, 0))[1] for key in keys)
    return events / denom if denom else 0.0


def cluster_bootstrap_diff(
    stats_a: dict[tuple, tuple[int, int]],
    stats_b: dict[tuple, tuple[int, int]],
    iterations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    keys = sorted(set(stats_a) | set(stats_b))
    if not keys:
        return 0.0, 0.0, 0.0, 1.0
    rng = random.Random(seed)
    observed = rate_from_cluster_sample(keys, stats_a) - rate_from_cluster_sample(keys, stats_b)
    samples = []
    for _ in range(iterations):
        draw = [keys[rng.randrange(len(keys))] for _ in keys]
        samples.append(rate_from_cluster_sample(draw, stats_a) - rate_from_cluster_sample(draw, stats_b))
    samples.sort()
    low = samples[int(0.025 * (iterations - 1))]
    high = samples[int(0.975 * (iterations - 1))]

    paired_keys = [key for key in keys if stats_a.get(key, (0, 0))[1] > 0 and stats_b.get(key, (0, 0))[1] > 0]
    diffs = []
    weights = []
    for key in paired_keys:
        a_events, a_denom = stats_a[key]
        b_events, b_denom = stats_b[key]
        diffs.append(a_events / a_denom - b_events / b_denom)
        weights.append(min(a_denom, b_denom))
    if not diffs:
        return observed, low, high, 1.0
    obs_signed = sum(diff * weight for diff, weight in zip(diffs, weights)) / sum(weights)
    extreme = 0
    for _ in range(iterations):
        signed = sum((diff if rng.random() < 0.5 else -diff) * weight for diff, weight in zip(diffs, weights)) / sum(weights)
        if abs(signed) >= abs(obs_signed):
            extreme += 1
    p_value = (extreme + 1) / (iterations + 1)
    return observed, low, high, p_value


def write_temporal_contrasts(records: dict[str, list[dict]], out_dir: Path, iterations: int, seed: int) -> None:
    rows = []
    for branch in ["GT", "NGT"]:
        for mode in ["static", "adaptive"]:
            stage_stats = {
                stage: cluster_stage_stats(records, branch, mode, stage)
                for stage in ["single", "same_family", "heterogeneous"]
            }
            for idx, (contrast, a_stage, b_stage) in enumerate(
                [
                    ("heterogeneous_minus_single", "heterogeneous", "single"),
                    ("heterogeneous_minus_same_family", "heterogeneous", "same_family"),
                    ("same_family_minus_single", "same_family", "single"),
                ]
            ):
                stats_a = stage_stats[a_stage]
                stats_b = stage_stats[b_stage]
                keys = sorted(set(stats_a) | set(stats_b))
                events_a = sum(stats_a.get(key, (0, 0))[0] for key in keys)
                denom_a = sum(stats_a.get(key, (0, 0))[1] for key in keys)
                events_b = sum(stats_b.get(key, (0, 0))[0] for key in keys)
                denom_b = sum(stats_b.get(key, (0, 0))[1] for key in keys)
                diff, low, high, p_value = cluster_bootstrap_diff(stats_a, stats_b, iterations, seed + 100 * idx)
                rows.append(
                    {
                        "branch": branch,
                        "mode": mode,
                        "contrast": contrast,
                        "stage_a": a_stage,
                        "stage_b": b_stage,
                        "clusters": len(keys),
                        "rate_a_pct": pct(events_a / denom_a) if denom_a else 0.0,
                        "rate_b_pct": pct(events_b / denom_b) if denom_b else 0.0,
                        "diff_pp": pct(diff),
                        "ci95_low_pp": pct(low),
                        "ci95_high_pp": pct(high),
                        "p_value_sign_flip": p_value,
                        "events_a": events_a,
                        "denom_a": denom_a,
                        "events_b": events_b,
                        "denom_b": denom_b,
                        "ci_method": "cluster_bootstrap_over_model_item_units",
                        "test_method": "paired_cluster_sign_flip_when_matched",
                    }
                )
    write_csv(out_dir / "trigger_temporal_contrasts.csv", rows)


def write_samplek_ci(paths: list[Path], out_dir: Path) -> None:
    rows = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        for section in ["models", "by_success_mode", "by_cue_type"]:
            for row in summary.get(section, []):
                cell_groups = int(row.get("cell_groups") or row.get("valid_item_groups") or 0)
                if cell_groups <= 0:
                    continue
                for key, value in row.items():
                    if value is None:
                        continue
                    if key.startswith("sample_pct@"):
                        k = int(key.split("@", 1)[1])
                        denom = cell_groups * k
                    elif key.startswith("any_pct@"):
                        denom = cell_groups
                    elif key.startswith("est_pass_pct@") or key.startswith("pass@"):
                        denom = int(row.get(f"{key}_groups") or cell_groups)
                    else:
                        continue
                    events = round((float(value) / 100.0) * denom)
                    low, high = trigger_plot.wilson_interval(events, denom)
                    rows.append(
                        {
                            "source_file": path.name,
                            "section": section,
                            "model": row.get("model", ""),
                            "success_mode": row.get("success_mode", ""),
                            "cue_type": row.get("cue_type", ""),
                            "metric": key,
                            "denom": denom,
                            "approx_events": events,
                            "rate_pct": float(value),
                            "ci95_low_pct": pct(low),
                            "ci95_high_pct": pct(high),
                            "ci_method": "descriptive_wilson_from_aggregate_summary",
                            "inference_note": "approximate; do not use for paired or pass@k inferential claims",
                        }
                    )
    write_csv(out_dir / "samplek_summary_descriptive_ci.csv", rows)


def write_manifest(out_dir: Path, validation_rows: list[dict], args: argparse.Namespace) -> None:
    rows = [
        {
            "run_id": args.run_id,
            "results_dir": str(args.results_dir),
            "bootstrap_iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "normal_critical_value_95": NormalDist().inv_cdf(0.975),
        }
    ]
    write_csv(out_dir / "statistical_analysis_manifest.csv", rows)
    write_csv(out_dir / "trigger_run_file_validation.csv", validation_rows)


def main() -> None:
    args = parse_args()
    validation_rows = validate_official_trigger_run(args.results_dir, args.run_id)
    records = trigger_plot.collect_records(args.results_dir, args.run_id)
    record_validation_rows = validate_records(records)
    tables = trigger_plot.build_trigger_figure_tables(records)
    write_manifest(args.out_dir, validation_rows, args)
    write_csv(args.out_dir / "trigger_record_validation.csv", record_validation_rows)
    write_rate_ci(tables, args.out_dir)
    write_confidence_ci(tables, args.out_dir)
    write_paired_single_contrasts(records, args.out_dir, args.bootstrap_iterations, args.seed)
    write_temporal_contrasts(records, args.out_dir, args.bootstrap_iterations, args.seed)
    if args.samplek_summary_json:
        write_samplek_ci(args.samplek_summary_json, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
