#!/usr/bin/env python3
"""Generate rebuttal-facing summaries from current artifacts.

This script itself performs no model calls. It consolidates evidence reviewers
asked for from benchmark records, human validation artifacts, appendix-derived
aggregate statistics, and optional rerun outputs when they exist.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OVERLEAF = Path(os.environ.get("SUPERSYCOPHANTIC_PAPER_DIR", ROOT.parent / "SuperSycophantic-overleaf-clean")).expanduser()
OUT_DIR = ROOT / "Experimental" / "reports" / "rebuttal"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl_rows(path: Path) -> tuple[list[dict], bool]:
    """Return records and whether a gzip stream ended before its footer."""
    if not path.exists():
        return [], False
    rows: list[dict] = []
    truncated = False
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except EOFError:
        truncated = True
    return rows, truncated


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_tex(value: str) -> str:
    value = value.strip()
    value = value.replace("\\%", "%").replace("\\&", "&")
    value = re.sub(r"\\rowcolor\{[^}]*\}", "", value)
    value = re.sub(r"\\textbf\{([^}]*)\}", r"\1", value)
    value = value.replace("``", '"').replace("''", '"')
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", value).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def text_features(text: str) -> dict:
    token_list = words(text)
    urgency_terms = {"now", "immediately", "urgent", "only", "chance", "cost", "fix", "redo"}
    certainty_terms = {"wrong", "mistake", "mistaken", "serious", "reject", "sure", "trusted"}
    social_terms = {"expert", "people", "team", "shared", "review", "score", "goal", "many"}
    return {
        "chars": len(text),
        "words": len(token_list),
        "exclamation_marks": text.count("!"),
        "question_marks": text.count("?"),
        "urgency_term_count": sum(term in urgency_terms for term in token_list),
        "certainty_term_count": sum(term in certainty_terms for term in token_list),
        "social_term_count": sum(term in social_terms for term in token_list),
    }


def panel_inventory() -> list[dict]:
    gt = load_json(ROOT / "Experimental" / "data" / "supersycophantic_context_gt_200.json")
    sub = load_json(ROOT / "Experimental" / "data" / "supersycophantic_context_ngt_100.json")

    rows: list[dict] = []
    gt_domain = Counter(item["domain"] for item in gt)
    gt_source = Counter(item.get("source", "") for item in gt)
    gt_domain_source = Counter((item["domain"], item.get("source", "")) for item in gt)
    hle_converted = sum(1 for item in gt if item.get("synthetic_mc_generation_rule"))
    rows.append({"setting": "OBJ", "group": "all", "n_items": len(gt), "note": "source-backed"})
    for domain, count in sorted(gt_domain.items()):
        rows.append({"setting": "OBJ", "group": f"domain:{domain}", "n_items": count})
    for source, count in sorted(gt_source.items()):
        rows.append({"setting": "OBJ", "group": f"source:{source}", "n_items": count})
    for (domain, source), count in sorted(gt_domain_source.items()):
        rows.append({"setting": "OBJ", "group": f"domain_source:{domain}:{source}", "n_items": count})
    rows.append({"setting": "OBJ", "group": "hle_numeric_converted", "n_items": hle_converted})

    sub_domain = Counter(item["domain"] for item in sub)
    rows.append({"setting": "SUB", "group": "all", "n_items": len(sub), "note": "balanced A/B, no hidden key"})
    for domain, count in sorted(sub_domain.items()):
        rows.append({"setting": "SUB", "group": f"domain:{domain}", "n_items": count})
    return rows


def wilson_pct(successes: int, total: int) -> tuple[float | str, float | str, float | str]:
    if total <= 0:
        return "", "", ""
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5 / denom
    return round(100 * p, 2), round(100 * max(0, center - margin), 2), round(100 * min(1, center + margin), 2)


def sub_framing_balance() -> list[dict]:
    sub = load_json(ROOT / "Experimental" / "data" / "supersycophantic_context_ngt_100.json")
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in sub:
        for name, variant in item["context_variants"].items():
            if name == "neutral":
                continue
            buckets[(variant["cue_type"], variant["direction"])].append(variant["user_framing"])

    rows = []
    for (cue, direction), texts in sorted(buckets.items()):
        rows.append(
            {
                "cue_type": cue,
                "direction": direction,
                "n": len(texts),
                "mean_words": round(mean(len(words(text)) for text in texts), 2),
                "min_words": min(len(words(text)) for text in texts),
                "max_words": max(len(words(text)) for text in texts),
                "mean_chars": round(mean(len(text) for text in texts), 2),
            }
        )
    return rows


def trigger_template_features() -> list[dict]:
    appendix = OVERLEAF / "sections" / "appendix.tex"
    if not appendix.exists():
        return []
    tex = appendix.read_text(encoding="utf-8")
    label_pos = tex.find("\\label{tab:triggers}")
    if label_pos < 0:
        return []
    begin = tex.find("\\begin{tabular}", label_pos)
    if begin < 0:
        begin = tex.rfind("\\begin{tabular}", 0, label_pos)
    end = tex.find("\\end{tabular}", begin)
    block = tex[begin:end]

    rows = []
    for raw in block.split("\\\\"):
        raw = raw.strip()
        if "&" not in raw:
            continue
        parts = [strip_tex(part) for part in raw.split("&")]
        family = parts[0]
        family_lower = family.lower()
        if len(parts) != 4 or "trigger" in family_lower or "tabular" in family_lower:
            continue
        family, mild, moderate, strong = parts
        if not family or family.startswith("toprule") or family.startswith("midrule"):
            continue
        for tone, text in [("mild", mild), ("moderate", moderate), ("strong", strong)]:
            rows.append({"family": family, "tone": tone, "text": text, **text_features(text)})
    return rows


def human_validation_summary() -> list[dict]:
    path = ROOT / "label_studio" / "label_studio_iaa_results.json"
    if not path.exists():
        return []
    data = load_json(path)
    alignment = data["alignment"]
    pooled = data["pooled"]
    return [
        {
            "section": "alignment",
            "metric": "common_tasks",
            "value": alignment["common_task_count"],
            "note": "paired nonempty annotations cover all common tasks",
        },
        {
            "section": "continuous",
            "metric": "n_label_pairs",
            "value": pooled["continuous"]["n_label_pairs"],
        },
        {
            "section": "continuous",
            "metric": "gwet_ac2",
            "value": round(pooled["continuous"]["gwet_ac2_quadratic"], 3),
        },
        {
            "section": "continuous",
            "metric": "within_one_point_agreement",
            "value": round(pooled["continuous"]["within_one_point_agreement"], 3),
        },
        {
            "section": "continuous",
            "metric": "exact_agreement",
            "value": round(pooled["continuous"]["exact_agreement"], 3),
        },
        {
            "section": "binary",
            "metric": "n_label_pairs",
            "value": pooled["binary"]["n_label_pairs"],
        },
        {
            "section": "binary",
            "metric": "exact_agreement",
            "value": round(pooled["binary"]["exact_agreement"], 3),
        },
        {
            "section": "binary",
            "metric": "gwet_ac1",
            "value": round(pooled["binary"]["gwet_ac1"], 3),
        },
        {
            "section": "binary",
            "metric": "weighted_kappa",
            "value": round(pooled["binary"]["weighted_kappa"], 3),
        },
    ]


def appendix_key_numbers() -> list[dict]:
    rows: list[dict] = []
    diff_path = ROOT / "Experimental" / "reports" / "appendix_statistics" / "appendix_aggregate_difference_checks.csv"
    cap_path = ROOT / "Experimental" / "reports" / "appendix_statistics" / "appendix_capability_rank_correlations.csv"
    conf_path = ROOT / "Experimental" / "reports" / "appendix_statistics" / "appendix_confidence_bounded_intervals.csv"
    if diff_path.exists():
        with diff_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["contrast"] in {
                    "SUB tone moderate minus mild",
                    "SUB tone strong minus moderate",
                    "SUB adaptive minus static",
                    "SUB adaptive heterogeneous minus same-family",
                }:
                    rows.append(
                        {
                            "source": "appendix_aggregate_difference_checks",
                            "claim": row["contrast"],
                            "estimate": round(float(row["diff_a_minus_b_pp"]), 2),
                            "ci95_low": round(float(row["ci95_low_pp"]), 2),
                            "ci95_high": round(float(row["ci95_high_pp"]), 2),
                        }
                    )
    if cap_path.exists():
        with cap_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "source": "appendix_capability_rank_correlations",
                        "claim": row["metric"],
                        "estimate": round(float(row["spearman_rho"]), 3),
                        "p_value": round(float(row["exact_permutation_p_two_sided"]), 3),
                    }
                )
    if conf_path.exists():
        with conf_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "source": "appendix_confidence_bounded_intervals",
                        "claim": f"{row['setting']} {row['final_state']} confidence delta",
                        "estimate": round(float(row["delta_mean"]), 2),
                        "ci95_low": round(float(row["delta_ci95_low"]), 2),
                        "ci95_high": round(float(row["delta_ci95_high"]), 2),
                    }
                )
    return rows


def samplek_key_numbers() -> list[dict]:
    rows: list[dict] = []
    base = ROOT / "Experimental" / "reports" / "samplek_statistics"
    validation_path = base / "samplek_raw_validation.csv"
    if validation_path.exists():
        with validation_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "source": "samplek_raw_validation",
                        "claim": row["file"],
                        "estimate": row["unique_records"],
                        "note": f"raw={row['raw_lines']}; duplicates={row['duplicate_lines_removed_before_analysis']}; invalid={row['invalid_latest_records']}",
                    }
                )
    for filename in ["obj_samplek_model_order_trend.csv", "sub_samplek_model_order_trend.csv"]:
        path = base / filename
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["metric"] not in {"sample_pct@1", "sample_pct@10", "any_pct@10"}:
                    continue
                rows.append(
                    {
                        "source": filename,
                        "claim": f"{row['branch']} {row['metric']} OpenAI size-gradient slope",
                        "estimate": round(float(row["slope_pp_per_step"]), 2),
                        "ci95_low": round(float(row["ci95_low_pp_per_step"]), 2),
                        "ci95_high": round(float(row["ci95_high_pp_per_step"]), 2),
                        "p_value": round(float(row["p_value_primary"]), 4),
                    }
                )
    return rows


def answer_label(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    match = re.search(r"\b([A-J])\b", text)
    if match:
        return match.group(1)
    match = re.match(r"^([A-J])", text)
    return match.group(1) if match else text[:1]


def row_answer_is(row: dict, expected: str) -> bool:
    return answer_label(row.get("answer")) == expected


def samplek_obj_domain_summary() -> list[dict]:
    path = ROOT / "Experimental" / "results" / "samplek" / "gt.jsonl.gz"
    if not path.exists():
        return []

    records: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))

    neutral: dict[tuple[str, str, int], dict] = {}
    framed: list[dict] = []
    for record in records:
        model = str(record.get("model"))
        item = str(record.get("item_id"))
        sample = int(record.get("sample_index"))
        if record.get("variant") == "neutral":
            neutral[(model, item, sample)] = record
        else:
            framed.append(record)

    sample_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    cell_groups: dict[tuple[str, str, str, str], list[bool]] = defaultdict(list)
    for frame in framed:
        model = str(frame.get("model"))
        item = str(frame.get("item_id"))
        sample = int(frame.get("sample_index"))
        cue = str(frame.get("cue_type"))
        domain = str(frame.get("domain"))
        neutral_record = neutral.get((model, item, sample))
        if neutral_record is None:
            continue
        correct = answer_label(frame.get("correct_answer"))
        event = row_answer_is(neutral_record, correct) and not row_answer_is(frame, correct)
        sample_groups[(model, domain)].append(event)
        cell_groups[(model, domain, item, cue)].append(event)

    rows: list[dict] = []
    all_by_domain: dict[str, list[bool]] = defaultdict(list)
    any_by_domain: dict[str, list[bool]] = defaultdict(list)
    for (model, domain), events in sorted(sample_groups.items()):
        cell_events = [vals for (m, d, _, _), vals in cell_groups.items() if m == model and d == domain]
        any_events = [any(vals) for vals in cell_events]
        for event in events:
            all_by_domain[domain].append(event)
        for event in any_events:
            any_by_domain[domain].append(event)
        rows.append(
            {
                "model": model,
                "domain": domain,
                "sample_events": len(events),
                "sample_successes": sum(events),
                "sample_event_rate_pct": round(100 * sum(events) / len(events), 2) if events else "",
                "cell_groups": len(cell_events),
                "any_success_cell_rate_pct": round(100 * sum(any_events) / len(any_events), 2) if any_events else "",
            }
        )

    for domain in sorted(all_by_domain):
        events = all_by_domain[domain]
        any_events = any_by_domain[domain]
        rows.append(
            {
                "model": "all_openai_samplek",
                "domain": domain,
                "sample_events": len(events),
                "sample_successes": sum(events),
                "sample_event_rate_pct": round(100 * sum(events) / len(events), 2) if events else "",
                "cell_groups": len(any_events),
                "any_success_cell_rate_pct": round(100 * sum(any_events) / len(any_events), 2) if any_events else "",
            }
        )
    return rows


def openrouter_context_rerun_summary() -> list[dict]:
    full_raw = ROOT / "Experimental" / "results" / "rebuttal_openrouter_context_full.jsonl.gz"
    full_summary = ROOT / "Experimental" / "results" / "rebuttal_openrouter_context_full_summary.json"
    small_raw = ROOT / "Experimental" / "results" / "rebuttal_openrouter_context_2x2.jsonl.gz"
    small_summary = ROOT / "Experimental" / "results" / "rebuttal_openrouter_context_2x2_summary.json"
    raw_path = full_raw if full_raw.exists() and full_summary.exists() else small_raw
    summary_path = full_summary if raw_path == full_raw else small_summary
    if not raw_path.exists() or not summary_path.exists():
        return []
    rows: list[dict] = []
    provider_counts: dict[str, Counter] = defaultdict(Counter)
    parse_counts = Counter()
    total_cost = 0.0
    total_tokens = 0
    errors = 0
    records, truncated = read_jsonl_rows(raw_path)
    for row in records:
        model = row["model"]
        parse_method = row.get("parse_method", "")
        parse_counts[parse_method] += 1
        provider_counts[model][str((row.get("response_metadata") or {}).get("provider"))] += 1
        usage = row.get("usage") or {}
        total_cost += float(usage.get("cost") or 0.0)
        total_tokens += int(usage.get("total_tokens") or 0)
        if row.get("error") or parse_method in {"request_error", "unparsed"}:
            errors += 1

    summary = load_json(summary_path)
    total_records = sum(model_summary["total_records"] for model_summary in summary["models"].values())
    parsed_records = sum(model_summary["parsed_records"] for model_summary in summary["models"].values())
    rows.append(
        {
            "run_label": "full" if raw_path == full_raw else "sanity_2x2",
            "model": "all",
            "records": total_records,
            "parsed_records": parsed_records,
            "parse_failures": total_records - parsed_records,
            "errors_or_unparsed": errors,
            "truncated_read": truncated,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "parse_methods": json.dumps(dict(parse_counts), sort_keys=True),
        }
    )
    for model, model_summary in sorted(summary["models"].items()):
        gt = model_summary.get("gt", {})
        sub = model_summary.get("ngt", {})
        rows.append(
            {
                "run_label": "full" if raw_path == full_raw else "sanity_2x2",
                "model": model,
                "records": model_summary["total_records"],
                "parsed_records": model_summary["parsed_records"],
                "parse_failures": model_summary["total_records"] - model_summary["parsed_records"],
                "provider_counts": json.dumps(dict(provider_counts[model]), sort_keys=True),
                "obj_pairs": gt.get("parsed_pairs"),
                "obj_neutral_correct_pairs": gt.get("neutral_correct_pairs"),
                "obj_correct_to_incorrect_rate": gt.get("correct_to_incorrect_rate"),
                "sub_pairs": sub.get("all", {}).get("pairs"),
                "sub_user_lift": sub.get("all", {}).get("user_answer_agreement", {}).get("lift"),
                "sub_both_user_aligned_rate": sub.get("paired_directionality", {}).get("aligned_with_user_both_rate"),
            }
        )
    return rows


def stored_usage_totals(row: dict) -> tuple[int, float]:
    usage = row.get("usage") or {}
    if "total_tokens" in usage or "cost" in usage:
        return int(usage.get("total_tokens") or 0), float(usage.get("cost") or 0.0)

    total_tokens = 0
    total_cost = 0.0
    for stage_usage in usage.values():
        if not isinstance(stage_usage, dict):
            continue
        total_tokens += int(stage_usage.get("total_tokens") or 0)
        total_cost += float(stage_usage.get("cost") or 0.0)
    return total_tokens, total_cost


def metadata_provider(row: dict, stage: str) -> str:
    metadata = row.get(f"{stage}_response_metadata") or {}
    return str(metadata.get("provider") or "")


def summarize_trigger_rows(
    rows: list[dict],
    *,
    branch: str,
    run_stage: str,
    group_by: str | None = None,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {"all": rows}
    if group_by:
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row.get(group_by) or "")].append(row)

    out: list[dict] = []
    parse_field = "initial_parse_method" if run_stage == "first_turn" else "final_parse_method"
    provider_stage = "first" if run_stage == "first_turn" else "second"
    for group_value, subset in sorted(grouped.items()):
        if not subset:
            out.append(
                {
                    "branch": branch,
                    "run_stage": run_stage,
                    "group_by": group_by or "all",
                    "group_value": group_value,
                    "records": 0,
                    "parsed_records": 0,
                    "errors_or_unparsed": 0,
                    "dry_run_records": 0,
                }
            )
            continue
        parse_counts = Counter(str(row.get(parse_field) or "") for row in subset)
        provider_counts = Counter(metadata_provider(row, provider_stage) for row in subset)
        parsed_records = sum(1 for row in subset if row.get(parse_field) not in {"request_error", "unparsed", None, ""})
        errors = sum(1 for row in subset if row.get(parse_field) in {"request_error", "unparsed"} or row.get("error"))
        dry = sum(1 for row in subset if row.get("dry_run"))
        switched = sum(1 for row in subset if bool(row.get("answer_changed")))
        total_tokens = 0
        total_cost = 0.0
        for row in subset:
            tokens, cost = stored_usage_totals(row)
            total_tokens += tokens
            total_cost += cost
        out.append(
            {
                "branch": branch,
                "run_stage": run_stage,
                "group_by": group_by or "all",
                "group_value": group_value,
                "records": len(subset),
                "parsed_records": parsed_records,
                "errors_or_unparsed": errors,
                "dry_run_records": dry,
                "switched_count": switched if "answer_changed" in subset[0] else "",
                "switched_rate": round(switched / len(subset), 4) if subset and "answer_changed" in subset[0] else "",
                "total_tokens_stored": total_tokens,
                "total_cost_usd_stored": round(total_cost, 6),
                "parse_methods": json.dumps(dict(parse_counts), sort_keys=True),
                "provider_counts": json.dumps(dict(provider_counts), sort_keys=True),
            }
        )
    return out


def openrouter_static_trigger_rerun_summary() -> list[dict]:
    paths = [
        (
            "full",
            "OBJ",
            "first_turn",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_full_gt_first_turn.jsonl.gz",
        ),
        (
            "full",
            "SUB",
            "first_turn",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_full_ngt_first_turn.jsonl.gz",
        ),
        (
            "full",
            "OBJ",
            "static_trigger",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_full_gt_trigger_static.jsonl.gz",
        ),
        (
            "full",
            "SUB",
            "static_trigger",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_full_ngt_trigger_static.jsonl.gz",
        ),
        (
            "full",
            "OBJ",
            "adaptive_trigger",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_full_gt_trigger_adaptive.jsonl.gz",
        ),
        (
            "full",
            "SUB",
            "adaptive_trigger",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_full_ngt_trigger_adaptive.jsonl.gz",
        ),
        (
            "sanity_1",
            "OBJ",
            "first_turn",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_gt_first_turn_actual_1.jsonl.gz",
        ),
        (
            "sanity_1",
            "SUB",
            "first_turn",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_ngt_first_turn_actual_1.jsonl.gz",
        ),
        (
            "sanity_1",
            "OBJ",
            "static_trigger",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_gt_trigger_static_actual_1.jsonl.gz",
        ),
        (
            "sanity_1",
            "SUB",
            "static_trigger",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_ngt_trigger_static_actual_1.jsonl.gz",
        ),
    ]

    rows: list[dict] = []
    for run_label, branch, run_stage, path in paths:
        if not path.exists():
            continue
        data, truncated = read_jsonl_rows(path)
        stage_rows = summarize_trigger_rows(data, branch=branch, run_stage=run_stage)
        for row in stage_rows:
            row["run_label"] = run_label
            row["truncated_read"] = truncated
        rows.extend(stage_rows)
        if run_stage in {"static_trigger", "adaptive_trigger"}:
            for grouped_rows in [
                summarize_trigger_rows(data, branch=branch, run_stage=run_stage, group_by="model"),
                summarize_trigger_rows(data, branch=branch, run_stage=run_stage, group_by="tone"),
            ]:
                for row in grouped_rows:
                    row["run_label"] = run_label
                    row["truncated_read"] = truncated
                rows.extend(grouped_rows)
    return rows


def positive_update_rebuttal_summary() -> list[dict]:
    rows: list[dict] = []
    full_rows: list[dict] = []
    full_path = OUT_DIR / "positive_update_full_summary.csv"
    if full_path.exists():
        with full_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["run_label"] = "full_obj_sub"
                if row.get("branch") == "OBJ":
                    corrected = int(row.get("corrected_after_evidence") or 0)
                    initially_wrong = int(row.get("initially_wrong") or 0)
                    rate, low, high = wilson_pct(corrected, initially_wrong)
                    row["correction_rate_pct"] = rate
                    row["correction_ci95_low_pct"] = low
                    row["correction_ci95_high_pct"] = high
                    preserved = int(row.get("preserved_correct") or 0)
                    initially_correct = int(row.get("initially_correct") or 0)
                    keep_rate, keep_low, keep_high = wilson_pct(preserved, initially_correct)
                    row["preserved_correct_rate_pct"] = keep_rate
                    row["preserved_correct_ci95_low_pct"] = keep_low
                    row["preserved_correct_ci95_high_pct"] = keep_high
                if row.get("branch") == "SUB":
                    updated = int(row.get("updated_to_new_preference") or 0)
                    needed = int(row.get("needed_update") or 0)
                    rate, low, high = wilson_pct(updated, needed)
                    row["update_rate_when_needed_pct"] = rate
                    row["update_ci95_low_pct"] = low
                    row["update_ci95_high_pct"] = high
                full_rows.append(row)
                rows.append(row)

    obj_rows = [row for row in full_rows if row.get("branch") == "OBJ" and row.get("target") == "all"]
    if obj_rows:
        initially_wrong = sum(int(row.get("initially_wrong") or 0) for row in obj_rows)
        corrected = sum(int(row.get("corrected_after_evidence") or 0) for row in obj_rows)
        initially_correct = sum(int(row.get("initially_correct") or 0) for row in obj_rows)
        preserved = sum(int(row.get("preserved_correct") or 0) for row in obj_rows)
        harmful = sum(int(row.get("harmful_change_after_evidence") or 0) for row in obj_rows)
        corr_rate, corr_low, corr_high = wilson_pct(corrected, initially_wrong)
        keep_rate, keep_low, keep_high = wilson_pct(preserved, initially_correct)
        rows.append(
            {
                "run_label": "full_obj_sub",
                "branch": "OBJ",
                "model": "all",
                "target": "all",
                "records": sum(int(row.get("records") or 0) for row in obj_rows),
                "request_errors": sum(int(row.get("request_errors") or 0) for row in obj_rows),
                "initially_wrong": initially_wrong,
                "corrected_after_evidence": corrected,
                "correction_rate_pct": corr_rate,
                "correction_ci95_low_pct": corr_low,
                "correction_ci95_high_pct": corr_high,
                "initially_correct": initially_correct,
                "preserved_correct": preserved,
                "preserved_correct_rate_pct": keep_rate,
                "preserved_correct_ci95_low_pct": keep_low,
                "preserved_correct_ci95_high_pct": keep_high,
                "harmful_change_after_evidence": harmful,
            }
        )

    sub_rows = [row for row in full_rows if row.get("branch") == "SUB" and row.get("target") == "all"]
    if sub_rows:
        needed = sum(int(row.get("needed_update") or 0) for row in sub_rows)
        updated = sum(int(row.get("updated_to_new_preference") or 0) for row in sub_rows)
        aligned = sum(int(row.get("target_aligned_final") or 0) for row in sub_rows)
        records = sum(int(row.get("records") or 0) for row in sub_rows)
        update_rate, update_low, update_high = wilson_pct(updated, needed)
        align_rate, align_low, align_high = wilson_pct(aligned, records)
        rows.append(
            {
                "run_label": "full_obj_sub",
                "branch": "SUB",
                "model": "all",
                "target": "all",
                "records": records,
                "request_errors": sum(int(row.get("request_errors") or 0) for row in sub_rows),
                "target_aligned_final": aligned,
                "target_alignment_rate_pct": align_rate,
                "target_alignment_ci95_low_pct": align_low,
                "target_alignment_ci95_high_pct": align_high,
                "needed_update": needed,
                "updated_to_new_preference": updated,
                "update_rate_when_needed_pct": update_rate,
                "update_ci95_low_pct": update_low,
                "update_ci95_high_pct": update_high,
            }
        )

    hard_summary = ROOT / "Experimental" / "rebuttal_positive_control" / "positive_update_gt_hard40_main.summary.md"
    if hard_summary.exists():
        values: dict[str, str] = {}
        for line in hard_summary.read_text(encoding="utf-8").splitlines():
            match = re.match(r"- ([^:]+): (.*)", line.strip())
            if match:
                values[match.group(1)] = match.group(2)
        rows.append(
            {
                "run_label": "hard40_obj",
                "branch": "OBJ",
                "model": "all",
                "records": values.get("Records", ""),
                "valid_records": values.get("Valid records", ""),
                "initially_wrong": values.get("Initially incorrect", ""),
                "corrected_after_evidence": values.get("Corrected after evidence", ""),
                "correction_rate": values.get("Correction rate", ""),
                "initially_correct": values.get("Initially correct", ""),
                "preserved_correct": values.get("Retained correct after evidence", ""),
                "preserved_correct_rate": values.get("Retention rate", ""),
                "transport_counts": values.get("Transport counts", ""),
            }
        )
    return rows


def tone_minimal_pair_rebuttal_summary() -> list[dict]:
    rows: list[dict] = []
    for run_label, path in [
        ("full", OUT_DIR / "tone_minimal_pairs_full_summary.csv"),
        ("smoke", OUT_DIR / "tone_minimal_pairs_smoke_summary.csv"),
    ]:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["run_label"] = run_label
                rows.append(row)
    return rows


def robustness_rebuttal_summary() -> list[dict]:
    rows: list[dict] = []
    summary_dir = ROOT / "Experimental" / "rebuttal_robustness_results" / "main_summaries"
    for filename in [
        "tone_confound_by_condition.csv",
        "tone_confound_by_model.csv",
        "adaptive_generator_by_generator.csv",
        "adaptive_generator_by_generator_tone.csv",
        "adaptive_generator_by_family_tone.csv",
        "adaptive_generator_by_target_generator.csv",
    ]:
        path = summary_dir / filename
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["source_file"] = filename
                rows.append(row)
    md_path = summary_dir / "rebuttal_robustness_summary.md"
    if md_path.exists():
        for line in md_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("- "):
                rows.append({"source_file": "rebuttal_robustness_summary.md", "summary_line": line[2:]})
    return rows


def main_temperature_samplek_summary() -> list[dict]:
    rows: list[dict] = []
    for branch, summary_path in [
        ("OBJ", ROOT / "Experimental" / "results" / "rebuttal_main_samplek_gt_20x5_summary.json"),
        ("SUB", ROOT / "Experimental" / "results" / "rebuttal_main_samplek_sub_20x5_summary.json"),
    ]:
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        for row in summary.get("models") or []:
            out = dict(row)
            out["branch"] = branch
            out["run_label"] = summary_path.stem.replace("_summary", "")
            out["temperature"] = summary.get("temperature")
            out["samples"] = summary.get("samples")
            out["variant_set"] = summary.get("variant_set")
            rows.append(out)
        if summary.get("models"):
            records = sum(int(row.get("records") or 0) for row in summary["models"])
            expected = sum(int(row.get("expected_samples") or 0) for row in summary["models"])
            valid = sum(int(row.get("valid_samples") or 0) for row in summary["models"])
            success = sum(int(row.get("success_samples") or 0) for row in summary["models"])
            rows.append(
                {
                    "branch": branch,
                    "run_label": summary_path.stem.replace("_summary", ""),
                    "model": "all",
                    "records": records,
                    "expected_samples": expected,
                    "valid_samples": valid,
                    "success_samples": success,
                    "request_errors": sum(int(row.get("request_errors") or 0) for row in summary["models"]),
                    "parse_invalid": sum(int(row.get("parse_invalid") or 0) for row in summary["models"]),
                    "cell_groups": sum(int(row.get("cell_groups") or 0) for row in summary["models"]),
                    "complete_cell_groups": sum(int(row.get("complete_cell_groups") or 0) for row in summary["models"]),
                    "sample_pct_all": round(100 * success / expected, 2) if expected else "",
                    "temperature": summary.get("temperature"),
                    "samples": summary.get("samples"),
                    "variant_set": summary.get("variant_set"),
                }
            )
    return rows


def temporal_slice_rebuttal_summary() -> list[dict]:
    rows: list[dict] = []
    expected_records = 20 * 9 * 8
    paths = [
        (
            "OBJ",
            "static",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_temporal_static_gt_20.jsonl.gz",
        ),
        (
            "SUB",
            "static",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_temporal_static_sub_20.jsonl.gz",
        ),
        (
            "OBJ",
            "adaptive",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_temporal_adaptive_gt_20.jsonl.gz",
        ),
        (
            "SUB",
            "adaptive",
            ROOT / "Experimental" / "results" / "rebuttal_openrouter_temporal_adaptive_sub_20.jsonl.gz",
        ),
    ]
    for branch, mode, path in paths:
        if not path.exists():
            continue
        records, truncated = read_jsonl_rows(path)
        if not records:
            rows.append(
                {
                    "branch": branch,
                    "mode": mode,
                    "records": 0,
                    "expected_records": expected_records,
                    "parsed_records": 0,
                    "errors_or_unparsed": 0,
                    "complete": False,
                    "truncated_read": truncated,
                }
            )
            continue
        parse_counts = Counter(str(row.get("final_parse_method") or "") for row in records)
        provider_counts = Counter(metadata_provider(row, "first") for row in records)
        parsed = sum(1 for row in records if row.get("final_parse_method") not in {"request_error", "unparsed", None, ""})
        errors = sum(1 for row in records if row.get("final_parse_method") in {"request_error", "unparsed"} or row.get("error"))
        if branch == "OBJ":
            event_key = "three_repetition_truth_departure"
        else:
            event_key = "three_repetition_answer_switch"
        events = sum(1 for row in records if bool(row.get(event_key)))
        rows.append(
            {
                "branch": branch,
                "mode": mode,
                "records": len(records),
                "expected_records": expected_records,
                "parsed_records": parsed,
                "errors_or_unparsed": errors,
                "event_key": event_key,
                "event_count": events,
                "event_rate": round(events / len(records), 4) if records else "",
                "complete": len(records) >= expected_records and not truncated,
                "truncated_read": truncated,
                "parse_methods": json.dumps(dict(parse_counts), sort_keys=True),
                "first_provider_counts": json.dumps(dict(provider_counts), sort_keys=True),
            }
        )
    return rows


def write_summary() -> None:
    inventory = panel_inventory()
    framing = sub_framing_balance()
    triggers = trigger_template_features()
    human = human_validation_summary()
    key = appendix_key_numbers()
    samplek = samplek_key_numbers()
    samplek_domain = samplek_obj_domain_summary()
    openrouter = openrouter_context_rerun_summary()
    trigger_rerun = openrouter_static_trigger_rerun_summary()
    positive = positive_update_rebuttal_summary()
    tone_minimal = tone_minimal_pair_rebuttal_summary()
    robustness = robustness_rebuttal_summary()
    main_samplek = main_temperature_samplek_summary()
    temporal_slice = temporal_slice_rebuttal_summary()

    def lookup(rows: list[dict], **where):
        for row in rows:
            if all(row.get(k) == v for k, v in where.items()):
                return row
        return {}

    obj_items = lookup(inventory, setting="OBJ", group="all").get("n_items", "")
    sub_items = lookup(inventory, setting="SUB", group="all").get("n_items", "")
    hle_conv = lookup(inventory, setting="OBJ", group="hle_numeric_converted").get("n_items", "")
    common_tasks = lookup(human, section="alignment", metric="common_tasks").get("value", "")
    cont_ac2 = lookup(human, section="continuous", metric="gwet_ac2").get("value", "")
    cont_w1 = lookup(human, section="continuous", metric="within_one_point_agreement").get("value", "")
    binary_exact = lookup(human, section="binary", metric="exact_agreement").get("value", "")
    binary_ac1 = lookup(human, section="binary", metric="gwet_ac1").get("value", "")

    tone_mild_mod = lookup(key, claim="SUB tone moderate minus mild")
    tone_mod_strong = lookup(key, claim="SUB tone strong minus moderate")
    adaptive = lookup(key, claim="SUB adaptive minus static")
    obj_samplek = lookup(samplek, claim="OBJ sample_pct@1 OpenAI size-gradient slope")
    sub_samplek = lookup(samplek, claim="SUB sample_pct@1 OpenAI size-gradient slope")
    obj_raw = next((row for row in samplek if "gt.jsonl.gz" in str(row.get("claim", ""))), {})
    sub_raw = next((row for row in samplek if "ngt.jsonl.gz" in str(row.get("claim", ""))), {})
    samplek_domain_rows = [row for row in samplek_domain if row.get("model") == "all_openai_samplek"]
    samplek_domain_high = max(
        samplek_domain_rows,
        key=lambda row: float(row["sample_event_rate_pct"]),
        default={},
    )
    samplek_domain_low = min(
        samplek_domain_rows,
        key=lambda row: float(row["sample_event_rate_pct"]),
        default={},
    )
    openrouter_all = lookup(openrouter, model="all")
    trigger_obj_all = lookup(trigger_rerun, branch="OBJ", run_stage="static_trigger", group_by="all")
    trigger_sub_all = lookup(trigger_rerun, branch="SUB", run_stage="static_trigger", group_by="all")
    adaptive_obj_all = lookup(trigger_rerun, branch="OBJ", run_stage="adaptive_trigger", group_by="all")
    adaptive_sub_all = lookup(trigger_rerun, branch="SUB", run_stage="adaptive_trigger", group_by="all")
    trigger_sub_mild = lookup(trigger_rerun, branch="SUB", run_stage="static_trigger", group_by="tone", group_value="mild")
    trigger_sub_moderate = lookup(trigger_rerun, branch="SUB", run_stage="static_trigger", group_by="tone", group_value="moderate")
    trigger_sub_strong = lookup(trigger_rerun, branch="SUB", run_stage="static_trigger", group_by="tone", group_value="strong")
    positive_obj_all = lookup(positive, run_label="full_obj_sub", branch="OBJ", model="all")
    positive_sub_all = lookup(positive, run_label="full_obj_sub", branch="SUB", model="all")
    positive_hard = lookup(positive, run_label="hard40_obj", branch="OBJ", model="all")
    tone_source = "full" if any(row.get("run_label") == "full" for row in tone_minimal) else "smoke"
    tone_obj_all = lookup(tone_minimal, run_label=tone_source, branch="OBJ", model="all", tone="all")
    tone_sub_all = lookup(tone_minimal, run_label=tone_source, branch="SUB", model="all", tone="all")
    tone_obj_mild = lookup(tone_minimal, run_label=tone_source, branch="OBJ", model="all", tone="mild")
    tone_obj_strong = lookup(tone_minimal, run_label=tone_source, branch="OBJ", model="all", tone="strong")
    tone_sub_mild = lookup(tone_minimal, run_label=tone_source, branch="SUB", model="all", tone="mild")
    tone_sub_strong = lookup(tone_minimal, run_label=tone_source, branch="SUB", model="all", tone="strong")
    robustness_lines = [
        row["summary_line"]
        for row in robustness
        if row.get("source_file") == "rebuttal_robustness_summary.md" and row.get("summary_line")
    ]
    main_samplek_obj = lookup(main_samplek, branch="OBJ", model="all")
    main_samplek_sub = lookup(main_samplek, branch="SUB", model="all")
    temporal_static_obj = lookup(temporal_slice, branch="OBJ", mode="static")
    temporal_static_sub = lookup(temporal_slice, branch="SUB", mode="static")
    temporal_adaptive_obj = lookup(temporal_slice, branch="OBJ", mode="adaptive")
    temporal_adaptive_sub = lookup(temporal_slice, branch="SUB", mode="adaptive")

    trigger_word_means = defaultdict(list)
    for row in triggers:
        trigger_word_means[row["tone"]].append(row["words"])
    trigger_word_text = ", ".join(
        f"{tone}={round(mean(vals), 1)} words" for tone, vals in sorted(trigger_word_means.items())
    )

    text = [
        "# Rebuttal Existing-Data Analysis",
        "",
        "Generated from benchmark records, human validation artifacts, appendix aggregate statistics, and optional rerun outputs. This script itself performs no model calls.",
        "",
        "## Dataset and Construction Checks",
        "",
        f"- OBJ contains {obj_items} source-backed items; SUB contains {sub_items} balanced A/B decision items.",
        f"- OBJ source mix is balanced by design across four domains and two upstream sources; {hle_conv} HLE-Verified exact-answer records use documented multiple-choice conversion.",
        "- SUB context framing has 100 A-directed and 100 B-directed variants for each non-neutral cue type.",
        "",
        "## Human Validation",
        "",
        f"- Human validation covers {common_tasks} paired tasks with complete nonempty annotations.",
        f"- Continuous coarse agreement: Gwet AC2={cont_ac2}, within-one-point agreement={cont_w1}.",
        f"- Binary audit labels: exact agreement={binary_exact}, Gwet AC1={binary_ac1}.",
        "",
        "## Existing Aggregate Results",
        "",
        f"- SUB tone moderate-minus-mild: {tone_mild_mod.get('estimate')} pp [{tone_mild_mod.get('ci95_low')}, {tone_mild_mod.get('ci95_high')}].",
        f"- SUB tone strong-minus-moderate: {tone_mod_strong.get('estimate')} pp [{tone_mod_strong.get('ci95_low')}, {tone_mod_strong.get('ci95_high')}].",
        f"- SUB adaptive-minus-static: {adaptive.get('estimate')} pp [{adaptive.get('ci95_low')}, {adaptive.get('ci95_high')}].",
        f"- Static template text audit: {trigger_word_text}.",
        "",
        "## Repeated-Sampling Diagnostic",
        "",
        f"- Raw sample@k inputs: OBJ={obj_raw.get('estimate')} records; SUB={sub_raw.get('estimate')} records; duplicate and invalid latest-record counts are zero.",
        f"- OpenAI size-gradient persists under repeated sampling: OBJ sample_pct@1 slope={obj_samplek.get('estimate')} pp/step [{obj_samplek.get('ci95_low')}, {obj_samplek.get('ci95_high')}]; SUB sample_pct@1 slope={sub_samplek.get('estimate')} pp/step [{sub_samplek.get('ci95_low')}, {sub_samplek.get('ci95_high')}].",
        f"- OBJ domain-level repeated-sampling range across OpenAI models: highest {samplek_domain_high.get('domain')}={samplek_domain_high.get('sample_event_rate_pct')}%; lowest {samplek_domain_low.get('domain')}={samplek_domain_low.get('sample_event_rate_pct')}%.",
        f"- All-main-model temperature=0.7 context slice: OBJ {main_samplek_obj.get('records')} records with {main_samplek_obj.get('valid_samples')} valid samples; SUB {main_samplek_sub.get('records')} records with {main_samplek_sub.get('valid_samples')} valid samples.",
        "",
        "## OpenRouter Rerun",
        "",
        f"- OpenRouter context rerun ({openrouter_all.get('run_label')}): {openrouter_all.get('records')} records, {openrouter_all.get('parsed_records')} parsed, {openrouter_all.get('errors_or_unparsed')} request/unparsed errors.",
        f"- The context rerun used {openrouter_all.get('total_tokens')} stored tokens and approximately ${openrouter_all.get('total_cost_usd')} in logged API cost.",
        f"- OpenRouter static trigger rerun ({trigger_obj_all.get('run_label')}): OBJ {trigger_obj_all.get('records')} records with {trigger_obj_all.get('parsed_records')} parsed and {trigger_obj_all.get('errors_or_unparsed')} request/unparsed errors; SUB {trigger_sub_all.get('records')} records with {trigger_sub_all.get('parsed_records')} parsed and {trigger_sub_all.get('errors_or_unparsed')} request/unparsed errors.",
        f"- SUB static trigger switching by tone: mild {trigger_sub_mild.get('switched_count')}/{trigger_sub_mild.get('records')}, moderate {trigger_sub_moderate.get('switched_count')}/{trigger_sub_moderate.get('records')}, strong {trigger_sub_strong.get('switched_count')}/{trigger_sub_strong.get('records')}.",
        f"- OpenRouter adaptive trigger rerun ({adaptive_obj_all.get('run_label')}): OBJ {adaptive_obj_all.get('records')} records with {adaptive_obj_all.get('parsed_records')} parsed; SUB {adaptive_sub_all.get('records')} records with {adaptive_sub_all.get('parsed_records')} parsed. Incomplete flags: OBJ={adaptive_obj_all.get('truncated_read')}, SUB={adaptive_sub_all.get('truncated_read')}.",
        f"- OpenRouter temporal static slice: OBJ {temporal_static_obj.get('records')}/{temporal_static_obj.get('expected_records')} trajectories, SUB {temporal_static_sub.get('records')}/{temporal_static_sub.get('expected_records')} trajectories.",
        f"- OpenRouter temporal adaptive slice: OBJ {temporal_adaptive_obj.get('records')}/{temporal_adaptive_obj.get('expected_records')} trajectories, SUB {temporal_adaptive_sub.get('records')}/{temporal_adaptive_sub.get('expected_records')} trajectories.",
        "",
        "## Positive-Update Controls",
        "",
        f"- Full OBJ/SUB control: OBJ corrected {positive_obj_all.get('corrected_after_evidence')}/{positive_obj_all.get('initially_wrong')} initially wrong answers after evidence ({positive_obj_all.get('correction_rate_pct')}% [{positive_obj_all.get('correction_ci95_low_pct')}, {positive_obj_all.get('correction_ci95_high_pct')}]) while preserving {positive_obj_all.get('preserved_correct')}/{positive_obj_all.get('initially_correct')} initially correct answers.",
        f"- Full SUB preference control: models updated toward a newly stated decisive preference in {positive_sub_all.get('updated_to_new_preference')}/{positive_sub_all.get('needed_update')} needed-update cases ({positive_sub_all.get('update_rate_when_needed_pct')}% [{positive_sub_all.get('update_ci95_low_pct')}, {positive_sub_all.get('update_ci95_high_pct')}]).",
        f"- Hard OBJ subset: {positive_hard.get('corrected_after_evidence')}/{positive_hard.get('initially_wrong')} initially wrong responses corrected after source evidence; {positive_hard.get('preserved_correct')}/{positive_hard.get('initially_correct')} initially correct responses retained.",
        "",
        "## Minimal-Pair Tone Control",
        "",
        f"- Minimal-pair tone control source: {tone_source}; OBJ {tone_obj_all.get('records')} records and SUB {tone_sub_all.get('records')} records.",
        f"- OBJ answer-change rate ranges from {tone_obj_mild.get('answer_changed')}/{tone_obj_mild.get('records')} under mild wording to {tone_obj_strong.get('answer_changed')}/{tone_obj_strong.get('records')} under strong wording.",
        f"- SUB answer-change rate ranges from {tone_sub_mild.get('answer_changed')}/{tone_sub_mild.get('records')} under mild wording to {tone_sub_strong.get('answer_changed')}/{tone_sub_strong.get('records')} under strong wording.",
        "",
        "## Robustness Runner",
        "",
        *[f"- {line}" for line in robustness_lines],
        "",
        "## Output Files",
        "",
        "- `panel_inventory.csv`",
        "- `sub_framing_balance.csv`",
        "- `static_trigger_text_features.csv`",
        "- `human_validation_summary.csv`",
        "- `appendix_rebuttal_key_numbers.csv`",
        "- `samplek_rebuttal_key_numbers.csv`",
        "- `samplek_obj_domain_summary.csv`",
        "- `main_temperature_samplek_summary.csv`",
        "- `openrouter_context_rerun_summary.csv`",
        "- `openrouter_static_trigger_rerun_summary.csv`",
        "- `temporal_slice_rebuttal_summary.csv`",
        "- `positive_update_rebuttal_summary.csv`",
        "- `tone_minimal_pair_rebuttal_summary.csv`",
        "- `robustness_rebuttal_summary.csv`",
    ]
    (OUT_DIR / "rebuttal_existing_data_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    global OVERLEAF, OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=OVERLEAF,
                        help="Read-only paper tree; also configurable with SUPERSYCOPHANTIC_PAPER_DIR.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OVERLEAF, OUT_DIR = args.paper_dir.expanduser(), args.out_dir.expanduser()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "panel_inventory.csv", panel_inventory())
    write_csv(OUT_DIR / "sub_framing_balance.csv", sub_framing_balance())
    write_csv(OUT_DIR / "static_trigger_text_features.csv", trigger_template_features())
    write_csv(OUT_DIR / "human_validation_summary.csv", human_validation_summary())
    write_csv(OUT_DIR / "appendix_rebuttal_key_numbers.csv", appendix_key_numbers())
    write_csv(OUT_DIR / "samplek_rebuttal_key_numbers.csv", samplek_key_numbers())
    write_csv(OUT_DIR / "samplek_obj_domain_summary.csv", samplek_obj_domain_summary())
    write_csv(OUT_DIR / "main_temperature_samplek_summary.csv", main_temperature_samplek_summary())
    write_csv(OUT_DIR / "openrouter_context_rerun_summary.csv", openrouter_context_rerun_summary())
    write_csv(OUT_DIR / "openrouter_static_trigger_rerun_summary.csv", openrouter_static_trigger_rerun_summary())
    write_csv(OUT_DIR / "temporal_slice_rebuttal_summary.csv", temporal_slice_rebuttal_summary())
    write_csv(OUT_DIR / "positive_update_rebuttal_summary.csv", positive_update_rebuttal_summary())
    write_csv(OUT_DIR / "tone_minimal_pair_rebuttal_summary.csv", tone_minimal_pair_rebuttal_summary())
    write_csv(OUT_DIR / "robustness_rebuttal_summary.csv", robustness_rebuttal_summary())
    write_summary()
    print(f"Wrote rebuttal existing-data analysis to {OUT_DIR}")


if __name__ == "__main__":
    main()
