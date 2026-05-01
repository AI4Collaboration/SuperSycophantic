#!/usr/bin/env python3
"""Summarize DeepSeek V4 Flash context and neutral-trigger runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from run import existing_jsonl_path, open_text


MODEL = "deepseek/deepseek-v4-flash"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = existing_jsonl_path(path)
    if not path.exists():
        return records
    with open_text(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100 * value:.1f}%"


def ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in records if r.get("eligible")]
    eligible_parsed = [r for r in eligible if r.get("final_answer")]
    gt = [r for r in records if r.get("verifiability") == "GT"]
    ngt = [r for r in records if r.get("verifiability") == "NGT"]
    gt_initial_correct_parsed = [r for r in gt if r.get("initial_correct") is True and r.get("final_answer")]
    gt_final_parsed = [r for r in gt if r.get("final_answer")]
    return {
        "records": len(records),
        "parsed": sum(bool(r.get("final_answer")) for r in records),
        "eligible": len(eligible),
        "eligible_parsed": len(eligible_parsed),
        "switches": sum(bool(r.get("pressure_aligned_switch")) for r in records),
        "answer_changes": sum(bool(r.get("answer_changed")) for r in records),
        "gt_records": len(gt),
        "ngt_records": len(ngt),
        "gt_initial_correct_parsed": len(gt_initial_correct_parsed),
        "gt_truth_departures": sum(bool(r.get("truth_departure")) for r in gt),
        "gt_final_correct_parsed": sum(r.get("final_correct") is True for r in gt_final_parsed),
        "gt_final_parsed": len(gt_final_parsed),
        "ngt_switches": sum(bool(r.get("flip_flop_switch")) for r in ngt),
        "switch_rate": ratio(sum(bool(r.get("pressure_aligned_switch")) for r in records), len(eligible)),
        "switch_rate_parsed": ratio(sum(bool(r.get("pressure_aligned_switch")) for r in records), len(eligible_parsed)),
        "answer_change_rate_parsed": ratio(sum(bool(r.get("answer_changed")) for r in records), len(eligible_parsed)),
        "gt_truth_departure_rate": ratio(
            sum(bool(r.get("truth_departure")) for r in gt),
            len(gt_initial_correct_parsed),
        ),
        "gt_final_correct_rate": ratio(
            sum(r.get("final_correct") is True for r in gt_final_parsed),
            len(gt_final_parsed),
        ),
        "ngt_switch_rate": ratio(sum(bool(r.get("flip_flop_switch")) for r in ngt), sum(bool(r.get("eligible")) for r in ngt)),
    }


def group_by(records: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(record.get(key) if record.get(key) is not None else "--") for key in keys)].append(record)
    return grouped


def summary_row(name: str, records: list[dict[str, Any]]) -> list[str]:
    summary = summarize_records(records)
    return [
        name,
        str(summary["records"]),
        f"{summary['parsed']}/{summary['records']}",
        str(summary["eligible"]),
        str(summary["switches"]),
        pct(summary["switch_rate"]),
        pct(summary["switch_rate_parsed"]),
        pct(summary["answer_change_rate_parsed"]),
        pct(summary["gt_truth_departure_rate"]),
        pct(summary["gt_final_correct_rate"]),
    ]


def compact_context_summary(summary_path: Path) -> tuple[str, str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_summary = summary["models"][MODEL]
    gt = model_summary["gt"]
    ngt = model_summary["ngt"]
    all_ngt = ngt["all"]
    context_rows = [
        ["GT injected pairs", str(gt["injected_belief_pairs"])],
        ["GT neutral correct", pct(gt["neutral_correct_rate"])],
        ["GT injected incorrect", pct(gt["injected_incorrect_rate"])],
        ["GT correct to incorrect", pct(gt["correct_to_incorrect_rate"])],
        ["GT answer change", pct(gt["answer_change_rate"])],
        ["NGT framed user-side", pct(all_ngt["framed_user_side_rate"])],
        ["NGT framing alignment lift", pct(all_ngt["framing_alignment_lift"])],
        ["NGT other-to-user switch", pct(all_ngt["other_to_user_switch_rate"])],
    ]
    cue_rows = [
        [
            cue,
            str(values["pairs"]),
            pct(values["framed_user_side_rate"]),
            pct(values["framing_alignment_lift"]),
            pct(values["other_to_user_switch_rate"]),
        ]
        for cue, values in sorted(ngt["by_cue"].items())
    ]
    return (
        md_table(["Context metric", "Value"], context_rows),
        md_table(["Cue", "Pairs", "Framed user-side", "Alignment lift", "Other-to-user switch"], cue_rows),
    )


def trigger_tables(neutral: list[dict[str, Any]]) -> tuple[str, str, str]:
    all_records = neutral
    overview_rows = [summary_row("neutral / all", neutral)]
    for (condition, branch), rows in sorted(group_by(all_records, "context_condition", "verifiability").items()):
        overview_rows.append(summary_row(f"{condition} / {branch}", rows))

    trigger_rows = [
        summary_row(trigger, rows)
        for (trigger,), rows in sorted(group_by(all_records, "trigger").items())
    ]
    tone_rows = [
        summary_row(tone, rows)
        for (tone,), rows in sorted(group_by(all_records, "tone").items())
    ]
    headers = [
        "Slice",
        "Records",
        "Parsed",
        "Eligible",
        "Switches",
        "Switch/eligible",
        "Switch/parsed eligible",
        "Answer change/parsed eligible",
        "GT truth departure",
        "GT final correct",
    ]
    return (
        md_table(headers, overview_rows),
        md_table(headers, trigger_rows),
        md_table(headers, tone_rows),
    )


def parse_counts(records: list[dict[str, Any]]) -> str:
    counts = Counter(str(record.get("final_parse_method")) for record in records)
    return ", ".join(f"{key}: {counts[key]}" for key in sorted(counts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    output = args.output or results_dir / "deepseek_v4_flash_full_eval_summary.md"
    context_table, context_cue_table = compact_context_summary(results_dir / "deepseek_v4_flash_context_eval_summary.json")
    neutral = read_jsonl(results_dir / "deepseek_v4_flash_trigger_neutral_static_eval.jsonl.gz")
    trigger_overview, trigger_by_family, trigger_by_tone = trigger_tables(neutral)

    expected_neutral = 300 * 8 * 3
    body = f"""# DeepSeek V4 Flash Full Evaluation Summary

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Model: `{MODEL}`

Run settings:
- Context run: first-turn only, `temperature=0`.
- Result JSONL files are stored as `.jsonl.gz`.
- Trigger run: static trigger prompts, all 8 trigger families, tones `mild/moderate/strong`.
- Trigger OpenRouter settings: `max_tokens=1024`, `temperature=0`, `reasoning.effort=none`, `reasoning.exclude=true`.
- A small number of initially unparsed trigger trials were removed from the JSONL and rerun with `max_tokens=2048`; final trigger outputs have no unparsed records.
- Trigger first turns reused the context-run cache when available.

## Inputs And Completion

| Panel | Expected input | Completed records | Final parse methods |
| --- | ---: | ---: | --- |
| Context | 800 framing cells (1100 exported prompt rows) | see context summary JSON | see context summary JSON |
| Trigger neutral | {expected_neutral} | {len(neutral)} | {parse_counts(neutral) if neutral else "--"} |

## Context Framing

{context_table}

### NGT Context By Cue

{context_cue_table}

## Trigger Pressure

Switch rates use eligible trials as denominator. The parsed-eligible column excludes trials where the final answer was not parsed.

{trigger_overview}

### By Trigger Family

{trigger_by_family}

### By Tone

{trigger_by_tone}
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
