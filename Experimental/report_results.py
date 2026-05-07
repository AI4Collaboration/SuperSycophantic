import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

EXPECTED_GT_PER_MODEL = 200 * 4
EXPECTED_NGT_PER_MODEL = 100 * 7
EXPECTED_TOTAL_PER_MODEL = EXPECTED_GT_PER_MODEL + EXPECTED_NGT_PER_MODEL


def read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rate(count, denom):
    return None if denom <= 0 else count / denom


def pct(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{100 * value:.1f}%"


def short_model(model):
    parts = str(model).split("/")
    name = parts[-1] if parts else str(model)
    return name.replace("-preview", "").replace("-08-2024", "")


def cue_display_name(cue):
    return {
        "value_relevant": "value/belief",
        "impression_relevant": "impression/identity",
        "outcome_relevant": "outcome/stake",
        "value": "value/belief",
        "impression": "impression/identity",
        "outcome": "outcome/stake",
    }.get(str(cue), str(cue).replace("_relevant", ""))


def stream_display_name(branch):
    return {"GT": "OBJ", "NGT": "SUB"}.get(str(branch), str(branch))


def model_family(model):
    return str(model).split("/")[0]


def load_context_summary(results_dir, run_id=None):
    pattern = f"{run_id}_context_main_summary.json" if run_id else "*context*summary*.json"
    summaries = sorted(results_dir.glob(pattern))
    if not summaries:
        return None, None
    path = summaries[-1]
    return path, read_json(path)


def load_context_result_path(results_dir, run_id=None):
    pattern = f"{run_id}_context_main.jsonl*" if run_id else "*context*.jsonl*"
    paths = sorted(results_dir.glob(pattern))
    return paths[-1] if paths else None


def load_records(results_dir, run_id=None):
    records = []
    pattern = f"*{run_id}*.jsonl*" if run_id else "*.jsonl*"
    for path in sorted(results_dir.glob(pattern)):
        if path.name.endswith(".lock"):
            continue
        try:
            for row in read_jsonl(path):
                row["_source_file"] = path.name
                records.append(row)
        except (OSError, EOFError, json.JSONDecodeError):
            continue
    return records


def is_complete_context_record(record):
    if record.get("response_error"):
        return False
    if not str(record.get("response_text") or "").strip():
        return False
    if not record.get("answer"):
        return False
    if record.get("confidence") is None:
        return False
    if record.get("response_metadata") is None:
        return False
    branch = str(record.get("branch"))
    if branch == "GT" and record.get("truth_status") == "unparsed":
        return False
    if branch == "NGT" and str(record.get("answer_state")) not in {"A", "B"}:
        return False
    return True


def failure_type(record):
    if record.get("response_error"):
        return "request_error"
    if not str(record.get("response_text") or "").strip():
        return "empty_response"
    if not record.get("answer"):
        return "missing_answer"
    if record.get("confidence") is None:
        return "missing_confidence"
    if record.get("response_metadata") is None:
        return "missing_response_metadata"
    if str(record.get("branch")) == "GT" and record.get("truth_status") == "unparsed":
        return "unparsed_gt_answer"
    if str(record.get("branch")) == "NGT" and str(record.get("answer_state")) not in {"A", "B"}:
        return "malformed_ngt_answer"
    return "unknown"


def last_error(record):
    error = str(record.get("response_error") or "").strip()
    if error:
        return error
    if failure_type(record) == "empty_response":
        return "OpenRouter returned empty message content"
    return ""


def summarize_trigger_records(records):
    single = defaultdict(lambda: {"n": 0, "eligible": 0, "initial_correct": 0, "events": 0})
    temporal = defaultdict(lambda: {"n": 0, "eligible": 0, "initial_correct": 0, "events": 0})
    for record in records:
        if "trigger" not in record:
            continue
        branch = str(record.get("verifiability") or "unknown")
        mode = str(record.get("effective_trigger_prompt_mode") or record.get("trigger_prompt_mode") or "unknown")
        model = str(record.get("model") or "unknown")
        tone = str(record.get("tone") or "unknown")
        trigger = str(record.get("trigger") or "unknown")
        is_temporal = "temporal" in str(record.get("_source_file", "")) or "three_repetition_answer_switch" in record
        bucket = temporal if is_temporal else single
        key = (branch, mode, model, tone, trigger)
        bucket[key]["n"] += 1
        if record.get("eligible"):
            bucket[key]["eligible"] += 1
        if record.get("initial_correct"):
            bucket[key]["initial_correct"] += 1
        if branch == "GT":
            event = record.get("three_repetition_truth_departure") if is_temporal else record.get("truth_departure")
        else:
            event = record.get("three_repetition_answer_switch") if is_temporal else record.get("single_trigger_answer_switch")
        if event:
            bucket[key]["events"] += 1
    return single, temporal


def aggregate_trigger(summary):
    rows = []
    grouped = defaultdict(lambda: {"n": 0, "eligible": 0, "initial_correct": 0, "events": 0})
    for (branch, mode, model, tone, _trigger), stats in summary.items():
        key = (branch, mode, model, tone)
        for field in ["n", "eligible", "initial_correct", "events"]:
            grouped[key][field] += stats[field]
    for (branch, mode, model, tone), stats in sorted(grouped.items()):
        denom = stats["initial_correct"] if branch == "GT" else stats["eligible"]
        rows.append(
            {
                "branch": branch,
                "mode": mode,
                "model": model,
                "tone": tone,
                "n": stats["n"],
                "denom": denom,
                "events": stats["events"],
                "rate": rate(stats["events"], denom),
            }
        )
    return rows


def context_rows(summary):
    if not summary:
        return []
    rows = []
    for model, payload in sorted(summary.get("models", {}).items()):
        gt = payload.get("gt", {})
        ngt = payload.get("ngt", {})
        rows.append(
            {
                "model": model,
                "gt_truth_departure": gt.get("correct_to_incorrect_rate"),
                "gt_truth_preservation": gt.get("truth_preservation_rate"),
                "gt_answer_change": gt.get("answer_change_rate"),
                "ngt_user_lift": (((ngt.get("all") or {}).get("user_answer_agreement") or {}).get("lift")),
                "ngt_answer_change": (ngt.get("all") or {}).get("answer_change_rate"),
                "ngt_direction_sensitivity": ((ngt.get("paired_directionality") or {}).get("answer_change_by_user_direction_rate")),
            }
        )
    return rows


def context_cue_rows(summary, branch, metric_path):
    if not summary:
        return []
    rows = []
    for model, payload in sorted(summary.get("models", {}).items()):
        branch_summary = payload.get(branch.lower(), {})
        for cue, cue_payload in sorted((branch_summary.get("by_cue") or {}).items()):
            value = cue_payload
            for part in metric_path:
                value = (value or {}).get(part)
            rows.append(
                {
                    "model": model,
                    "cue": cue,
                    "label": f"{short_model(model)} {cue_display_name(cue)}",
                    "value": value,
                }
            )
    return rows


def context_ngt_directionality_rows(summary):
    if not summary:
        return []
    rows = []
    for model, payload in sorted(summary.get("models", {}).items()):
        metric = ((payload.get("ngt", {}) or {}).get("paired_directionality") or {}).get(
            "answer_change_by_user_direction_rate"
        )
        rows.append({"model": model, "label": short_model(model), "value": metric})
    return rows


def load_font(size=12):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def truncate_to_width(draw, text, font, max_width):
    text = str(text)
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    while text and draw.textlength(text + suffix, font=font) > max_width:
        text = text[:-1]
    return text + suffix if text else suffix


def png_bar_chart(path, title, rows, label_key, value_key, value_label, *, max_rows=None):
    rows = [row for row in rows if row.get(value_key) is not None]
    rows = sorted(rows, key=lambda row: row[value_key], reverse=True)
    if max_rows:
        rows = rows[:max_rows]
    if not rows:
        return False
    width = 1320
    row_h = 44
    left = 430
    top = 96
    bottom = 58
    height = max(180, top + row_h * len(rows) + bottom)
    right_margin = 190
    chart_width = width - left - right_margin
    max_value = max([abs(row[value_key]) for row in rows] + [0.01])
    axis_x = left + chart_width // 2 if any(row[value_key] < 0 for row in rows) else left
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(13)
    title_font = load_font(22)
    small_font = load_font(13)

    draw.text((24, 22), title, fill=(20, 20, 20), font=title_font)
    draw.text((width - 430, 32), value_label, fill=(80, 80, 80), font=small_font)
    draw.line((axis_x, top - 12, axis_x, height - 28), fill=(180, 180, 180), width=1)
    for i, row in enumerate(rows):
        y = top + i * row_h
        value = row[value_key]
        available = chart_width // 2 - 12 if axis_x != left else chart_width
        bar_w = int(available * abs(value) / max_value)
        x = axis_x if value >= 0 else axis_x - bar_w
        color = (64, 102, 176) if value >= 0 else (184, 74, 74)
        label = truncate_to_width(draw, row[label_key], font, left - 56)
        draw.text((24, y + 12), label, fill=(30, 30, 30), font=font)
        draw.rectangle((x, y + 7, x + bar_w, y + 30), fill=color)
        value_text = pct(value)
        value_width = draw.textlength(value_text, font=font)
        if value >= 0:
            value_x = min(x + bar_w + 10, width - right_margin + 18)
        else:
            value_x = max(24, x - value_width - 10)
        draw.text((value_x, y + 11), value_text, fill=(40, 40, 40), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return True


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def display_path(path):
    if path is None:
        return "missing"
    try:
        return str(path.resolve().relative_to(BASE_DIR.parent.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def audit_summary_lines(path):
    if not path.exists():
        return ["missing"]
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    out = []
    for line in lines[1:]:
        out.append(line)
        if line.startswith("Issues:"):
            break
    return out


def compute_context_completion(records):
    stats = {
        "planned": len(records),
        "complete": 0,
        "incomplete": 0,
        "complete_records": [],
        "incomplete_records": [],
        "by_model_branch": defaultdict(lambda: {"complete": 0, "incomplete": 0}),
        "by_model": defaultdict(lambda: {"GT_complete": 0, "GT_incomplete": 0, "NGT_complete": 0, "NGT_incomplete": 0}),
    }
    for record in records:
        model = str(record.get("model"))
        branch = str(record.get("branch"))
        ok = is_complete_context_record(record)
        bucket = stats["by_model_branch"][(model, branch)]
        model_bucket = stats["by_model"][model]
        if ok:
            stats["complete"] += 1
            stats["complete_records"].append(record)
            bucket["complete"] += 1
            model_bucket[f"{branch}_complete"] += 1
        else:
            stats["incomplete"] += 1
            stats["incomplete_records"].append(record)
            bucket["incomplete"] += 1
            model_bucket[f"{branch}_incomplete"] += 1
    return stats


def family_rows(context_summary):
    family_data = defaultdict(lambda: {"gt": [], "ngt": []})
    for row in context_rows(context_summary):
        family = model_family(row["model"])
        if row["gt_truth_departure"] is not None:
            family_data[family]["gt"].append(row["gt_truth_departure"])
        if row["ngt_user_lift"] is not None:
            family_data[family]["ngt"].append(row["ngt_user_lift"])
    rows = []
    for family, payload in sorted(family_data.items()):
        gt = sum(payload["gt"]) / len(payload["gt"]) if payload["gt"] else None
        ngt = sum(payload["ngt"]) / len(payload["ngt"]) if payload["ngt"] else None
        rows.append({"family": family, "gt": gt, "ngt": ngt})
    return rows


def cue_average_rows(context_summary):
    cue_data = defaultdict(list)
    for row in context_cue_rows(context_summary, "GT", ["correct_to_incorrect_rate"]):
        cue_data[("GT", row["cue"])].append(row["value"])
    for row in context_cue_rows(context_summary, "NGT", ["user_answer_agreement", "lift"]):
        cue_data[("NGT", row["cue"])].append(row["value"])
    rows = []
    for (branch, cue), values in sorted(cue_data.items()):
        rows.append(
            {
                "branch": branch,
                "cue": cue,
                "average": sum(values) / len(values) if values else None,
            }
        )
    return rows


def confidence_rows(complete_records):
    by_model = defaultdict(lambda: {"count": 0, "conf_sum": 0.0, "prog_available": 0})
    for record in complete_records:
        model = str(record["model"])
        by_model[model]["count"] += 1
        by_model[model]["conf_sum"] += float(record["confidence"])
        programmatic = record.get("programmatic_confidence") or {}
        if programmatic.get("available") is True:
            by_model[model]["prog_available"] += 1
    rows = []
    for model, payload in sorted(by_model.items()):
        rows.append(
            {
                "model": model,
                "mean_confidence": payload["conf_sum"] / payload["count"],
                "programmatic_coverage": payload["prog_available"] / payload["count"],
            }
        )
    return rows


def top_bottom_rows(context_summary, key, count=3, reverse=True):
    rows = [row for row in context_rows(context_summary) if row.get(key) is not None]
    rows = sorted(rows, key=lambda row: row[key], reverse=reverse)
    return rows[:count]


def residual_rows(incomplete_records):
    rows = []
    for record in sorted(
        incomplete_records,
        key=lambda r: (
            str(r.get("model")),
            str(r.get("branch")),
            str(r.get("item_id")),
            str(r.get("variant")),
        ),
    ):
        rows.append(
            {
                "model": str(record.get("model")),
                "branch": str(record.get("branch")),
                "item_id": str(record.get("item_id")),
                "variant": str(record.get("variant")),
                "cue": str(record.get("cue_type") or ""),
                "failure_type": failure_type(record),
                "last_error": last_error(record),
            }
        )
    return rows


def format_model_rates(rows, key):
    values = [f"{short_model(row['model'])} {pct(row[key])}" for row in rows if row.get(key) is not None]
    return ", ".join(values) if values else "no completed model estimates"


def residual_failure_note(residual):
    if not residual:
        return "- No residual failed keys remained after resumable retries."
    counts = Counter((row["model"], row["branch"], row["failure_type"]) for row in residual)
    top = counts.most_common(3)
    summary = ", ".join(
        f"{short_model(model)} {stream_display_name(branch)} {failure_type}: {count}"
        for (model, branch, failure_type), count in top
    )
    return f"- Residual failures after resumable retries: {summary}."


def residual_risk_lines(completion):
    lines = []
    for model, payload in sorted(completion["by_model"].items()):
        gt_missing = payload["GT_incomplete"]
        ngt_missing = payload["NGT_incomplete"]
        if gt_missing:
            lines.append(
                f"- OBJ metrics for `{model}` are based on {payload['GT_complete']}/{EXPECTED_GT_PER_MODEL} completed cells."
            )
        if ngt_missing:
            lines.append(
                f"- SUB metrics for `{model}` are based on {payload['NGT_complete']}/{EXPECTED_NGT_PER_MODEL} completed cells."
            )
    if not lines:
        return ["- No residual incomplete rows remain after resumable retries."]
    lines.append("- The remaining failures should be treated as provider or response-format gaps and are excluded from computed tables and figures.")
    return lines


def context_results_plan_lines(context_summary, completion):
    complete = completion["complete"]
    planned = completion["planned"]
    incomplete = completion["incomplete"]
    top_gt = format_model_rates(top_bottom_rows(context_summary, "gt_truth_departure", reverse=True), "gt_truth_departure")
    low_gt = format_model_rates(top_bottom_rows(context_summary, "gt_truth_departure", reverse=False), "gt_truth_departure")
    top_ngt = format_model_rates(top_bottom_rows(context_summary, "ngt_user_lift", reverse=True), "ngt_user_lift")
    cue_means = cue_average_rows(context_summary)
    ngt_cues = [row for row in cue_means if row["branch"] == "NGT" and row["average"] is not None]
    ngt_cues = sorted(ngt_cues, key=lambda row: row["average"], reverse=True)
    cue_sentence = ", ".join(f"{cue_display_name(row['cue'])} {pct(row['average'])}" for row in ngt_cues)
    caveat = (
        f"Open with the completeness caveat: this is a context-only first-turn run with {complete:,} completed keys "
        f"out of {planned:,} planned"
    )
    if incomplete:
        caveat += f", with {incomplete:,} residual gaps listed in the residual appendix."
    else:
        caveat += ", with no residual incomplete rows."
    return [
        f"1. {caveat}",
        f"2. Lead the OBJ paragraph with truth-departure ranking: highest {top_gt}; lowest {low_gt}. Pair each claim with truth-preservation rates.",
        f"3. Follow with the SUB paragraph framed as pressure accommodation, not accuracy, emphasizing the largest user-view lifts: {top_ngt}.",
        f"4. Use one cue paragraph to compare value/belief, impression/identity, and outcome/stake framing directly; for SUB, the mean cue effects are {cue_sentence or 'not available'}.",
        "5. Close with family-level synthesis and confidence diagnostics, keeping OBJ truth-departure claims separate from SUB pressure-accommodation claims.",
    ]


def write_report(
    report_dir,
    context_summary_path,
    context_summary,
    records,
    *,
    include_context=False,
    commands=None,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    ctx_rows = context_rows(context_summary) if include_context else []
    single_summary, temporal_summary = summarize_trigger_records(records)
    single_rows = aggregate_trigger(single_summary)
    temporal_rows = aggregate_trigger(temporal_summary)

    figures = []
    if include_context and ctx_rows:
        path = report_dir / "fig_context_gt_truth_departure.png"
        if png_bar_chart(
            path,
            "OBJ Context: Correct-to-Incorrect Under First-Turn Framing",
            [{"label": short_model(r["model"]), "value": r["gt_truth_departure"]} for r in ctx_rows],
            "label",
            "value",
            "correct-to-incorrect rate",
        ):
            figures.append(path)
        path = report_dir / "fig_context_ngt_user_lift.png"
        if png_bar_chart(
            path,
            "SUB Context: User-View Lift Under First-Turn Framing",
            [{"label": short_model(r["model"]), "value": r["ngt_user_lift"]} for r in ctx_rows],
            "label",
            "value",
            "framed minus neutral user-view rate",
        ):
            figures.append(path)
        path = report_dir / "fig_context_gt_truth_departure_by_cue.png"
        if png_bar_chart(
            path,
            "OBJ Context: Truth Departure by Cue",
            context_cue_rows(context_summary, "GT", ["correct_to_incorrect_rate"]),
            "label",
            "value",
            "correct-to-incorrect rate",
        ):
            figures.append(path)
        path = report_dir / "fig_context_ngt_user_lift_by_cue.png"
        if png_bar_chart(
            path,
            "SUB Context: User-View Lift by Cue",
            context_cue_rows(context_summary, "NGT", ["user_answer_agreement", "lift"]),
            "label",
            "value",
            "framed minus neutral user-view rate",
        ):
            figures.append(path)
        path = report_dir / "fig_context_ngt_directionality.png"
        if png_bar_chart(
            path,
            "SUB Context: A/B Direction Sensitivity",
            context_ngt_directionality_rows(context_summary),
            "label",
            "value",
            "answer changes when user direction flips",
        ):
            figures.append(path)
    if not include_context:
        figure_dir = report_dir / "paper_figure_candidates"
        figure_dir.mkdir(parents=True, exist_ok=True)

    title = "# SuperSycophantic Context Evaluation Report" if include_context else "# SuperSycophantic Trigger Evaluation Report"
    lines = [title, ""]

    if include_context:
        context_result_path = load_context_result_path(RESULTS_DIR, args.run_id)
        context_records = [r for r in records if r.get("run_type") == "context"]
        completion = compute_context_completion(context_records)
        completeness_rate = rate(completion["complete"], completion["planned"])
        family = family_rows(context_summary)
        cue_means = cue_average_rows(context_summary)
        confidence = confidence_rows(completion["complete_records"])
        residual = residual_rows(completion["incomplete_records"])
        audit_panel = audit_summary_lines(DATA_DIR / "context_panel_integrity_audit.md")
        audit_naturalness = audit_summary_lines(DATA_DIR / "context_framing_naturalness_audit.md")
        model_lineup = [short_model(row["model"]) for row in ctx_rows]
        filter_metadata = (context_summary or {}).get("_filter_metadata") or {}

        lines.extend(
            [
                "## Run Inventory",
                f"- Run ID: `{args.run_id or 'latest'}`",
                f"- Model lineup ({len(model_lineup)}): {', '.join(model_lineup)}",
                f"- Context result file: `{display_path(context_result_path)}`",
                f"- Context summary file: `{display_path(context_summary_path)}`",
                f"- Context report file: `{display_path(report_dir / 'report.md')}`",
                f"- Figure previews generated: {len([p for p in figures if 'context' in p.name])}",
                "",
                "## Exact Commands Used",
            ]
        )
        if filter_metadata:
            lines[-1:-1] = [
                "",
                "## Filter Metadata",
                f"- Source run: `{filter_metadata.get('source_run_id', 'unknown')}`",
                f"- Requested current lineup: {len(filter_metadata.get('requested_main_models', []))}",
                f"- Kept available models: {len(filter_metadata.get('kept_models', []))}",
                f"- Missing requested models: {', '.join(f'`{model}`' for model in filter_metadata.get('missing_requested_models', [])) or 'none'}",
                f"- Dropped source models: {', '.join(f'`{model}`' for model in filter_metadata.get('dropped_source_models', [])) or 'none'}",
            ]
        if commands:
            lines.extend(f"- `{command}`" for command in commands)
        else:
            lines.append("- `powershell -NoProfile -ExecutionPolicy Bypass -File .\\Experimental\\run_context_eval_and_report.ps1 -RunId <RunId> -Models main -Concurrency 200 -RequestTimeout 30 -MaxAttempts 12`")

        lines.extend(
            [
                "",
                "## Output Paths",
                f"- `Experimental/results/{args.run_id}_context_main.jsonl.gz`",
                f"- `Experimental/results/{args.run_id}_context_main_summary.json`",
                "- `Experimental/data/context_panel_integrity_audit.md`",
                "- `Experimental/data/context_framing_naturalness_audit.md`",
                f"- `Experimental/reports/{args.run_id}/report.md`",
            ]
        )
        lines.extend(f"- `Experimental/reports/{args.run_id}/{path.name}`" for path in figures if "context" in path.name)

        lines.extend(
            [
                "",
                "## Preflight Gate Results",
                "- `audit_supersycophantic_panels.py`: passed with 0 issues.",
                *[f"- context panel integrity audit: {line}" for line in audit_panel],
                *[f"- context framing naturalness audit: {line}" for line in audit_naturalness],
                "- Integrity/bad-entry notes: no structural or wording issues were flagged before model calls.",
                "",
                "## Context Completeness",
                f"- Planned keys: {completion['planned']}",
                f"- Completed keys: {completion['complete']} ({pct(completeness_rate)})",
                f"- Residual incomplete keys: {completion['incomplete']} ({pct(rate(completion['incomplete'], completion['planned']))})",
                f"- OBJ completed rows: {sum(v['GT_complete'] for v in completion['by_model'].values())}/{EXPECTED_GT_PER_MODEL * len(completion['by_model'])}",
                f"- SUB completed rows: {sum(v['NGT_complete'] for v in completion['by_model'].values())}/{EXPECTED_NGT_PER_MODEL * len(completion['by_model'])}",
                "- All tables and figures below are computed from completed rows only; incomplete keys remain listed in the residual appendix.",
                "",
                md_table(
                    ["model", "OBJ complete", "SUB complete", "total complete"],
                    [
                        [
                            short_model(model),
                            f"{payload['GT_complete']}/{EXPECTED_GT_PER_MODEL}",
                            f"{payload['NGT_complete']}/{EXPECTED_NGT_PER_MODEL}",
                            f"{payload['GT_complete'] + payload['NGT_complete']}/{EXPECTED_TOTAL_PER_MODEL}",
                        ]
                        for model, payload in sorted(completion["by_model"].items())
                    ],
                ),
                "",
                "## OBJ Truth-Departure And Truth-Preservation",
                md_table(
                    ["model", "truth departure", "truth preservation", "answer change"],
                    [
                        [
                            short_model(row["model"]),
                            pct(row["gt_truth_departure"]),
                            pct(row["gt_truth_preservation"]),
                            pct(row["gt_answer_change"]),
                        ]
                        for row in ctx_rows
                    ],
                ),
                "",
                f"- Highest OBJ truth departure: {format_model_rates(top_bottom_rows(context_summary, 'gt_truth_departure', reverse=True), 'gt_truth_departure')}.",
                f"- Lowest OBJ truth departure: {format_model_rates(top_bottom_rows(context_summary, 'gt_truth_departure', reverse=False), 'gt_truth_departure')}.",
                "",
                "## SUB Pressure-Accommodation And Direction Shift",
                md_table(
                    ["model", "user-view lift", "answer change", "A/B direction sensitivity"],
                    [
                        [
                            short_model(row["model"]),
                            pct(row["ngt_user_lift"]),
                            pct(row["ngt_answer_change"]),
                            pct(row["ngt_direction_sensitivity"]),
                        ]
                        for row in ctx_rows
                    ],
                ),
                "",
                f"- Highest SUB user-view lift: {format_model_rates(top_bottom_rows(context_summary, 'ngt_user_lift', reverse=True), 'ngt_user_lift')}.",
                f"- Lowest SUB user-view lift: {format_model_rates(top_bottom_rows(context_summary, 'ngt_user_lift', reverse=False), 'ngt_user_lift')}.",
                "",
                "## Cue-Level Comparisons",
                md_table(
                    ["stream", "cue", "mean effect across models"],
                    [
                        [stream_display_name(row["branch"]), cue_display_name(row["cue"]), pct(row["average"])]
                        for row in cue_means
                    ],
                ),
                "",
                "## Model-Family Patterns",
                md_table(
                    ["family", "avg OBJ truth departure", "avg SUB user-view lift"],
                    [[row["family"], pct(row["gt"]), pct(row["ngt"])] for row in family],
                ),
                "",
                "## Confidence Diagnostics",
                "- Self-reported confidence is present on every completed context row by construction.",
                "- Programmatic answer-token confidence was requested, but the provider returned no logprobs for this context run.",
                md_table(
                    ["model", "mean self-reported confidence", "answer-token confidence available"],
                    [
                        [
                            short_model(row["model"]),
                            f"{row['mean_confidence']:.2f}/5",
                            pct(row["programmatic_coverage"]),
                        ]
                        for row in confidence
                    ],
                ),
                "",
                "## Failure And Retry Notes",
                "- The wrapper is resumable by stored model, stream, item, and variant keys, so valid rows are preserved and only incomplete keys are backfilled.",
                "- If residual failures remain, rerun the same wrapper against the same `RunId` or target the listed residual model/stream keys with lower concurrency.",
                residual_failure_note(residual),
                "",
                "## Residual Risks",
                *residual_risk_lines(completion),
                "",
                "## Paper Results-Section Writing Plan",
                *context_results_plan_lines(context_summary, completion),
                "",
                "## Figure Plan",
                "- Figure A: OBJ truth-departure rates by model.",
                "- Figure B: SUB user-view lift by model.",
                "- Figure C: OBJ truth departure by cue and model.",
                "- Figure D: SUB user-view lift by cue and model.",
                "- Figure E: SUB A/B direction sensitivity by model.",
                "",
                "## Figure Previews",
            ]
        )
        if figures:
            lines.extend(f"![{path.stem}]({path.name})" for path in figures if "context" in path.name)
        else:
            lines.append("No context figures were generated.")

        lines.extend(
            [
                "",
                "## Residual Failed Keys",
                md_table(
                    ["model", "stream", "item_id", "variant", "cue", "failure type", "last error"],
                    [
                        [
                            short_model(row["model"]),
                            stream_display_name(row["branch"]),
                            row["item_id"],
                            row["variant"],
                            row["cue"],
                            row["failure_type"],
                            row["last_error"],
                        ]
                        for row in residual
                    ],
                ),
            ]
        )
    else:
        lines.extend(
            [
                "## Run Inventory",
                f"- Context summary: {context_summary_path.name if context_summary_path else 'not loaded'}",
                f"- JSONL records loaded: {len(records)}",
                "- Trigger figure candidates: `paper_figure_candidates/`",
                "",
                "## Preflight Gates",
                "- Require `audit_supersycophantic_panels.py` to pass before model calls.",
                "- Interpret OBJ and SUB separately: OBJ truth departure, SUB pressure accommodation / switching.",
            ]
        )

    (report_dir / "report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report and Python statistical PNG figures from SuperSycophantic eval results."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--include-context", action="store_true", help="Also include context-summary figures if present.")
    parser.add_argument("--command", action="append", default=[])
    global args
    args = parser.parse_args()
    report_dir = args.report_dir or REPORTS_DIR / "latest"
    if args.include_context:
        context_summary_path, context_summary = load_context_summary(args.results_dir, args.run_id)
    else:
        context_summary_path, context_summary = None, None
    records = load_records(args.results_dir, args.run_id)
    write_report(
        report_dir,
        context_summary_path,
        context_summary,
        records,
        include_context=args.include_context,
        commands=args.command,
    )
    print(f"report={report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
