import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

try:
    import plot_context_results as context_plot
    import plot_trigger_figures as trigger_plot
except ImportError:
    from Experimental import plot_context_results as context_plot
    from Experimental import plot_trigger_figures as trigger_plot


REPO_ROOT = Path(__file__).resolve().parents[1]
INK = "#17202A"
MUTED = "#637083"
GRID = "#D9E0E8"
LIGHT = "#EEF3F8"
PANEL = "#F8FAFC"
WHITE = "#FFFFFF"
STATIC = "#0072B2"
ADAPTIVE = "#CC79A7"
GT = "#009E73"
NGT = "#D55E00"
ORANGE = "#E69F00"
PURPLE = "#6A3D9A"
GRAY = "#A7B1C2"
RED = "#C74332"

MODELS = trigger_plot.MODELS
MODEL_LABELS = trigger_plot.MODEL_SHORT_LABELS
TRIGGERS = trigger_plot.TRIGGER_LABELS
TONES = trigger_plot.TONES
CIALDINI = set(TRIGGERS) - {"simple_baseline"}
CONTEXT_MODELS = context_plot.MODEL_ORDER
CONTEXT_LABELS = {m: context_plot.SHORT_LABELS[m].replace("\n", " ") for m in CONTEXT_MODELS}
CUE_ORDER = [
    ("value_relevant", "Belief"),
    ("impression_relevant", "Identity"),
    ("outcome_relevant", "Stake"),
]
GT_DOMAINS = ["Mathematical Science", "Physical Science", "Chemical Science", "Biomedical Science"]
NGT_DOMAINS = ["policy", "moral dilemma", "interpersonal", "personal choice"]
SOURCE_LABELS = {
    "TIGER-Lab/MMLU-Pro": "MMLU-Pro",
    "skylenage-ai/HLE-Verified": "HLE-Verified",
    "mmlu_pro": "MMLU-Pro",
    "hle_verified": "HLE-Verified",
}


def font(size, bold=False):
    return trigger_plot.load_font(size, bold)


FONT_TITLE = font(44, True)
FONT_PANEL = font(30, True)
FONT_LABEL = font(24, True)
FONT = font(22)
FONT_SMALL = font(18)
FONT_CELL = font(28, True)


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value):
    return 100.0 * value


def draw_text(draw, xy, text, fnt=FONT, fill=INK, anchor=None):
    draw.text(xy, str(text), font=fnt, fill=fill, anchor=anchor)


def text_w(draw, text, fnt=FONT):
    box = draw.textbbox((0, 0), str(text), font=fnt)
    return box[2] - box[0]


def draw_header(draw, title, width):
    draw.rounded_rectangle((44, 24, width - 44, 108), radius=18, fill=PANEL, outline="#D6DEE8", width=2)
    draw_text(draw, (width / 2, 66), title, FONT_TITLE, anchor="mm")


def save(im, path, padding=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    trigger_plot.save_tight(im.convert("RGB"), path, padding=padding)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def blend(c1, c2, t):
    a = hex_to_rgb(c1) if isinstance(c1, str) else c1
    b = hex_to_rgb(c2) if isinstance(c2, str) else c2
    t = max(0.0, min(1.0, t))
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat(value, max_value, color):
    return blend("#F6FAFD", color, value / max_value if max_value else 0.0)


def is_true(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def denom_ok(record, branch):
    return bool(record.get("initial_correct")) if branch == "GT" else bool(record.get("eligible"))


def single_event(record, branch):
    if branch == "GT":
        return bool(record.get("truth_departure"))
    return bool(record.get("single_trigger_answer_switch") or record.get("answer_changed"))


def temporal_event(record, branch):
    if branch == "GT":
        return bool(record.get("three_repetition_truth_departure"))
    return bool(record.get("three_repetition_answer_switch"))


def round_event(round_record, branch):
    if branch == "GT":
        return bool(round_record.get("truth_departure"))
    return bool(round_record.get("answer_switch_from_initial"))


def is_cialdini(record):
    return record.get("trigger") in CIALDINI


def collect_trigger(results_dir, run_id):
    records = {"single": [], "temporal": []}
    for branch in ["gt", "ngt"]:
        branch_label = branch.upper()
        for mode in ["static", "adaptive"]:
            for source, suffix in [("single", "trigger"), ("temporal", "trigger_temporal")]:
                path = results_dir / f"{run_id}_{branch}_{suffix}_{mode}.jsonl.gz"
                for record in read_jsonl_gz(path):
                    record["_branch"] = branch_label
                    record["_mode"] = mode
                    record["_source_kind"] = source
                    records[source].append(record)
    return records


def rate(rows, event_fn, denom_fn=lambda r: True):
    denom = 0
    events = 0
    for row in rows:
        if denom_fn(row):
            denom += 1
            events += int(event_fn(row))
    return (events / denom if denom else 0.0), denom, events


def draw_percent_axis(draw, x0, y0, x1, y1, max_pct=100, ticks=(0, 25, 50, 75, 100)):
    for tick in ticks:
        x = x0 + (x1 - x0) * tick / max_pct
        draw.line((x, y0, x, y1), fill=LIGHT, width=1)
        draw_text(draw, (x, y1 + 10), str(tick), FONT_SMALL, MUTED, anchor="ma")
    draw.line((x0, y1, x1, y1), fill=GRID, width=2)


def bar_x(x0, x1, value, max_pct=100):
    return x0 + (x1 - x0) * max(0.0, min(max_pct, value)) / max_pct


def draw_legend(draw, entries, x, y, gap=190):
    cursor = x
    for label, color in entries:
        draw.rounded_rectangle((cursor, y - 11, cursor + 32, y + 11), radius=5, fill=color)
        draw_text(draw, (cursor + 42, y), label, FONT_SMALL, MUTED, anchor="lm")
        cursor += gap


def draw_inline_confidence_legend(draw, width, y, colors):
    title = "final self-reported confidence distribution (%)"
    entries = [("<4", colors["<4"]), ("4", colors["4"]), ("5", colors["5"])]
    swatch_w, swatch_h = 44, 25
    title_gap = 34
    item_gap = 34
    text_gap = 12
    total = text_w(draw, title, FONT_LABEL) + title_gap
    for i, (label, _) in enumerate(entries):
        total += swatch_w + text_gap + text_w(draw, label, FONT_LABEL)
        if i < len(entries) - 1:
            total += item_gap
    cursor = (width - total) / 2
    draw_text(draw, (cursor, y), title, FONT_LABEL, INK, anchor="lm")
    cursor += text_w(draw, title, FONT_LABEL) + title_gap
    for i, (label, color) in enumerate(entries):
        draw.rounded_rectangle(
            (cursor, y - swatch_h / 2, cursor + swatch_w, y + swatch_h / 2),
            radius=6,
            fill=color,
        )
        cursor += swatch_w + text_gap
        draw_text(draw, (cursor, y), label, FONT_LABEL, MUTED, anchor="lm")
        cursor += text_w(draw, label, FONT_LABEL)
        if i < len(entries) - 1:
            cursor += item_gap


def figure_temporal_state_paths(records, out_path):
    width, height = 2500, 980
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Temporal state paths and recovery", width)
    rows = []
    for branch in ["GT", "NGT"]:
        for mode in ["static", "adaptive"]:
            subset = [
                r
                for r in records["temporal"]
                if r["_branch"] == branch and r["_mode"] == mode and denom_ok(r, branch)
            ]
            counts = Counter()
            for record in subset:
                bits = [round_event(rr, branch) for rr in record.get("rounds", [])[:3]]
                if len(bits) != 3:
                    continue
                if not any(bits):
                    counts["Never moved"] += 1
                elif bits[-1]:
                    counts["Final moved"] += 1
                else:
                    counts["Recovered"] += 1
            total = sum(counts.values())
            rows.append({"label": f"{branch} {mode.title()}", "total": total, **counts})

    x_label, x0, x1 = 170, 560, 1960
    y0, row_h = 210, 135
    draw_percent_axis(draw, x0, y0 - 30, x1, y0 + row_h * len(rows) + 10)
    colors = {"Never moved": "#C9D2DE", "Recovered": STATIC, "Final moved": NGT}
    for i, row in enumerate(rows):
        y = y0 + i * row_h
        draw_text(draw, (x_label, y + 38), row["label"], FONT_LABEL, anchor="lm")
        cursor = x0
        cumulative = 0.0
        for key in ["Never moved", "Recovered", "Final moved"]:
            value = pct(row.get(key, 0) / row["total"]) if row["total"] else 0.0
            cumulative += value
            xx = bar_x(x0, x1, cumulative)
            if xx > cursor:
                draw.rounded_rectangle((cursor, y + 12, xx, y + 66), radius=12, fill=colors[key])
            if value >= 8:
                draw_text(draw, ((cursor + xx) / 2, y + 39), f"{value:.0f}", FONT_SMALL, WHITE, anchor="mm")
            cursor = xx
        ever = pct((row.get("Recovered", 0) + row.get("Final moved", 0)) / row["total"]) if row["total"] else 0.0
        final = pct(row.get("Final moved", 0) / row["total"]) if row["total"] else 0.0
        draw.line((2040, y + 39, 2240, y + 39), fill=GRID, width=5)
        draw.ellipse((2032, y + 31, 2048, y + 47), fill=STATIC)
        draw.ellipse((2230, y + 29, 2250, y + 49), fill=NGT)
        draw_text(draw, (2070, y + 39), f"ever {ever:.0f}", FONT_SMALL, MUTED, anchor="lm")
        draw_text(draw, (2270, y + 39), f"final {final:.0f}", FONT_SMALL, MUTED, anchor="lm")
    draw_legend(
        draw,
        [("never", colors["Never moved"]), ("recovered", colors["Recovered"]), ("final moved", colors["Final moved"])],
        x0,
        height - 86,
        gap=230,
    )
    draw_text(draw, ((x0 + x1) / 2, height - 42), "share of temporal trials (%)", FONT_LABEL, anchor="mm")
    save(im, out_path)


def judge_mean(row, field):
    values = []
    for suffix in ["_a", "_b"]:
        value = row.get(field + suffix)
        if value in (None, ""):
            continue
        if str(value).lower() in {"true", "false"}:
            values.append(1.0 if str(value).lower() == "true" else 0.0)
        else:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def load_judge_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def figure_judge_mechanism_outcome(judge_rows, out_path):
    width, height = 2500, 1120
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Judge diagnostics separate changed and stable answers", width)

    binary = [
        ("rationalized_change", "Rationalized"),
        ("contradicted_itself", "Contradicted"),
        ("redo_question_by_reasoning_or_calculation", "Redid check"),
    ]
    likert = [
        ("supported_user_viewpoint", "Supports user"),
        ("disagreed_with_user_points_reverse", "Does not push back"),
        ("eager_to_please_user", "Eager to please"),
        ("sought_user_good_graces", "Seeks favor"),
        ("excessive_flattery", "Flattery"),
    ]

    def collect(field, changed):
        vals = [judge_mean(row, field) for row in judge_rows if is_true(row["answer_changed"]) == changed]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def panel(x, y, w, h, title, fields, max_value, percent):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, y + 36), title, FONT_PANEL, anchor="ma")
        x0, x1 = x + 320, x + w - 70
        yy0 = y + 120
        row_h = (h - 190) / len(fields)
        draw_percent_axis(draw, x0, yy0 - 20, x1, yy0 + row_h * len(fields), max_value, (0, max_value / 2, max_value))
        for i, (field, label) in enumerate(fields):
            yy = yy0 + i * row_h + row_h / 2
            stable = collect(field, False)
            changed = collect(field, True)
            if percent:
                stable *= 100
                changed *= 100
            draw_text(draw, (x + 28, yy), label, FONT, anchor="lm")
            xs = bar_x(x0, x1, stable, max_value)
            xc = bar_x(x0, x1, changed, max_value)
            draw.line((xs, yy, xc, yy), fill=GRID, width=6)
            draw.ellipse((xs - 10, yy - 10, xs + 10, yy + 10), fill=STATIC)
            draw.ellipse((xc - 12, yy - 12, xc + 12, yy + 12), fill=NGT)
            draw_text(draw, (xc + 14, yy), f"{changed:.1f}", FONT_SMALL, NGT, anchor="lm")
        draw_legend(draw, [("stable", STATIC), ("changed", NGT)], x0, y + h - 42, gap=150)

    panel(70, 155, 1135, 860, "Binary process labels (%)", binary, 100, True)
    panel(1295, 155, 1135, 860, "Social-response scores (1-5)", likert, 5, False)
    save(im, out_path)


def figure_trigger_confidence_high_risk(records, out_path):
    width, height = 2500, 990
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "High confidence remains common after sycophantic outcomes", width)

    def summarize(source, branch, outcome):
        rows = records[source]
        values = []
        for row in rows:
            if row["_mode"] != "adaptive" or row["_branch"] != branch or not denom_ok(row, branch):
                continue
            if source == "single" and not is_cialdini(row):
                continue
            event = single_event(row, branch) if source == "single" else temporal_event(row, branch)
            wanted = event if outcome in {"departed", "switched"} else not event
            if not wanted:
                continue
            conf = row.get("final_confidence")
            if isinstance(conf, (int, float)):
                values.append(int(conf))
        total = len(values)
        return {
            "n": total,
            "<4": sum(1 for v in values if v < 4) / total if total else 0,
            "4": sum(1 for v in values if v == 4) / total if total else 0,
            "5": sum(1 for v in values if v == 5) / total if total else 0,
        }

    rows = [
        ("GT stable", "GT", "stable"),
        ("GT departed", "GT", "departed"),
        ("NGT held", "NGT", "held"),
        ("NGT switched", "NGT", "switched"),
    ]
    colors = {"<4": "#CDD6E2", "4": ORANGE, "5": RED}

    for pi, (source, title) in enumerate([("single", "Single follow-up"), ("temporal", "Three-turn final")]):
        x = 90 + pi * 1200
        y = 160
        w, h = 1120, 800
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, y + 38), title, FONT_PANEL, anchor="ma")
        x0, x1 = x + 260, x + w - 90
        y0 = y + 130
        draw_percent_axis(draw, x0, y0 - 20, x1, y0 + 4 * 135 + 10)
        for i, (label, branch, outcome) in enumerate(rows):
            yy = y0 + i * 135
            stats = summarize(source, branch, outcome)
            draw_text(draw, (x + 28, yy + 32), label, FONT_LABEL, anchor="lm")
            cursor = x0
            cumulative = 0.0
            for key in ["<4", "4", "5"]:
                value = pct(stats[key])
                cumulative += value
                xx = bar_x(x0, x1, cumulative)
                if xx > cursor:
                    draw.rounded_rectangle((cursor, yy + 8, xx, yy + 58), radius=10, fill=colors[key])
                cursor = xx
            hi = pct(stats["4"] + stats["5"])
            draw_text(draw, (x1 + 18, yy + 33), f"{hi:.1f} high", FONT_SMALL, NGT if outcome in {"departed", "switched"} else MUTED, anchor="lm")
    draw_inline_confidence_legend(draw, width, 926, colors)
    save(im, out_path)


def figure_context_ngt_directionality(summary, out_path):
    width, height = 2550, 820
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "NGT paired A/B directionality by cue and model", width)
    x0, y0 = 350, 250
    cell_w, cell_h = 225, 132
    for ci, model in enumerate(CONTEXT_MODELS):
        draw.multiline_text(
            (x0 + ci * cell_w + cell_w / 2, 176),
            context_plot.COLUMN_LABELS[model],
            font=FONT_SMALL,
            fill=INK,
            anchor="ma",
            align="center",
            spacing=2,
        )
    for ri, (cue, label) in enumerate(CUE_ORDER):
        y = y0 + ri * (cell_h + 18)
        draw_text(draw, (x0 - 30, y + cell_h / 2), label, FONT_LABEL, GT, anchor="rm")
        for ci, model in enumerate(CONTEXT_MODELS):
            data = summary["models"][model]["ngt"]["paired_directionality_by_cue"][cue]
            direction = pct(data["answer_change_by_user_direction_rate"])
            both = pct(data["aligned_with_user_both_rate"])
            x = x0 + ci * cell_w
            draw.rounded_rectangle((x, y, x + cell_w - 12, y + cell_h), radius=14, fill=heat(direction, 80, GT))
            draw_text(draw, (x + cell_w / 2 - 6, y + 50), f"{direction:.0f}", FONT_CELL, INK, anchor="mm")
            draw_text(draw, (x + cell_w / 2 - 6, y + 92), f"both {both:.0f}", FONT_SMALL, MUTED, anchor="mm")
    draw_text(draw, (width / 2, height - 44), "cell: answer changes when user direction flips; small text: aligned with user in both directions (%)", FONT, MUTED, anchor="mm")
    save(im, out_path)


def figure_validation_coverage(records, context_summary, trigger_judge_summary, context_judge_summary, out_path):
    width, height = 2250, 960
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Validation and coverage checks", width)

    context_total = sum(m["total_records"] for m in context_summary["models"].values())
    context_parsed = sum(m["parsed_records"] for m in context_summary["models"].values())

    def parsed_trigger(row):
        return bool(row.get("initial_answer")) and bool(row.get("final_answer"))

    single_total = len(records["single"])
    single_parsed = sum(1 for r in records["single"] if parsed_trigger(r))
    single_analyzed = sum(1 for r in records["single"] if denom_ok(r, r["_branch"]))
    temporal_total = len(records["temporal"])
    temporal_parsed = sum(1 for r in records["temporal"] if parsed_trigger(r))
    temporal_analyzed = sum(1 for r in records["temporal"] if denom_ok(r, r["_branch"]))

    rows = [
        ("Context eval", context_total, context_parsed, context_parsed, context_total - context_parsed),
        ("Trigger single", single_total, single_parsed, single_analyzed, single_total - single_parsed),
        ("Trigger temporal", temporal_total, temporal_parsed, temporal_analyzed, temporal_total - temporal_parsed),
        ("Judge context", context_judge_summary["manifest_count"], context_judge_summary["judge_a_pass"], context_judge_summary["paired_count"], context_judge_summary["missing_any_count"]),
        ("Judge trigger", trigger_judge_summary["manifest_count"], trigger_judge_summary["judge_a_pass"], trigger_judge_summary["paired_count"], trigger_judge_summary["missing_any_count"]),
    ]
    headers = ["Raw/input", "Parsed/pass", "Analyzed/paired", "Missing"]
    x0, y0 = 430, 205
    cell_w, cell_h = 390, 105
    for ci, header in enumerate(headers):
        draw_text(draw, (x0 + ci * cell_w + cell_w / 2, y0 - 44), header, FONT_LABEL, anchor="ma")
    max_count = max(row[1] for row in rows)
    for ri, row in enumerate(rows):
        y = y0 + ri * (cell_h + 18)
        draw_text(draw, (70, y + cell_h / 2), row[0], FONT_LABEL, anchor="lm")
        for ci, value in enumerate(row[1:]):
            x = x0 + ci * cell_w
            color = heat(value, max_count, GT if ci != 3 else RED)
            if ci == 3 and value == 0:
                color = "#F0F4F8"
            draw.rounded_rectangle((x, y, x + cell_w - 16, y + cell_h), radius=14, fill=color)
            draw_text(draw, (x + cell_w / 2 - 8, y + cell_h / 2), f"{value:,}", FONT_CELL, INK, anchor="mm")
    draw_text(draw, (width / 2, height - 48), "GT outcome denominators use initially correct trials; NGT uses eligible parsed trials.", FONT, MUTED, anchor="mm")
    save(im, out_path)


def figure_item_concentration(records, out_path):
    width, height = 2450, 1180
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Item-level susceptibility is broadly distributed", width)

    specs = [
        ("GT single", "single", "GT"),
        ("NGT single", "single", "NGT"),
        ("GT temporal", "temporal", "GT"),
        ("NGT temporal", "temporal", "NGT"),
    ]
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]

    def item_rates(source, branch):
        totals = Counter()
        events = Counter()
        for r in records[source]:
            if r["_mode"] != "adaptive" or r["_branch"] != branch or not denom_ok(r, branch):
                continue
            if source == "single" and not is_cialdini(r):
                continue
            item = r["item_id"]
            totals[item] += 1
            events[item] += int(single_event(r, branch) if source == "single" else temporal_event(r, branch))
        return [pct(events[item] / totals[item]) for item in totals if totals[item]]

    for pi, (title, source, branch) in enumerate(specs):
        x = 90 + (pi % 2) * 1180
        y = 160 + (pi // 2) * 470
        w, h = 1080, 390
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, y + 34), title, FONT_PANEL, anchor="ma")
        rates = item_rates(source, branch)
        counts = []
        for lo, hi in bins:
            counts.append(sum(1 for v in rates if lo <= v < hi))
        max_count = max(counts) if counts else 1
        x0, x1, ybase = x + 95, x + w - 60, y + h - 75
        bar_gap = 24
        bar_w = (x1 - x0 - bar_gap * (len(bins) - 1)) / len(bins)
        for i, ((lo, hi), count) in enumerate(zip(bins, counts)):
            bx = x0 + i * (bar_w + bar_gap)
            bh = 210 * count / max_count
            color = GT if branch == "GT" else NGT
            draw.rounded_rectangle((bx, ybase - bh, bx + bar_w, ybase), radius=9, fill=color)
            draw_text(draw, (bx + bar_w / 2, ybase - bh - 12), str(count), FONT_SMALL, MUTED, anchor="mm")
            label = f"{lo}-{hi - 1}" if hi <= 100 else f"{lo}-100"
            draw_text(draw, (bx + bar_w / 2, ybase + 14), label, FONT_SMALL, MUTED, anchor="ma")
        ge50 = sum(1 for v in rates if v >= 50)
        draw_text(draw, (x + w - 70, y + 78), f"{ge50}/{len(rates)} items >=50%", FONT_LABEL, color, anchor="ra")
    draw_text(draw, (width / 2, height - 42), "item-level outcome rate bins under adaptive pressure (%)", FONT_LABEL, anchor="mm")
    save(im, out_path)


def figure_trigger_domain_source(records, out_path):
    width, height = 2600, 980
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Trigger susceptibility by source and decision domain", width)

    gt_counts = defaultdict(lambda: [0, 0])
    ngt_counts = defaultdict(lambda: [0, 0])
    for r in records["single"]:
        if r["_mode"] != "adaptive" or not is_cialdini(r) or not denom_ok(r, r["_branch"]):
            continue
        if r["_branch"] == "GT":
            key = (SOURCE_LABELS.get(r.get("source_dataset"), r.get("source_dataset")), r.get("domain"))
            gt_counts[key][0] += 1
            gt_counts[key][1] += int(single_event(r, "GT"))
        else:
            key = r.get("domain")
            ngt_counts[key][0] += 1
            ngt_counts[key][1] += int(single_event(r, "NGT"))

    # GT heatmap
    x, y = 80, 160
    w, h = 1160, 700
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
    draw_text(draw, (x + w / 2, y + 36), "GT truth departure", FONT_PANEL, anchor="ma")
    sources = ["MMLU-Pro", "HLE-Verified"]
    cell_w, cell_h = 260, 118
    gx, gy = x + 390, y + 150
    for ci, src in enumerate(sources):
        draw_text(draw, (gx + ci * cell_w + cell_w / 2, gy - 52), src, FONT_LABEL, anchor="ma")
    for ri, domain in enumerate(GT_DOMAINS):
        yy = gy + ri * (cell_h + 16)
        draw_text(draw, (gx - 24, yy + cell_h / 2), domain.replace(" Science", ""), FONT, anchor="rm")
        for ci, src in enumerate(sources):
            denom, events = gt_counts[(src, domain)]
            value = pct(events / denom) if denom else 0.0
            xx = gx + ci * cell_w
            draw.rounded_rectangle((xx, yy, xx + cell_w - 16, yy + cell_h), radius=14, fill=heat(value, 70, NGT))
            draw_text(draw, (xx + cell_w / 2 - 8, yy + cell_h / 2), f"{value:.0f}", FONT_CELL, anchor="mm")

    # NGT bars
    x = 1350
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
    draw_text(draw, (x + w / 2, y + 36), "NGT Flip-Flop", FONT_PANEL, anchor="ma")
    x0, x1 = x + 280, x + w - 70
    y0 = y + 155
    draw_percent_axis(draw, x0, y0 - 25, x1, y0 + 4 * 110)
    for i, domain in enumerate(NGT_DOMAINS):
        denom, events = ngt_counts[domain]
        value = pct(events / denom) if denom else 0.0
        yy = y0 + i * 110
        draw_text(draw, (x + 32, yy + 26), domain.title(), FONT, anchor="lm")
        xx = bar_x(x0, x1, value)
        draw.rounded_rectangle((x0, yy, xx, yy + 50), radius=11, fill=GT)
        draw_text(draw, (xx + 12, yy + 25), f"{value:.1f}", FONT_SMALL, GT, anchor="lm")
    save(im, out_path)


def figure_adaptive_family_lift(records, out_path):
    width, height = 2500, 1120
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Adaptive lift varies by trigger family", width)
    families = [k for k in TRIGGERS if k != "simple_baseline"]

    def family_rate(branch, mode, family):
        rows = [
            r
            for r in records["single"]
            if r["_branch"] == branch
            and r["_mode"] == mode
            and r.get("trigger") == family
            and denom_ok(r, branch)
        ]
        return rate(rows, lambda r: single_event(r, branch))[0]

    for pi, branch in enumerate(["GT", "NGT"]):
        x = 90 + pi * 1200
        y = 160
        w, h = 1120, 830
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, y + 36), "GT truth departure" if branch == "GT" else "NGT Flip-Flop", FONT_PANEL, anchor="ma")
        x0, x1 = x + 300, x + w - 70
        y0 = y + 112
        row_h = 86
        draw_percent_axis(draw, x0, y0 - 20, x1, y0 + row_h * len(families), 80, (0, 40, 80))
        for i, family in enumerate(families):
            static = pct(family_rate(branch, "static", family))
            adaptive = pct(family_rate(branch, "adaptive", family))
            yy = y0 + i * row_h + row_h / 2
            draw_text(draw, (x + 28, yy), TRIGGERS[family], FONT, anchor="lm")
            xs = bar_x(x0, x1, static, 80)
            xa = bar_x(x0, x1, adaptive, 80)
            draw.line((xs, yy, xa, yy), fill=GRID, width=6)
            draw.ellipse((xs - 9, yy - 9, xs + 9, yy + 9), fill=STATIC)
            draw.ellipse((xa - 11, yy - 11, xa + 11, yy + 11), fill=ADAPTIVE)
            delta = adaptive - static
            draw_text(draw, (x1 + 10, yy), f"{delta:+.1f}", FONT_SMALL, ADAPTIVE if delta >= 0 else NGT, anchor="lm")
    draw_legend(draw, [("static", STATIC), ("adaptive", ADAPTIVE)], 1000, height - 66, gap=175)
    save(im, out_path)


def load_trigger_judge_inputs(path):
    mapping = {}
    for row in read_jsonl_gz(path):
        meta = row.get("trigger_metadata") or {}
        mapping[row["transcript_key"]] = {
            "tone": meta.get("tone"),
            "source_kind": row.get("source_kind"),
            "model": row.get("target_model"),
            "branch": row.get("branch"),
            "trigger": meta.get("trigger"),
            "prompt_mode": meta.get("trigger_prompt_mode") or row.get("prompt_mode"),
        }
    return mapping


def figure_strong_recheck_boundary(records, judge_rows, judge_meta, out_path):
    width, height = 2450, 980
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Strong pressure can trigger rechecking", width)

    def group(model):
        return "Claude" if model and model.startswith("anthropic/") else "Other"

    outcome = defaultdict(lambda: [0, 0])
    for r in records["single"]:
        if r["_mode"] != "adaptive" or not is_cialdini(r) or not denom_ok(r, r["_branch"]):
            continue
        key = (group(r["model"]), r["_branch"], r.get("tone"))
        outcome[key][0] += 1
        outcome[key][1] += int(single_event(r, r["_branch"]))

    redo = defaultdict(lambda: [0.0, 0])
    for row in judge_rows:
        meta = judge_meta.get(row["transcript_key"])
        if not meta or meta["source_kind"] != "single" or meta.get("trigger") not in CIALDINI:
            continue
        value = judge_mean(row, "redo_question_by_reasoning_or_calculation")
        if value is None:
            continue
        key = (group(meta["model"]), meta.get("tone"))
        redo[key][0] += value
        redo[key][1] += 1

    def panel(x, y, w, h, title, metric_kind):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, y + 36), title, FONT_PANEL, anchor="ma")
        x0, x1 = x + 190, x + w - 80
        y0, y1 = y + 150, y + h - 105
        max_rate = 70 if metric_kind == "outcome" else 30
        draw_percent_axis(draw, x0, y0 - 25, x1, y1, max_rate, (0, max_rate / 2, max_rate))
        tone_x = {tone: x0 + (x1 - x0) * i / 2 for i, tone in enumerate(TONES)}
        for tone, xx in tone_x.items():
            draw_text(draw, (xx, y1 + 38), tone.title(), FONT_SMALL, MUTED, anchor="ma")
        for gi, (grp, color) in enumerate([("Claude", NGT), ("Other", STATIC)]):
            points = []
            for tone in TONES:
                if metric_kind == "outcome":
                    vals = []
                    for branch in ["GT", "NGT"]:
                        denom, events = outcome[(grp, branch, tone)]
                        if denom:
                            vals.append(events / denom)
                    value = pct(sum(vals) / len(vals)) if vals else 0.0
                else:
                    total, n = redo[(grp, tone)]
                    value = pct(total / n) if n else 0.0
                xx = tone_x[tone]
                yy = y1 - (y1 - y0) * min(value, max_rate) / max_rate
                points.append((xx, yy, value))
            for a, b in zip(points, points[1:]):
                draw.line((a[0], a[1], b[0], b[1]), fill=color, width=5)
            for xx, yy, value in points:
                draw.ellipse((xx - 11, yy - 11, xx + 11, yy + 11), fill=color)
                draw_text(draw, (xx + 15, yy), f"{value:.1f}", FONT_SMALL, color, anchor="lm")
        draw_legend(draw, [("Claude", NGT), ("Other", STATIC)], x + 360, y + 90, gap=165)

    panel(80, 155, 1110, 740, "Sycophantic outcome rate", "outcome")
    panel(1270, 155, 1110, 740, "Redo/rechecking label rate", "redo")
    save(im, out_path)


def figure_confidence_risk_calibration(records, out_path):
    width, height = 2400, 1000
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Initial confidence does not eliminate later switching", width)

    def bin_rates(source, branch):
        counts = defaultdict(lambda: [0, 0])
        for r in records[source]:
            if r["_mode"] != "adaptive" or r["_branch"] != branch or not denom_ok(r, branch):
                continue
            if source == "single" and not is_cialdini(r):
                continue
            conf = r.get("initial_confidence")
            if not isinstance(conf, (int, float)):
                continue
            counts[int(conf)][0] += 1
            counts[int(conf)][1] += int(single_event(r, branch) if source == "single" else temporal_event(r, branch))
        return {k: pct(v[1] / v[0]) if v[0] else 0.0 for k, v in counts.items()}

    for pi, (source, title) in enumerate([("single", "Single follow-up"), ("temporal", "Three-turn final")]):
        x = 90 + pi * 1130
        y = 160
        w, h = 1040, 740
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, y + 36), title, FONT_PANEL, anchor="ma")
        x0, x1, y0, y1 = x + 120, x + w - 70, y + 120, y + h - 105
        draw_percent_axis(draw, x0, y0, x1, y1, 100, (0, 50, 100))
        conf_x = {c: x0 + (x1 - x0) * (c - 1) / 4 for c in range(1, 6)}
        for c, xx in conf_x.items():
            draw_text(draw, (xx, y1 + 40), str(c), FONT_SMALL, MUTED, anchor="ma")
        for branch, color in [("GT", GT), ("NGT", NGT)]:
            rates = bin_rates(source, branch)
            points = []
            for c in range(1, 6):
                value = rates.get(c, 0.0)
                xx = conf_x[c]
                yy = y1 - (y1 - y0) * value / 100
                points.append((xx, yy, value))
            for a, b in zip(points, points[1:]):
                draw.line((a[0], a[1], b[0], b[1]), fill=color, width=5)
            for xx, yy, value in points:
                draw.ellipse((xx - 10, yy - 10, xx + 10, yy + 10), fill=color)
                draw_text(draw, (xx + 13, yy), f"{value:.0f}", FONT_SMALL, color, anchor="lm")
        draw_legend(draw, [("GT", GT), ("NGT", NGT)], x + 360, y + h - 44, gap=110)
    draw_text(draw, (width / 2, height - 42), "outcome rate by initial self-reported confidence (%)", FONT_LABEL, anchor="mm")
    save(im, out_path)


def load_context_rows(path):
    return list(read_jsonl_gz(path))


def context_pairs(rows):
    by_key = {}
    for r in rows:
        by_key[(r["model"], r["item_id"], r["variant"])] = r
    gt_pairs = []
    ngt_pairs = []
    for r in rows:
        if r["variant"] == "neutral":
            continue
        neutral = by_key.get((r["model"], r["item_id"], "neutral"))
        if not neutral:
            continue
        if r["branch"] == "GT":
            gt_pairs.append((neutral, r))
        else:
            suffix = r["variant"].rsplit("_", 1)[-1]
            if suffix in {"A", "B"}:
                ngt_pairs.append((neutral, r, suffix))
    return gt_pairs, ngt_pairs


def figure_context_gt_source_decomposition(gt_pairs, out_path):
    width, height = 2500, 920
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Context GT truth departure by source and domain", width)
    sources = ["MMLU-Pro", "HLE-Verified"]
    x0, y0 = 520, 240
    cell_w, cell_h = 330, 120
    counts = defaultdict(lambda: [0, 0])
    for neutral, framed in gt_pairs:
        if neutral.get("truth_status") != "correct":
            continue
        src = SOURCE_LABELS.get(framed.get("source"), framed.get("source"))
        key = (src, framed.get("domain"), framed.get("cue_type"))
        counts[key][0] += 1
        counts[key][1] += int(framed.get("truth_status") != "correct")
    for ci, src in enumerate(sources):
        draw_text(draw, (x0 + ci * cell_w + cell_w / 2, y0 - 52), src, FONT_LABEL, anchor="ma")
    for ri, domain in enumerate(GT_DOMAINS):
        y = y0 + ri * (cell_h + 18)
        draw_text(draw, (x0 - 28, y + cell_h / 2), domain.replace(" Science", ""), FONT, anchor="rm")
        for ci, src in enumerate(sources):
            den = ev = 0
            for cue, _ in CUE_ORDER:
                d, e = counts[(src, domain, cue)]
                den += d
                ev += e
            value = pct(ev / den) if den else 0.0
            x = x0 + ci * cell_w
            draw.rounded_rectangle((x, y, x + cell_w - 18, y + cell_h), radius=14, fill=heat(value, 80, NGT))
            draw_text(draw, (x + cell_w / 2 - 9, y + cell_h / 2), f"{value:.0f}", FONT_CELL, anchor="mm")
    # cue marginal bars
    x = 1350
    draw.rounded_rectangle((x, 170, x + 1020, 690), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
    draw_text(draw, (x + 510, 210), "Cue marginal", FONT_PANEL, anchor="ma")
    ax0, ax1, ay0 = x + 270, x + 930, 300
    draw_percent_axis(draw, ax0, ay0 - 25, ax1, ay0 + 3 * 95, 80, (0, 40, 80))
    for i, (cue, label) in enumerate(CUE_ORDER):
        den = ev = 0
        for src in sources:
            for domain in GT_DOMAINS:
                d, e = counts[(src, domain, cue)]
                den += d
                ev += e
        value = pct(ev / den) if den else 0.0
        y = ay0 + i * 95
        draw_text(draw, (x + 40, y + 26), label, FONT_LABEL, anchor="lm")
        xx = bar_x(ax0, ax1, value, 80)
        draw.rounded_rectangle((ax0, y, xx, y + 50), radius=11, fill=NGT)
        draw_text(draw, (xx + 12, y + 25), f"{value:.1f}", FONT_SMALL, NGT, anchor="lm")
    save(im, out_path)


def figure_context_change_decomposition(gt_pairs, out_path):
    width, height = 2400, 1020
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Context GT answer changes decompose into failure and harmless movement", width)
    counts = defaultdict(lambda: Counter())
    for neutral, framed in gt_pairs:
        if neutral.get("truth_status") != "correct":
            continue
        model = framed["model"]
        changed = framed.get("answer") != neutral.get("answer")
        departed = framed.get("truth_status") != "correct"
        if departed:
            counts[model]["truth departure"] += 1
        elif changed:
            counts[model]["changed, still correct"] += 1
        else:
            counts[model]["held correct"] += 1
    x0, x1 = 500, 2200
    y0, row_h = 190, 78
    draw_percent_axis(draw, x0, y0 - 24, x1, y0 + row_h * len(CONTEXT_MODELS), 100)
    colors = {"held correct": "#C9D2DE", "changed, still correct": STATIC, "truth departure": NGT}
    for i, model in enumerate(CONTEXT_MODELS):
        y = y0 + i * row_h
        draw_text(draw, (70, y + 28), CONTEXT_LABELS[model], FONT, anchor="lm")
        total = sum(counts[model].values())
        cursor = x0
        cumulative = 0.0
        for key in ["held correct", "changed, still correct", "truth departure"]:
            value = pct(counts[model][key] / total) if total else 0.0
            cumulative += value
            xx = bar_x(x0, x1, cumulative)
            if xx > cursor:
                draw.rounded_rectangle((cursor, y + 8, xx, y + 48), radius=9, fill=colors[key])
            cursor = xx
        dep = pct(counts[model]["truth departure"] / total) if total else 0.0
        draw_text(draw, (x1 + 15, y + 28), f"{dep:.1f}", FONT_SMALL, NGT, anchor="lm")
    draw_legend(draw, [(k, colors[k]) for k in ["held correct", "changed, still correct", "truth departure"]], 640, height - 72, gap=285)
    save(im, out_path)


def figure_ngt_domain_cue(ngt_pairs, out_path):
    width, height = 1900, 850
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "NGT framing lift by decision domain and cue", width)
    counts = defaultdict(lambda: [0, 0])
    for neutral, framed, user_dir in ngt_pairs:
        cue = framed.get("cue_type")
        domain = framed.get("domain")
        neutral_user = neutral.get("answer") == user_dir
        framed_user = framed.get("answer") == user_dir
        counts[(domain, cue)][0] += 1
        counts[(domain, cue)][1] += int(framed_user) - int(neutral_user)
    x0, y0 = 440, 230
    cell_w, cell_h = 310, 115
    for ci, (_, label) in enumerate(CUE_ORDER):
        draw_text(draw, (x0 + ci * cell_w + cell_w / 2, y0 - 50), label, FONT_LABEL, anchor="ma")
    for ri, domain in enumerate(NGT_DOMAINS):
        y = y0 + ri * (cell_h + 18)
        draw_text(draw, (x0 - 24, y + cell_h / 2), domain.title(), FONT_LABEL, anchor="rm")
        for ci, (cue, _) in enumerate(CUE_ORDER):
            denom, net = counts[(domain, cue)]
            value = pct(net / denom) if denom else 0.0
            x = x0 + ci * cell_w
            color = heat(max(0, value), 55, GT) if value >= 0 else heat(abs(value), 20, STATIC)
            draw.rounded_rectangle((x, y, x + cell_w - 18, y + cell_h), radius=14, fill=color)
            draw_text(draw, (x + cell_w / 2 - 9, y + cell_h / 2), f"{value:+.0f}", FONT_CELL, anchor="mm")
    draw_text(draw, (width / 2, height - 46), "framed user-view selection minus matched neutral selection (percentage points)", FONT, MUTED, anchor="mm")
    save(im, out_path)


def figure_context_confidence_outcome(gt_pairs, ngt_pairs, out_path):
    width, height = 2050, 880
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Context confidence by movement outcome", width)
    groups = defaultdict(list)
    for neutral, framed in gt_pairs:
        if neutral.get("truth_status") != "correct":
            continue
        conf = framed.get("confidence")
        if not isinstance(conf, (int, float)):
            continue
        groups["GT departed" if framed.get("truth_status") != "correct" else "GT preserved"].append(float(conf))
    for neutral, framed, user_dir in ngt_pairs:
        conf = framed.get("confidence")
        if not isinstance(conf, (int, float)):
            continue
        changed = framed.get("answer") != neutral.get("answer")
        user_aligned = framed.get("answer") == user_dir
        label = "NGT switched to user" if changed and user_aligned else "NGT other/held"
        groups[label].append(float(conf))
    labels = ["GT preserved", "GT departed", "NGT other/held", "NGT switched to user"]
    x0, x1 = 650, 1800
    y0, row_h = 210, 115
    draw_percent_axis(draw, x0, y0 - 25, x1, y0 + row_h * len(labels), 5, (1, 3, 5))
    for i, label in enumerate(labels):
        values = groups[label]
        mean = sum(values) / len(values) if values else 0.0
        high = pct(sum(1 for v in values if v >= 4) / len(values)) if values else 0.0
        y = y0 + i * row_h
        color = NGT if "departed" in label or "switched" in label else STATIC
        draw_text(draw, (80, y + 28), label, FONT_LABEL, anchor="lm")
        xx = bar_x(x0, x1, mean, 5)
        draw.rounded_rectangle((x0, y + 5, xx, y + 51), radius=10, fill=color)
        draw_text(draw, (xx + 14, y + 28), f"{mean:.2f} mean, {high:.0f}% high", FONT_SMALL, color, anchor="lm")
    draw_text(draw, ((x0 + x1) / 2, height - 52), "framed-turn confidence (1-5)", FONT_LABEL, anchor="mm")
    save(im, out_path)


def figure_judge_reliability_triage(trigger_summary, context_summary, out_path):
    width, height = 2500, 1180
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Judge reliability differs by diagnostic dimension", width)
    rows = []
    for section, fields in [
        ("binary", ["redo_question_by_reasoning_or_calculation", "rationalized_change", "contradicted_itself"]),
        ("likert", ["supported_user_viewpoint", "disagreed_with_user_points_reverse", "questioned_user_assumptions_reverse", "excessive_flattery", "eager_to_please_user", "sought_user_good_graces", "enthusiastic", "excited_to_engage"]),
        ("factor_scores", ["uncritical_agreement", "obsequiousness", "excitement"]),
    ]:
        for field in fields:
            if section == "binary":
                metric_name = "cohen_kappa"
            elif section == "likert":
                metric_name = "pearson"
            else:
                metric_name = "pearson"
            rows.append(
                {
                    "label": field.replace("_", " ").replace(" by reasoning or calculation", ""),
                    "trigger": trigger_summary[section][field][metric_name],
                    "context": context_summary[section][field][metric_name],
                }
            )
    x0, x1 = 930, 2260
    y0, row_h = 170, 58
    draw_percent_axis(draw, x0, y0 - 25, x1, y0 + row_h * len(rows), 1, (0, 0.5, 1))
    for i, row in enumerate(rows):
        y = y0 + i * row_h + row_h / 2
        draw_text(draw, (70, y), row["label"], FONT_SMALL, anchor="lm")
        xt = bar_x(x0, x1, row["trigger"], 1)
        xc = bar_x(x0, x1, row["context"], 1)
        draw.line((xt, y, xc, y), fill=GRID, width=4)
        draw.ellipse((xt - 8, y - 8, xt + 8, y + 8), fill=ADAPTIVE)
        draw.ellipse((xc - 8, y - 8, xc + 8, y + 8), fill=GT)
    draw_legend(draw, [("trigger", ADAPTIVE), ("context", GT)], 1120, height - 58, gap=160)
    draw_text(draw, ((x0 + x1) / 2, height - 28), "kappa or Pearson reliability", FONT_LABEL, anchor="mm")
    save(im, out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate extra appendix diagnostic figures from official results.")
    parser.add_argument("--run-id", default="trigger_20260504_070840")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--context-summary", type=Path, required=True)
    parser.add_argument("--context-raw", type=Path, required=True)
    parser.add_argument("--judge-trigger-summary", type=Path, required=True)
    parser.add_argument("--judge-context-summary", type=Path, required=True)
    parser.add_argument("--judge-trigger-csv", type=Path, required=True)
    parser.add_argument("--judge-trigger-inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("images/results/appendix"))
    args = parser.parse_args()

    records = collect_trigger(args.results_dir, args.run_id)
    context_summary = load_json(args.context_summary)
    trigger_judge_summary = load_json(args.judge_trigger_summary)
    context_judge_summary = load_json(args.judge_context_summary)
    judge_rows = load_judge_rows(args.judge_trigger_csv)
    judge_meta = load_trigger_judge_inputs(args.judge_trigger_inputs)
    gt_pairs, ngt_pairs = context_pairs(load_context_rows(args.context_raw))

    outputs = [
        ("temporal_state_paths.png", lambda p: figure_temporal_state_paths(records, p)),
        ("judge_mechanism_outcome.png", lambda p: figure_judge_mechanism_outcome(judge_rows, p)),
        ("trigger_confidence_high_risk.png", lambda p: figure_trigger_confidence_high_risk(records, p)),
        ("context_ngt_directionality.png", lambda p: figure_context_ngt_directionality(context_summary, p)),
        ("validation_coverage_funnel.png", lambda p: figure_validation_coverage(records, context_summary, trigger_judge_summary, context_judge_summary, p)),
        ("trigger_item_concentration.png", lambda p: figure_item_concentration(records, p)),
        ("trigger_domain_source_susceptibility.png", lambda p: figure_trigger_domain_source(records, p)),
        ("trigger_adaptive_family_lift.png", lambda p: figure_adaptive_family_lift(records, p)),
        ("trigger_strong_recheck_boundary.png", lambda p: figure_strong_recheck_boundary(records, judge_rows, judge_meta, p)),
        ("trigger_confidence_risk_calibration.png", lambda p: figure_confidence_risk_calibration(records, p)),
        ("context_gt_source_decomposition.png", lambda p: figure_context_gt_source_decomposition(gt_pairs, p)),
        ("context_change_decomposition.png", lambda p: figure_context_change_decomposition(gt_pairs, p)),
        ("context_ngt_domain_cue.png", lambda p: figure_ngt_domain_cue(ngt_pairs, p)),
        ("context_confidence_outcome.png", lambda p: figure_context_confidence_outcome(gt_pairs, ngt_pairs, p)),
        ("judge_reliability_triage.png", lambda p: figure_judge_reliability_triage(trigger_judge_summary, context_judge_summary, p)),
    ]
    for filename, fn in outputs:
        path = args.out_dir / filename
        fn(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
