import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

try:
    from plot_context_results import make_badge
except ImportError:
    from Experimental.plot_context_results import make_badge


MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
    "mistralai/mistral-medium-3.1",
    "cohere/command-r-08-2024",
]

MODEL_LABELS = {
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "GPT-5.4 Mini",
    "openai/gpt-5.4-nano": "GPT-5.4 Nano",
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "google/gemini-3.1-flash-lite-preview": "Gemini Flash Lite",
    "mistralai/mistral-medium-3.1": "Mistral Medium 3.1",
    "cohere/command-r-08-2024": "Command R",
}

MODEL_SHORT_LABELS = {
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "GPT-5.4\nMini",
    "openai/gpt-5.4-nano": "GPT-5.4\nNano",
    "anthropic/claude-opus-4.5": "Opus 4.5",
    "anthropic/claude-sonnet-4.5": "Sonnet 4.5",
    "anthropic/claude-haiku-4.5": "Haiku 4.5",
    "google/gemini-3.1-flash-lite-preview": "Gemini\nFlash Lite",
    "mistralai/mistral-medium-3.1": "Mistral\nMedium",
    "cohere/command-r-08-2024": "Command R",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGO_FILES = {
    "openai": "openai.png",
    "anthropic": "claude.png",
    "google": "gemini.png",
    "mistralai": "mistral.png",
    "cohere": "cohere.png",
}
MODEL_BADGE_FILES = {
    "openai/gpt-5.4": "gpt-5.4.png",
    "openai/gpt-5.4-mini": "gpt-5.4-mini.png",
    "openai/gpt-5.4-nano": "gpt-5.4-nano.png",
    "anthropic/claude-sonnet-4.5": "sonnet-4.5.png",
    "anthropic/claude-haiku-4.5": "haiku-4.5.png",
    "google/gemini-3.1-flash-lite-preview": "gemini-flash-lite.png",
    "mistralai/mistral-medium-3.1": "mistral-medium.png",
    "cohere/command-r-08-2024": "command-r.png",
}
FAMILY_COLORS = {
    "openai": "#6F55D9",
    "anthropic": "#D96C42",
    "google": "#357AE8",
    "mistralai": "#E65F2B",
    "cohere": "#35685A",
}

TRIGGER_LABELS = {
    "simple_baseline": "Baseline",
    "authority": "Authority",
    "social_proof": "Social proof",
    "consistency": "Consistency",
    "liking": "Liking",
    "reciprocity": "Reciprocity",
    "scarcity": "Scarcity",
    "unity": "Unity",
}

TONE_LABELS = {"mild": "Mild", "moderate": "Moderate", "strong": "Strong"}
TONES = ["mild", "moderate", "strong"]
CIALDINI_TRIGGERS = set(TRIGGER_LABELS) - {"simple_baseline"}

INK = "#17202A"
MUTED = "#637083"
GRID = "#D9E0E8"
LIGHT_GRID = "#EEF2F6"
STATIC = "#5B7DB8"
ADAPTIVE = "#D07A45"
GT = "#4B8B6A"
NGT = "#8B5EA7"
ACCENT = "#C94C4C"
PAPER_BG = "#FFFFFF"
PANEL_BG = "#FAFBFC"


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = load_font(46, True)
FONT_SUBTITLE = load_font(27)
FONT_PANEL = load_font(27, True)
FONT_AXIS = load_font(22)
FONT_AXIS_BOLD = load_font(22, True)
FONT_SMALL = load_font(19)
FONT_TINY = load_font(16)

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-oriented trigger figure candidates from pass@1-clean results."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--clean", action="store_true", help="Remove stale trigger figures before writing new ones.")
    return parser.parse_args()


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def metric(record, branch, temporal=False):
    if temporal:
        if branch == "GT":
            return bool(record.get("three_repetition_truth_departure"))
        return bool(record.get("three_repetition_answer_switch"))
    if branch == "GT":
        return bool(record.get("truth_departure"))
    return bool(record.get("single_trigger_answer_switch") or record.get("answer_changed"))


def denom_ok(record, branch):
    if branch == "GT":
        return bool(record.get("initial_correct"))
    return bool(record.get("eligible"))


def pct(value):
    return 100.0 * value


def fmt_pct(value, digits=1):
    return f"{pct(value):.{digits}f}%"


def draw_text(draw, xy, text, font, fill=INK, anchor=None):
    draw.text(xy, str(text), font=font, fill=fill, anchor=anchor)


def text_width(draw, text, font):
    box = draw.textbbox((0, 0), str(text), font=font)
    return box[2] - box[0]


def wrap_text(draw, text, font, max_width):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if not current or text_width(draw, trial, font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw, text, x, y, max_width, font, fill=INK, line_gap=3, anchor_center=False):
    lines = wrap_text(draw, text, font, max_width)
    for line in lines:
        if anchor_center:
            draw_text(draw, (x + max_width / 2, y), line, font, fill, anchor="ma")
        else:
            draw_text(draw, (x, y), line, font, fill)
        y += font.size + line_gap
    return y


def draw_header(draw, title, subtitle, width):
    draw_text(draw, (65, 38), title, FONT_TITLE)
    if subtitle:
        draw_text(draw, (65, 94), subtitle, FONT_SUBTITLE, MUTED)
        line_y = 145
    else:
        line_y = 122
    draw.line((65, line_y, width - 65, line_y), fill=LIGHT_GRID, width=3)


def draw_axis(draw, x0, y0, width, height, max_rate, ticks=None):
    ticks = ticks or [0, 0.25, 0.5, 0.75, 1.0]
    for t in ticks:
        x = x0 + width * (t * max_rate / max_rate)
        draw.line((x, y0, x, y0 + height), fill=LIGHT_GRID, width=1)
        draw_text(draw, (x, y0 + height + 8), f"{int(t * max_rate * 100)}", FONT_TINY, MUTED, anchor="ma")
    draw.line((x0, y0 + height, x0 + width, y0 + height), fill=GRID, width=1)


def scale_x(value, x0, width, max_rate):
    return x0 + width * min(value, max_rate) / max_rate


def make_logo_badge(family, diameter=64):
    color = FAMILY_COLORS.get(family, ADAPTIVE)
    rgb = ImageColor.getrgb(color)
    margin = max(8, diameter // 7)
    size = diameter + 2 * margin
    center = size // 2
    radius = diameter // 2
    badge = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(badge)
    draw.ellipse(
        (center - radius + 3, center - radius + 5, center + radius + 3, center + radius + 5),
        fill=(15, 23, 42, 36),
    )
    draw.ellipse(
        (center - radius, center - radius, center + radius, center + radius),
        fill=(255, 255, 255, 246),
        outline=(*rgb, 230),
        width=max(3, diameter // 18),
    )

    logo_file = LOGO_FILES.get(family)
    logo_path = REPO_ROOT / "images" / "logos" / logo_file if logo_file else None
    if logo_path and logo_path.exists():
        logo = Image.open(logo_path).convert("RGBA")
        logo.thumbnail((int(diameter * 0.56), int(diameter * 0.56)), RESAMPLE_LANCZOS)
        badge.alpha_composite(logo, (center - logo.width // 2, center - logo.height // 2))
    else:
        draw_text(draw, (center, center), family[:1].upper(), load_font(int(diameter * 0.42), True), color, anchor="mm")
    return badge


def load_model_badge(model_id, diameter=76):
    filename = MODEL_BADGE_FILES.get(model_id)
    if filename:
        path = REPO_ROOT / "images" / "model_logos" / "model_badges" / filename
        if path.exists():
            badge = Image.open(path).convert("RGBA")
            badge.thumbnail((diameter, diameter), RESAMPLE_LANCZOS)
            return badge
    return make_logo_badge(model_id.split("/")[0], diameter)


def collect_records(results_dir, run_id):
    out = {"single": [], "temporal": []}
    for branch in ["gt", "ngt"]:
        branch_label = branch.upper()
        for mode in ["static", "adaptive"]:
            single_path = results_dir / f"{run_id}_{branch}_trigger_{mode}.jsonl.gz"
            temporal_path = results_dir / f"{run_id}_{branch}_trigger_temporal_{mode}.jsonl.gz"
            for record in read_jsonl_gz(single_path):
                record["_branch"] = branch_label
                record["_mode"] = mode
                out["single"].append(record)
            for record in read_jsonl_gz(temporal_path):
                record["_branch"] = branch_label
                record["_mode"] = mode
                out["temporal"].append(record)
    return out


def grouped_rate(records, predicate):
    denom = 0
    events = 0
    for record in records:
        if predicate(record) and denom_ok(record, record["_branch"]):
            denom += 1
            events += int(metric(record, record["_branch"], "trigger_sequence" in record))
    return events / denom if denom else 0.0, denom, events


def build_tables(records):
    single = records["single"]
    temporal = records["temporal"]
    headline = []
    for model in MODELS:
        for branch in ["GT", "NGT"]:
            for mode in ["static", "adaptive"]:
                for source, source_records in [("single", single), ("temporal", temporal)]:
                    value, denom, events = grouped_rate(
                        source_records,
                        lambda r, model=model, branch=branch, mode=mode: r["_branch"] == branch
                        and r["_mode"] == mode
                        and r["model"] == model,
                    )
                    headline.append(
                        {
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "branch": branch,
                            "mode": mode,
                            "source": source,
                            "denom": denom,
                            "events": events,
                            "rate": value,
                        }
                    )

    tone = []
    for branch in ["GT", "NGT"]:
        for mode in ["static", "adaptive"]:
            for tone_name in TONES:
                value, denom, events = grouped_rate(
                    single,
                    lambda r, branch=branch, mode=mode, tone_name=tone_name: r["_branch"] == branch
                    and r["_mode"] == mode
                    and r.get("tone") == tone_name,
                )
                tone.append(
                    {
                        "branch": branch,
                        "mode": mode,
                        "tone": tone_name,
                        "denom": denom,
                        "events": events,
                        "rate": value,
                    }
                )

    family = []
    for model in MODELS:
        for trigger in TRIGGER_LABELS:
            value, denom, events = grouped_rate(
                single,
                lambda r, model=model, trigger=trigger: r["_branch"] == "NGT"
                and r["_mode"] == "adaptive"
                and r["model"] == model
                and r.get("trigger") == trigger,
            )
            family.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "trigger": trigger,
                    "trigger_label": TRIGGER_LABELS[trigger],
                    "denom": denom,
                    "events": events,
                    "rate": value,
                }
            )

    sequence = []
    sequences = sorted({">".join(record.get("trigger_sequence") or []) for record in temporal})
    for seq in sequences:
        for mode in ["static", "adaptive"]:
            value, denom, events = grouped_rate(
                temporal,
                lambda r, seq=seq, mode=mode: r["_branch"] == "NGT"
                and r["_mode"] == mode
                and ">".join(r.get("trigger_sequence") or []) == seq,
            )
            sequence.append(
                {
                    "sequence": seq,
                    "sequence_label": sequence_label(seq),
                    "mode": mode,
                    "denom": denom,
                    "events": events,
                    "rate": value,
                }
            )
    return headline, tone, family, sequence


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_lookup(rows, **filters):
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            return row
    raise KeyError(filters)


def cialdini_single(record):
    return record.get("trigger") in CIALDINI_TRIGGERS


def temporal_stage(record):
    sequence = record.get("trigger_sequence") or []
    if len(set(sequence)) == 1:
        if sequence and sequence[0] != "simple_baseline":
            return "same_family"
        return "baseline"
    return "heterogeneous"


def aggregate_rate(records, branch, predicate, temporal=False):
    denom = 0
    events = 0
    for record in records:
        if predicate(record) and denom_ok(record, branch):
            denom += 1
            events += int(metric(record, branch, temporal))
    return events / denom if denom else 0.0, denom, events


def aggregate_answer_change(records, predicate):
    denom = 0
    events = 0
    for record in records:
        if predicate(record):
            denom += 1
            events += int(record.get("single_trigger_answer_switch") or record.get("answer_changed"))
    return events / denom if denom else 0.0, denom, events


def build_trigger_figure_tables(records):
    single = records["single"]
    temporal = records["temporal"]

    model = []
    for model_id in MODELS:
        row = {
            "model": model_id,
            "model_label": MODEL_LABELS[model_id],
            "model_short_label": MODEL_SHORT_LABELS[model_id].replace("\n", " "),
            "family": model_id.split("/")[0],
        }
        change_rate, change_denom, change_events = aggregate_answer_change(
            single,
            lambda r, model_id=model_id: r["_branch"] == "GT"
            and r["model"] == model_id
            and cialdini_single(r),
        )
        row["gt_change_rate"] = change_rate
        row["gt_change_denom"] = change_denom
        row["gt_change_events"] = change_events
        for branch in ["GT", "NGT"]:
            rate, denom, events = aggregate_rate(
                single,
                branch,
                lambda r, model_id=model_id, branch=branch: r["_branch"] == branch
                and r["model"] == model_id
                and cialdini_single(r),
            )
            row[f"{branch.lower()}_rate"] = rate
            row[f"{branch.lower()}_denom"] = denom
            row[f"{branch.lower()}_events"] = events
        model.append(row)

    tone = []
    for branch in ["GT", "NGT"]:
        for group, label, group_predicate in [
            ("all", "All models", lambda r: True),
            ("opus", "Opus 4.5", lambda r: r["model"] == "anthropic/claude-opus-4.5"),
        ]:
            for tone_name in TONES:
                rate, denom, events = aggregate_rate(
                    single,
                    branch,
                    lambda r, branch=branch, group_predicate=group_predicate, tone_name=tone_name: r["_branch"] == branch
                    and cialdini_single(r)
                    and r.get("tone") == tone_name
                    and group_predicate(r),
                )
                tone.append(
                    {
                        "branch": branch,
                        "group": group,
                        "group_label": label,
                        "tone": tone_name,
                        "rate": rate,
                        "denom": denom,
                        "events": events,
                    }
                )

    static_adaptive = []
    for branch in ["GT", "NGT"]:
        for mode in ["static", "adaptive"]:
            rate, denom, events = aggregate_rate(
                single,
                branch,
                lambda r, branch=branch, mode=mode: r["_branch"] == branch
                and r["_mode"] == mode
                and cialdini_single(r),
            )
            static_adaptive.append(
                {
                    "branch": branch,
                    "mode": mode,
                    "rate": rate,
                    "denom": denom,
                    "events": events,
                }
            )

    temporal_pressure = []
    for branch in ["GT", "NGT"]:
        for mode in ["static", "adaptive"]:
            single_rate, single_denom, single_events = aggregate_rate(
                single,
                branch,
                lambda r, branch=branch, mode=mode: r["_branch"] == branch
                and r["_mode"] == mode
                and cialdini_single(r),
            )
            temporal_pressure.append(
                {
                    "branch": branch,
                    "mode": mode,
                    "stage": "single",
                    "stage_label": "Single",
                    "rate": single_rate,
                    "denom": single_denom,
                    "events": single_events,
                }
            )
            for stage, stage_label in [
                ("same_family", "Same-family x3"),
                ("heterogeneous", "Mixed x3"),
            ]:
                rate, denom, events = aggregate_rate(
                    temporal,
                    branch,
                    lambda r, branch=branch, mode=mode, stage=stage: r["_branch"] == branch
                    and r["_mode"] == mode
                    and temporal_stage(r) == stage,
                    temporal=True,
                )
                temporal_pressure.append(
                    {
                        "branch": branch,
                        "mode": mode,
                        "stage": stage,
                        "stage_label": stage_label,
                        "rate": rate,
                        "denom": denom,
                        "events": events,
                    }
                )

    confidence = []
    for branch in ["GT", "NGT"]:
        for category in (["preserved", "departed"] if branch == "GT" else ["held", "switched"]):
            by_turn = {turn: [] for turn in range(4)}
            for record in temporal:
                if record["_branch"] != branch or not denom_ok(record, branch):
                    continue
                event = metric(record, branch, temporal=True)
                if branch == "GT":
                    wanted = event if category == "departed" else not event
                else:
                    wanted = event if category == "switched" else not event
                if not wanted:
                    continue
                values = [record.get("initial_confidence")]
                values.extend(round_record.get("confidence") for round_record in record.get("rounds") or [])
                if len(values) != 4:
                    continue
                for turn, value in enumerate(values):
                    if isinstance(value, (int, float)):
                        by_turn[turn].append(float(value))
            for turn, values in by_turn.items():
                confidence.append(
                    {
                        "branch": branch,
                        "category": category,
                        "turn": turn,
                        "turn_label": ["Initial", "Turn 1", "Turn 2", "Turn 3"][turn],
                        "mean_confidence": sum(values) / len(values) if values else 0.0,
                        "n": len(values),
                    }
                )
    return {
        "model_comparison": model,
        "tone_gradient_opus": tone,
        "static_vs_adaptive": static_adaptive,
        "temporal_pressure": temporal_pressure,
        "confidence_trajectory": confidence,
    }


def draw_dumbbell_panel(draw, rows, x, y, w, h, title, subtitle, max_rate=1.0, show_labels=False):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=PANEL_BG, outline=LIGHT_GRID, width=2)
    draw_text(draw, (x + 30, y + 25), title, FONT_PANEL)
    draw_text(draw, (x + 30, y + 60), subtitle, FONT_SMALL, MUTED)
    axis_x = x + (235 if show_labels else 36)
    axis_y = y + 122
    axis_w = w - (285 if show_labels else 92)
    row_h = (h - 132) / len(rows)
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        xx = axis_x + axis_w * (t * max_rate / max_rate)
        draw.line((xx, axis_y - 8, xx, axis_y + row_h * len(rows) - 3), fill=LIGHT_GRID, width=1)
    for i, row in enumerate(rows):
        yy = axis_y + i * row_h + row_h / 2
        if show_labels:
            draw_text(draw, (x + 22, yy), row["label"], FONT_AXIS, INK, anchor="lm")
        xs = scale_x(row["static"], axis_x, axis_w, max_rate)
        xa = scale_x(row["adaptive"], axis_x, axis_w, max_rate)
        draw.line((xs, yy, xa, yy), fill=GRID, width=5)
        draw.ellipse((xs - 8, yy - 8, xs + 8, yy + 8), fill=STATIC)
        draw.ellipse((xa - 10, yy - 10, xa + 10, yy + 10), fill=ADAPTIVE)
        label = fmt_pct(row["adaptive"])
        label_x = min(axis_x + axis_w + 6, max(xs, xa) + 10)
        draw_text(draw, (label_x, yy), label, FONT_TINY, MUTED, anchor="lm")
    for t in [0, 0.5, 1.0]:
        xx = axis_x + axis_w * t
        draw_text(draw, (xx, y + h - 32), f"{int(t * max_rate * 100)}", FONT_TINY, MUTED, anchor="ma")
    draw_text(draw, (axis_x + axis_w / 2, y + h - 17), "rate (%)", FONT_TINY, MUTED, anchor="ma")


def figure_headline(path, headline):
    width, height = 1900, 1180
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(
        draw,
        "Trigger-induced sycophancy",
        "",
        width,
    )

    grid_x, grid_y = 170, 205
    col_w, row_h = 770, 395
    gutter_x, gutter_y = 90, 86

    def pooled(branch, source, mode):
        rows = [row for row in headline if row["branch"] == branch and row["source"] == source and row["mode"] == mode]
        denom = sum(row["denom"] for row in rows)
        events = sum(row["events"] for row in rows)
        return events / denom if denom else 0.0

    def panel(x, y, branch, source, title, accent):
        draw.rectangle((x, y, x + col_w, y + row_h), fill="white", outline="#D8DEE8", width=2)
        draw_text(draw, (x + 28, y + 28), title, load_font(30, True), INK)
        static_rate = pooled(branch, source, "static")
        adaptive_rate = pooled(branch, source, "adaptive")
        rates = [("Static", static_rate, STATIC), ("Adaptive", adaptive_rate, accent)]
        axis_x, axis_y = x + 145, y + 112
        axis_w = col_w - 220
        bar_h = 58
        row_gap = 92
        for tick in [0, 0.25, 0.50, 0.75, 1.0]:
            xx = axis_x + axis_w * tick
            draw.line((xx, axis_y - 18, xx, axis_y + row_gap + bar_h + 22), fill="#EDF1F6", width=1)
            draw_text(draw, (xx, axis_y + row_gap + bar_h + 40), f"{int(tick * 100)}", FONT_SMALL, MUTED, anchor="ma")
        for i, (label, value, color) in enumerate(rates):
            yy = axis_y + i * row_gap
            draw_text(draw, (x + 34, yy + bar_h / 2), label, FONT_AXIS_BOLD, INK, anchor="lm")
            draw.rectangle((axis_x, yy, axis_x + axis_w, yy + bar_h), fill="#F0F3F7")
            fill_w = axis_w * value
            draw.rectangle((axis_x, yy, axis_x + fill_w, yy + bar_h), fill=color)
            draw_text(draw, (axis_x + fill_w + 12, yy + bar_h / 2), fmt_pct(value), load_font(24, True), INK, anchor="lm")
        draw_text(draw, (axis_x + axis_w / 2, y + row_h - 42), "Rate (%)", FONT_SMALL, MUTED, anchor="ma")

    panels = [
        (grid_x, grid_y, "GT", "single", "GT: wrong turns", "#B8443F"),
        (grid_x + col_w + gutter_x, grid_y, "GT", "temporal", "GT: after pressure", "#B8443F"),
        (grid_x, grid_y + row_h + gutter_y, "NGT", "single", "NGT: decision shifts", "#2870A8"),
        (grid_x + col_w + gutter_x, grid_y + row_h + gutter_y, "NGT", "temporal", "NGT: after pressure", "#2870A8"),
    ]
    for args in panels:
        panel(*args)
    im.save(path)


def figure_tone_gradient(path, tone_rows):
    width, height = 1800, 980
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(
        draw,
        "Tone is the most influential factor",
        "",
        width,
    )
    panels = [
        ("GT", 115, 210, "Wrong turns on factual questions", 0.35, [0, 0.1, 0.2, 0.3]),
        ("NGT", 955, 210, "Decision shifts", 1.0, [0, 0.25, 0.5, 0.75, 1.0]),
    ]
    mode_specs = [("static", "Static", STATIC), ("adaptive", "Adaptive", ADAPTIVE)]
    for branch, x, y, title, max_rate, ticks in panels:
        w, h = 720, 600
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D8DEE8", width=2)
        draw_text(draw, (x + 24, y + 24), title, load_font(31, True), INK)
        px0, py0 = x + 105, y + 100
        pw, ph = w - 175, h - 205
        for tick in ticks:
            yy = py0 + ph - ph * tick / max_rate
            draw.line((px0, yy, px0 + pw, yy), fill="#E8EDF3", width=1)
            draw_text(draw, (px0 - 16, yy), f"{int(tick * 100)}", FONT_SMALL, MUTED, anchor="rm")
        for mode, label, color in mode_specs:
            values = [row_lookup(tone_rows, branch=branch, mode=mode, tone=tone)["rate"] for tone in TONES]
            pts = []
            for i, value in enumerate(values):
                xx = px0 + i * pw / 2
                yy = py0 + ph - ph * value / max_rate
                pts.append((xx, yy))
            draw.line(pts, fill=color, width=7)
            for xx, yy in pts:
                draw.ellipse((xx - 12, yy - 12, xx + 12, yy + 12), fill=color)
        legend_x, legend_y = x + w - 230, y + 56
        for i, (_, label, color) in enumerate(mode_specs):
            yy = legend_y + i * 32
            draw.line((legend_x, yy, legend_x + 42, yy), fill=color, width=7)
            draw_text(draw, (legend_x + 54, yy), label, FONT_AXIS_BOLD, color, anchor="lm")
        for i, tone in enumerate(TONES):
            xx = px0 + i * pw / 2
            draw_text(draw, (xx, py0 + ph + 34), TONE_LABELS[tone], FONT_AXIS_BOLD, INK, anchor="ma")
        draw_text(draw, (px0 + pw / 2, y + h - 34), "Tone", FONT_AXIS_BOLD, INK, anchor="ma")
        y_label = Image.new("RGBA", (210, 34), (255, 255, 255, 0))
        yd = ImageDraw.Draw(y_label)
        yd.text((0, 0), "Rate (%)", font=FONT_AXIS_BOLD, fill=INK)
        y_label = y_label.rotate(90, expand=True)
        im.paste(y_label, (x + 20, y + 245), y_label)
        draw = ImageDraw.Draw(im)
    im.save(path)


def heat_color(value, max_value):
    lo = (247, 249, 252)
    hi = (196, 78, 78)
    t = 0.0 if max_value <= 0 else min(1.0, value / max_value)
    return tuple(int(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))


def figure_family_heatmap(path, family_rows):
    averages = {}
    for trigger in TRIGGER_LABELS:
        vals = [row["rate"] for row in family_rows if row["trigger"] == trigger]
        averages[trigger] = sum(vals) / len(vals)
    triggers = sorted(TRIGGER_LABELS, key=lambda trigger: averages[trigger], reverse=True)
    width, height = 2100, 900
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(
        draw,
        "Trigger-family landscape",
        "",
        width,
    )
    left, top = 290, 215
    cell_w, cell_h = 175, 68
    max_value = max(row["rate"] for row in family_rows)
    ordered_models = sorted(
        MODELS,
        key=lambda model: sum(row["rate"] for row in family_rows if row["model"] == model) / len(TRIGGER_LABELS),
        reverse=True,
    )
    for ci, model in enumerate(ordered_models):
        draw_wrapped(draw, MODEL_SHORT_LABELS[model], left + ci * cell_w, top - 70, cell_w - 12, FONT_SMALL, INK, anchor_center=True)
    for ri, trigger in enumerate(triggers):
        y = top + ri * cell_h
        draw_text(draw, (left - 24, y + cell_h / 2), TRIGGER_LABELS[trigger], FONT_AXIS_BOLD, INK, anchor="rm")
        for ci, model in enumerate(ordered_models):
            row = row_lookup(family_rows, model=model, trigger=trigger)
            x = left + ci * cell_w
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), fill=heat_color(row["rate"], max_value))
        avg_x = left + len(ordered_models) * cell_w + 18
        avg_w = 150
        draw.rectangle((avg_x, y, avg_x + avg_w, y + cell_h - 4), fill="#F0F3F7")
        draw.rectangle((avg_x, y, avg_x + avg_w * averages[trigger], y + cell_h - 4), fill=ACCENT)
        draw_text(draw, (avg_x + avg_w + 14, y + cell_h / 2), fmt_pct(averages[trigger]), FONT_AXIS_BOLD, INK, anchor="lm")
    draw_text(draw, (left + len(ordered_models) * cell_w + 18, top - 54), "Mean", FONT_AXIS_BOLD, INK, anchor="lm")
    draw_text(draw, (left + (len(ordered_models) * cell_w) / 2, height - 55), "Darker cells indicate more NGT decision shifts", FONT_SMALL, MUTED, anchor="ma")
    im.save(path)


def sequence_label(sequence):
    parts = sequence.split(">")
    if len(set(parts)) == 1:
        return f"{TRIGGER_LABELS.get(parts[0], parts[0])} x3"
    return " -> ".join(TRIGGER_LABELS.get(part, part) for part in parts)


def figure_temporal_sequences(path, sequence_rows):
    adaptive = {
        row["sequence"]: row["rate"]
        for row in sequence_rows
        if row["mode"] == "adaptive"
    }
    sequences = sorted(adaptive, key=adaptive.get, reverse=True)
    width, height = 2040, 1220
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(
        draw,
        "Three-turn trajectories amplify pressure accommodation",
        "NGT temporal answer-switch rate by sequence; same-family and mixed-family sequences share the same mild -> moderate -> strong tone ramp.",
        width,
    )
    x0, y0 = 610, 220
    plot_w = 1200
    row_h = 64
    max_rate = 1.0
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        xx = x0 + plot_w * t
        draw.line((xx, y0 - 8, xx, y0 + row_h * len(sequences)), fill=LIGHT_GRID, width=1)
        draw_text(draw, (xx, y0 + row_h * len(sequences) + 10), f"{int(t * 100)}", FONT_TINY, MUTED, anchor="ma")
    draw.ellipse((x0, 172, x0 + 20, 192), fill=STATIC)
    draw_text(draw, (x0 + 32, 182), "Static", FONT_AXIS, MUTED, anchor="lm")
    draw.ellipse((x0 + 150, 171, x0 + 172, 193), fill=ADAPTIVE)
    draw_text(draw, (x0 + 188, 182), "Adaptive", FONT_AXIS, MUTED, anchor="lm")
    for i, sequence in enumerate(sequences):
        y = y0 + i * row_h + row_h / 2
        label = sequence_label(sequence)
        draw_text(draw, (x0 - 24, y), label, FONT_AXIS, INK, anchor="rm")
        static = row_lookup(sequence_rows, sequence=sequence, mode="static")["rate"]
        adapt = row_lookup(sequence_rows, sequence=sequence, mode="adaptive")["rate"]
        xs = scale_x(static, x0, plot_w, max_rate)
        xa = scale_x(adapt, x0, plot_w, max_rate)
        draw.line((xs, y, xa, y), fill=GRID, width=6)
        draw.ellipse((xs - 9, y - 9, xs + 9, y + 9), fill=STATIC)
        draw.ellipse((xa - 11, y - 11, xa + 11, y + 11), fill=ADAPTIVE)
        draw_text(draw, (x0 + plot_w + 18, y), fmt_pct(adapt), FONT_AXIS_BOLD, ADAPTIVE, anchor="lm")
    draw_text(draw, (x0 + plot_w / 2, height - 58), "NGT answer-switch rate (%)", FONT_SMALL, MUTED, anchor="ma")
    im.save(path)


def figure_scatter(path, headline):
    width, height = 1800, 1120
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(
        draw,
        "Model comparison",
        "",
        width,
    )
    rows = []
    for model in MODELS:
        gt_rate = row_lookup(headline, model=model, branch="GT", mode="adaptive", source="single")["rate"]
        ngt_rate = row_lookup(headline, model=model, branch="NGT", mode="adaptive", source="single")["rate"]
        rows.append(
            {
                "model": model,
                "label": MODEL_SHORT_LABELS[model],
                "family": model.split("/")[0],
                "gt": gt_rate,
                "ngt": ngt_rate,
            }
        )
    left, top = 210, 185
    plot_w, plot_h = 1360, 760
    min_x, max_x = 0.05, 0.38
    min_y, max_y = 0.32, 0.95
    x_cut, y_cut = 0.20, 0.60

    def x_pos(value):
        return left + plot_w * (value - min_x) / (max_x - min_x)

    def y_pos(value):
        return top + plot_h - plot_h * (value - min_y) / (max_y - min_y)

    # Soft quadrant fields, deliberately more visual than a plain scatter background.
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    od = ImageDraw.Draw(overlay)
    x_mid, y_mid = x_pos(x_cut), y_pos(y_cut)
    od.rectangle((left, top, x_mid, y_mid), fill=(*ImageColor.getrgb("#EAF4EF"), 150))
    od.rectangle((x_mid, top, left + plot_w, y_mid), fill=(*ImageColor.getrgb("#FFF3E8"), 150))
    od.rectangle((left, y_mid, x_mid, top + plot_h), fill=(*ImageColor.getrgb("#EEF5FB"), 150))
    od.rectangle((x_mid, y_mid, left + plot_w, top + plot_h), fill=(*ImageColor.getrgb("#F9EEF1"), 135))
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(im)

    for value in [0.10, 0.20, 0.30]:
        xx = x_pos(value)
        draw.line((xx, top, xx, top + plot_h), fill="#DCE3EC", width=1)
        draw_text(draw, (xx, top + plot_h + 18), f"{int(value * 100)}%", FONT_SMALL, MUTED, anchor="ma")
    for value in [0.40, 0.60, 0.80]:
        yy = y_pos(value)
        draw.line((left, yy, left + plot_w, yy), fill="#DCE3EC", width=1)
        draw_text(draw, (left - 16, yy), f"{int(value * 100)}%", FONT_SMALL, MUTED, anchor="rm")
    draw.line((x_mid, top, x_mid, top + plot_h), fill="#586474", width=2)
    draw.line((left, y_mid, left + plot_w, y_mid), fill="#586474", width=2)
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline="#1A2028", width=2)

    label_specs = {
        "openai/gpt-5.4": (-18, -42, "rm"),
        "openai/gpt-5.4-mini": (24, 16, "lm"),
        "openai/gpt-5.4-nano": (24, 14, "lm"),
        "anthropic/claude-sonnet-4.5": (22, 18, "lm"),
        "anthropic/claude-haiku-4.5": (22, -42, "lm"),
        "google/gemini-3.1-flash-lite-preview": (22, 18, "lm"),
        "mistralai/mistral-medium-3.1": (24, -38, "lm"),
        "cohere/command-r-08-2024": (24, 16, "lm"),
    }
    for row in rows:
        x = x_pos(row["gt"])
        y = y_pos(row["ngt"])
        badge = make_logo_badge(row["family"], 64)
        im.paste(badge, (int(x - badge.width / 2), int(y - badge.height / 2)), badge)
        draw = ImageDraw.Draw(im)
        dx, dy, anchor = label_specs.get(row["model"], (12, 8, "lm"))
        draw.multiline_text((x + dx, y + dy), row["label"], font=FONT_SMALL, fill=INK, anchor=anchor, spacing=2, align="center")

    draw_text(draw, (left + plot_w / 2, top + plot_h + 80), "Wrong turns on factual questions (%)", FONT_AXIS_BOLD, INK, anchor="ma")
    y_label = Image.new("RGBA", (430, 42), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 0), "Decision shifts (%)", font=FONT_AXIS_BOLD, fill=INK)
    y_label = y_label.rotate(90, expand=True)
    im.paste(y_label, (72, top + 170), y_label)
    im.save(path)


def figure_model_comparison(path, rows):
    width, height = 2200, 1280
    im = Image.new("RGBA", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_text(draw, (width / 2, 44), "Trigger Model Comparison", load_font(58, True), INK, anchor="ma")
    draw_text(draw, (width / 2, 112), "Single-follow-up Cialdini triggers, pass@1-clean", load_font(32), MUTED, anchor="ma")
    draw.line((65, 155, width - 65, 155), fill=LIGHT_GRID, width=3)

    left, top = 285, 195
    plot_w, plot_h = 1600, 840
    x_min, x_max = 0.05, 0.36
    y_min, y_max = 0.30, 0.90

    def x_pos(value):
        return left + plot_w * (value - x_min) / (x_max - x_min)

    def y_pos(value):
        return top + plot_h - plot_h * (value - y_min) / (y_max - y_min)

    draw.rounded_rectangle((left - 48, top - 52, left + plot_w + 48, top + plot_h + 100), radius=28, fill="#F7FAFC")

    for value in [0.10, 0.20, 0.30]:
        xx = x_pos(value)
        draw.line((xx, top, xx, top + plot_h), fill="#DCE4ED", width=2)
        draw_text(draw, (xx, top + plot_h + 24), f"{int(value * 100)}%", load_font(30), MUTED, anchor="ma")
    for value in [0.40, 0.60, 0.80]:
        yy = y_pos(value)
        draw.line((left, yy, left + plot_w, yy), fill="#DCE4ED", width=2)
        draw_text(draw, (left - 20, yy), f"{int(value * 100)}%", load_font(30), MUTED, anchor="rm")
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#758294", width=4)
    draw.line((left, top, left, top + plot_h), fill="#758294", width=4)

    label_font = load_font(38, True)
    offsets = {
        "openai/gpt-5.4": (-160, -56),
        "openai/gpt-5.4-mini": (-74, 166),
        "openai/gpt-5.4-nano": (-80, -210),
        "anthropic/claude-opus-4.5": (-186, 124),
        "anthropic/claude-sonnet-4.5": (120, -32),
        "anthropic/claude-haiku-4.5": (168, 72),
        "google/gemini-3.1-flash-lite-preview": (88, -80),
        "mistralai/mistral-medium-3.1": (-40, 80),
        "cohere/command-r-08-2024": (10, -150),
    }

    boxes = []
    for row in sorted(rows, key=lambda r: r["ngt_rate"]):
        anchor_x = x_pos(row["gt_rate"])
        anchor_y = y_pos(row["ngt_rate"])
        dx, dy = offsets[row["model"]]
        cx = max(left + 95, min(left + plot_w - 95, anchor_x + dx))
        cy = max(top + 92, min(top + plot_h - 82, anchor_y + dy))
        badge = make_badge(row["model"])
        total_h = badge.height + 14 + (78 if "\n" in MODEL_SHORT_LABELS[row["model"]] else 40)
        x = int(cx - badge.width / 2)
        y = int(cy - total_h / 2)
        if abs(cx - anchor_x) > 24 or abs(cy - anchor_y) > 24:
            draw.line((anchor_x, anchor_y, cx, y + badge.height / 2), fill="#B5C0CE", width=2)
        im.alpha_composite(badge, (x, y))
        draw = ImageDraw.Draw(im)
        label = MODEL_SHORT_LABELS[row["model"]]
        draw.multiline_text((cx, y + badge.height + 14), label, font=label_font, fill=INK, anchor="ma", spacing=4, align="center")
        boxes.append((row["model"], (x, y, x + badge.width, y + total_h)))

    for i, (model_a, box_a) in enumerate(boxes):
        for model_b, box_b in boxes[i + 1 :]:
            if not (box_a[2] + 6 <= box_b[0] or box_b[2] + 6 <= box_a[0] or box_a[3] + 6 <= box_b[1] or box_b[3] + 6 <= box_a[1]):
                raise RuntimeError(f"trigger scatter label overlap: {model_a} vs {model_b}")

    draw_text(draw, (left + plot_w / 2, top + plot_h + 88), "GT correct-to-wrong (%)", load_font(40, True), INK, anchor="ma")
    y_label = Image.new("RGBA", (620, 58), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 0), "NGT flip (%)", font=load_font(40, True), fill=INK)
    y_label = y_label.rotate(90, expand=True)
    im.alpha_composite(y_label, (92, top + 230))
    im.convert("RGB").save(path)


def figure_tone_opus(path, rows):
    width, height = 1780, 980
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Tone gradient + Opus special", "", width)
    panels = [
        ("GT", 130, 205, "GT correct-to-wrong", 0.35, [0, 0.1, 0.2, 0.3]),
        ("NGT", 950, 205, "NGT flip", 0.9, [0, 0.3, 0.6, 0.9]),
    ]
    for branch, x, y, title, max_rate, ticks in panels:
        w, h = 700, 590
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 24), title, FONT_PANEL, INK)
        px0, py0 = x + 108, y + 102
        pw, ph = w - 178, h - 210
        for tick in ticks:
            yy = py0 + ph - ph * tick / max_rate
            draw.line((px0, yy, px0 + pw, yy), fill="#E7EDF3", width=1)
            draw_text(draw, (px0 - 16, yy), f"{int(tick * 100)}", FONT_SMALL, MUTED, anchor="rm")
        specs = [("all", "All models", "#263A5E", 8, 14), ("opus", "Opus 4.5", "#D96C42", 8, 20)]
        for group, label, color, line_w, dot_r in specs:
            pts = []
            for i, tone in enumerate(TONES):
                value = row_lookup(rows, branch=branch, group=group, tone=tone)["rate"]
                xx = px0 + i * pw / 2
                yy = py0 + ph - ph * value / max_rate
                pts.append((xx, yy))
            draw.line(pts, fill=color, width=line_w)
            for xx, yy in pts:
                draw.ellipse((xx - dot_r, yy - dot_r, xx + dot_r, yy + dot_r), fill=color, outline="white", width=3)
        for i, tone in enumerate(TONES):
            xx = px0 + i * pw / 2
            draw_text(draw, (xx, py0 + ph + 36), TONE_LABELS[tone], FONT_AXIS_BOLD, INK, anchor="ma")
        legend_x, legend_y = x + w - 235, y + 52
        for i, (_, label, color, _, _) in enumerate(specs):
            yy = legend_y + i * 34
            draw.line((legend_x, yy, legend_x + 44, yy), fill=color, width=8)
            draw_text(draw, (legend_x + 56, yy), label, FONT_SMALL, color, anchor="lm")
        draw_text(draw, (px0 + pw / 2, y + h - 34), "Tone", FONT_AXIS_BOLD, INK, anchor="ma")
    im.save(path)


def figure_static_vs_adaptive(path, rows):
    width, height = 1500, 760
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Static vs. adaptive", "", width)
    panels = [("GT", 130, 190, "GT correct-to-wrong", 0.26), ("NGT", 800, 190, "NGT flip", 0.75)]
    for branch, x, y, title, max_rate in panels:
        w, h = 560, 410
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 26), title, FONT_PANEL, INK)
        axis_x, axis_y = x + 138, y + 118
        axis_w = w - 206
        bar_h, gap = 56, 86
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            xx = axis_x + axis_w * tick
            draw.line((xx, axis_y - 14, xx, axis_y + gap + bar_h + 15), fill="#E8EEF4", width=1)
            draw_text(draw, (xx, axis_y + gap + bar_h + 32), f"{int(tick * max_rate * 100)}", FONT_TINY, MUTED, anchor="ma")
        for i, (mode, label, color) in enumerate([("static", "Static", STATIC), ("adaptive", "Adaptive", ADAPTIVE)]):
            row = row_lookup(rows, branch=branch, mode=mode)
            yy = axis_y + i * gap
            draw_text(draw, (x + 28, yy + bar_h / 2), label, FONT_AXIS_BOLD, INK, anchor="lm")
            draw.rectangle((axis_x, yy, axis_x + axis_w, yy + bar_h), fill="#EEF2F6")
            fill = axis_w * row["rate"] / max_rate
            draw.rectangle((axis_x, yy, axis_x + fill, yy + bar_h), fill=color)
            draw_text(draw, (axis_x + fill + 10, yy + bar_h / 2), fmt_pct(row["rate"]), FONT_AXIS_BOLD, INK, anchor="lm")
        draw_text(draw, (axis_x + axis_w / 2, y + h - 32), "Rate (%)", FONT_SMALL, MUTED, anchor="ma")
    im.save(path)


def figure_temporal_pressure(path, rows):
    width, height = 1780, 930
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Temporal pressure", "", width)
    panels = [
        ("GT", 125, 205, "GT final correct-to-wrong", 0.32, [0, 0.1, 0.2, 0.3]),
        ("NGT", 940, 205, "NGT final flip", 0.75, [0, 0.25, 0.5, 0.75]),
    ]
    stages = ["single", "same_family", "heterogeneous"]
    mode_specs = [("static", "Static", STATIC), ("adaptive", "Adaptive", ADAPTIVE)]
    for branch, x, y, title, max_rate, ticks in panels:
        w, h = 720, 560
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 24), title, FONT_PANEL, INK)
        px0, py0 = x + 105, y + 102
        pw, ph = w - 175, h - 200
        for tick in ticks:
            yy = py0 + ph - ph * tick / max_rate
            draw.line((px0, yy, px0 + pw, yy), fill="#E7EDF3", width=1)
            draw_text(draw, (px0 - 16, yy), f"{int(tick * 100)}", FONT_SMALL, MUTED, anchor="rm")
        for mode, label, color in mode_specs:
            pts = []
            for i, stage in enumerate(stages):
                value = row_lookup(rows, branch=branch, mode=mode, stage=stage)["rate"]
                xx = px0 + i * pw / 2
                yy = py0 + ph - ph * value / max_rate
                pts.append((xx, yy))
            draw.line(pts, fill=color, width=7)
            for xx, yy in pts:
                draw.ellipse((xx - 12, yy - 12, xx + 12, yy + 12), fill=color)
        for i, stage in enumerate(stages):
            xx = px0 + i * pw / 2
            label = row_lookup(rows, branch=branch, mode="static", stage=stage)["stage_label"]
            draw_wrapped(draw, label, xx - 82, py0 + ph + 32, 164, FONT_SMALL, INK, anchor_center=True)
        legend_x, legend_y = x + w - 225, y + 52
        for i, (_, label, color) in enumerate(mode_specs):
            yy = legend_y + i * 34
            draw.line((legend_x, yy, legend_x + 44, yy), fill=color, width=7)
            draw_text(draw, (legend_x + 56, yy), label, FONT_SMALL, color, anchor="lm")
    im.save(path)


def figure_confidence_trajectory(path, rows):
    width, height = 1780, 900
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Confidence trajectory", "", width)
    panels = [
        ("GT", 130, 200, [("preserved", "Preserved", GT), ("departed", "Departed", ACCENT)]),
        ("NGT", 950, 200, [("held", "Held", GT), ("switched", "Switched", ACCENT)]),
    ]
    turns = [0, 1, 2, 3]
    for branch, x, y, specs in panels:
        w, h = 700, 540
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 24), "GT confidence" if branch == "GT" else "NGT confidence", FONT_PANEL, INK)
        px0, py0 = x + 105, y + 95
        pw, ph = w - 175, h - 190
        min_c, max_c = 2.8, 5.0
        for tick in [3, 4, 5]:
            yy = py0 + ph - ph * (tick - min_c) / (max_c - min_c)
            draw.line((px0, yy, px0 + pw, yy), fill="#E7EDF3", width=1)
            draw_text(draw, (px0 - 16, yy), f"{tick}", FONT_SMALL, MUTED, anchor="rm")
        for category, label, color in specs:
            pts = []
            for i, turn in enumerate(turns):
                value = row_lookup(rows, branch=branch, category=category, turn=turn)["mean_confidence"]
                xx = px0 + i * pw / 3
                yy = py0 + ph - ph * (value - min_c) / (max_c - min_c)
                pts.append((xx, yy))
            draw.line(pts, fill=color, width=7)
            for xx, yy in pts:
                draw.ellipse((xx - 11, yy - 11, xx + 11, yy + 11), fill=color)
        for i, turn in enumerate(turns):
            xx = px0 + i * pw / 3
            draw_text(draw, (xx, py0 + ph + 32), ["Initial", "T1", "T2", "T3"][turn], FONT_AXIS_BOLD, INK, anchor="ma")
        legend_x, legend_y = x + w - 225, y + 52
        for i, (_, label, color) in enumerate(specs):
            yy = legend_y + i * 34
            draw.line((legend_x, yy, legend_x + 44, yy), fill=color, width=7)
            draw_text(draw, (legend_x + 56, yy), label, FONT_SMALL, color, anchor="lm")
        draw_text(draw, (px0 + pw / 2, y + h - 32), "Turn", FONT_SMALL, MUTED, anchor="ma")
    im.save(path)


def clean_outputs(report_dir, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.png", "*.csv", "*.json", "README.md"]:
        for path in figure_dir.glob(pattern):
            path.unlink()
    for name in ["fig_trigger_single_rates.png", "fig_temporal_rates.png"]:
        path = report_dir / name
        if path.exists():
            path.unlink()


def main():
    args = parse_args()
    report_dir = args.report_dir or Path("Experimental/reports") / args.run_id
    figure_dir = report_dir / "paper_figure_candidates"
    if args.clean:
        clean_outputs(report_dir, figure_dir)
    else:
        figure_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.results_dir, args.run_id)
    tables = build_trigger_figure_tables(records)

    table_names = []
    for name, rows in tables.items():
        filename = f"{name}.csv"
        write_csv(figure_dir / filename, rows)
        table_names.append(filename)

    figures = [
        ("trigger_model_comparison.png", figure_model_comparison, tables["model_comparison"]),
        ("trigger_tone_gradient_opus.png", figure_tone_opus, tables["tone_gradient_opus"]),
        ("trigger_static_vs_adaptive.png", figure_static_vs_adaptive, tables["static_vs_adaptive"]),
        ("trigger_temporal_pressure.png", figure_temporal_pressure, tables["temporal_pressure"]),
        ("trigger_confidence_trajectory.png", figure_confidence_trajectory, tables["confidence_trajectory"]),
    ]
    written = []
    for filename, fn, rows in figures:
        path = figure_dir / filename
        fn(path, rows)
        written.append(filename)

    summary = {
        "run_id": args.run_id,
        "source": "official pass@1-clean trigger result files",
        "figure_dir": str(figure_dir.resolve()),
        "figures": written,
        "tables": table_names,
    }
    (figure_dir / "figure_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (figure_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Trigger Figure Candidates: {args.run_id}",
                "",
                "This directory is the single canonical location for trigger figure candidates.",
                "All figures are Python-generated PNG charts from the official pass@1-clean result files.",
                "",
                "- `trigger_model_comparison.png`: model-level GT answer change vs GT correct-to-wrong, with NGT flip encoded by marker size.",
                "- `trigger_tone_gradient_opus.png`: mild/moderate/strong tone gradients with Opus 4.5 highlighted.",
                "- `trigger_static_vs_adaptive.png`: static vs adaptive GT and NGT rates.",
                "- `trigger_temporal_pressure.png`: single, same-family escalation, and heterogeneous temporal pressure.",
                "- `trigger_confidence_trajectory.png`: confidence over turns for preserved/departed GT and held/switched NGT trajectories.",
                "",
                "The CSV files in this directory contain the exact plotted aggregates.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
