import argparse
import csv
import gzip
import json
import math
import shutil
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
MODEL_COLORS = {
    "openai/gpt-5.4": "#5E4ACF",
    "openai/gpt-5.4-mini": "#8B6FE8",
    "openai/gpt-5.4-nano": "#B19AF4",
    "anthropic/claude-opus-4.5": "#C85F39",
    "anthropic/claude-sonnet-4.5": "#E08A57",
    "anthropic/claude-haiku-4.5": "#F0B37E",
    "google/gemini-3.1-flash-lite-preview": "#2F80ED",
    "mistralai/mistral-medium-3.1": "#E84D2A",
    "cohere/command-r-08-2024": "#2D6A5A",
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
    parser.add_argument("--publish-dir", type=Path, default=None, help="Optional manuscript image directory to update.")
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
    draw_text(draw, (width / 2, 38), title, FONT_TITLE, anchor="ma")
    if subtitle:
        draw_text(draw, (width / 2, 94), subtitle, FONT_SUBTITLE, MUTED, anchor="ma")
        line_y = 145
    else:
        line_y = 122
    draw.line((65, line_y, width - 65, line_y), fill=LIGHT_GRID, width=3)


def save_tight(im, path, padding=16):
    if Path(path).suffix.lower() != ".png":
        im.save(path)
        return
    rgb = im.convert("RGB")
    pix = rgb.load()
    width, height = rgb.size

    def nonwhite_pixel(x, y):
        return pix[x, y] != (255, 255, 255)

    top = next((y for y in range(height) if any(nonwhite_pixel(x, y) for x in range(width))), 0)
    bottom = next((y for y in range(height - 1, -1, -1) if any(nonwhite_pixel(x, y) for x in range(width))), height - 1)
    left = next((x for x in range(width) if any(nonwhite_pixel(x, y) for y in range(height))), 0)
    right = next((x for x in range(width - 1, -1, -1) if any(nonwhite_pixel(x, y) for y in range(height))), width - 1)
    crop = (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding + 1),
        min(height, bottom + padding + 1),
    )
    im.crop(crop).save(path)


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
        for model_id in MODELS:
            for tone_name in TONES:
                rate, denom, events = aggregate_rate(
                    single,
                    branch,
                    lambda r, branch=branch, model_id=model_id, tone_name=tone_name: r["_branch"] == branch
                    and r["model"] == model_id
                    and cialdini_single(r)
                    and r.get("tone") == tone_name,
                )
                tone.append(
                    {
                        "branch": branch,
                        "model": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "model_short_label": MODEL_SHORT_LABELS[model_id].replace("\n", " "),
                        "tone": tone_name,
                        "rate": rate,
                        "denom": denom,
                        "events": events,
                    }
                )

    static_adaptive = []
    for branch in ["GT", "NGT"]:
        for model_id in MODELS:
            for mode in ["static", "adaptive"]:
                rate, denom, events = aggregate_rate(
                    single,
                    branch,
                    lambda r, branch=branch, model_id=model_id, mode=mode: r["_branch"] == branch
                    and r["model"] == model_id
                    and r["_mode"] == mode
                    and cialdini_single(r),
                )
                static_adaptive.append(
                    {
                        "branch": branch,
                        "model": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "model_short_label": MODEL_SHORT_LABELS[model_id].replace("\n", " "),
                        "mode": mode,
                        "rate": rate,
                        "denom": denom,
                        "events": events,
                    }
                )

    temporal_pressure = []
    for branch in ["GT", "NGT"]:
        for model_id in MODELS:
            for mode in ["static", "adaptive"]:
                single_rate, single_denom, single_events = aggregate_rate(
                    single,
                    branch,
                    lambda r, branch=branch, model_id=model_id, mode=mode: r["_branch"] == branch
                    and r["model"] == model_id
                    and r["_mode"] == mode
                    and cialdini_single(r),
                )
                temporal_pressure.append(
                    {
                        "branch": branch,
                        "model": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "model_short_label": MODEL_SHORT_LABELS[model_id].replace("\n", " "),
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
                        lambda r, branch=branch, model_id=model_id, mode=mode, stage=stage: r["_branch"] == branch
                        and r["model"] == model_id
                        and r["_mode"] == mode
                        and temporal_stage(r) == stage,
                        temporal=True,
                    )
                    temporal_pressure.append(
                        {
                            "branch": branch,
                            "model": model_id,
                            "model_label": MODEL_LABELS[model_id],
                            "model_short_label": MODEL_SHORT_LABELS[model_id].replace("\n", " "),
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
        for model_id in MODELS:
            for mode in ["static", "adaptive"]:
                for category in (["preserved", "departed"] if branch == "GT" else ["held", "switched"]):
                    by_turn = {turn: [] for turn in range(4)}
                    for record in temporal:
                        if (
                            record["_branch"] != branch
                            or record["model"] != model_id
                            or record["_mode"] != mode
                            or not denom_ok(record, branch)
                        ):
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
                                "model": model_id,
                                "model_label": MODEL_LABELS[model_id],
                                "model_short_label": MODEL_SHORT_LABELS[model_id].replace("\n", " "),
                                "mode": mode,
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
    save_tight(im, path)


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
    save_tight(im, path)


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
    save_tight(im, path)


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
    save_tight(im, path)


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
    save_tight(im, path)


def figure_model_comparison(path, rows):
    width, height = 1780, 980
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Model comparison", "", width)

    left, top = 185, 170
    plot_w, plot_h = 1280, 640
    min_x, max_x = 0.10, 0.55
    min_y, max_y = 0.00, 0.36

    def x_pos(value):
        return left + plot_w * (value - min_x) / (max_x - min_x)

    def y_pos(value):
        return top + plot_h - plot_h * (value - min_y) / (max_y - min_y)

    draw.rectangle((left, top, left + plot_w, top + plot_h), fill="#FBFCFE", outline="#18212B", width=2)
    for tick in [0.10, 0.20, 0.30, 0.40, 0.50]:
        xx = x_pos(tick)
        draw.line((xx, top, xx, top + plot_h), fill="#DDE5EE", width=1)
        draw_text(draw, (xx, top + plot_h + 20), f"{int(tick * 100)}%", FONT_SMALL, MUTED, anchor="ma")
    for tick in [0.00, 0.10, 0.20, 0.30]:
        yy = y_pos(tick)
        draw.line((left, yy, left + plot_w, yy), fill="#DDE5EE", width=1)
        draw_text(draw, (left - 16, yy), f"{int(tick * 100)}%", FONT_SMALL, MUTED, anchor="rm")
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#18212B", width=3)
    draw.line((left, top, left, top + plot_h), fill="#18212B", width=3)

    # Correct-to-wrong turns are a stricter subset of answer movement.
    diag_start, diag_end = max(min_x, min_y), min(max_x, max_y)
    draw.line((x_pos(diag_start), y_pos(diag_start), x_pos(diag_end), y_pos(diag_end)), fill="#AEB8C6", width=3)
    draw_text(draw, (x_pos(0.36) - 18, y_pos(0.36) + 22), "y = x", FONT_TINY, MUTED, anchor="rm")

    badge_offsets = {
        "openai/gpt-5.4": (-66, -34),
        "openai/gpt-5.4-mini": (-94, -48),
        "anthropic/claude-opus-4.5": (2, 72),
        "mistralai/mistral-medium-3.1": (56, 34),
        "cohere/command-r-08-2024": (-98, 38),
    }
    label_specs = {
        "openai/gpt-5.4": (0, -56, "ma"),
        "openai/gpt-5.4-mini": (0, -58, "ma"),
        "openai/gpt-5.4-nano": (44, -16, "lm"),
        "anthropic/claude-opus-4.5": (-46, 12, "rm"),
        "anthropic/claude-sonnet-4.5": (54, -10, "lm"),
        "anthropic/claude-haiku-4.5": (-70, -26, "rm"),
        "google/gemini-3.1-flash-lite-preview": (54, 18, "lm"),
        "mistralai/mistral-medium-3.1": (0, 54, "ma"),
        "cohere/command-r-08-2024": (0, 54, "ma"),
    }

    badge_centers = {}
    for row in sorted(rows, key=lambda item: item["ngt_rate"], reverse=True):
        model = row["model"]
        x = x_pos(row["gt_change_rate"])
        y = y_pos(row["gt_rate"])
        dx, dy = badge_offsets.get(model, (0, 0))
        bx, by = x + dx, y + dy
        if dx or dy:
            draw.line((x, y, bx, by), fill="#AEB8C6", width=2)
        badge = make_badge(model)
        badge.thumbnail((78, 78), RESAMPLE_LANCZOS)
        im.paste(badge, (int(bx - badge.width / 2), int(by - badge.height / 2)), badge)
        badge_centers[model] = (bx, by)
        draw = ImageDraw.Draw(im)

    for row in rows:
        model = row["model"]
        x, y = badge_centers[model]
        dx, dy, anchor = label_specs.get(model, (24, 20, "lm"))
        draw.multiline_text(
            (x + dx, y + dy),
            MODEL_SHORT_LABELS[model],
            font=FONT_SMALL,
            fill=INK,
            anchor=anchor,
            spacing=2,
            align="center",
        )

    draw_text(draw, (left + plot_w / 2, top + plot_h + 54), "Answer change after 1 trigger (%)", FONT_AXIS_BOLD, INK, anchor="ma")
    y_label = Image.new("RGBA", (520, 40), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 0), "Right-to-wrong after 1 trigger (%)", font=FONT_AXIS_BOLD, fill=INK)
    y_label = y_label.rotate(90, expand=True)
    im.paste(y_label, (92, top + 130), y_label)
    save_tight(im, path)


def blend_color(hex_color, t, low=(248, 250, 252)):
    hi = ImageColor.getrgb(hex_color)
    t = max(0.0, min(1.0, t))
    return tuple(int(low[i] + (hi[i] - low[i]) * t) for i in range(3))


def branch_rate_scale(branch):
    return 0.55 if branch == "GT" else 0.90


def model_display(model):
    return MODEL_SHORT_LABELS[model].replace("\n", " ")


def draw_rate_cell(draw, rect, value, max_rate, color, font=FONT_SMALL):
    t = 0 if max_rate <= 0 else min(1.0, value / max_rate)
    fill = blend_color(color, 0.18 + 0.72 * t)
    draw.rounded_rectangle(rect, radius=8, fill=fill)
    text_fill = "white" if t > 0.63 else INK
    draw_text(draw, ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2), fmt_pct(value, 0), font, text_fill, anchor="mm")


def fmt_score_delta(value, digits=1):
    if abs(value) < 0.5 * (10 ** -digits):
        value = 0.0
    return f"{value:+.{digits}f}"


def draw_delta_cell(draw, rect, value, max_abs, font=FONT_SMALL, confidence=False):
    if abs(value) < 0.002:
        fill = "#F2F4F7"
        t = 0.0
    else:
        if confidence:
            color = GT if value > 0 else ACCENT
        else:
            color = ACCENT if value > 0 else "#3D7EA6"
        t = min(1.0, abs(value) / max_abs)
        fill = blend_color(color, 0.16 + 0.78 * t)
    draw.rounded_rectangle(rect, radius=8, fill=fill)
    text_fill = "white" if t > 0.58 else INK
    label = fmt_score_delta(value) if confidence else f"{pct(value):+.1f}"
    draw_text(draw, ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2), label, font, text_fill, anchor="mm")


def draw_confidence_cell(draw, rect, value, font=FONT_SMALL):
    t = max(0.0, min(1.0, (value - 1.0) / 4.0))
    fill = blend_color("#6F55D9", 0.12 + 0.68 * t)
    draw.rounded_rectangle(rect, radius=8, fill=fill)
    text_fill = "white" if t > 0.70 else INK
    draw_text(draw, ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2), f"{value:.1f}", font, text_fill, anchor="mm")


def figure_tone_opus(path, rows):
    width, height = 1640, 1036
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Tone gradient by model", "", width)
    panels = [
        ("GT", 110, 180, "GT correct-to-wrong", 0.55, [0, 0.15, 0.30, 0.45]),
        ("NGT", 870, 180, "NGT flip", 0.9, [0, 0.3, 0.6, 0.9]),
    ]
    for branch, x, y, title, max_rate, ticks in panels:
        w, h = 650, 610
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 24), title, FONT_PANEL, INK)
        px0, py0 = x + 108, y + 112
        pw, ph = w - 178, h - 225
        for tick in ticks:
            yy = py0 + ph - ph * tick / max_rate
            draw.line((px0, yy, px0 + pw, yy), fill="#E7EDF3", width=1)
            draw_text(draw, (px0 - 16, yy), f"{int(tick * 100)}", FONT_SMALL, MUTED, anchor="rm")
        for pass_is_claude in [False, True]:
            for model in MODELS:
                is_claude = model.startswith("anthropic/")
                if is_claude != pass_is_claude:
                    continue
                color = MODEL_COLORS[model] if is_claude else "#B9C3D0"
                line_w = 9 if model == "anthropic/claude-opus-4.5" else 6 if is_claude else 3
                pts = []
                for i, tone in enumerate(TONES):
                    value = row_lookup(rows, branch=branch, model=model, tone=tone)["rate"]
                    xx = px0 + i * pw / 2
                    yy = py0 + ph - ph * value / max_rate
                    pts.append((xx, yy))
                draw.line(pts, fill=color, width=line_w)
        if branch == "NGT":
            draw_text(
                draw,
                (x + w - 28, y + 78),
                "Claude-family reversal",
                FONT_SMALL,
                MODEL_COLORS["anthropic/claude-opus-4.5"],
                anchor="ra",
            )
        for i, tone in enumerate(TONES):
            xx = px0 + i * pw / 2
            draw_text(draw, (xx, py0 + ph + 36), TONE_LABELS[tone], FONT_AXIS_BOLD, INK, anchor="ma")
        draw_text(draw, (px0 + pw / 2, y + h - 34), "Tone", FONT_AXIS_BOLD, INK, anchor="ma")
    legend_x, legend_y = 170, 823
    legend_items = [
        ("anthropic/claude-opus-4.5", "Opus 4.5", 9),
        ("anthropic/claude-sonnet-4.5", "Sonnet 4.5", 6),
        ("anthropic/claude-haiku-4.5", "Haiku 4.5", 6),
        (None, "Other models", 3),
    ]
    for i, (model, label, width_line) in enumerate(legend_items):
        x = legend_x + i * 340
        color = MODEL_COLORS[model] if model else "#B9C3D0"
        draw.line((x, legend_y, x + 58, legend_y), fill=color, width=width_line)
        draw_text(draw, (x + 76, legend_y), label, FONT_SMALL, INK, anchor="lm")
    save_tight(im, path)


def figure_model_tone(path, model_rows, tone_rows):
    width, height = 3900, 1450
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    panel_title = load_font(50, True)
    axis_font = load_font(30)
    axis_bold = load_font(34, True)
    label_font = load_font(30, True)
    small_font = load_font(27)
    tiny_font = load_font(23)

    # Left panel: model-level trigger susceptibility.
    panel_x, panel_y, panel_w, panel_h = 80, 70, 1715, 1290
    draw.rounded_rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        radius=22,
        fill="#F8FAFC",
        outline="#D9E0E8",
        width=2,
    )
    draw_text(draw, (panel_x + panel_w / 2, panel_y + 44), "Model comparison", panel_title, INK, anchor="ma")

    left, top = panel_x + 170, panel_y + 150
    plot_w, plot_h = 1425, 900
    min_x, max_x = 0.10, 0.55
    min_y, max_y = 0.00, 0.36

    def x_pos(value):
        return left + plot_w * (value - min_x) / (max_x - min_x)

    def y_pos(value):
        return top + plot_h - plot_h * (value - min_y) / (max_y - min_y)

    draw.rectangle((left, top, left + plot_w, top + plot_h), fill="#FBFCFE", outline="#18212B", width=3)
    for tick in [0.10, 0.20, 0.30, 0.40, 0.50]:
        xx = x_pos(tick)
        draw.line((xx, top, xx, top + plot_h), fill="#DDE5EE", width=2)
        draw_text(draw, (xx, top + plot_h + 27), f"{int(tick * 100)}%", axis_font, MUTED, anchor="ma")
    for tick in [0.00, 0.10, 0.20, 0.30]:
        yy = y_pos(tick)
        draw.line((left, yy, left + plot_w, yy), fill="#DDE5EE", width=2)
        draw_text(draw, (left - 20, yy), f"{int(tick * 100)}%", axis_font, MUTED, anchor="rm")
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#18212B", width=4)
    draw.line((left, top, left, top + plot_h), fill="#18212B", width=4)

    diag_start, diag_end = max(min_x, min_y), min(max_x, max_y)
    draw.line((x_pos(diag_start), y_pos(diag_start), x_pos(diag_end), y_pos(diag_end)), fill="#AEB8C6", width=4)
    draw_text(draw, (x_pos(0.36) - 12, y_pos(0.36) + 30), "y = x", tiny_font, MUTED, anchor="rm")

    badge_offsets = {
        "openai/gpt-5.4": (-78, -44),
        "openai/gpt-5.4-mini": (-110, -58),
        "anthropic/claude-opus-4.5": (2, 88),
        "mistralai/mistral-medium-3.1": (66, 42),
        "cohere/command-r-08-2024": (-108, 46),
    }
    label_specs = {
        "openai/gpt-5.4": (0, -70, "ma"),
        "openai/gpt-5.4-mini": (0, -72, "ma"),
        "openai/gpt-5.4-nano": (58, -18, "lm"),
        "anthropic/claude-opus-4.5": (-58, 16, "rm"),
        "anthropic/claude-sonnet-4.5": (66, -12, "lm"),
        "anthropic/claude-haiku-4.5": (-84, -28, "rm"),
        "google/gemini-3.1-flash-lite-preview": (70, 20, "lm"),
        "mistralai/mistral-medium-3.1": (0, 66, "ma"),
        "cohere/command-r-08-2024": (0, 66, "ma"),
    }

    badge_centers = {}
    for row in sorted(model_rows, key=lambda item: item["ngt_rate"], reverse=True):
        model = row["model"]
        x = x_pos(row["gt_change_rate"])
        y = y_pos(row["gt_rate"])
        dx, dy = badge_offsets.get(model, (0, 0))
        bx, by = x + dx, y + dy
        if dx or dy:
            draw.line((x, y, bx, by), fill="#AEB8C6", width=2)
        badge = make_badge(model)
        badge.thumbnail((112, 112), RESAMPLE_LANCZOS)
        im.paste(badge, (int(bx - badge.width / 2), int(by - badge.height / 2)), badge)
        badge_centers[model] = (bx, by)
        draw = ImageDraw.Draw(im)

    for row in model_rows:
        model = row["model"]
        x, y = badge_centers[model]
        dx, dy, anchor = label_specs.get(model, (28, 24, "lm"))
        draw.multiline_text(
            (x + dx, y + dy),
            MODEL_SHORT_LABELS[model],
            font=label_font,
            fill=INK,
            anchor=anchor,
            spacing=2,
            align="center",
        )

    draw_text(
        draw,
        (left + plot_w / 2, top + plot_h + 86),
        "Answer change after 1 trigger (%)",
        axis_bold,
        INK,
        anchor="ma",
    )
    y_label = Image.new("RGBA", (720, 50), (255, 255, 255, 0))
    yd = ImageDraw.Draw(y_label)
    yd.text((0, 0), "Right-to-wrong after 1 trigger (%)", font=axis_bold, fill=INK)
    y_label = y_label.rotate(90, expand=True)
    im.paste(y_label, (int(left - 128), int(top + plot_h / 2 - y_label.height / 2)), y_label)

    # Right panel: tone gradient, stacked to avoid the quarter-page mini-panel problem.
    right_x, right_y, right_w, right_h = 1840, 70, 1980, 1290
    draw.rounded_rectangle(
        (right_x, right_y, right_x + right_w, right_y + right_h),
        radius=22,
        fill="#F8FAFC",
        outline="#D9E0E8",
        width=2,
    )
    draw_text(draw, (right_x + right_w / 2, right_y + 44), "Tone gradient", panel_title, INK, anchor="ma")

    def tone_panel(branch, x, y, title, max_rate, ticks, show_x_label=False):
        w, h = right_w - 150, 505
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 30, y + 24), title, axis_bold, INK)
        px0, py0 = x + 150, y + 92
        pw, ph = w - 245, h - 185
        for tick in ticks:
            yy = py0 + ph - ph * tick / max_rate
            draw.line((px0, yy, px0 + pw, yy), fill="#E7EDF3", width=2)
            draw_text(draw, (px0 - 18, yy), f"{int(tick * 100)}", axis_font, MUTED, anchor="rm")
        for pass_is_claude in [False, True]:
            for model in MODELS:
                is_claude = model.startswith("anthropic/")
                if is_claude != pass_is_claude:
                    continue
                color = MODEL_COLORS[model] if is_claude else "#B9C3D0"
                line_w = 11 if model == "anthropic/claude-opus-4.5" else 8 if is_claude else 4
                dot_r = 11 if is_claude else 7
                pts = []
                for i, tone in enumerate(TONES):
                    value = row_lookup(tone_rows, branch=branch, model=model, tone=tone)["rate"]
                    xx = px0 + i * pw / 2
                    yy = py0 + ph - ph * value / max_rate
                    pts.append((xx, yy))
                draw.line(pts, fill=color, width=line_w)
                if is_claude:
                    for xx, yy in pts:
                        draw.ellipse((xx - dot_r, yy - dot_r, xx + dot_r, yy + dot_r), fill=color, outline="white", width=3)
        if branch == "NGT":
            draw_text(
                draw,
                (x + w - 30, y + 54),
                "Claude-family reversal",
                small_font,
                MODEL_COLORS["anthropic/claude-opus-4.5"],
                anchor="ra",
            )
        for i, tone in enumerate(TONES):
            xx = px0 + i * pw / 2
            draw_text(draw, (xx, py0 + ph + 33), TONE_LABELS[tone], axis_bold, INK, anchor="ma")
        if show_x_label:
            draw_text(draw, (px0 + pw / 2, y + h - 25), "Tone", axis_bold, INK, anchor="ma")

    tone_x = right_x + 75
    tone_panel("GT", tone_x, right_y + 122, "GT correct-to-wrong", 0.55, [0, 0.15, 0.30, 0.45])
    tone_panel("NGT", tone_x, right_y + 672, "NGT flip", 0.9, [0, 0.3, 0.6, 0.9], show_x_label=True)

    legend_items = [
        ("anthropic/claude-opus-4.5", "Opus 4.5", 11),
        ("anthropic/claude-sonnet-4.5", "Sonnet 4.5", 8),
        ("anthropic/claude-haiku-4.5", "Haiku 4.5", 8),
        (None, "Other models", 4),
    ]
    legend_total_w = 1420
    legend_x = right_x + (right_w - legend_total_w) / 2
    legend_y = right_y + right_h - 44
    for i, (model, label, width_line) in enumerate(legend_items):
        x = legend_x + i * 360
        color = MODEL_COLORS[model] if model else "#B9C3D0"
        draw.line((x, legend_y, x + 78, legend_y), fill=color, width=width_line)
        draw_text(draw, (x + 98, legend_y), label, small_font, INK, anchor="lm")

    save_tight(im, path, padding=8)


def figure_trigger_dynamics(path, tone_rows, temporal_rows, confidence_rows):
    width, height = 3900, 1320
    im = Image.new("RGBA", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im, "RGBA")
    panel_font = load_font(46, True)
    axis_font = load_font(25)
    axis_bold = load_font(27, True)
    label_font = load_font(24, True)

    margin = 80
    gap = 80
    panel_y = 70
    panel_h = 1145
    panel_w = (width - 2 * margin - gap) / 2
    panel_specs = [
        ("GT", margin, "GT Truth Departure", 0.0, 0.50, [0.0, 0.10, 0.20, 0.30, 0.40, 0.50], STATIC),
        ("NGT", margin + panel_w + gap, "NGT Flip-Flop", 0.25, 0.95, [0.25, 0.40, 0.55, 0.70, 0.85, 0.95], ADAPTIVE),
    ]

    def color_alpha(hex_color, alpha):
        return (*ImageColor.getrgb(hex_color), alpha)

    offsets = {
        "GT": {
            "openai/gpt-5.4": (-115, -72),
            "openai/gpt-5.4-mini": (112, -70),
            "openai/gpt-5.4-nano": (-18, 78),
            "anthropic/claude-opus-4.5": (-104, 54),
            "anthropic/claude-sonnet-4.5": (72, -82),
            "anthropic/claude-haiku-4.5": (64, 52),
            "google/gemini-3.1-flash-lite-preview": (-82, -8),
            "mistralai/mistral-medium-3.1": (96, 54),
            "cohere/command-r-08-2024": (-65, -118),
        },
        "NGT": {
            "openai/gpt-5.4": (150, 20),
            "openai/gpt-5.4-mini": (-150, -60),
            "openai/gpt-5.4-nano": (0, 78),
            "anthropic/claude-opus-4.5": (25, -96),
            "anthropic/claude-sonnet-4.5": (-20, -160),
            "anthropic/claude-haiku-4.5": (115, -85),
            "google/gemini-3.1-flash-lite-preview": (135, 92),
            "mistralai/mistral-medium-3.1": (45, -125),
            "cohere/command-r-08-2024": (-88, 36),
        },
    }

    def draw_dashed_line(p0, p1, fill, width=4, dash=28, gap_len=16):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        pos = 0
        while pos < length:
            end = min(pos + dash, length)
            draw.line((x0 + ux * pos, y0 + uy * pos, x0 + ux * end, y0 + uy * end), fill=fill, width=width)
            pos += dash + gap_len

    def scatter_panel(branch, panel_x, title, v_min, v_max, ticks, accent):
        draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=22, fill="#F8FAFC", outline="#D9E0E8", width=2)
        draw_text(draw, (panel_x + panel_w / 2, panel_y + 36), title, panel_font, INK, anchor="ma")
        plot_x = panel_x + 205
        plot_y = panel_y + 205
        plot_w = panel_w - 315
        plot_h = 760

        def x_pos(value):
            return plot_x + plot_w * (value - v_min) / (v_max - v_min)

        def y_pos(value):
            return plot_y + plot_h - plot_h * (value - v_min) / (v_max - v_min)

        for tick in ticks:
            xx = x_pos(tick)
            yy = y_pos(tick)
            draw.line((xx, plot_y, xx, plot_y + plot_h), fill="#E4EAF2", width=1)
            draw.line((plot_x, yy, plot_x + plot_w, yy), fill="#E4EAF2", width=1)
            draw_text(draw, (xx, plot_y + plot_h + 22), f"{int(round(tick * 100))}", axis_font, MUTED, anchor="ma")
            draw_text(draw, (plot_x - 18, yy), f"{int(round(tick * 100))}", axis_font, MUTED, anchor="rm")
        draw.line((plot_x, plot_y, plot_x, plot_y + plot_h), fill="#9AA7B7", width=3)
        draw.line((plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h), fill="#9AA7B7", width=3)
        draw_dashed_line((plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y), "#AAB5C3", width=4)

        for model in MODELS:
            single = row_lookup(temporal_rows, branch=branch, model=model, mode="adaptive", stage="single")["rate"]
            mixed = row_lookup(temporal_rows, branch=branch, model=model, mode="adaptive", stage="heterogeneous")["rate"]
            px = x_pos(single)
            py = y_pos(mixed)
            dx, dy = offsets[branch].get(model, (0, 0))
            cx, cy = px + dx, py + dy
            if abs(dx) > 8 or abs(dy) > 8:
                draw.line((px, py, cx, cy), fill="#AEB8C6", width=2)
            badge = make_badge(model)
            badge.thumbnail((82, 82), RESAMPLE_LANCZOS)
            im.alpha_composite(badge, (int(cx - badge.width / 2), int(cy - badge.height / 2)))
            draw.multiline_text(
                (cx, cy + badge.height / 2 + 8),
                MODEL_SHORT_LABELS[model],
                font=label_font,
                fill=INK,
                anchor="ma",
                align="center",
                spacing=2,
            )

        draw_text(draw, (plot_x + plot_w / 2, panel_y + panel_h - 62), "Single-follow-up movement (%)", axis_bold, INK, anchor="ma")
        y_label = Image.new("RGBA", (560, 44), (255, 255, 255, 0))
        yd = ImageDraw.Draw(y_label)
        yd.text((0, 0), "Mixed three-turn movement (%)", font=axis_bold, fill=INK)
        y_label = y_label.rotate(90, expand=True)
        im.alpha_composite(y_label, (int(panel_x + 52), int(plot_y + 125)))

    for spec in panel_specs:
        scatter_panel(*spec)

    save_tight(im.convert("RGB"), path)


def figure_tone_temporal(path, tone_rows, temporal_rows):
    width, height = 3900, 1500
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_text(draw, (1000, 58), "Tone Gradient by Model", load_font(64, True), INK, anchor="ma")
    draw_text(draw, (2930, 58), "Temporal Pressure by Model", load_font(64, True), INK, anchor="ma")
    draw.line((70, 140, width - 70, 140), fill=LIGHT_GRID, width=4)

    def tone_panel(branch, x, y, title, max_rate, ticks):
        w, h = 1750, 470
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 28, y + 22), title, FONT_PANEL, INK)
        px0, py0 = x + 118, y + 84
        pw, ph = w - 205, h - 160
        for tick in ticks:
            yy = py0 + ph - ph * tick / max_rate
            draw.line((px0, yy, px0 + pw, yy), fill="#E7EDF3", width=1)
            draw_text(draw, (px0 - 18, yy), f"{int(tick * 100)}", FONT_SMALL, MUTED, anchor="rm")
        for model in MODELS:
            color = MODEL_COLORS[model]
            line_w = 8 if model == "anthropic/claude-opus-4.5" else 5
            dot_r = 12 if model == "anthropic/claude-opus-4.5" else 8
            pts = []
            for i, tone in enumerate(TONES):
                value = row_lookup(tone_rows, branch=branch, model=model, tone=tone)["rate"]
                xx = px0 + i * pw / 2
                yy = py0 + ph - ph * value / max_rate
                pts.append((xx, yy))
            draw.line(pts, fill=color, width=line_w)
            for xx, yy in pts:
                draw.ellipse((xx - dot_r, yy - dot_r, xx + dot_r, yy + dot_r), fill=color, outline="white", width=3)
        for i, tone in enumerate(TONES):
            xx = px0 + i * pw / 2
            draw_text(draw, (xx, py0 + ph + 34), TONE_LABELS[tone], FONT_AXIS_BOLD, INK, anchor="ma")

    tone_panel("GT", 120, 210, "GT correct-to-wrong", 0.55, [0, 0.15, 0.30, 0.45])
    tone_panel("NGT", 120, 725, "NGT flip", 0.90, [0, 0.30, 0.60, 0.90])
    legend_x, legend_y = 150, 1260
    col_w, row_h = 575, 42
    for i, model in enumerate(MODELS):
        col, row = i % 3, i // 3
        x = legend_x + col * col_w
        y = legend_y + row * row_h
        color = MODEL_COLORS[model]
        width_line = 8 if model == "anthropic/claude-opus-4.5" else 5
        draw.line((x, y, x + 58, y), fill=color, width=width_line)
        draw.ellipse((x + 27, y - 7, x + 41, y + 7), fill=color, outline="white", width=2)
        draw_text(draw, (x + 76, y), model_display(model), FONT_SMALL, INK, anchor="lm")

    def temporal_panel(branch, x, y, title, max_rate, color):
        w, h = 815, 1080
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 24), title, FONT_PANEL, INK)
        draw_text(draw, (x + w - 26, y + 36), "Adaptive", FONT_SMALL, MUTED, anchor="rm")
        label_x = x + 26
        cell_x = x + 260
        top = y + 128
        row_h = 106
        cell_w = 145
        cell_h = 64
        gap = 12
        for ci, stage in enumerate(["single", "same_family", "heterogeneous"]):
            label = row_lookup(temporal_rows, branch=branch, model=MODELS[0], mode="adaptive", stage=stage)["stage_label"]
            draw_wrapped(draw, label, cell_x + ci * (cell_w + gap) - 8, top - 62, cell_w + 16, FONT_TINY, INK, anchor_center=True)
        for ri, model in enumerate(MODELS):
            yy = top + ri * row_h
            draw_text(draw, (label_x, yy + cell_h / 2), model_display(model), FONT_SMALL, INK, anchor="lm")
            for ci, stage in enumerate(["single", "same_family", "heterogeneous"]):
                value = row_lookup(temporal_rows, branch=branch, model=model, mode="adaptive", stage=stage)["rate"]
                xx = cell_x + ci * (cell_w + gap)
                draw_rate_cell(draw, (xx, yy, xx + cell_w, yy + cell_h), value, max_rate, color, FONT_AXIS_BOLD)

    temporal_panel("GT", 2010, 210, "GT final correct-to-wrong", 0.55, STATIC)
    temporal_panel("NGT", 2910, 210, "NGT final flip", 0.90, ADAPTIVE)
    save_tight(im, path)


def figure_static_vs_adaptive(path, rows):
    width, height = 1780, 1000
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Static vs. adaptive by model", "", width)
    panels = [("GT", 110, 170, "GT correct-to-wrong", 0.55, STATIC), ("NGT", 935, 170, "NGT flip", 0.90, ADAPTIVE)]

    def pooled_rate(branch, mode):
        subset = [row for row in rows if row["branch"] == branch and row["mode"] == mode]
        denom = sum(row["denom"] for row in subset)
        events = sum(row["events"] for row in subset)
        return events / denom if denom else 0.0

    for branch, x, y, title, max_rate, _ in panels:
        w, h = 735, 720
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 26), title, FONT_PANEL, INK)
        label_x = x + 26
        cell_x = x + 285
        top = y + 110
        row_h = 56
        cell_w = 148
        gap = 10
        for ci, mode in enumerate(["static", "adaptive"]):
            draw_text(draw, (cell_x + ci * (cell_w + gap) + cell_w / 2, top - 34), mode.capitalize(), FONT_AXIS_BOLD, INK, anchor="ma")
        delta_x = cell_x + 2 * (cell_w + gap) + 8
        draw_text(draw, (delta_x + 56, top - 34), "Delta pp", FONT_AXIS_BOLD, INK, anchor="ma")
        display_rows = [{"model": model, "label": model_display(model), "aggregate": False} for model in MODELS]
        display_rows.append({"model": None, "label": "All models", "aggregate": True})
        max_delta = 0.08 if branch == "GT" else 0.22
        for ri, item in enumerate(display_rows):
            model = item["model"]
            yy = top + ri * row_h
            if item["aggregate"]:
                draw.line((label_x, yy - 8, x + w - 26, yy - 8), fill="#C9D2DE", width=2)
                font = FONT_AXIS_BOLD
                static = pooled_rate(branch, "static")
                adaptive = pooled_rate(branch, "adaptive")
            else:
                font = FONT_SMALL
                static = row_lookup(rows, branch=branch, model=model, mode="static")["rate"]
                adaptive = row_lookup(rows, branch=branch, model=model, mode="adaptive")["rate"]
            draw_text(draw, (label_x, yy + 23), item["label"], font, INK, anchor="lm")
            for ci, (mode, value) in enumerate([("static", static), ("adaptive", adaptive)]):
                xx = cell_x + ci * (cell_w + gap)
                color = STATIC if mode == "static" else ADAPTIVE
                draw_rate_cell(draw, (xx, yy, xx + cell_w, yy + 46), value, max_rate, color, FONT_AXIS_BOLD)
            delta = adaptive - static
            draw_delta_cell(draw, (delta_x, yy, delta_x + 112, yy + 46), delta, max_delta, FONT_AXIS_BOLD)
    save_tight(im, path)


def figure_temporal_pressure(path, rows):
    width, height = 1780, 1000
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Temporal pressure by model", "", width)
    panels = [
        ("GT", 105, 170, "GT final correct-to-wrong", 0.55, STATIC, 0.14),
        ("NGT", 920, 170, "NGT final flip", 0.90, ADAPTIVE, 0.40),
    ]
    stages = ["single", "same_family", "heterogeneous"]
    for branch, x, y, title, max_rate, color, max_delta in panels:
        w, h = 755, 720
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 24), title, FONT_PANEL, INK)
        draw_text(draw, (x + w - 26, y + 36), "Adaptive", FONT_SMALL, MUTED, anchor="rm")
        label_x = x + 26
        cell_x = x + 250
        top = y + 110
        row_h = 60
        cell_w = 108
        gap = 10
        for ci, stage in enumerate(stages):
            label = row_lookup(rows, branch=branch, model=MODELS[0], mode="adaptive", stage=stage)["stage_label"]
            draw_wrapped(draw, label, cell_x + ci * (cell_w + gap) - 8, top - 52, cell_w + 16, FONT_TINY, INK, anchor_center=True)
        delta_x = cell_x + 3 * (cell_w + gap) + 10
        draw_wrapped(draw, "Mixed - single", delta_x, top - 57, 122, FONT_TINY, INK, anchor_center=True)
        for ri, model in enumerate(MODELS):
            yy = top + ri * row_h
            draw_text(draw, (label_x, yy + 23), model_display(model), FONT_SMALL, INK, anchor="lm")
            values = []
            for ci, stage in enumerate(stages):
                value = row_lookup(rows, branch=branch, model=model, mode="adaptive", stage=stage)["rate"]
                values.append(value)
                xx = cell_x + ci * (cell_w + gap)
                draw_rate_cell(draw, (xx, yy, xx + cell_w, yy + 46), value, max_rate, color, FONT_AXIS_BOLD)
            draw_delta_cell(draw, (delta_x, yy, delta_x + 122, yy + 46), values[-1] - values[0], max_delta, FONT_AXIS_BOLD)
        draw_text(draw, (x + w - 26, y + h - 28), "red = more final movement; blue = recovery", FONT_TINY, MUTED, anchor="rm")
    save_tight(im, path)


def figure_confidence_trajectory(path, rows):
    width, height = 3900, 1125
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    title_font = load_font(58, True)
    subtitle_font = load_font(30)
    panel_font = load_font(37, True)
    axis_font = load_font(27)
    axis_bold = load_font(29, True)
    label_font = load_font(27, True)
    turns = [0, 1, 2, 3]
    turn_labels = ["Initial", "T1", "T2", "T3"]

    def weighted_mean(predicate, turn):
        subset = [row for row in rows if row["turn"] == turn and predicate(row)]
        denom = sum(row.get("n", 0) for row in subset)
        if denom <= 0:
            return 0.0
        return sum(row["mean_confidence"] * row.get("n", 0) for row in subset) / denom

    draw_text(draw, (width / 2, 48), "Confidence as an indicator", title_font, INK, anchor="ma")
    draw_text(draw, (width / 2, 104), "Self-reported confidence trends under temporal trigger pressure", subtitle_font, MUTED, anchor="ma")

    def draw_line_panel(panel_x, panel_y, panel_w, panel_h, title, subtitle, line_defs, show_y_label=False):
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
            radius=22,
            fill="#F8FAFC",
            outline="#D9E0E8",
            width=2,
        )
        draw_text(draw, (panel_x + panel_w / 2, panel_y + 42), title, panel_font, INK, anchor="ma")
        draw_text(draw, (panel_x + panel_w / 2, panel_y + 88), subtitle, axis_font, MUTED, anchor="ma")

        plot_x = panel_x + 145
        plot_y = panel_y + 170
        plot_w = panel_w - 355
        plot_h = panel_h - 330
        y_min, y_max = 2.4, 5.0

        def x_pos(turn_index):
            return plot_x + plot_w * turn_index / 3

        def y_pos(value):
            return plot_y + plot_h - plot_h * (value - y_min) / (y_max - y_min)

        for tick in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            yy = y_pos(tick)
            draw.line((plot_x, yy, plot_x + plot_w, yy), fill="#E3EAF2", width=2)
            draw_text(draw, (plot_x - 18, yy), f"{tick:.1f}", axis_font, MUTED, anchor="rm")
        for i, label in enumerate(turn_labels):
            xx = x_pos(i)
            draw.line((xx, plot_y, xx, plot_y + plot_h), fill="#EEF2F6", width=1)
            draw_text(draw, (xx, plot_y + plot_h + 34), label, axis_bold, INK, anchor="ma")
        draw.line((plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h), fill="#9AA7B7", width=4)
        draw.line((plot_x, plot_y, plot_x, plot_y + plot_h), fill="#9AA7B7", width=4)

        for line_def in line_defs:
            label, predicate, color, line_w = line_def[:4]
            label_dy = line_def[4] if len(line_def) > 4 else 0
            values = [weighted_mean(predicate, turn) for turn in turns]
            points = [(x_pos(i), y_pos(value)) for i, value in enumerate(values)]
            draw.line(points, fill=color, width=line_w)
            for xx, yy in points:
                draw.ellipse((xx - 11, yy - 11, xx + 11, yy + 11), fill=color, outline="white", width=3)
            label_x = points[-1][0] + 22
            label_y = points[-1][1] + label_dy
            draw_text(draw, (label_x, label_y), label, label_font, color, anchor="lm")

        draw_text(draw, (plot_x + plot_w / 2, panel_y + panel_h - 55), "Assistant turn", axis_bold, INK, anchor="ma")
        if show_y_label:
            y_label = Image.new("RGBA", (740, 50), (255, 255, 255, 0))
            yd = ImageDraw.Draw(y_label)
            yd.text((0, 0), "Mean self-rated confidence (1-5)", font=axis_bold, fill=INK)
            y_label = y_label.rotate(90, expand=True)
            im.paste(y_label, (int(plot_x - 118), int(plot_y + plot_h / 2 - y_label.height / 2)), y_label)

    draw_line_panel(
        65,
        145,
        1215,
        960,
        "Branch trend",
        "All temporal runs",
        [
            ("GT all", lambda row: row["branch"] == "GT", GT, 10),
            ("NGT all", lambda row: row["branch"] == "NGT", NGT, 10),
        ],
        show_y_label=True,
    )
    draw_line_panel(
        1342,
        145,
        1215,
        960,
        "Mode trend",
        "Static versus adaptive",
        [
            ("Static", lambda row: row["mode"] == "static", STATIC, 10, -20),
            ("Adaptive", lambda row: row["mode"] == "adaptive", ADAPTIVE, 10, 16),
        ],
    )
    draw_line_panel(
        2619,
        145,
        1215,
        960,
        "Sycophancy trend",
        "Stable versus moved",
        [
            (
                "Stable",
                lambda row: (row["branch"] == "GT" and row["category"] == "preserved")
                or (row["branch"] == "NGT" and row["category"] == "held"),
                "#3C8A66",
                10,
            ),
            (
                "Moved",
                lambda row: (row["branch"] == "GT" and row["category"] == "departed")
                or (row["branch"] == "NGT" and row["category"] == "switched"),
                "#C64B4B",
                10,
            ),
        ],
    )

    save_tight(im, path, padding=8)


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
        ("trigger_static_vs_adaptive.png", figure_static_vs_adaptive, tables["static_vs_adaptive"]),
        ("trigger_temporal_pressure.png", figure_temporal_pressure, tables["temporal_pressure"]),
        ("trigger_confidence_trajectory.png", figure_confidence_trajectory, tables["confidence_trajectory"]),
    ]
    written = []
    figure_model_tone(
        figure_dir / "trigger_model_tone.png",
        tables["model_comparison"],
        tables["tone_gradient_opus"],
    )
    written.append("trigger_model_tone.png")
    for filename, fn, rows in figures:
        path = figure_dir / filename
        fn(path, rows)
        written.append(filename)
    figure_trigger_dynamics(
        figure_dir / "trigger_dynamics_summary.png",
        tables["tone_gradient_opus"],
        tables["temporal_pressure"],
        tables["confidence_trajectory"],
    )
    written.append("trigger_dynamics_summary.png")

    published = []
    if args.publish_dir:
        args.publish_dir.mkdir(parents=True, exist_ok=True)
        for filename in written:
            source = figure_dir / filename
            if source.exists():
                shutil.copy2(source, args.publish_dir / filename)
                published.append(filename)

    summary = {
        "run_id": args.run_id,
        "source": "official pass@1-clean trigger result files",
        "figure_dir": str(figure_dir.resolve()),
        "figures": written,
        "tables": table_names,
        "published_to": str(args.publish_dir.resolve()) if args.publish_dir else None,
        "published_figures": published,
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
                "- `trigger_model_tone.png`: main-text Figure 6, combining model-level GT answer change versus right-to-wrong movement with model-separated tone gradients.",
                "- `trigger_static_vs_adaptive.png`: per-model static vs adaptive GT and NGT rates.",
                "- `trigger_temporal_pressure.png`: per-model adaptive single, same-family escalation, heterogeneous temporal pressure, and mixed-minus-single deltas.",
                "- `trigger_dynamics_summary.png`: 1x2 main-text scatter comparing single-follow-up and mixed three-turn movement by model for GT and NGT.",
                "- `trigger_confidence_trajectory.png`: three-panel confidence trends by branch, trigger mode, and stable versus moved final trajectories.",
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
