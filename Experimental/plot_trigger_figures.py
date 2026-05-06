import argparse
import csv
import gzip
import json
import math
import os
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
CLAUDE_MODELS = [model for model in MODELS if model.startswith("anthropic/")]
OTHER_MODELS = [model for model in MODELS if model not in CLAUDE_MODELS]

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


def release_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


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
    font_dir = Path(os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "") / "Fonts"
    candidates = []
    if bold:
        candidates.extend(
            [
                font_dir / "arialbd.ttf",
                font_dir / "segoeuib.ttf",
            ]
        )
    candidates.extend(
        [
            font_dir / "arial.ttf",
            font_dir / "segoeui.ttf",
            font_dir / "calibri.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
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
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing result file: {path}. Restore the ignored Experimental/results raw run files before regenerating figures."
        )
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


def draw_rotated_label(im, center, text, font, fill=INK, angle=90, pad_x=18, pad_y=14):
    label_box = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    ld = ImageDraw.Draw(label_box)
    box = ld.textbbox((0, 0), text, font=font)
    label = Image.new("RGBA", (box[2] - box[0] + 2 * pad_x, box[3] - box[1] + 2 * pad_y), (255, 255, 255, 0))
    ld = ImageDraw.Draw(label)
    ld.text((pad_x - box[0], pad_y - box[1]), text, font=font, fill=fill)
    rotated = label.rotate(angle, expand=True)
    im.paste(rotated, (int(center[0] - rotated.width / 2), int(center[1] - rotated.height / 2)), rotated)


def save_tight(im, path, padding=34):
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
            for category in (["preserved", "departed"] if branch == "GT" else ["held", "switched"]):
                by_turn = {turn: [] for turn in range(4)}
                for record in temporal:
                    if record["_branch"] != branch or record["model"] != model_id or not denom_ok(record, branch):
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
        "Trigger-induced outcomes",
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
        (grid_x + col_w + gutter_x, grid_y, "GT", "temporal", "GT: cave over time", "#B8443F"),
        (grid_x, grid_y + row_h + gutter_y, "NGT", "single", "NGT: decision shifts", "#2870A8"),
        (grid_x + col_w + gutter_x, grid_y + row_h + gutter_y, "NGT", "temporal", "NGT: cave over time", "#2870A8"),
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
    width, height = 2440, 1090
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Adaptive triggers by a small model can mislead larger models", "", width)

    top = 165
    panel_w, panel_h = 1025, 790
    panel_gap = 115
    panel_lefts = [145, 145 + panel_w + panel_gap]
    plot_pad_left, plot_pad_top = 116, 92
    plot_pad_right, plot_pad_bottom = 58, 112
    plot_w = panel_w - plot_pad_left - plot_pad_right
    plot_h = panel_h - plot_pad_top - plot_pad_bottom
    min_x, max_x = 0.05, 0.38
    min_y, max_y = 0.25, 0.95
    x_cut, y_cut = 0.20, 0.60

    label_specs = {
        "static": {
            "openai/gpt-5.4": (-46, -54, "rm"),
            "openai/gpt-5.4-mini": (-16, 58, "ma"),
            "openai/gpt-5.4-nano": (34, -4, "lm"),
            "anthropic/claude-opus-4.5": (-40, -50, "rm"),
            "anthropic/claude-sonnet-4.5": (34, -44, "lm"),
            "anthropic/claude-haiku-4.5": (36, 24, "lm"),
            "google/gemini-3.1-flash-lite-preview": (32, -44, "lm"),
            "mistralai/mistral-medium-3.1": (36, -10, "lm"),
            "cohere/command-r-08-2024": (34, 18, "lm"),
        },
        "adaptive": {
            "openai/gpt-5.4": (0, -70, "ma"),
            "openai/gpt-5.4-mini": (-46, 50, "rm"),
            "openai/gpt-5.4-nano": (30, 50, "lm"),
            "anthropic/claude-opus-4.5": (-42, 18, "rm"),
            "anthropic/claude-sonnet-4.5": (36, -46, "lm"),
            "anthropic/claude-haiku-4.5": (36, 18, "lm"),
            "google/gemini-3.1-flash-lite-preview": (34, -44, "lm"),
            "mistralai/mistral-medium-3.1": (38, 20, "lm"),
            "cohere/command-r-08-2024": (36, 18, "lm"),
        },
    }

    def draw_panel(panel_left, mode, title):
        nonlocal draw
        panel_top = top
        draw.rounded_rectangle(
            (panel_left, panel_top, panel_left + panel_w, panel_top + panel_h),
            radius=18,
            fill="#FBFCFE",
            outline="#D8E0EA",
            width=2,
        )
        draw_text(draw, (panel_left + panel_w / 2, panel_top + 36), title, FONT_PANEL, INK, anchor="ma")

        left = panel_left + plot_pad_left
        plot_top = panel_top + plot_pad_top

        def x_pos(value):
            return left + plot_w * (value - min_x) / (max_x - min_x)

        def y_pos(value):
            return plot_top + plot_h - plot_h * (value - min_y) / (max_y - min_y)

        overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        od = ImageDraw.Draw(overlay)
        x_mid, y_mid = x_pos(x_cut), y_pos(y_cut)
        od.rectangle((left, plot_top, x_mid, y_mid), fill=(*ImageColor.getrgb("#EAF4EF"), 150))
        od.rectangle((x_mid, plot_top, left + plot_w, y_mid), fill=(*ImageColor.getrgb("#FFF3E8"), 150))
        od.rectangle((left, y_mid, x_mid, plot_top + plot_h), fill=(*ImageColor.getrgb("#EEF5FB"), 150))
        od.rectangle((x_mid, y_mid, left + plot_w, plot_top + plot_h), fill=(*ImageColor.getrgb("#F9EEF1"), 135))
        base = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        im.paste(base)
        draw = ImageDraw.Draw(im)

        for value in [0.10, 0.20, 0.30]:
            xx = x_pos(value)
            draw.line((xx, plot_top, xx, plot_top + plot_h), fill="#DCE3EC", width=1)
            draw_text(draw, (xx, plot_top + plot_h + 16), f"{int(value * 100)}%", FONT_SMALL, MUTED, anchor="ma")
        for value in [0.40, 0.60, 0.80]:
            yy = y_pos(value)
            draw.line((left, yy, left + plot_w, yy), fill="#DCE3EC", width=1)
            draw_text(draw, (left - 15, yy), f"{int(value * 100)}%", FONT_SMALL, MUTED, anchor="rm")
        draw.line((x_mid, plot_top, x_mid, plot_top + plot_h), fill="#586474", width=2)
        draw.line((left, y_mid, left + plot_w, y_mid), fill="#586474", width=2)
        draw.rectangle((left, plot_top, left + plot_w, plot_top + plot_h), outline="#1A2028", width=2)

        rows = []
        for model in MODELS:
            rows.append(
                {
                    "model": model,
                    "label": MODEL_SHORT_LABELS[model],
                    "gt": row_lookup(headline, model=model, branch="GT", mode=mode, source="single")["rate"],
                    "ngt": row_lookup(headline, model=model, branch="NGT", mode=mode, source="single")["rate"],
                }
            )

        for row in sorted(rows, key=lambda item: item["ngt"]):
            x = x_pos(row["gt"])
            y = y_pos(row["ngt"])
            badge = make_badge(row["model"])
            badge.thumbnail((62, 62), RESAMPLE_LANCZOS)
            im.paste(badge, (int(x - badge.width / 2), int(y - badge.height / 2)), badge)
            draw = ImageDraw.Draw(im)
            dx, dy, anchor = label_specs[mode].get(row["model"], (24, 18, "lm"))
            lx, ly = x + dx, y + dy
            if abs(dx) > 34 or abs(dy) > 34:
                edge_x = x + (badge.width / 2 if dx > 0 else -badge.width / 2 if dx < 0 else 0)
                edge_y = y + (badge.height / 2 if dy > 0 else -badge.height / 2 if dy < 0 else 0)
                draw.line((edge_x, edge_y, lx, ly), fill="#AEB8C6", width=1)
            draw.multiline_text((lx, ly), row["label"], font=FONT_TINY, fill=INK, anchor=anchor, spacing=1, align="center")

        draw_text(
            draw,
            (left + plot_w / 2, plot_top + plot_h + 56),
            "Model changes from right to wrong after 1 trigger (%)",
            FONT_AXIS_BOLD,
            INK,
            anchor="ma",
        )
        draw_rotated_label(im, (left - 42, plot_top + plot_h / 2), "Model changes answer after 1 trigger (%)", FONT_AXIS_BOLD)
        draw = ImageDraw.Draw(im)

    draw_panel(panel_lefts[0], "static", "Static")
    draw_panel(panel_lefts[1], "adaptive", "Adaptive")
    save_tight(im, path)


def figure_model_comparison(path, rows):
    width, height = 1780, 1036
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Model comparison", "", width)

    def panel(x, y, title, rate_key, max_rate, color, ticks):
        w, h = 700, 760
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 24, y + 24), title, FONT_PANEL, INK)
        label_x = x + 26
        bar_x = x + 235
        bar_w = w - 305
        top = y + 118
        row_h = 66
        bar_h = 38
        for tick in ticks:
            xx = bar_x + bar_w * tick / max_rate
            draw.line((xx, top - 14, xx, top + row_h * len(MODELS) - 8), fill="#E6ECF3", width=1)
            draw_text(draw, (xx, top + row_h * len(MODELS) + 8), f"{int(round(tick * 100))}", FONT_TINY, MUTED, anchor="ma")
        for ri, model in enumerate(MODELS):
            row = row_lookup(rows, model=model)
            value = row[rate_key]
            yy = top + ri * row_h
            model_color = MODEL_COLORS[model]
            draw.line((label_x, yy + bar_h / 2, label_x + 32, yy + bar_h / 2), fill=model_color, width=5)
            draw.ellipse((label_x + 11, yy + bar_h / 2 - 6, label_x + 23, yy + bar_h / 2 + 6), fill=model_color, outline="white", width=2)
            draw_text(draw, (label_x + 44, yy + bar_h / 2), model_display(model), FONT_SMALL, INK, anchor="lm")
            draw.rounded_rectangle((bar_x, yy, bar_x + bar_w, yy + bar_h), radius=8, fill="#EEF2F6")
            fill_w = bar_w * min(value, max_rate) / max_rate
            draw.rounded_rectangle((bar_x, yy, bar_x + fill_w, yy + bar_h), radius=8, fill=blend_color(color, 0.36 + 0.52 * value / max_rate))
            label = fmt_pct(value, 0)
            if fill_w > 72:
                draw_text(draw, (bar_x + fill_w - 10, yy + bar_h / 2), label, FONT_SMALL, "white", anchor="rm")
            else:
                draw_text(draw, (bar_x + fill_w + 8, yy + bar_h / 2), label, FONT_SMALL, INK, anchor="lm")
        draw_text(draw, (bar_x + bar_w / 2, y + h - 32), "Rate (%)", FONT_AXIS_BOLD, MUTED, anchor="ma")

    panel(100, 195, "GT correct-to-wrong", "gt_rate", 0.55, STATIC, [0.0, 0.15, 0.30, 0.45, 0.55])
    panel(960, 195, "NGT Flip-Flop", "ngt_rate", 0.90, ADAPTIVE, [0.0, 0.30, 0.60, 0.90])
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
    width, height = 1780, 940
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Claude often peaks before the strongest pressure", "", width)
    panels = [
        ("GT", 130, 185, "OBJ", 0.55, [0, 0.15, 0.30, 0.45]),
        ("NGT", 950, 185, "SUB", 0.9, [0, 0.3, 0.6, 0.9]),
    ]

    def all_model_mean(branch, tone):
        subset = [row for row in rows if row["branch"] == branch and row["tone"] == tone and row["model"] in MODELS]
        denom = sum(row.get("denom", 0) for row in subset)
        if denom <= 0:
            raise ValueError(f"Missing tone rows for branch={branch}, tone={tone}")
        return sum(row["rate"] * row.get("denom", 0) for row in subset) / denom

    line_specs = [
        ("All-model baseline", "#17202A", 8, 13, None),
        ("Opus 4.5", MODEL_COLORS["anthropic/claude-opus-4.5"], 8, 13, "anthropic/claude-opus-4.5"),
        ("Sonnet 4.5", MODEL_COLORS["anthropic/claude-sonnet-4.5"], 6, 10, "anthropic/claude-sonnet-4.5"),
        ("Haiku 4.5", MODEL_COLORS["anthropic/claude-haiku-4.5"], 6, 10, "anthropic/claude-haiku-4.5"),
    ]

    for branch, x, y, title, max_rate, ticks in panels:
        w, h = 700, 610
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + w / 2, y + 36), title, load_font(33, True), INK, anchor="ma")
        px0, py0 = x + 108, y + 112
        pw, ph = w - 178, h - 225
        for tick in ticks:
            yy = py0 + ph - ph * tick / max_rate
            draw.line((px0, yy, px0 + pw, yy), fill="#E7EDF3", width=1)
            draw_text(draw, (px0 - 16, yy), f"{int(tick * 100)}", FONT_SMALL, MUTED, anchor="rm")
        for label, color, line_w, dot_r, model in line_specs:
            pts = []
            for i, tone in enumerate(TONES):
                value = all_model_mean(branch, tone) if model is None else row_lookup(rows, branch=branch, model=model, tone=tone)["rate"]
                xx = px0 + i * pw / 2
                yy = py0 + ph - ph * value / max_rate
                pts.append((xx, yy))
            draw.line(pts, fill=color, width=line_w)
            for xx, yy in pts:
                draw.ellipse((xx - dot_r, yy - dot_r, xx + dot_r, yy + dot_r), fill=color, outline="white", width=3)
        for i, tone in enumerate(TONES):
            xx = px0 + i * pw / 2
            draw_text(draw, (xx, py0 + ph + 36), TONE_LABELS[tone], FONT_AXIS_BOLD, INK, anchor="ma")
        draw_text(draw, (px0 + pw / 2, py0 + ph + 68), "Tone", FONT_AXIS_BOLD, INK, anchor="ma")
        draw_rotated_label(im, (px0 - 74, py0 + ph / 2), "Sycophantic rate (%)", FONT_AXIS_BOLD)
        draw = ImageDraw.Draw(im)
    save_tight(im, path)


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
            color = MODEL_COLORS[model]
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color_alpha(color, 230), outline=(255, 255, 255, 255), width=2)
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
    tone_panel("NGT", 120, 725, "NGT Flip-Flop", 0.90, [0, 0.30, 0.60, 0.90])
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
    temporal_panel("NGT", 2910, 210, "NGT final Flip-Flop", 0.90, ADAPTIVE)
    save_tight(im, path)


def figure_static_vs_adaptive(path, rows):
    width, height = 1780, 1100
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Static vs. adaptive by model", "", width)
    panels = [("GT", 110, 195, "GT correct-to-wrong", 0.55, STATIC), ("NGT", 935, 195, "NGT Flip-Flop", 0.90, ADAPTIVE)]
    for branch, x, y, title, max_rate, _ in panels:
        w, h = 735, 780
        draw.rectangle((x, y, x + w, y + h), fill="white", outline="#D9E0E8", width=2)
        draw_text(draw, (x + 26, y + 26), title, FONT_PANEL, INK)
        label_x = x + 26
        cell_x = x + 285
        top = y + 110
        row_h = 60
        cell_w = 155
        gap = 10
        for ci, mode in enumerate(["static", "adaptive"]):
            draw_text(draw, (cell_x + ci * (cell_w + gap) + cell_w / 2, top - 34), mode.capitalize(), FONT_AXIS_BOLD, INK, anchor="ma")
        draw_text(draw, (cell_x + 2 * (cell_w + gap) + 35, top - 34), "Delta", FONT_AXIS_BOLD, INK, anchor="ma")
        for ri, model in enumerate(MODELS):
            yy = top + ri * row_h
            draw_text(draw, (label_x, yy + 23), model_display(model), FONT_SMALL, INK, anchor="lm")
            static = row_lookup(rows, branch=branch, model=model, mode="static")["rate"]
            adaptive = row_lookup(rows, branch=branch, model=model, mode="adaptive")["rate"]
            for ci, (mode, value) in enumerate([("static", static), ("adaptive", adaptive)]):
                xx = cell_x + ci * (cell_w + gap)
                color = STATIC if mode == "static" else ADAPTIVE
                draw_rate_cell(draw, (xx, yy, xx + cell_w, yy + 46), value, max_rate, color, FONT_AXIS_BOLD)
            delta = adaptive - static
            delta_x = cell_x + 2 * (cell_w + gap) + 8
            sign = "+" if delta >= 0 else ""
            draw_text(draw, (delta_x + 35, yy + 23), f"{sign}{pct(delta):.1f}", FONT_SMALL, ADAPTIVE if delta >= 0 else ACCENT, anchor="mm")
    save_tight(im, path)


def figure_temporal_pressure(path, rows):
    width, height = 1780, 1100
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Temporal pressure by model", "", width)
    panels = [
        ("GT", 105, 195, "GT final correct-to-wrong", 0.55, STATIC, 0.14),
        ("NGT", 920, 195, "NGT final Flip-Flop", 0.90, ADAPTIVE, 0.40),
    ]
    stages = ["single", "same_family", "heterogeneous"]
    for branch, x, y, title, max_rate, color, max_delta in panels:
        w, h = 755, 780
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
        draw_text(draw, (delta_x + 61, top - 34), "Delta pp", FONT_AXIS_BOLD, INK, anchor="ma")
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
    save_tight(im, path)


def figure_confidence_trajectory(path, rows):
    width, height = 2620, 980
    im = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(im)
    panel_font = load_font(43, True)
    axis_font = load_font(31)
    axis_bold = load_font(34, True)
    label_font = load_font(33, True)
    turns = [0, 1, 2, 3]
    turn_labels = ["Initial", "T1", "T2", "T3"]

    def weighted_mean(predicate, turn):
        subset = [row for row in rows if row["turn"] == turn and predicate(row)]
        denom = sum(row.get("n", 0) for row in subset)
        if denom <= 0:
            raise ValueError(f"Missing confidence rows for turn={turn}")
        return sum(row["mean_confidence"] * row.get("n", 0) for row in subset) / denom

    def draw_line_panel(panel_x, panel_y, panel_w, panel_h, title, line_defs, show_y_label=False):
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
            radius=22,
            fill="#F8FAFC",
            outline="#D9E0E8",
            width=2,
        )
        draw_text(draw, (panel_x + panel_w / 2, panel_y + 46), title, panel_font, INK, anchor="ma")

        plot_x = panel_x + 145
        plot_y = panel_y + 126
        plot_w = panel_w - 355
        plot_h = panel_h - 255
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
            draw_text(draw, (xx, plot_y + plot_h + 32), label, axis_bold, INK, anchor="ma")
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
            if label:
                draw_text(draw, (points[-1][0] + 22, points[-1][1] + label_dy), label, label_font, color, anchor="lm")

        draw_text(draw, (plot_x + plot_w / 2, plot_y + plot_h + 62), "Assistant turn", axis_bold, INK, anchor="ma")
        if show_y_label:
            draw_rotated_label(im, (plot_x - 48, plot_y + plot_h / 2), "Mean self-rated confidence (1-5)", axis_bold)

    draw_line_panel(
        65,
        35,
        1215,
        860,
        "OBJ vs. SUB",
        [
            ("OBJ", lambda row: row["branch"] == "GT", "#009E73", 10),
            ("SUB", lambda row: row["branch"] == "NGT", "#D55E00", 10),
        ],
        show_y_label=True,
    )
    draw_line_panel(
        1340,
        35,
        1215,
        860,
        "Sycophantic vs. stable",
        [
            (
                "Stable",
                lambda row: (row["branch"] == "GT" and row["category"] == "preserved")
                or (row["branch"] == "NGT" and row["category"] == "held"),
                "#3C8A66",
                10,
            ),
            (
                "Sycophantic",
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
    headline, _, _, _ = build_tables(records)
    tables = build_trigger_figure_tables(records)

    table_names = []
    for name, rows in tables.items():
        filename = f"{name}.csv"
        write_csv(figure_dir / filename, rows)
        table_names.append(filename)

    figures = [
        ("trigger_model_quadrant.png", figure_scatter, headline),
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
    figure_tone_temporal(
        figure_dir / "trigger_tone_temporal.png",
        tables["tone_gradient_opus"],
        tables["temporal_pressure"],
    )
    written.append("trigger_tone_temporal.png")
    figure_trigger_dynamics(
        figure_dir / "trigger_dynamics_summary.png",
        tables["tone_gradient_opus"],
        tables["temporal_pressure"],
        tables["confidence_trajectory"],
    )
    written.append("trigger_dynamics_summary.png")

    summary = {
        "run_id": args.run_id,
        "source": "official pass@1-clean trigger result files",
        "figure_dir": release_relative_path(figure_dir),
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
                "- `trigger_model_quadrant.png`: main-text static/adaptive changes from right to wrong versus answer changes quadrant view.",
                "- `trigger_model_comparison.png`: side-by-side GT correct-to-wrong and NGT Flip-Flop rates by model.",
                "- `trigger_tone_gradient_opus.png`: Claude mild/moderate/strong tone gradients against the denominator-weighted all-model baseline.",
                "- `trigger_static_vs_adaptive.png`: per-model static vs adaptive GT and NGT rates.",
                "- `trigger_temporal_pressure.png`: per-model adaptive single, same-family escalation, heterogeneous temporal pressure, and mixed-minus-single deltas.",
                "- `trigger_tone_temporal.png`: compact main-text composite of per-model tone gradients and adaptive temporal pressure.",
                "- `trigger_dynamics_summary.png`: 1x2 main-text scatter comparing single-follow-up and mixed three-turn movement by model for GT and NGT.",
                "- `trigger_confidence_trajectory.png`: two-panel confidence trends for OBJ/SUB and stable/sycophantic temporal trajectories.",
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
