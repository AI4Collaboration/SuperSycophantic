import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CORE_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
    "mistralai/mistral-medium-3.1",
    "cohere/command-r-08-2024",
]
OPUS_MODEL = "anthropic/claude-opus-4.5"

MODEL_LABELS = {
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "GPT-5.4 Mini",
    "openai/gpt-5.4-nano": "GPT-5.4 Nano",
    "anthropic/claude-opus-4.5": "Opus 4.5",
    "anthropic/claude-sonnet-4.5": "Sonnet 4.5",
    "anthropic/claude-haiku-4.5": "Haiku 4.5",
    "google/gemini-3.1-flash-lite-preview": "Gemini Flash Lite",
    "mistralai/mistral-medium-3.1": "Mistral Medium",
    "cohere/command-r-08-2024": "Command R",
}

TONES = ["mild", "moderate", "strong"]
TONE_LABELS = {"mild": "Mild", "moderate": "Moderate", "strong": "Strong"}

W, H = 1120, 720
MARGIN = 56
INK = "#1B2430"
MUTED = "#687385"
GRID = "#E2E8F0"
LIGHT = "#F6F8FB"
GT = "#2E8B6D"
NGT = "#7A55A3"
OPUS = "#D97706"
ACCENT = "#BC3C4A"


def font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend([r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf"])
    candidates.extend([r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


TITLE = font(31, True)
LABEL = font(18)
LABEL_B = font(18, True)
SMALL = font(15)
TINY = font(13)


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        while True:
            try:
                line = next(handle)
            except StopIteration:
                break
            except EOFError:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def pct(rate):
    return 100.0 * rate if rate is not None else None


def rate(events, denom):
    return None if denom == 0 else events / denom


def add_rate(stats, key, event, denom_ok=True):
    if not denom_ok:
        return
    stats[key]["denom"] += 1
    if event:
        stats[key]["events"] += 1


def record_event(record, branch, temporal=False):
    if branch == "gt":
        denom_ok = bool(record.get("initial_correct"))
        event = bool(
            record.get("truth_departure_round") is not None
            if temporal
            else record.get("truth_departure")
        )
    else:
        denom_ok = bool(record.get("eligible"))
        event = bool(
            record.get("answer_switch_round") is not None
            if temporal
            else (record.get("single_trigger_answer_switch") or record.get("answer_changed"))
        )
    return event, denom_ok


def load_records(results_dir, run_id, branch, mode, temporal=False):
    kind = "trigger_temporal" if temporal else "trigger"
    path = results_dir / f"{run_id}_{branch}_{kind}_{mode}.jsonl.gz"
    return list(read_jsonl_gz(path))


def draw_title(draw, title):
    draw.text((W / 2, 28), title, font=TITLE, fill=INK, anchor="ma")
    draw.line((MARGIN, 82, W - MARGIN, 82), fill=GRID, width=2)


def draw_axis_x(draw, x0, y0, width, max_pct, ticks):
    for tick in ticks:
        x = x0 + width * tick / max_pct
        draw.line((x, y0, x, y0 + 1), fill=GRID, width=1)
        draw.text((x, y0 + 8), f"{tick:.0f}", font=TINY, fill=MUTED, anchor="ma")
    draw.text((x0 + width, y0 + 28), "%", font=TINY, fill=MUTED, anchor="ra")


def draw_legend(draw, entries, x, y):
    for label, color in entries:
        draw.rounded_rectangle((x, y + 4, x + 22, y + 16), radius=4, fill=color)
        draw.text((x + 29, y), label, font=SMALL, fill=INK)
        x += 29 + int(draw.textlength(label, font=SMALL)) + 34


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def aggregate_single(results_dir, run_id, models):
    stats = defaultdict(lambda: {"events": 0, "denom": 0})
    for branch in ["gt", "ngt"]:
        for mode in ["static", "adaptive"]:
            for record in load_records(results_dir, run_id, branch, mode, temporal=False):
                model = record.get("model")
                if model not in models:
                    continue
                event, denom_ok = record_event(record, branch, temporal=False)
                add_rate(stats, (branch, model), event, denom_ok)
    return stats


def figure_model_comparison(path, results_dir, run_id):
    stats = aggregate_single(results_dir, run_id, CORE_MODELS)
    rows = []
    for model in CORE_MODELS:
        gt_rate = rate(stats[("gt", model)]["events"], stats[("gt", model)]["denom"])
        ngt_rate = rate(stats[("ngt", model)]["events"], stats[("ngt", model)]["denom"])
        rows.append((model, gt_rate or 0, ngt_rate or 0))
    rows.sort(key=lambda row: (row[1] + row[2]) / 2, reverse=True)

    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)
    draw_title(draw, "Model susceptibility under follow-up pressure")
    draw_legend(draw, [("GT truth departure", GT), ("NGT flip", NGT)], 710, 93)

    x0, x1 = 284, W - 92
    y0, row_h = 125, 59
    max_pct = 92
    for tick in [0, 25, 50, 75, 90]:
        x = x0 + (x1 - x0) * tick / max_pct
        draw.line((x, y0 - 16, x, y0 + row_h * len(rows) - 4), fill=GRID if tick else "#B8C2CC", width=1)
        draw.text((x, y0 + row_h * len(rows) + 5), str(tick), font=TINY, fill=MUTED, anchor="ma")
    draw.text((x1, y0 + row_h * len(rows) + 27), "%", font=TINY, fill=MUTED, anchor="ra")

    for i, (model, gt_rate, ngt_rate) in enumerate(rows):
        y = y0 + i * row_h
        draw.text((MARGIN, y + 12), MODEL_LABELS.get(model, model), font=LABEL_B, fill=INK)
        for j, (value, color) in enumerate([(pct(gt_rate), GT), (pct(ngt_rate), NGT)]):
            by = y + 5 + j * 25
            bw = (x1 - x0) * value / max_pct
            draw.rounded_rectangle((x0, by, x0 + bw, by + 17), radius=6, fill=color)
            draw.text((x0 + bw + 7, by - 1), f"{value:.1f}", font=TINY, fill=INK)
        draw.line((MARGIN, y + row_h - 8, W - MARGIN, y + row_h - 8), fill=LIGHT, width=1)
    save_png(im, path)


def aggregate_tone(results_dir, run_id):
    stats = defaultdict(lambda: {"events": 0, "denom": 0})
    models_for_branch = {"gt": set(CORE_MODELS), "ngt": set(CORE_MODELS)}
    for branch in ["gt", "ngt"]:
        for mode in ["static", "adaptive"]:
            for record in load_records(results_dir, run_id, branch, mode, temporal=False):
                model = record.get("model")
                tone = record.get("tone")
                if tone not in TONES:
                    continue
                if model in models_for_branch[branch]:
                    event, denom_ok = record_event(record, branch, temporal=False)
                    add_rate(stats, (branch, tone), event, denom_ok)
                if branch == "gt" and model == OPUS_MODEL:
                    event, denom_ok = record_event(record, branch, temporal=False)
                    add_rate(stats, ("opus_gt", tone), event, denom_ok)
    return stats


def figure_tone_gradient(path, results_dir, run_id):
    stats = aggregate_tone(results_dir, run_id)
    series = [
        ("GT", GT, [rate(stats[("gt", tone)]["events"], stats[("gt", tone)]["denom"]) for tone in TONES]),
        ("NGT", NGT, [rate(stats[("ngt", tone)]["events"], stats[("ngt", tone)]["denom"]) for tone in TONES]),
        ("Opus GT-only", OPUS, [rate(stats[("opus_gt", tone)]["events"], stats[("opus_gt", tone)]["denom"]) for tone in TONES]),
    ]

    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)
    draw_title(draw, "Tone gradient is key, but Opus is special")
    draw_legend(draw, [(label, color) for label, color, _ in series], 670, 93)

    px0, py0, pw, ph = 135, 125, 850, 465
    max_pct = 90
    for tick in [0, 20, 40, 60, 80]:
        y = py0 + ph - ph * tick / max_pct
        draw.line((px0, y, px0 + pw, y), fill=GRID, width=1)
        draw.text((px0 - 16, y), str(tick), font=TINY, fill=MUTED, anchor="rm")
    draw.text((px0 - 12, py0 - 8), "%", font=TINY, fill=MUTED, anchor="rm")

    x_positions = [px0 + 80, px0 + pw / 2, px0 + pw - 80]
    for x, tone in zip(x_positions, TONES):
        draw.text((x, py0 + ph + 24), TONE_LABELS[tone], font=LABEL_B, fill=INK, anchor="ma")
        draw.line((x, py0, x, py0 + ph), fill=LIGHT, width=1)

    for label, color, values in series:
        points = []
        for x, value in zip(x_positions, values):
            value = 0 if value is None else value
            y = py0 + ph - ph * pct(value) / max_pct
            points.append((x, y, value))
        line_width = 5 if label != "Opus GT-only" else 4
        for (x1, y1, _), (x2, y2, _) in zip(points, points[1:]):
            draw.line((x1, y1, x2, y2), fill=color, width=line_width)
        for x, y, value in points:
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill="white", outline=color, width=4)
            draw.text((x, y - 29), f"{pct(value):.1f}", font=TINY, fill=color, anchor="ma")
        draw.text((points[-1][0] + 24, points[-1][1] - 4), label, font=SMALL, fill=color)

    draw.rounded_rectangle((650, 500, 1015, 585), radius=14, fill="#FFF7ED", outline="#FED7AA", width=2)
    draw.text((670, 515), "Opus stays lower and flatter", font=LABEL_B, fill=OPUS)
    draw.text((670, 542), "GT truth departure changes little", font=SMALL, fill=INK)
    draw.text((670, 564), "from moderate to strong tone.", font=SMALL, fill=INK)
    save_png(im, path)


def sequence_category(record):
    seq = record.get("trigger_sequence") or []
    if len(set(seq)) <= 1:
        return "Same-family escalation"
    return "Heterogeneous sequence"


def aggregate_temporal(results_dir, run_id):
    stats = defaultdict(lambda: {"events": 0, "denom": 0})
    single = aggregate_single(results_dir, run_id, CORE_MODELS)
    for branch in ["gt", "ngt"]:
        for model in CORE_MODELS:
            stats[(branch, "Single follow-up")]["events"] += single[(branch, model)]["events"]
            stats[(branch, "Single follow-up")]["denom"] += single[(branch, model)]["denom"]
        for mode in ["static", "adaptive"]:
            for record in load_records(results_dir, run_id, branch, mode, temporal=True):
                if record.get("model") not in CORE_MODELS:
                    continue
                event, denom_ok = record_event(record, branch, temporal=True)
                add_rate(stats, (branch, sequence_category(record)), event, denom_ok)
    return stats


def figure_temporal(path, results_dir, run_id):
    stats = aggregate_temporal(results_dir, run_id)
    cats = ["Single follow-up", "Same-family escalation", "Heterogeneous sequence"]
    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)
    draw_title(draw, "Temporal pressure creates more chances to switch")
    draw_legend(draw, [("GT ever truth departure", GT), ("NGT ever flip", NGT)], 625, 93)

    x0, x1 = 205, W - 105
    y0, group_h = 135, 138
    max_pct = 96
    for tick in [0, 25, 50, 75, 95]:
        x = x0 + (x1 - x0) * tick / max_pct
        draw.line((x, y0 - 18, x, y0 + group_h * len(cats) - 18), fill=GRID if tick else "#B8C2CC", width=1)
        draw.text((x, y0 + group_h * len(cats) + 3), str(tick), font=TINY, fill=MUTED, anchor="ma")
    draw.text((x1, y0 + group_h * len(cats) + 25), "%", font=TINY, fill=MUTED, anchor="ra")

    for i, cat in enumerate(cats):
        y = y0 + i * group_h
        draw.text((MARGIN, y + 24), cat.replace(" ", "\n", 1), font=LABEL_B, fill=INK)
        for j, (branch, color) in enumerate([("gt", GT), ("ngt", NGT)]):
            r = rate(stats[(branch, cat)]["events"], stats[(branch, cat)]["denom"]) or 0
            value = pct(r)
            by = y + 10 + j * 43
            bw = (x1 - x0) * value / max_pct
            draw.rounded_rectangle((x0, by, x0 + bw, by + 28), radius=8, fill=color)
            draw.text((x0 + bw + 8, by + 5), f"{value:.1f}", font=SMALL, fill=INK)
        draw.line((MARGIN, y + group_h - 20, W - MARGIN, y + group_h - 20), fill=LIGHT, width=1)
    save_png(im, path)


def aggregate_confidence(results_dir, run_id):
    series = defaultdict(lambda: [[], [], [], []])
    for branch in ["gt", "ngt"]:
        for mode in ["static", "adaptive"]:
            for record in load_records(results_dir, run_id, branch, mode, temporal=True):
                if record.get("model") not in CORE_MODELS:
                    continue
                if branch == "gt":
                    if not record.get("initial_correct"):
                        continue
                    key = "GT departed" if record.get("truth_departure_round") is not None else "GT preserved"
                else:
                    if not record.get("eligible"):
                        continue
                    key = "NGT switched" if record.get("answer_switch_round") is not None else "NGT held"
                rounds = record.get("rounds") or []
                values = [record.get("initial_confidence")] + [round_.get("confidence") for round_ in rounds[:3]]
                if len(values) != 4 or not all(isinstance(value, int) for value in values):
                    continue
                for idx, value in enumerate(values):
                    series[key][idx].append(value)
    return {
        key: [sum(values) / len(values) if values else None for values in by_turn]
        for key, by_turn in series.items()
    }


def figure_confidence(path, results_dir, run_id):
    series = aggregate_confidence(results_dir, run_id)
    order = [
        ("GT preserved", GT),
        ("GT departed", ACCENT),
        ("NGT held", "#6B7280"),
        ("NGT switched", NGT),
    ]
    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)
    draw_title(draw, "Confidence erodes before answer movement")

    px0, py0, pw, ph = 125, 120, 820, 470
    ymin, ymax = 3.0, 5.0
    for tick in [3.0, 3.5, 4.0, 4.5, 5.0]:
        y = py0 + ph - ph * (tick - ymin) / (ymax - ymin)
        draw.line((px0, y, px0 + pw, y), fill=GRID, width=1)
        draw.text((px0 - 16, y), f"{tick:.1f}", font=TINY, fill=MUTED, anchor="rm")
    draw.text((px0 - 15, py0 - 10), "conf.", font=TINY, fill=MUTED, anchor="rm")

    xs = [px0 + 60, px0 + 300, px0 + 540, px0 + 780]
    xlabels = ["Initial", "Turn 1", "Turn 2", "Turn 3"]
    for x, label in zip(xs, xlabels):
        draw.line((x, py0, x, py0 + ph), fill=LIGHT, width=1)
        draw.text((x, py0 + ph + 23), label, font=LABEL_B, fill=INK, anchor="ma")

    for label, color in order:
        values = series.get(label)
        if not values or any(value is None for value in values):
            continue
        points = []
        for x, value in zip(xs, values):
            y = py0 + ph - ph * (value - ymin) / (ymax - ymin)
            points.append((x, y, value))
        for (x1, y1, _), (x2, y2, _) in zip(points, points[1:]):
            draw.line((x1, y1, x2, y2), fill=color, width=5)
        for x, y, value in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="white", outline=color, width=4)
        draw.text((points[-1][0] + 22, points[-1][1] - 8), label, font=SMALL, fill=color)
    draw.rounded_rectangle((650, 500, 1015, 587), radius=14, fill="#F8FAFC", outline=GRID, width=2)
    draw.text((670, 515), "Confidence is not a guardrail", font=LABEL_B, fill=INK)
    draw.text((670, 542), "Moved runs erode under pressure,", font=SMALL, fill=INK)
    draw.text((670, 564), "while stable runs stay higher.", font=SMALL, fill=INK)
    save_png(im, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="trigger_20260504_070840")
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("images/results"))
    return parser.parse_args()


def main():
    args = parse_args()
    figure_model_comparison(args.output_dir / "trigger_model_comparison.png", args.results_dir, args.run_id)
    figure_tone_gradient(args.output_dir / "trigger_tone_gradient_opus.png", args.results_dir, args.run_id)
    figure_temporal(args.output_dir / "trigger_temporal_comparison.png", args.results_dir, args.run_id)
    figure_confidence(args.output_dir / "trigger_confidence_trajectory.png", args.results_dir, args.run_id)


if __name__ == "__main__":
    main()
