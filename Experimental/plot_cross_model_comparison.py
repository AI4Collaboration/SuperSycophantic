import argparse
import json
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

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
PANEL_BG = "#F8FAFC"
GT_COLOR = "#5B7DB8"
NGT_COLOR = "#D07A45"


def load_font(size, bold=False):
    return context_plot.font("bold" if bold else "regular", size)


def draw_text(draw, xy, text, font, fill=INK, anchor="la"):
    draw.text(xy, str(text), font=font, fill=fill, anchor=anchor)


def blend_color(hex_color, t, low=(248, 250, 252)):
    hi = ImageColor.getrgb(hex_color)
    t = max(0.0, min(1.0, t))
    return tuple(int(low[i] + (hi[i] - low[i]) * t) for i in range(3))


def cell_fill(value, scale, color):
    return blend_color(color, 0.16 + 0.74 * max(0.0, min(1.0, value / scale)))


def text_color(value, scale):
    return "white" if value / scale > 0.62 else INK


def context_metrics(summary):
    out = {}
    for model in context_plot.MODEL_ORDER:
        row = summary["models"][model]
        out[model] = {
            "context_gt": row["gt"]["correct_to_incorrect_rate"] * 100,
            "context_ngt": row["ngt"]["all"]["user_answer_agreement"]["lift"] * 100,
        }
    return out


def trigger_metrics(results_dir, run_id):
    records = trigger_plot.collect_records(results_dir, run_id)
    rows = trigger_plot.build_trigger_figure_tables(records)["model_comparison"]
    return {
        row["model"]: {
            "trigger_gt": row["gt_rate"] * 100,
            "trigger_ngt": row["ngt_rate"] * 100,
        }
        for row in rows
    }


def draw_cross_model(path, rows):
    width, height = 3900, 1620
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    title = load_font(68, True)
    subtitle = load_font(34)
    group_font = load_font(38, True)
    header_font = load_font(31, True)
    model_font = load_font(30)
    value_font = load_font(34, True)
    note_font = load_font(24)

    draw_text(draw, (width / 2, 54), "Cross-Model Sycophancy Comparison", title, anchor="ma")
    draw_text(draw, (width / 2, 125), "Branch-specific movement rates; darker cells mean more movement within each column", subtitle, MUTED, anchor="ma")
    draw.line((80, 174, width - 80, 174), fill="#EEF2F6", width=4)

    x0, y0 = 140, 230
    table_w, table_h = width - 280, 1280
    draw.rounded_rectangle((x0, y0, x0 + table_w, y0 + table_h), radius=24, fill=PANEL_BG, outline="#D4DEE9", width=2)

    label_w = 650
    cell_w = 680
    gap = 22
    cell_h = 84
    row_gap = 20
    top = y0 + 218
    model_x = x0 + 55
    grid_x = x0 + label_w

    metrics = [
        ("context_gt", "GT truth\ndeparture", "Context", GT_COLOR, 75.0, "%"),
        ("context_ngt", "NGT user-view\nlift", "Context", NGT_COLOR, 40.0, "pp"),
        ("trigger_gt", "GT correct-\nto-wrong", "Trigger", GT_COLOR, 35.0, "%"),
        ("trigger_ngt", "NGT flip", "Trigger", NGT_COLOR, 90.0, "%"),
    ]
    group_spans = {
        "Context": (0, 1),
        "Trigger": (2, 3),
    }

    for group, (start, end) in group_spans.items():
        gx0 = grid_x + start * (cell_w + gap)
        gx1 = grid_x + end * (cell_w + gap) + cell_w
        color = GT_COLOR if group == "Context" else NGT_COLOR
        draw.rounded_rectangle((gx0, y0 + 44, gx1, y0 + 102), radius=12, fill=blend_color(color, 0.18), outline=blend_color(color, 0.52), width=2)
        draw_text(draw, ((gx0 + gx1) / 2, y0 + 74), group, group_font, INK, anchor="mm")

    for ci, (_, label, _, color, _, _) in enumerate(metrics):
        cx = grid_x + ci * (cell_w + gap) + cell_w / 2
        draw.multiline_text((cx, y0 + 130), label, font=header_font, fill=color, anchor="ma", spacing=5, align="center")

    for ri, model in enumerate(context_plot.MODEL_ORDER):
        y = top + ri * (cell_h + row_gap)
        badge = context_plot.make_badge(model)
        badge.thumbnail((62, 62), Image.Resampling.LANCZOS)
        im.paste(badge, (model_x, int(y + (cell_h - badge.height) / 2)), badge)
        label = context_plot.SHORT_LABELS[model].replace("\n", " ")
        draw_text(draw, (model_x + 85, y + cell_h / 2), label, model_font, INK, anchor="lm")
        row = rows[model]
        for ci, (key, _, _, color, scale, unit) in enumerate(metrics):
            x = grid_x + ci * (cell_w + gap)
            value = row[key]
            rect = (x, y, x + cell_w, y + cell_h)
            draw.rounded_rectangle(rect, radius=11, fill=cell_fill(value, scale, color))
            label_text = f"{value:.0f}{unit}"
            draw_text(draw, (x + cell_w / 2, y + cell_h / 2), label_text, value_font, text_color(value, scale), anchor="mm")

    legend_y = y0 + table_h - 64
    draw_text(draw, (x0 + 55, legend_y), "Context: first-turn framed movement. Trigger: single-follow-up movement after a committed neutral answer. Each column uses its own scale.", note_font, MUTED, anchor="la")

    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-summary",
        type=Path,
        default=REPO_ROOT / "Experimental" / "results" / "context_20260504_184050_context_main_summary.json",
    )
    parser.add_argument("--trigger-results-dir", type=Path, default=REPO_ROOT / "Experimental" / "results")
    parser.add_argument("--trigger-run-id", default="trigger_20260504_070840")
    parser.add_argument("--out-png", type=Path, default=REPO_ROOT / "images" / "results" / "cross_model_comparison.png")
    parser.add_argument("--out-pdf", type=Path, default=REPO_ROOT / "images" / "results" / "cross_model_comparison.pdf")
    args = parser.parse_args()

    summary = json.loads(args.context_summary.read_text(encoding="utf-8"))
    ctx = context_metrics(summary)
    trig = trigger_metrics(args.trigger_results_dir, args.trigger_run_id)
    rows = {}
    for model in context_plot.MODEL_ORDER:
        if model not in trig:
            raise RuntimeError(f"Trigger results missing model: {model}")
        rows[model] = {**ctx[model], **trig[model]}

    draw_cross_model(args.out_png, rows)
    Image.open(args.out_png).convert("RGB").save(args.out_pdf)


if __name__ == "__main__":
    main()
