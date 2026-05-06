import argparse
import json
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
GRID = "#E1E8F0"
PANEL = "#F8FAFC"
WHITE = "#FFFFFF"


def font(size, bold=False):
    return trigger_plot.load_font(size, bold)


TITLE = font(48, True)
PANEL_TITLE = font(34, True)
LABEL = font(26, True)
SMALL = font(23)
CELL = font(30, True)


MODEL_ORDER = context_plot.MODEL_ORDER
MODEL_LABELS = {model: context_plot.SHORT_LABELS[model].replace("\n", " ") for model in MODEL_ORDER}
TONE_COLORS = {
    "gt": "#D55E00",
    "ngt": "#009E73",
    "neutral": "#0072B2",
    "change": "#CC79A7",
    "direction": "#E69F00",
}


def pct(value):
    return 100.0 * value


def draw_text(draw, xy, text, fill=INK, fnt=SMALL, anchor=None):
    draw.text(xy, str(text), font=fnt, fill=fill, anchor=anchor)


def draw_header(draw, title, width):
    draw.rounded_rectangle((40, 24, width - 40, 108), radius=20, fill=PANEL, outline="#D8E0EA", width=2)
    draw_text(draw, (width / 2, 66), title, fnt=TITLE, anchor="mm")


def save(im, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    trigger_plot.save_tight(im.convert("RGB"), path, padding=10)


def load_context_summary(path):
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [model for model in MODEL_ORDER if model not in summary["models"]]
    if missing:
        raise RuntimeError(f"Context summary missing models: {missing}")
    return summary


def draw_axis(draw, x0, y0, x1, y1, max_value, ticks):
    draw.line((x0, y1, x1, y1), fill="#9AA7B7", width=3)
    for tick in ticks:
        x = x0 + (x1 - x0) * tick / max_value
        draw.line((x, y0, x, y1), fill=GRID, width=1)
        draw_text(draw, (x, y1 + 16), f"{int(tick)}", fill=MUTED, fnt=SMALL, anchor="ma")


def draw_grouped_bar_panel(draw, x, y, w, h, title, rows, max_value, legend):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
    draw_text(draw, (x + w / 2, y + 32), title, fnt=PANEL_TITLE, anchor="ma")
    label_w = 220
    axis_x0 = x + label_w
    axis_x1 = x + w - 36
    axis_y0 = y + 90
    axis_y1 = y + h - 86
    draw_axis(draw, axis_x0, axis_y0, axis_x1, axis_y1, max_value, [0, max_value / 2, max_value])
    row_h = (axis_y1 - axis_y0) / len(rows)
    bar_h = min(20, row_h / (len(legend) + 0.6))
    gap = bar_h * 0.45
    for ri, row in enumerate(rows):
        yc = axis_y0 + ri * row_h + row_h / 2
        draw_text(draw, (x + 22, yc), row["label"], fnt=SMALL, anchor="lm")
        total_h = len(legend) * bar_h + (len(legend) - 1) * gap
        y_start = yc - total_h / 2
        for li, (key, label, color) in enumerate(legend):
            value = row[key]
            yy = y_start + li * (bar_h + gap)
            xx = axis_x0 + (axis_x1 - axis_x0) * value / max_value
            draw.rounded_rectangle((axis_x0, yy, xx, yy + bar_h), radius=5, fill=color)
            draw_text(draw, (xx + 8, yy + bar_h / 2), f"{value:.0f}", fill=color, fnt=SMALL, anchor="lm")
    legend_x = axis_x0
    legend_y = y + h - 42
    for key, label, color in legend:
        draw.rounded_rectangle((legend_x, legend_y - 11, legend_x + 32, legend_y + 11), radius=5, fill=color)
        draw_text(draw, (legend_x + 42, legend_y), label, fnt=SMALL, anchor="lm")
        legend_x += 250


def figure_context_model_bars(summary, out_path):
    width, height = 2800, 1160
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Context: Model-Level Diagnostics", width)

    gt_rows = []
    ngt_rows = []
    for model in MODEL_ORDER:
        row = summary["models"][model]
        gt = row["gt"]
        ngt = row["ngt"]["all"]
        direction = row["ngt"]["paired_directionality"]
        gt_rows.append(
            {
                "label": MODEL_LABELS[model],
                "truth": pct(gt["correct_to_incorrect_rate"]),
                "change": pct(gt["answer_change_rate"]),
            }
        )
        ngt_rows.append(
            {
                "label": MODEL_LABELS[model],
                "lift": pct(ngt["user_answer_agreement"]["lift"]),
                "change": pct(ngt["answer_change_rate"]),
                "direction": pct(direction["answer_change_by_user_direction_rate"]),
            }
        )

    draw_grouped_bar_panel(
        draw,
        70,
        145,
        1300,
        930,
        "OBJ",
        gt_rows,
        80,
        [("truth", "truth departure", TONE_COLORS["gt"]), ("change", "answer change", TONE_COLORS["change"])],
    )
    draw_grouped_bar_panel(
        draw,
        1430,
        145,
        1300,
        930,
        "SUB",
        ngt_rows,
        80,
        [
            ("lift", "user-view lift", TONE_COLORS["ngt"]),
            ("change", "answer change", TONE_COLORS["change"]),
            ("direction", "A/B direction", TONE_COLORS["direction"]),
        ],
    )
    save(im, out_path)


def heat_color(value, max_value, branch):
    if branch == "gt":
        lo, hi = (255, 245, 242), (196, 62, 50)
    else:
        lo, hi = (239, 250, 243), (48, 179, 107)
    t = max(0.0, min(1.0, value / max_value))
    return tuple(round(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))


def figure_context_cue_detail(summary, out_path):
    width, height = 3000, 860
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Context: Cue-Level Detail", width)
    cues = [("value_relevant", "Belief"), ("impression_relevant", "Identity"), ("outcome_relevant", "Stake")]

    def panel(x, title, branch, max_value):
        w = 1390
        draw.rounded_rectangle((x, 150, x + w, 790), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, 184), title, fnt=PANEL_TITLE, anchor="ma")
        cell_w, cell_h = 124, 112
        grid_x, grid_y = x + 190, 285
        for ci, model in enumerate(MODEL_ORDER):
            draw.multiline_text(
                (grid_x + ci * cell_w + cell_w / 2, 220),
                context_plot.COLUMN_LABELS[model],
                font=SMALL,
                fill=INK,
                anchor="ma",
                align="center",
                spacing=1,
            )
        for ri, (cue, cue_label) in enumerate(cues):
            y = grid_y + ri * (cell_h + 14)
            draw_text(draw, (grid_x - 22, y + cell_h / 2), cue_label, fill=TONE_COLORS[branch], fnt=LABEL, anchor="rm")
            for ci, model in enumerate(MODEL_ORDER):
                row = summary["models"][model]
                if branch == "gt":
                    value = pct(row["gt"]["by_cue"][cue]["correct_to_incorrect_rate"])
                else:
                    value = pct(row["ngt"]["by_cue"][cue]["user_answer_agreement"]["lift"])
                x0 = grid_x + ci * cell_w
                draw.rounded_rectangle(
                    (x0, y, x0 + cell_w - 10, y + cell_h),
                    radius=13,
                    fill=heat_color(value, max_value, branch),
                )
                draw_text(draw, (x0 + cell_w / 2 - 5, y + cell_h / 2), f"{value:.0f}", fnt=CELL, anchor="mm")

    panel(70, "OBJ truth departure", "gt", 80)
    panel(1540, "SUB user-view lift", "ngt", 45)
    save(im, out_path)


def figure_trigger_confidence_by_model(conf_rows, out_path):
    width, height = 2000, 980
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Trigger: Final Confidence by Model", width)
    cols = [
        ("OBJ preserved", "GT", "preserved"),
        ("OBJ departed", "GT", "departed"),
        ("SUB held", "NGT", "held"),
        ("SUB switched", "NGT", "switched"),
    ]
    x0, y0 = 360, 210
    cell_w, cell_h = 310, 66
    for ci, (label, _, _) in enumerate(cols):
        draw_text(draw, (x0 + ci * cell_w + cell_w / 2, y0 - 48), label, fnt=LABEL, anchor="ma")
    for ri, model in enumerate(trigger_plot.MODELS):
        y = y0 + ri * (cell_h + 12)
        draw_text(draw, (70, y + cell_h / 2), trigger_plot.MODEL_LABELS[model], fnt=SMALL, anchor="lm")
        for ci, (_, branch, category) in enumerate(cols):
            values = [
                row
                for row in conf_rows
                if row["model"] == model and row["branch"] == branch and row["category"] == category and row["turn"] == 3
            ]
            denom = sum(row["n"] for row in values)
            mean = sum(row["mean_confidence"] * row["n"] for row in values) / denom if denom else 0.0
            t = max(0.0, min(1.0, (mean - 2.5) / 2.5))
            fill = tuple(round(245 + (54 - 245) * t) for _ in range(1))
            color = (255 - round(64 * t), 245 - round(110 * t), 232 - round(158 * t))
            x = x0 + ci * cell_w
            draw.rounded_rectangle((x, y, x + cell_w - 14, y + cell_h), radius=12, fill=color)
            draw_text(draw, (x + cell_w / 2 - 7, y + cell_h / 2), f"{mean:.2f}", fnt=CELL, anchor="mm")
    save(im, out_path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def figure_judge_agreement(trigger_summary, context_summary, out_path):
    width, height = 2300, 900
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Judge Agreement Diagnostics", width)

    def bar_panel(x, y, w, h, title, rows, max_value=1.0):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, y + 34), title, fnt=PANEL_TITLE, anchor="ma")
        label_w = 270
        axis_x0, axis_x1 = x + label_w, x + w - 60
        axis_y0, axis_y1 = y + 110, y + h - 90
        for tick in [0, 0.5, 1.0]:
            xx = axis_x0 + (axis_x1 - axis_x0) * tick / max_value
            draw.line((xx, axis_y0, xx, axis_y1), fill=GRID, width=1)
            draw_text(draw, (xx, axis_y1 + 18), f"{tick:.1f}", fill=MUTED, fnt=SMALL, anchor="ma")
        row_h = (axis_y1 - axis_y0) / len(rows)
        for ri, row in enumerate(rows):
            yc = axis_y0 + ri * row_h + row_h / 2
            draw_text(draw, (x + 25, yc), row["label"], fnt=SMALL, anchor="lm")
            for bi, (key, color, offset) in enumerate([("trigger", "#CC79A7", -10), ("context", "#009E73", 16)]):
                value = row[key]
                yy = yc + offset
                xx = axis_x0 + (axis_x1 - axis_x0) * value / max_value
                draw.rounded_rectangle((axis_x0, yy - 10, xx, yy + 10), radius=5, fill=color)
                draw_text(draw, (xx + 8, yy), f"{value:.2f}", fill=color, fnt=SMALL, anchor="lm")
        legend_y = y + h - 38
        for lx, label, color in [(axis_x0, "Trigger", "#CC79A7"), (axis_x0 + 185, "Context", "#009E73")]:
            draw.rounded_rectangle((lx, legend_y - 10, lx + 32, legend_y + 10), radius=5, fill=color)
            draw_text(draw, (lx + 42, legend_y), label, fnt=SMALL, anchor="lm")

    factors = [
        ("Uncritical", "uncritical_agreement"),
        ("Obsequious", "obsequiousness"),
        ("Excitement", "excitement"),
    ]
    factor_rows = [
        {
            "label": label,
            "trigger": trigger_summary["factor_scores"][key]["pearson"],
            "context": context_summary["factor_scores"][key]["pearson"],
        }
        for label, key in factors
    ]
    binary = [
        ("Redo", "redo_question_by_reasoning_or_calculation"),
        ("Rationalized", "rationalized_change"),
        ("Contradicted", "contradicted_itself"),
    ]
    binary_rows = [
        {
            "label": label,
            "trigger": trigger_summary["binary"][key]["cohen_kappa"],
            "context": context_summary["binary"][key]["cohen_kappa"],
        }
        for label, key in binary
    ]
    bar_panel(70, 145, 1045, 680, "Factor-score correlation", factor_rows)
    bar_panel(1185, 145, 1045, 680, "Binary-label Cohen kappa", binary_rows)
    save(im, out_path)


def figure_temporal_strategy_by_model(rows, out_path):
    width, height = 2700, 1110
    im = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(im)
    draw_header(draw, "Temporal Strategy Comparison", width)

    stages = [
        ("single", "Single", "#0072B2"),
        ("same_family", "Same-family x3", "#009E73"),
        ("heterogeneous", "Mixed x3", "#CC79A7"),
    ]

    def lookup(branch, model, stage):
        for row in rows:
            if row["branch"] == branch and row["model"] == model and row["mode"] == "adaptive" and row["stage"] == stage:
                return row["rate"] * 100.0
        raise KeyError((branch, model, stage))

    def panel(x, y, w, h, branch, title, x_max, ticks):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#D8E0EA", width=2)
        draw_text(draw, (x + w / 2, y + 36), title, fnt=PANEL_TITLE, anchor="ma")
        label_w = 245
        axis_x0 = x + label_w
        axis_x1 = x + w - 74
        axis_y0 = y + 120
        axis_y1 = y + h - 92
        for tick in ticks:
            xx = axis_x0 + (axis_x1 - axis_x0) * tick / x_max
            draw.line((xx, axis_y0, xx, axis_y1), fill=GRID, width=1)
            draw_text(draw, (xx, axis_y1 + 18), f"{int(tick)}", fill=MUTED, fnt=SMALL, anchor="ma")
        row_h = (axis_y1 - axis_y0) / len(trigger_plot.MODELS)
        for ri, model in enumerate(trigger_plot.MODELS):
            yc = axis_y0 + ri * row_h + row_h / 2
            draw_text(draw, (x + 24, yc), trigger_plot.MODEL_SHORT_LABELS[model].replace("\n", " "), fnt=SMALL, anchor="lm")
            values = [lookup(branch, model, stage) for stage, _, _ in stages]
            xs = [axis_x0 + (axis_x1 - axis_x0) * value / x_max for value in values]
            draw.line((min(xs), yc, max(xs), yc), fill="#D7DEE8", width=7)
            for (stage, label, color), xx, value in zip(stages, xs, values):
                radius = 13 if stage == "heterogeneous" else 10
                draw.ellipse((xx - radius, yc - radius, xx + radius, yc + radius), fill=color, outline=WHITE, width=3)
            mixed_x = xs[-1]
            anchor = "lm" if mixed_x < axis_x1 - 55 else "rm"
            text_x = mixed_x + 18 if anchor == "lm" else mixed_x - 18
            draw_text(draw, (text_x, yc), f"{values[-1]:.1f}", fill=stages[-1][2], fnt=SMALL, anchor=anchor)
        draw.line((axis_x0, axis_y1, axis_x1, axis_y1), fill="#9AA7B7", width=3)
        draw_text(draw, (axis_x0 + (axis_x1 - axis_x0) / 2, y + h - 34), "Final-state rate (%)", fnt=LABEL, anchor="ma")

    panel(70, 150, 1260, 860, "GT", "OBJ right-to-wrong", 55, [0, 15, 30, 45])
    panel(1370, 150, 1260, 860, "NGT", "SUB switching", 95, [0, 25, 50, 75, 95])

    legend_y = 120
    legend_x = 835
    for stage, label, color in stages:
        draw.ellipse((legend_x, legend_y - 11, legend_x + 22, legend_y + 11), fill=color)
        draw_text(draw, (legend_x + 35, legend_y), label, fill=MUTED, fnt=SMALL, anchor="lm")
        legend_x += 300
    save(im, out_path)


def generate_trigger_appendix(results_dir, run_id, out_dir):
    records = trigger_plot.collect_records(Path(results_dir), run_id)
    headline, tone, family, sequence = trigger_plot.build_tables(records)
    detailed = trigger_plot.build_trigger_figure_tables(records)
    outputs = []

    specs = [
        ("appendix_trigger_headline_rates.png", trigger_plot.figure_headline, headline),
        ("appendix_trigger_family_landscape.png", trigger_plot.figure_family_heatmap, family),
    ]
    for filename, fn, rows in specs:
        path = out_dir / filename
        fn(path, rows)
        outputs.append(path)

    conf_path = out_dir / "appendix_trigger_confidence_by_model.png"
    figure_trigger_confidence_by_model(detailed["confidence_trajectory"], conf_path)
    outputs.append(conf_path)
    temporal_path = out_dir / "appendix_trigger_temporal_strategy.png"
    figure_temporal_strategy_by_model(detailed["temporal_pressure"], temporal_path)
    outputs.append(temporal_path)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-summary", type=Path, required=True)
    parser.add_argument("--trigger-run-id", default="trigger_20260504_070840")
    parser.add_argument("--trigger-results-dir", type=Path, required=True)
    parser.add_argument("--judge-trigger-summary", type=Path)
    parser.add_argument("--judge-context-summary", type=Path)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "images" / "results" / "appendix")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_context_summary(args.context_summary)

    outputs = []
    for filename, fn in [
        ("appendix_context_model_bars.png", figure_context_model_bars),
        ("appendix_context_cue_detail.png", figure_context_cue_detail),
    ]:
        path = args.out_dir / filename
        fn(summary, path)
        outputs.append(path)
    outputs.extend(generate_trigger_appendix(args.trigger_results_dir, args.trigger_run_id, args.out_dir))
    if args.judge_trigger_summary and args.judge_context_summary:
        judge_path = args.out_dir / "appendix_judge_agreement.png"
        figure_judge_agreement(load_json(args.judge_trigger_summary), load_json(args.judge_context_summary), judge_path)
        outputs.append(judge_path)

    extra_diagnostics = [
        "context_change_decomposition.png",
        "context_confidence_outcome.png",
        "context_gt_source_decomposition.png",
        "context_ngt_directionality.png",
        "context_ngt_domain_cue.png",
        "judge_mechanism_outcome.png",
        "judge_reliability_triage.png",
        "temporal_state_paths.png",
        "trigger_adaptive_family_lift.png",
        "trigger_confidence_high_risk.png",
        "trigger_confidence_risk_calibration.png",
        "trigger_domain_source_susceptibility.png",
        "trigger_item_concentration.png",
        "validation_coverage_funnel.png",
    ]
    output_names = [path.name for path in outputs]
    appendix_names = output_names + [
        name for name in extra_diagnostics if name not in output_names and (args.out_dir / name).exists()
    ]
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Appendix Result Figures",
                "",
                "All figures in this directory are generated from the latest official raw result files:",
                f"- context summary: `{args.context_summary}`",
                f"- trigger run: `{args.trigger_run_id}` under `{args.trigger_results_dir}`",
                "",
                "These figures are appendix-only diagnostics and do not replace the main-text figures.",
                "",
                *[f"- `{name}`" for name in appendix_names],
                "",
                "All generated appendix diagnostics are directly referenced from `sections/appendix.tex`; there is no separate draft figure pool.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"figures": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
