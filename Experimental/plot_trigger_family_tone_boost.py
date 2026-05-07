import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

try:
    import plot_trigger_figures as trigger_plot
except ImportError:
    from Experimental import plot_trigger_figures as trigger_plot


REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = [
    "authority",
    "social_proof",
    "consistency",
    "liking",
    "reciprocity",
    "scarcity",
    "unity",
]
FAMILY_LABELS = {
    "authority": "Authority",
    "social_proof": "Social",
    "consistency": "Consistency",
    "liking": "Liking",
    "reciprocity": "Reciprocity",
    "scarcity": "Scarcity",
    "unity": "Unity",
}
FAMILY_COLORS = {
    "authority": "#0072B2",
    "social_proof": "#009E73",
    "consistency": "#CC79A7",
    "liking": "#E69F00",
    "reciprocity": "#D55E00",
    "scarcity": "#56B4E9",
    "unity": "#6A3D9A",
}
INK = "#17202A"
MUTED = "#637083"
GRID = "#E4E9F0"
PAPER = "#FFFFFF"
BOOST_MODERATE = "#0072B2"
BOOST_STRONG = "#D62728"


def font(size, bold=False):
    return trigger_plot.load_font(size, bold)


def draw_text(draw, xy, text, size, fill=INK, bold=False, anchor=None):
    draw.text(xy, str(text), font=font(size, bold), fill=fill, anchor=anchor)


def y_at(rate, y0, h, max_rate):
    return y0 + h - h * min(max(rate, 0.0), max_rate) / max_rate


def tone_arrow(draw, x, y_from, y_to, color):
    if abs(y_to - y_from) < 8:
        return
    if y_to < y_from:
        shaft_end = y_to + 22
        draw.line((x, y_from, x, shaft_end), fill=color, width=10)
        head = [(x, y_to), (x - 18, y_to + 32), (x + 18, y_to + 32)]
        draw.polygon(head, fill=color)
    else:
        shaft_end = y_to - 22
        draw.line((x, y_from, x, shaft_end), fill=color, width=10)
        head = [(x, y_to), (x - 18, y_to - 32), (x + 18, y_to - 32)]
        draw.polygon(head, fill=color)


def tone_step(draw, x, y_mild, y_moderate, y_strong):
    tone_arrow(draw, x, y_mild, y_moderate, BOOST_MODERATE)
    tone_arrow(draw, x, y_moderate, y_strong, BOOST_STRONG)


def aggregate_tone(results_dir, run_id, mode):
    modes = ["static", "adaptive"] if mode == "all" else [mode]
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"denom": 0, "events": 0})))
    for selected_mode in modes:
        for branch in ["gt", "ngt"]:
            path = results_dir / f"{run_id}_{branch}_trigger_{selected_mode}.jsonl.gz"
            for record in trigger_plot.read_jsonl_gz(path):
                family = record.get("trigger")
                tone = record.get("tone")
                if record.get("eligible") and family in FAMILIES and tone in {"mild", "moderate", "strong"}:
                    cell = grouped[(record["model"], family)][tone][branch]
                    cell["denom"] += 1
                    if branch == "gt":
                        event = bool(record.get("single_trigger_truth_departure"))
                    else:
                        event = bool(record.get("single_trigger_answer_switch"))
                    cell["events"] += int(event)
    out = {}
    for key, by_tone in grouped.items():
        rates = {}
        denoms = []
        for tone in ["mild", "moderate", "strong"]:
            branch_rates = []
            for branch in ["gt", "ngt"]:
                cell = by_tone[tone][branch]
                if cell["denom"]:
                    branch_rates.append(cell["events"] / cell["denom"])
                    denoms.append(cell["denom"])
            rates[tone] = sum(branch_rates) / len(branch_rates) if branch_rates else 0.0
        mild_rate = rates["mild"]
        moderate_rate = rates["moderate"]
        strong_rate = rates["strong"]
        out[key] = {
            "mild": mild_rate,
            "moderate": moderate_rate,
            "strong": strong_rate,
            "delta_moderate": moderate_rate - mild_rate,
            "delta_strong": strong_rate - moderate_rate,
            "delta_total": strong_rate - mild_rate,
            "denom": min(denoms) if denoms else 0,
        }
    return out


def draw_legend(draw, x, y):
    col_w = 265
    row_h = 42
    for i, family in enumerate(FAMILIES):
        col = i % 4
        row = i // 4
        xx = x + col * col_w
        yy = y + row * row_h
        draw.rounded_rectangle((xx, yy, xx + 36, yy + 26), radius=6, fill=FAMILY_COLORS[family])
        draw_text(draw, (xx + 48, yy + 13), FAMILY_LABELS[family], 30, anchor="lm")
    arrow_y = y + row_h * 2 + 16
    for i, (label, color) in enumerate([("Mild to moderate", BOOST_MODERATE), ("Moderate to strong", BOOST_STRONG)]):
        xx = x + i * 510
        draw.line((xx, arrow_y + 14, xx + 78, arrow_y + 14), fill=color, width=10)
        draw.polygon([(xx + 102, arrow_y + 14), (xx + 72, arrow_y - 3), (xx + 72, arrow_y + 31)], fill=color)
        draw_text(draw, (xx + 122, arrow_y + 14), label, 30, anchor="lm")


def draw_model_label(im, draw, model, cx, y):
    badge = trigger_plot.make_badge(model)
    badge.thumbnail((78, 78), Image.Resampling.LANCZOS)
    im.paste(badge, (int(cx - badge.width / 2), int(y)), badge)
    label = trigger_plot.MODEL_SHORT_LABELS[model].replace("\n", " ")
    if len(label) > 12:
        parts = label.split()
        label = " ".join(parts[:1]) + "\n" + " ".join(parts[1:])
    draw.multiline_text((cx, y + 78), label, font=font(36, True), fill=INK, anchor="ma", align="center", spacing=3)


def draw_figure(rows, out_path):
    width, height = 3900, 1160
    im = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(im)

    draw.rounded_rectangle((70, 22, width - 70, height - 34), radius=26, fill="#F8FAFC", outline="#D9E0E8", width=2)
    draw_text(draw, (width / 2, 82), "Tone Is the Most Influential Trigger Factor", 66, bold=True, anchor="mm")

    draw_legend(draw, width - 1215, 132)

    x0, y0 = 205, 300
    plot_w, plot_h = width - 360, 595
    max_rate = 0.75
    bottom = y0 + plot_h

    for tick in [0, 0.25, 0.50, 0.75]:
        yy = y_at(tick, y0, plot_h, max_rate)
        draw.line((x0, yy, x0 + plot_w, yy), fill=GRID, width=2 if tick else 4)
        draw_text(draw, (x0 - 22, yy), f"{int(tick * 100)}", 38, MUTED, bold=tick in [0, 0.75], anchor="rm")
    draw.line((x0, y0, x0, bottom), fill="#B7C1CE", width=4)
    trigger_plot.draw_rotated_label(
        im,
        (x0 - 72, y0 + plot_h / 2),
        "Sycophancy rate (%)",
        font(40, True),
    )
    draw = ImageDraw.Draw(im)

    cluster_w = plot_w / len(trigger_plot.MODELS)
    bar_w = 34
    bar_gap = 13
    group_w = len(FAMILIES) * bar_w + (len(FAMILIES) - 1) * bar_gap

    for mi, model in enumerate(trigger_plot.MODELS):
        cx = x0 + mi * cluster_w + cluster_w / 2
        gx = cx - group_w / 2
        if mi:
            xx = x0 + mi * cluster_w
            draw.line((xx, y0 + 18, xx, bottom + 18), fill="#F0F3F7", width=2)
        for fi, family in enumerate(FAMILIES):
            value = rows.get((model, family))
            if value is None or value["denom"] <= 0:
                raise ValueError(f"Missing tone aggregate for model={model}, family={family}")
            x = gx + fi * (bar_w + bar_gap)
            y_mild = y_at(value["mild"], y0, plot_h, max_rate)
            y_moderate = y_at(value["moderate"], y0, plot_h, max_rate)
            y_strong = y_at(value["strong"], y0, plot_h, max_rate)
            fill = FAMILY_COLORS[family]
            draw.rounded_rectangle((x, y_mild, x + bar_w, bottom), radius=5, fill=fill)
            draw.line((x, bottom, x + bar_w, bottom), fill="#9DA8B5", width=2)
            tone_step(draw, x + bar_w / 2, y_mild, y_moderate, y_strong)
        draw_model_label(im, draw, model, cx, bottom + 54)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trigger_plot.save_tight(im, out_path, padding=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Current trigger run id under --results-dir.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Experimental" / "results")
    parser.add_argument("--mode", choices=["static", "adaptive", "all"], default="all")
    parser.add_argument("--out-png", type=Path, default=REPO_ROOT / "Experimental" / "reports" / "trigger_family_tone_boost.png")
    parser.add_argument("--out-pdf", type=Path)
    args = parser.parse_args()

    rows = aggregate_tone(args.results_dir, args.run_id, args.mode)
    draw_figure(rows, args.out_png)
    if args.out_pdf:
        args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
        Image.open(args.out_png).convert("RGB").save(args.out_pdf)


if __name__ == "__main__":
    main()
