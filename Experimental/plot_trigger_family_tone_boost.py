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
    "authority": "#4B8B6A",
    "social_proof": "#2D86B8",
    "consistency": "#22A699",
    "liking": "#F2A340",
    "reciprocity": "#E64B3C",
    "scarcity": "#8E63B6",
    "unity": "#D8B13F",
}
INK = "#17202A"
MUTED = "#637083"
GRID = "#E4E9F0"
PAPER = "#FFFFFF"
BOOST_MODERATE = "#FFD84D"
BOOST_STRONG = "#F7A63B"


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
        shaft_end = y_to + 18
        draw.line((x, y_from, x, shaft_end), fill=color, width=7)
        head = [(x, y_to), (x - 14, y_to + 25), (x + 14, y_to + 25)]
        draw.polygon(head, fill=color)
    else:
        shaft_end = y_to - 18
        draw.line((x, y_from, x, shaft_end), fill=color, width=7)
        head = [(x, y_to), (x - 14, y_to - 25), (x + 14, y_to - 25)]
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
    for i, family in enumerate(FAMILIES):
        xx = x + i * 310
        yy = y
        draw.rounded_rectangle((xx, yy, xx + 36, yy + 28), radius=6, fill=FAMILY_COLORS[family])
        draw_text(draw, (xx + 50, yy + 14), FAMILY_LABELS[family], 27, anchor="lm")
    arrow_y = y
    for i, (label, color) in enumerate([("Mild to moderate", BOOST_MODERATE), ("Moderate to strong", BOOST_STRONG)]):
        xx = x + 2225 + i * 595
        draw.line((xx, arrow_y + 15, xx + 84, arrow_y + 15), fill=color, width=8)
        draw.polygon([(xx + 104, arrow_y + 15), (xx + 76, arrow_y), (xx + 76, arrow_y + 30)], fill=color)
        draw_text(draw, (xx + 126, arrow_y + 15), label, 27, anchor="lm")


def draw_model_label(im, draw, model, cx, y):
    badge = trigger_plot.make_badge(model)
    badge.thumbnail((62, 62), Image.Resampling.LANCZOS)
    im.paste(badge, (int(cx - badge.width / 2), int(y)), badge)
    label = trigger_plot.MODEL_SHORT_LABELS[model].replace("\n", " ")
    if len(label) > 12:
        parts = label.split()
        label = " ".join(parts[:1]) + "\n" + " ".join(parts[1:])
    draw.multiline_text((cx, y + 78), label, font=font(25, True), fill=INK, anchor="ma", align="center", spacing=3)


def draw_figure(rows, out_path):
    width, height = 3900, 1248
    im = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(im)

    draw.rounded_rectangle((70, 22, width - 70, height - 34), radius=26, fill="#F8FAFC", outline="#D9E0E8", width=2)
    draw_text(draw, (width / 2, 82), "Tone Is the Most Influential Trigger Factor", 66, bold=True, anchor="mm")

    x0, y0 = 190, 210
    plot_w, plot_h = width - 330, 680
    max_rate = 1.08
    bottom = y0 + plot_h

    for tick in [0, 0.25, 0.50, 0.75, 1.0]:
        yy = y_at(tick, y0, plot_h, max_rate)
        draw.line((x0, yy, x0 + plot_w, yy), fill=GRID, width=2 if tick else 4)
        draw_text(draw, (x0 - 22, yy), f"{int(tick * 100)}", 26, MUTED, bold=tick in [0, 1.0], anchor="rm")
    draw.line((x0, y0, x0, bottom), fill="#B7C1CE", width=4)
    draw_text(draw, (x0, y0 - 36), "GT+NGT sycophantic rate (%)", 31, INK, bold=True, anchor="la")

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
            value = rows.get(
                (model, family),
                {"mild": 0, "moderate": 0, "strong": 0, "delta_moderate": 0, "delta_strong": 0, "delta_total": 0, "denom": 0},
            )
            x = gx + fi * (bar_w + bar_gap)
            y_mild = y_at(value["mild"], y0, plot_h, max_rate)
            y_moderate = y_at(value["moderate"], y0, plot_h, max_rate)
            y_strong = y_at(value["strong"], y0, plot_h, max_rate)
            fill = FAMILY_COLORS[family]
            draw.rounded_rectangle((x, y_mild, x + bar_w, bottom), radius=5, fill=fill)
            draw.line((x, bottom, x + bar_w, bottom), fill="#9DA8B5", width=2)
            tone_step(draw, x + bar_w / 2, y_mild, y_moderate, y_strong)
        draw_model_label(im, draw, model, cx, bottom + 54)

    draw_legend(draw, 300, 1165)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trigger_plot.save_tight(im, out_path, padding=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="trigger_20260504_070840")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Experimental" / "results")
    parser.add_argument("--mode", choices=["static", "adaptive", "all"], default="all")
    parser.add_argument("--out-png", type=Path, default=REPO_ROOT / "images" / "results" / "trigger_family_tone_boost.png")
    parser.add_argument("--out-pdf", type=Path, default=REPO_ROOT / "images" / "results" / "trigger_family_tone_boost.pdf")
    args = parser.parse_args()

    rows = aggregate_tone(args.results_dir, args.run_id, args.mode)
    draw_figure(rows, args.out_png)
    Image.open(args.out_png).convert("RGB").save(args.out_pdf)


if __name__ == "__main__":
    main()
