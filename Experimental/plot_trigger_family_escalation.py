import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

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
BOOST = "#FFD84D"
BOOST_TEXT = "#08751B"
DROP = "#D23B35"


def font(size, bold=False):
    return trigger_plot.load_font(size, bold)


def draw_text(draw, xy, text, size, fill=INK, bold=False, anchor=None):
    draw.text(xy, str(text), font=font(size, bold), fill=fill, anchor=anchor)


def y_at(rate, y0, h, max_rate):
    return y0 + h - h * min(max(rate, 0.0), max_rate) / max_rate


def boost_arrow(draw, x, y_from, y_to):
    if y_to >= y_from - 8:
        return
    shaft_end = y_to + 24
    draw.line((x, y_from, x, shaft_end), fill=BOOST, width=13)
    head = [(x, y_to), (x - 24, y_to + 38), (x + 24, y_to + 38)]
    draw.polygon(head, fill=BOOST)


def aggregate_temporal(results_dir, run_id, mode):
    path = results_dir / f"{run_id}_ngt_trigger_temporal_{mode}.jsonl.gz"
    grouped = defaultdict(lambda: {"denom": 0, "first": 0, "final": 0})
    for record in trigger_plot.read_jsonl_gz(path):
        family = record.get("trigger")
        sequence = record.get("trigger_sequence") or []
        if (
            record.get("eligible")
            and family in FAMILIES
            and sequence
            and len(set(sequence)) == 1
            and sequence[0] == family
        ):
            cell = grouped[(record["model"], family)]
            cell["denom"] += 1
            cell["first"] += int(bool(record.get("single_trigger_answer_switch")))
            cell["final"] += int(bool(record.get("three_repetition_answer_switch")))
    out = {}
    for key, value in grouped.items():
        denom = value["denom"]
        out[key] = {
            "first": value["first"] / denom if denom else 0.0,
            "final": value["final"] / denom if denom else 0.0,
            "delta": (value["final"] - value["first"]) / denom if denom else 0.0,
            "denom": denom,
        }
    return out


def draw_legend(draw, x, y):
    for i, family in enumerate(FAMILIES):
        row = 0 if i < 4 else 1
        col = i if i < 4 else i - 4
        xx = x + col * 230
        yy = y + row * 40
        draw.rounded_rectangle((xx, yy, xx + 30, yy + 24), radius=5, fill=FAMILY_COLORS[family])
        draw_text(draw, (xx + 42, yy + 12), FAMILY_LABELS[family], 23, anchor="lm")


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
    width, height = 3900, 1380
    im = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(im)

    draw_text(draw, (width / 2, 38), "Trigger Escalation Moves Models Unevenly", 66, bold=True, anchor="ma")
    draw_text(
        draw,
        (width / 2, 112),
        "NGT flip under same-family mild to moderate to strong pressure; yellow arrows mark escalation boost",
        34,
        MUTED,
        anchor="ma",
    )
    draw.line((90, 172, width - 90, 172), fill=GRID, width=4)
    draw_legend(draw, width - 1040, 205)

    x0, y0 = 190, 295
    plot_w, plot_h = width - 330, 720
    max_rate = 1.0
    bottom = y0 + plot_h

    for tick in [0, 0.25, 0.50, 0.75, 1.0]:
        yy = y_at(tick, y0, plot_h, max_rate)
        draw.line((x0, yy, x0 + plot_w, yy), fill=GRID, width=2 if tick else 4)
        draw_text(draw, (x0 - 22, yy), f"{int(tick * 100)}", 26, MUTED, bold=tick in [0, 1.0], anchor="rm")
    draw.line((x0, y0, x0, bottom), fill="#B7C1CE", width=4)
    draw_text(draw, (x0, y0 - 36), "NGT flip rate (%)", 31, INK, bold=True, anchor="la")

    cluster_w = plot_w / len(trigger_plot.MODELS)
    bar_w = 37
    bar_gap = 9
    group_w = len(FAMILIES) * bar_w + (len(FAMILIES) - 1) * bar_gap

    for mi, model in enumerate(trigger_plot.MODELS):
        cx = x0 + mi * cluster_w + cluster_w / 2
        gx = cx - group_w / 2
        if mi:
            xx = x0 + mi * cluster_w
            draw.line((xx, y0 + 18, xx, bottom + 18), fill="#F0F3F7", width=2)
        for fi, family in enumerate(FAMILIES):
            value = rows.get((model, family), {"first": 0, "final": 0, "delta": 0, "denom": 0})
            x = gx + fi * (bar_w + bar_gap)
            y_final = y_at(value["final"], y0, plot_h, max_rate)
            y_first = y_at(value["first"], y0, plot_h, max_rate)
            fill = FAMILY_COLORS[family]
            draw.rounded_rectangle((x, y_final, x + bar_w, bottom), radius=5, fill=fill)
            draw.line((x, bottom, x + bar_w, bottom), fill="#9DA8B5", width=2)
            delta = value["delta"]
            if delta > 0.025:
                boost_arrow(draw, x + bar_w / 2, y_first, y_final)
            if abs(delta) >= 0.08:
                label = f"{delta * 100:+.0f}"
                label_y = y_final - 18 if delta >= 0 else max(y0 + 18, y_final - 18)
                label_fill = BOOST_TEXT if delta >= 0 else DROP
                draw_text(draw, (x + bar_w / 2, label_y), label, 18, label_fill, bold=True, anchor="mm")
        draw_model_label(im, draw, model, cx, bottom + 54)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="trigger_20260504_070840")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Experimental" / "results")
    parser.add_argument("--mode", choices=["static", "adaptive"], default="static")
    parser.add_argument("--out-png", type=Path, default=REPO_ROOT / "images" / "results" / "trigger_family_escalation.png")
    parser.add_argument("--out-pdf", type=Path, default=REPO_ROOT / "images" / "results" / "trigger_family_escalation.pdf")
    args = parser.parse_args()

    rows = aggregate_temporal(args.results_dir, args.run_id, args.mode)
    draw_figure(rows, args.out_png)
    Image.open(args.out_png).convert("RGB").save(args.out_pdf)


if __name__ == "__main__":
    main()
