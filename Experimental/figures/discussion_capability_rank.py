"""Generate discussion_capability_rank.png from appendix aggregate tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

from _common import ensure_parent

import plot_trigger_figures as trigger_plot


INK = "#17202A"
MUTED = "#566679"
GRID = "#DCE5EE"
RED = "#D94C35"
BLUE = "#1F7EBB"

DISPLAY_NAMES = {
    "GPT-5.4": "GPT-5.4",
    "GPT-5.4-Mini": "GPT-5.4-mini",
    "GPT-5.4-Nano": "GPT-5.4-nano",
    "Opus-4.5": "Opus-4.5",
    "Sonnet-4.5": "Sonnet-4.5",
    "Haiku-4.5": "Haiku-4.5",
    "Gemini-3.1-Flash-Lite": "Gemini-3.1-Flash-Lite",
    "Mistral-Medium-3.1": "Mistral-Medium-3.1",
    "Command-R": "Command-R",
}


def draw_text(draw, xy, text, font, fill=INK, anchor=None):
    draw.text(xy, str(text), font=font, fill=fill, anchor=anchor)


def table_block(tex: str, label: str) -> str:
    marker = f"\\label{{{label}}}"
    label_index = tex.find(marker)
    if label_index < 0:
        raise ValueError(f"Cannot find appendix table label: {label}")
    start = tex.rfind("\\begin{tabular}", 0, label_index)
    end = tex.find("\\end{tabular}", start)
    if start < 0 or end < 0:
        raise ValueError(f"Cannot find tabular block for: {label}")
    return tex[start:end]


def data_rows(block: str) -> list[list[str]]:
    rows = []
    in_body = False
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line == "\\midrule":
            in_body = True
            continue
        if line == "\\bottomrule":
            break
        if not in_body or "&" not in line:
            continue
        line = re.sub(r"\\\\\s*$", "", line)
        rows.append([cell.strip() for cell in line.split("&")])
    return rows


def parse_float(cell: str) -> float:
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", cell)
    cleaned = re.sub(r"\$.*?\$", "", cleaned)
    cleaned = cleaned.replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError(f"Cannot parse numeric cell: {cell}")
    return float(match.group(0))


def load_models_from_appendix(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing appendix source: {path}")
    tex = path.read_text(encoding="utf-8")

    context = {}
    for row in data_rows(table_block(tex, "tab:context_model_rates")):
        if len(row) != 6:
            raise ValueError(f"Unexpected context table row: {row}")
        context[row[0]] = {
            "context_obj": parse_float(row[3]),
            "context_sub": parse_float(row[5]),
        }

    trigger = {}
    for row in data_rows(table_block(tex, "tab:trigger_model_rates")):
        if len(row) != 4:
            raise ValueError(f"Unexpected trigger table row: {row}")
        trigger[row[0]] = {
            "trigger_obj": parse_float(row[2]),
            "trigger_sub": parse_float(row[3]),
        }

    models = []
    for row in data_rows(table_block(tex, "tab:capability_rank")):
        if len(row) != 4:
            raise ValueError(f"Unexpected capability-rank row: {row}")
        rank = int(parse_float(row[0]))
        name = row[1]
        if name not in context or name not in trigger:
            raise ValueError(f"Capability model missing from aggregate tables: {name}")
        models.append(
            {
                "name": DISPLAY_NAMES.get(name, name),
                "rank": rank,
                **context[name],
                **trigger[name],
            }
        )
    return sorted(models, key=lambda row: row["rank"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate capability-rank discussion figure.")
    parser.add_argument("--out", type=Path, default=Path("Experimental/reports/discussion_capability_rank.png"))
    parser.add_argument(
        "--appendix-tex",
        type=Path,
        default=Path("sections/appendix.tex"),
        help="Appendix TeX file containing context, trigger, and capability-rank aggregate tables.",
    )
    args = parser.parse_args()
    models = load_models_from_appendix(args.appendix_tex)

    width, height = 3300, 1520
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    title = trigger_plot.load_font(62, True)
    panel_title = trigger_plot.load_font(52, True)
    label_font = trigger_plot.load_font(34)
    axis_font = trigger_plot.load_font(34, True)
    axis_small = trigger_plot.load_font(30, True)
    tick_font = trigger_plot.load_font(32)
    draw_text(draw, (width / 2, 58), "Capability rank does not predict sycophancy risk", title, anchor="ma")
    draw.line((65, 130, width - 65, 130), fill="#EEF2F6", width=4)

    def panel(x0, y0, w, h, title_text, red_key, blue_key):
        draw_text(draw, (x0 + w / 2, y0 - 36), title_text, panel_title, anchor="ma")
        plot_x0 = x0 + 520
        plot_x1 = x0 + w - 80
        row_top = y0 + 35
        row_h = 96
        max_x = 90
        draw_text(
            draw,
            (plot_x0 - 24, row_top - 55),
            "Artificial Analysis rank (1 = strongest)",
            axis_small,
            INK,
            anchor="rs",
        )
        for tick in [0, 25, 50, 75]:
            xx = plot_x0 + (plot_x1 - plot_x0) * tick / max_x
            draw.line((xx, row_top - 38, xx, row_top + row_h * len(models) - 18), fill=GRID, width=2)
            draw_text(draw, (xx, row_top + row_h * len(models) + 6), f"{tick}%", tick_font, MUTED, anchor="ma")
        draw.line((plot_x0, row_top - 38, plot_x0, row_top + row_h * len(models) - 18), fill=MUTED, width=4)
        draw.line((plot_x0, row_top + row_h * len(models) - 18, plot_x1, row_top + row_h * len(models) - 18), fill=MUTED, width=4)
        draw_text(
            draw,
            (plot_x0 + (plot_x1 - plot_x0) / 2, row_top + row_h * len(models) + 38),
            "Sycophantic outcome rate (%)",
            axis_font,
            INK,
            anchor="ma",
        )

        for i, row in enumerate(models):
            name, rank = row["name"], row["rank"]
            red_value = row[red_key]
            blue_value = row[blue_key]
            y = row_top + i * row_h
            draw.line((plot_x0, y, plot_x1, y), fill="#EEF2F6", width=2)
            draw_text(draw, (plot_x0 - 24, y), f"{rank}  {name}", label_font, MUTED, anchor="rm")
            xr = plot_x0 + (plot_x1 - plot_x0) * red_value / max_x
            xb = plot_x0 + (plot_x1 - plot_x0) * blue_value / max_x
            draw.line((min(xr, xb), y, max(xr, xb), y), fill="#CCD6E2", width=6)
            draw.ellipse((xr - 16, y - 16, xr + 16, y + 16), fill=RED, outline="white", width=3)
            draw.ellipse((xb - 16, y - 16, xb + 16, y + 16), fill=BLUE, outline="white", width=3)

    panel(70, 225, 1540, 930, "Context framing", "context_obj", "context_sub")
    panel(1690, 225, 1540, 930, "Trigger pressure", "trigger_obj", "trigger_sub")

    out = ensure_parent(args.out)
    trigger_plot.save_tight(im, out, padding=12)
    print(out)


if __name__ == "__main__":
    main()
