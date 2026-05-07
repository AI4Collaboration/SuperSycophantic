"""Generate trigger_confidence_trajectory.png."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

from _common import default_figure_path, ensure_parent

import plot_trigger_figures as trigger_plot


INK = "#17202A"
MUTED = "#566679"
GRID = "#DCE5EE"


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
    cleaned = cell.replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        raise ValueError(f"Cannot parse numeric cell: {cell}")
    return float(match.group(0))


def load_confidence_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing appendix source: {path}")
    tex = path.read_text(encoding="utf-8")
    rows = []
    for row in data_rows(table_block(tex, "tab:trigger_confidence_rates")):
        if len(row) != 6:
            raise ValueError(f"Unexpected confidence table row: {row}")
        rows.append(
            {
                "setting": row[0],
                "state": row[1],
                "initial": parse_float(row[2]),
                "final": parse_float(row[3]),
                "n": int(parse_float(row[5])),
            }
        )
    expected = {("OBJ", "Preserved"), ("OBJ", "Departed"), ("SUB", "Held"), ("SUB", "Switched")}
    observed = {(row["setting"], row["state"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Unexpected confidence rows: {sorted(observed)}")
    return rows


def figure_from_appendix(rows, out_path: Path) -> None:
    width, height = 2500, 910
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    panel_title = trigger_plot.load_font(45, True)
    axis_font = trigger_plot.load_font(33)
    axis_bold = trigger_plot.load_font(35, True)
    label_font = trigger_plot.load_font(34, True)

    colors = {
        "Preserved": "#2D8B57",
        "Departed": "#C84C4C",
        "Held": "#1F7EBB",
        "Switched": "#D56A00",
    }

    def lookup(setting, state):
        for row in rows:
            if row["setting"] == setting and row["state"] == state:
                return row
        raise ValueError((setting, state))

    def panel(x0, y0, w, h, setting, states):
        nonlocal draw
        draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=20, fill="#F8FAFC", outline="#D9E0E8", width=2)
        draw_text(draw, (x0 + w / 2, y0 + 46), f"{setting} final-state groups", panel_title, anchor="ma")
        plot_x0 = x0 + 150
        plot_x1 = x0 + w - 150
        plot_y0 = y0 + 150
        plot_y1 = y0 + h - 125
        y_min, y_max = 3.0, 5.0

        def y_pos(value):
            return plot_y1 - (plot_y1 - plot_y0) * (value - y_min) / (y_max - y_min)

        x_initial = plot_x0 + 110
        x_final = plot_x1 - 110
        for tick in [3.0, 3.5, 4.0, 4.5, 5.0]:
            yy = y_pos(tick)
            draw.line((plot_x0, yy, plot_x1, yy), fill=GRID, width=2)
            draw_text(draw, (plot_x0 - 18, yy), f"{tick:.1f}", axis_font, MUTED, anchor="rm")
        for xx, label in [(x_initial, "Initial"), (x_final, "Final")]:
            draw.line((xx, plot_y0, xx, plot_y1), fill="#EEF2F6", width=2)
            draw_text(draw, (xx, plot_y1 + 44), label, axis_bold, INK, anchor="ma")
        draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#9AA7B7", width=4)
        draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="#9AA7B7", width=4)

        legend_x = x0 + w - 365
        legend_y = y0 + 100
        for i, state in enumerate(states):
            yy = legend_y + i * 42
            color = colors[state]
            draw.line((legend_x, yy, legend_x + 48, yy), fill=color, width=8)
            draw.ellipse((legend_x + 18, yy - 8, legend_x + 34, yy + 8), fill=color, outline="white", width=2)
            draw_text(draw, (legend_x + 64, yy), state, label_font, color, anchor="lm")

        for state in states:
            row = lookup(setting, state)
            color = colors[state]
            pts = [(x_initial, y_pos(row["initial"])), (x_final, y_pos(row["final"]))]
            draw.line(pts, fill=color, width=9)
            for xx, yy in pts:
                draw.ellipse((xx - 13, yy - 13, xx + 13, yy + 13), fill=color, outline="white", width=3)
            draw_text(draw, (x_final + 24, pts[-1][1]), f"{row['final']:.2f}", label_font, color, anchor="lm")

        draw_text(draw, ((plot_x0 + plot_x1) / 2, plot_y1 + 86), "Assistant turn", axis_bold, INK, anchor="ma")
        trigger_plot.draw_rotated_label(
            im,
            (plot_x0 - 105, (plot_y0 + plot_y1) / 2),
            "Mean self-rated confidence (1-5)",
            axis_bold,
        )
        draw = ImageDraw.Draw(im)

    panel(70, 30, 1150, 785, "OBJ", ["Preserved", "Departed"])
    panel(1238, 30, 1150, 785, "SUB", ["Held", "Switched"])

    trigger_plot.save_tight(im, out_path, padding=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the main-text trigger confidence trajectory figure.")
    parser.add_argument("--run-id", required=True, help="Current trigger run id under --results-dir.")
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--appendix-tex",
        type=Path,
        help="Optional appendix TeX source. When provided, generate the initial-to-final confidence figure from tab:trigger_confidence_rates.",
    )
    args = parser.parse_args()

    out = ensure_parent(args.out or default_figure_path(args.run_id, "trigger_confidence_trajectory.png"))
    if args.appendix_tex:
        figure_from_appendix(load_confidence_rows(args.appendix_tex), out)
    else:
        records = trigger_plot.collect_records(args.results_dir, args.run_id)
        tables = trigger_plot.build_trigger_figure_tables(records)
        trigger_plot.figure_confidence_trajectory(out, tables["confidence_trajectory"])
    print(out)


if __name__ == "__main__":
    main()
