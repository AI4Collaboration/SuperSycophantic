"""Generate trigger_tone_gradient_opus.png with Claude models separated."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from _common import DEFAULT_TRIGGER_RUN_ID, default_figure_path, ensure_parent

import plot_trigger_figures as trigger_plot


TITLE = "Claude often peaks before the strongest pressure"


def add_title_block(source: Path, out: Path) -> None:
    source_im = Image.open(source).convert("RGB")
    width, source_height = source_im.size
    top_pad = 132
    im = Image.new("RGB", (width, source_height + top_pad), trigger_plot.PAPER_BG)
    draw = ImageDraw.Draw(im)
    trigger_plot.draw_text(draw, (width / 2, 38), TITLE, trigger_plot.FONT_TITLE, anchor="ma")
    draw.line((65, 122, width - 65, 122), fill=trigger_plot.LIGHT_GRID, width=3)
    im.paste(source_im, (0, top_pad))
    trigger_plot.save_tight(im, out, padding=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Claude-vs-all-model-baseline trigger tone trajectory figure.")
    parser.add_argument("--run-id", default=DEFAULT_TRIGGER_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--source-image",
        type=Path,
        help="Existing generated figure to reuse when ignored raw trigger result files are unavailable.",
    )
    args = parser.parse_args()

    out = ensure_parent(args.out or default_figure_path(args.run_id, "trigger_tone_gradient_opus.png"))
    if args.source_image:
        add_title_block(args.source_image, out)
    else:
        records = trigger_plot.collect_records(args.results_dir, args.run_id)
        tables = trigger_plot.build_trigger_figure_tables(records)
        trigger_plot.figure_tone_opus(out, tables["tone_gradient_opus"])
    print(out)


if __name__ == "__main__":
    main()
