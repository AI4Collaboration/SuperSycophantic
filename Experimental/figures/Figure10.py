"""Generate Figure10.png, the Opus/Sonnet/Haiku tone-detail trajectory figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from PIL import ImageDraw

from _common import default_figure_path, ensure_parent

import plot_trigger_figures as trigger_plot


def add_claude_legend(image: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = trigger_plot.load_font(21)
    entries = [
        ("All-model baseline", "#17202A", 5),
        ("Opus-4.5", "#C85F39", 5),
        ("Sonnet-4.5", "#E08A57", 4),
        ("Haiku-4.5", "#F0B37E", 4),
    ]
    total_w = 0
    widths = []
    for label, _, _ in entries:
        bbox = draw.textbbox((0, 0), label, font=font)
        width = 44 + (bbox[2] - bbox[0]) + 36
        widths.append(width)
        total_w += width
    x = max(30, int((image.width - total_w) / 2))
    y = 102 if image.height < 850 else 122
    for (label, color, line_w), width in zip(entries, widths):
        draw.line((x, y, x + 34, y), fill=color, width=line_w)
        draw.ellipse((x + 14, y - 5, x + 24, y + 5), fill=color, outline="white", width=2)
        draw.text((x + 44, y), label, font=font, fill="#17202A", anchor="lm")
        x += width
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the main-text Opus/Sonnet/Haiku tone-detail figure.")
    parser.add_argument("--run-id", required=True, help="Current trigger run id under --results-dir.")
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--source-image",
        type=Path,
        help="Existing trigger_tone_claude_detail.png to reuse when ignored raw trigger results are unavailable.",
    )
    args = parser.parse_args()

    out = ensure_parent(args.out or default_figure_path(args.run_id, "Figure10.png"))
    if args.source_image:
        add_claude_legend(Image.open(args.source_image).convert("RGB")).save(out)
    else:
        records = trigger_plot.collect_records(args.results_dir, args.run_id)
        tables = trigger_plot.build_trigger_figure_tables(records)
        trigger_plot.figure_tone_opus(out, tables["tone_gradient_opus"])
    print(out)


if __name__ == "__main__":
    main()
