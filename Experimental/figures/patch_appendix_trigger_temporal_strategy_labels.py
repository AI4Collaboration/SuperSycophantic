"""Patch release-facing labels on the temporal strategy PNG without changing plotted data.

Use this only when the ignored raw temporal trigger result files are unavailable
and the manuscript image must preserve the existing per-model plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from _common import ensure_parent

import plot_trigger_figures as trigger_plot


INK = "#17202A"
WHITE = "#FFFFFF"


def patch_labels(source: Path, out: Path) -> None:
    im = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(im)
    title_font = trigger_plot.load_font(34, True)

    width, _ = im.size
    panel_centers = [width * 0.25, width * 0.75]
    labels = ["OBJ right-to-wrong", "SUB switching"]
    title_y = 170
    cover_h = 72
    cover_w = 760

    for center_x, label in zip(panel_centers, labels):
        draw.rectangle(
            (
                center_x - cover_w / 2,
                title_y - cover_h / 2,
                center_x + cover_w / 2,
                title_y + cover_h / 2,
            ),
            fill=WHITE,
        )
        draw.text((center_x, title_y), label, font=title_font, fill=INK, anchor="mm")

    ensure_parent(out)
    im.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch OBJ/SUB labels onto an existing temporal strategy PNG while preserving all plotted values."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    patch_labels(args.source, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
