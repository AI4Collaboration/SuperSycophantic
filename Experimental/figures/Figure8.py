"""Generate Figure8.png, the main-text confidence trajectory figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from _common import DEFAULT_TRIGGER_RUN_ID, default_figure_path, ensure_parent

import plot_trigger_figures as trigger_plot
import trigger_confidence_trajectory as confidence_trajectory


def trim_center_gap(image: Image.Image, pixels: int) -> Image.Image:
    if pixels <= 0:
        return image
    if pixels >= image.width:
        raise ValueError(f"Cannot trim {pixels} pixels from image width {image.width}")
    left_width = (image.width - pixels) // 2
    right_start = left_width + pixels
    trimmed = Image.new(image.mode, (image.width - pixels, image.height), "white")
    trimmed.paste(image.crop((0, 0, left_width, image.height)), (0, 0))
    trimmed.paste(image.crop((right_start, 0, image.width, image.height)), (left_width, 0))
    return trimmed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate main-text Figure 8.")
    parser.add_argument("--run-id", default=DEFAULT_TRIGGER_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--appendix-tex",
        type=Path,
        help="Optional appendix TeX source. When provided, regenerate from tab:trigger_confidence_rates.",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        help="Optional existing trajectory PNG to republish as Figure8.png.",
    )
    parser.add_argument(
        "--trim-center-gap-px",
        type=int,
        default=0,
        help="Trim this many pixels from the center gap when republishing an existing source image.",
    )
    args = parser.parse_args()

    out = ensure_parent(args.out or default_figure_path(args.run_id, "Figure8.png"))
    if args.source_image:
        trim_center_gap(Image.open(args.source_image).convert("RGB"), args.trim_center_gap_px).save(out)
    elif args.appendix_tex:
        confidence_trajectory.figure_from_appendix(
            confidence_trajectory.load_confidence_rows(args.appendix_tex),
            out,
        )
    else:
        records = trigger_plot.collect_records(args.results_dir, args.run_id)
        tables = trigger_plot.build_trigger_figure_tables(records)
        trigger_plot.figure_confidence_trajectory(out, tables["confidence_trajectory"])
    print(out)


if __name__ == "__main__":
    main()
