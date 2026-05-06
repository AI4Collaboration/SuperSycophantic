"""Generate Figure10.png, the Claude tone-detail trajectory figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from _common import DEFAULT_TRIGGER_RUN_ID, default_figure_path, ensure_parent

import plot_trigger_figures as trigger_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the main-text Claude tone-detail figure.")
    parser.add_argument("--run-id", default=DEFAULT_TRIGGER_RUN_ID)
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
        Image.open(args.source_image).convert("RGB").save(out)
    else:
        records = trigger_plot.collect_records(args.results_dir, args.run_id)
        tables = trigger_plot.build_trigger_figure_tables(records)
        trigger_plot.figure_tone_opus(out, tables["tone_gradient_opus"])
    print(out)


if __name__ == "__main__":
    main()
