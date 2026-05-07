"""Generate trigger_model_quadrant.png from raw trigger result files."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import default_figure_path, ensure_parent

import plot_trigger_figures as trigger_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the main-text static/adaptive trigger quadrant.")
    parser.add_argument("--run-id", required=True, help="Current trigger run id under --results-dir.")
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    records = trigger_plot.collect_records(args.results_dir, args.run_id)
    headline, _, _, _ = trigger_plot.build_tables(records)
    out = ensure_parent(args.out or default_figure_path(args.run_id, "trigger_model_quadrant.png"))
    trigger_plot.figure_scatter(out, headline)
    print(out)


if __name__ == "__main__":
    main()
