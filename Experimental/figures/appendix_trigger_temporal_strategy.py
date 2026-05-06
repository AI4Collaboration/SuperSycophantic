"""Generate appendix_trigger_temporal_strategy.png from raw trigger result files."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import DEFAULT_TRIGGER_RUN_ID, default_figure_path, ensure_parent

import plot_appendix_figures as appendix_plot
import plot_trigger_figures as trigger_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the temporal strategy figure used in the main text.")
    parser.add_argument("--run-id", default=DEFAULT_TRIGGER_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    records = trigger_plot.collect_records(args.results_dir, args.run_id)
    tables = trigger_plot.build_trigger_figure_tables(records)
    out = ensure_parent(args.out or default_figure_path(args.run_id, "appendix_trigger_temporal_strategy.png"))
    appendix_plot.figure_temporal_strategy_by_model(tables["temporal_pressure"], out)
    print(out)


if __name__ == "__main__":
    main()
