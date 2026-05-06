"""Generate trigger_family_tone_boost.png from raw trigger result files."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import DEFAULT_TRIGGER_RUN_ID, default_figure_path, ensure_parent

import plot_trigger_family_tone_boost as tone_boost


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the trigger family/tone boost figure.")
    parser.add_argument("--run-id", default=DEFAULT_TRIGGER_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--mode", choices=["static", "adaptive", "all"], default="all")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = tone_boost.aggregate_tone(args.results_dir, args.run_id, args.mode)
    out = ensure_parent(args.out or default_figure_path(args.run_id, "trigger_family_tone_boost.png"))
    tone_boost.draw_figure(rows, out)
    print(out)


if __name__ == "__main__":
    main()
