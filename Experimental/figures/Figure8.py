"""Generate Figure8.png, the main-text confidence distribution figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import DEFAULT_TRIGGER_RUN_ID, ensure_parent

import plot_supplementary_diagnostics as supplementary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate main-text Figure 8.")
    parser.add_argument("--run-id", default=DEFAULT_TRIGGER_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--out", type=Path, default=Path("Experimental/reports/Figure8.png"))
    args = parser.parse_args()

    records = supplementary.collect_trigger(args.results_dir, args.run_id)
    supplementary.figure_trigger_confidence_high_risk(records, ensure_parent(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
