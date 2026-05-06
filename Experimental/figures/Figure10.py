"""Generate Figure10.png, the Claude strong-pressure rechecking figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import DEFAULT_TRIGGER_RUN_ID, ensure_parent

import plot_supplementary_diagnostics as supplementary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate main-text Figure 10.")
    parser.add_argument("--run-id", default=DEFAULT_TRIGGER_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=Path("Experimental/results"))
    parser.add_argument("--judge-trigger-csv", type=Path, required=True)
    parser.add_argument("--judge-trigger-inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("Experimental/reports/Figure10.png"))
    args = parser.parse_args()

    records = supplementary.collect_trigger(args.results_dir, args.run_id)
    judge_rows = supplementary.load_judge_rows(args.judge_trigger_csv)
    judge_meta = supplementary.load_trigger_judge_inputs(args.judge_trigger_inputs)
    supplementary.figure_strong_recheck_boundary(records, judge_rows, judge_meta, ensure_parent(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
