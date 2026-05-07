"""Generate context_neutral_shift.pdf/png from the context summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ensure_parent

import plot_context_results as context_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the main-text context neutral-shift figure.")
    parser.add_argument("--summary", type=Path, required=True, help="Current context summary JSON to plot.")
    parser.add_argument("--out-pdf", type=Path, default=Path("Experimental/reports/context_neutral_shift.pdf"))
    parser.add_argument("--out-png", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    context_plot.save_shift_plot(summary, ensure_parent(args.out_png) if args.out_png else None, ensure_parent(args.out_pdf))
    print(args.out_pdf)


if __name__ == "__main__":
    main()
