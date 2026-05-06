"""Shared helpers for one-file figure regeneration entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL_ROOT = REPO_ROOT / "Experimental"
DEFAULT_TRIGGER_RUN_ID = "trigger_20260504_070840"
DEFAULT_CONTEXT_SUMMARY = EXPERIMENTAL_ROOT / "results" / "context_20260504_184050_context_main_summary.json"

if str(EXPERIMENTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def default_figure_path(run_id: str, filename: str) -> Path:
    return EXPERIMENTAL_ROOT / "reports" / run_id / "paper_figure_candidates" / filename


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
