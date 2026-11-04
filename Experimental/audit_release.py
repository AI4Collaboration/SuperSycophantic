"""Check release files without opening ignored credentials or raw run data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import subprocess


PRIVATE_DIRS = (
    "Experimental/results/", "Experimental/reports/", "Experimental/logs/",
    "Experimental/rebuttal_adaptive_audit/", "Experimental/rebuttal_positive_control/",
    "Experimental/rebuttal_robustness_results/", "Experimental/data/hle_verified/data/",
    "images/results/", "figure_png_for_upscale/", "output/", "sections/", "tables/",
)
PRIVATE_FILES = {"main.tex", "main.bib", "main.pdf", "neurips_checklist.tex", "secrets.env"}
PATTERNS = (
    ("provider credential", re.compile(r"\bsk-(?:or-v1-|ant-api\d+-|proj-)?[A-Za-z0-9_-]{24,}")),
    ("GitHub credential", re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{24,}")),
    ("Slack credential", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    ("credential in URL", re.compile(r"https?://[^\s/:@]+:[^\s/@]+@", re.I)),
    ("private manuscript URL", re.compile(r"https?://(?:git@)?git\.overleaf\.com/[^\s]+", re.I)),
    ("absolute user path", re.compile(r"(?:[A-Z]:[/\\]+Users[/\\]+[^\s/\\\"']+|/(?:Users|home)/[^\s/\"']+)", re.I)),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    reason: str


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    base = path.name
    if name in PRIVATE_FILES or name.startswith(PRIVATE_DIRS):
        return True
    if any(part in {".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
           for part in path.parts):
        return True
    if base == ".env" or (base.startswith(".env.") and base != ".env.example"):
        return True
    if path.suffix.lower() in {".pem", ".key", ".pyc", ".pyo", ".aux", ".log", ".bak", ".orig"}:
        return True
    if base.startswith(("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")) or base.endswith(".local.json"):
        return True
    return bool(
        re.match(r"images/Figure\d+\.(png|pdf|svg)$", name, re.I)
        or re.match(r"label_studio/label_studio_export_.*\.json$", name)
        or re.match(r"Experimental/IAA/(llm_judge_[^/]*|[^/]*-8-model)/", name)
        or re.match(r"Experimental/data/[^/]*_audit\.md$", name)
    )


def inspect_file(name: str, data: bytes) -> list[Finding]:
    findings = [Finding(name, 0, "private or generated artifact")] if private_path(name) else []
    if b"\0" in data:
        return findings
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return findings
    for number, line in enumerate(text.splitlines(), 1):
        for reason, pattern in PATTERNS:
            if pattern.search(line):
                findings.append(Finding(name, number, reason))
    return findings


def release_files(root: Path, include_untracked: bool) -> list[str]:
    command = ["git", "ls-files", "-z", "--cached"]
    if include_untracked:
        command.extend(["--others", "--exclude-standard"])
    result = subprocess.run(command, cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        names = release_files(root, args.include_untracked)
    except (OSError, subprocess.CalledProcessError):
        parser.exit(2, "Release checks require Git and a repository checkout.\n")
    findings = []
    for name in names:
        path = root / name
        if path.is_symlink():
            findings.append(Finding(name, 0, "symlink requires explicit release review"))
        elif path.is_file():
            findings.extend(inspect_file(name, path.read_bytes()))
        else:
            findings.append(Finding(name, 0, "tracked file is missing"))
    for finding in findings:
        # Never echo the matching line: it may contain a real credential.
        print(f"{finding.path}:{finding.line}: {finding.reason}")
    print(f"Checked {len(names)} release files; {len(findings)} finding(s).")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
