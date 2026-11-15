"""Check release files without opening ignored credentials or raw run data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


PRIVATE_DIRS = (
    "Experimental/results/", "Experimental/reports/", "Experimental/logs/",
    "Experimental/rebuttal_adaptive_audit/", "Experimental/rebuttal_positive_control/",
    "Experimental/rebuttal_robustness_results/", "Experimental/data/hle_verified/data/",
    "images/results/", "figure_png_for_upscale/", "output/", "sections/", "tables/",
)
PRIVATE_FILES = {
    "main.tex", "main.bib", "main.pdf", "neurips_checklist.tex", "secrets.env", "AGENTS.md",
    "label_studio/label_studio_iaa_results.json", "label_studio/label_studio_iaa_summary.md",
}
OUTPUT_DIRS = {"logs", "trajectory", "trajectories", "transcripts", "traces", "raw_responses"}
RESPONSE_FIELDS = {
    "assistant_response", "assistant_responses", "raw_response", "raw_responses",
    "response_text", "reasoning_content", "initial_response_text", "final_response_text",
}
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
    if any(part.lower() in OUTPUT_DIRS for part in path.parts[:-1]):
        return True
    if (base.endswith(".jsonl.gz") or re.search(r"\.log(?:\.|$)", base, re.I)
            or re.search(r"_(?:stdout|stderr)\.txt$", base, re.I)
            or re.search(r"_(?:trajector(?:y|ies)|transcripts?|responses)\.jsonl?$", base, re.I)
            or re.match(r"(?:trajector(?:y|ies)|transcripts?|responses).*\.jsonl?$", base, re.I)):
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
        or re.match(r"Experimental/IAA/[^/]+\.json$", name)
        or re.match(r"Experimental/IAA/(llm_judge_[^/]*|[^/]*-8-model)/", name)
        or re.match(r"Experimental/data/[^/]*_audit\.md$", name)
    )


def contains_model_output(value: object) -> bool:
    if isinstance(value, dict):
        if (RESPONSE_FIELDS.intersection(value) or value.get("role") == "assistant"
                or isinstance(value.get("assistant"), (str, dict, list))
                or {"initial_answer", "final_answer"}.issubset(value)):
            return True
        return any(contains_model_output(child) for child in value.values())
    if isinstance(value, list):
        return any(contains_model_output(child) for child in value)
    return False


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
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        try:
            records = ([json.loads(text)] if suffix == ".json" else
                       [json.loads(line) for line in text.splitlines() if line.strip()])
            if any(contains_model_output(record) for record in records):
                findings.append(Finding(name, 0, "model response or conversation output"))
        except (ValueError, RecursionError):
            findings.append(Finding(name, 0, "invalid structured release data"))
    return findings


def release_files(root: Path, include_untracked: bool) -> list[str]:
    command = ["git", "ls-files", "-z", "--cached"]
    if include_untracked:
        command.extend(["--others", "--exclude-standard"])
    result = subprocess.run(command, cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})


def inspect_revision(root: Path, revision: str) -> tuple[list[str], list[Finding]]:
    """Inspect committed blobs, independently of local edits or ignored files."""
    tree = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{tree}}"],
        cwd=root, stderr=subprocess.PIPE,
    ).decode().strip()
    listing = subprocess.check_output(["git", "ls-tree", "-rz", tree], cwd=root)
    names, blobs, findings = [], [], []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        name = raw_name.decode("utf-8")
        names.append(name)
        if mode == "120000" or kind != "blob":
            findings.append(Finding(name, 0, "symlink or submodule requires explicit release review"))
        else:
            blobs.append((name, oid))
    if blobs:
        output = subprocess.check_output(
            ["git", "cat-file", "--batch"], cwd=root,
            input="".join(f"{oid}\n" for _, oid in blobs).encode(),
        )
        stream = io.BytesIO(output)
        for name, oid in blobs:
            actual_oid, kind, size = stream.readline().decode().split()
            if actual_oid != oid or kind != "blob":
                raise ValueError("Unexpected Git blob response")
            data = stream.read(int(size))
            if len(data) != int(size) or stream.read(1) != b"\n":
                raise ValueError("Incomplete Git blob response")
            findings.extend(inspect_file(name, data))
    return names, findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--include-untracked", action="store_true")
    source.add_argument("--revision", help="inspect a Git revision, such as HEAD or origin/main")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.revision:
            names, findings = inspect_revision(root, args.revision)
        else:
            names = release_files(root, args.include_untracked)
            findings = []
    except (OSError, subprocess.CalledProcessError, ValueError):
        parser.exit(2, "Release checks require Git and a repository checkout.\n")
    for name in ([] if args.revision else names):
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
