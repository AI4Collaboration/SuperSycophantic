# AGENTS.md

## Repo

Paper-first LaTeX repo. Most work is manuscript editing.

## Git

- Remote: `origin` -> `https://github.com/AI4Collaboration/SuperSycophantic.git`
- Default branch: `main`
- At the start of each day, and whenever the gap since the last conversation is long, pull from GitHub first so local work is synced with `origin/main` before making new edits.
- Before pushing, make sure the worktree only contains intended changes.
- If the user says to push, push the full current intended worktree, including `.tex` edits, unless they explicitly ask for a partial push.
- Unless the user asks otherwise, commit and push directly to `origin/main`.

## File map

- `main.tex`: top-level entry point
- `sections/1-Intro.tex`: abstract + introduction
- `sections/2-RelatedWork.tex`: related work
- `sections/3-Method.tex`: benchmark design, evaluation, ablations
- `sections/4-MechInterp.tex`: mechanistic interpretability
- `sections/appendix.tex`: annotation details, experimental details, extended text
- `tables/BenchScope.tex`: benchmark comparison table
- `Experimental/gpqa_trigger_screen/`: standalone external benchmark runners, tracked panels, and local gitignored results
- `main.bib`: bibliography

## Windows workflow notes

- Keep repo paths free of spaces when possible. The experimental code lives under `Experimental/` rather than a directory with spaces so PowerShell, `Start-Process`, and Python subprocess calls do not need extra quoting.
- Prefer `rg` first for search. If `rg.exe` fails on Windows with `Access is denied`, falls silent unexpectedly, or otherwise does not return usable output, fall back to PowerShell: `Get-ChildItem -Recurse -File | Select-String -Pattern "..."`.
- For background Python runs on Windows, quote script paths carefully or avoid spaces entirely.

## Figure 1

- Source of truth: `images/Figure1.svg`
- Manuscript uses `images/Figure1.pdf` in `sections/1-Intro.tex`; do not keep a repo PNG copy
- Current layout is a single-row spectrum: left `Context-induced`, right `Trigger-induced`
- Current card order is `Context -> Verifiability -> Framing -> Trigger -> Temporal -> Sycophancy Scale`
- Trigger layer keeps 8 triggers, with colored flaticon-style icons and black trigger labels
- Keep the middle logo stack centered in the gap between framing and trigger
- Export `Figure1.pdf` from a self-contained/no-scrollbar render and crop tightly; remove extra bottom/right whitespace
- Correct export flow:
  1. Build a temporary self-contained SVG by embedding every external `<image href="...">` asset as a data URI
  2. Render that embedded SVG with headless Edge using `--hide-scrollbars`, overscan the viewport slightly, then crop back to the exact target canvas
  3. Generate `images/Figure1.pdf` from the final cropped PNG with `reportlab`; do not use browser `print-to-pdf` directly for this figure

## Editing rules

- Default to text-only manuscript edits
- Preserve existing LaTeX structure and macros unless needed
- Do not edit `neurips_2026.sty` or `neurips_checklist.tex` unless asked
- Do not introduce multimodal or image-resolution framing unless asked
- If a methodological decision changes, sync abstract, intro, method, appendix, and affected tables
- Most benchmark-construction edits belong in `sections/3-Method.tex` and `sections/appendix.tex`
- Check `sections/1-Intro.tex` for summary wording and `tables/BenchScope.tex` for prior-work comparisons.
- Watch for legacy or placeholder text, especially in the appendix and tables.

## Current benchmark conventions

- Core trigger taxonomy: **8 triggers**, including `simple baseline`
- Trigger taxonomy stays aligned with **Cialdini-style influence**
- Temporal modes: `simple repetition`, `combined-trigger`, `escalation`, `de-escalation`
- `Evidence-bearing` variants are an ablation, not a new trigger family
- Small internal headings in `sections/3-Method.tex` use `\paragraph{...}`
- Tone is the operational label for interpersonal force
- Primary behavioral endpoint is **answer change under unsupported pressure**, not raw accuracy alone
- GT keeps a ground-truth label for secondary analyses such as truth preservation and truth departure
- Eligibility is based on a committed initial answer, not initial correctness
- Both GT and NGT should be trackable as two stable answer states so switch events can be measured uniformly
- External benchmark adaptations should prioritize answer revision under pressure over accuracy-only scoring
- Current external transfer shortlist: `MMLU-Pro`, `ARC-Challenge`, `GPQA`, `HLE-Verified`, and `TruthfulQA`
- These external benchmarks are auxiliary stress tests, not replacements for the main benchmark

## Bibliography

- Any new `bibtex` entry must be taken from **arXiv** or **DBLP**
- Every new entry must be manually verified so that authors, title, venue/journal, year, identifier, and links are real and accurate

## Verification

- Do not assume LaTeX tooling is installed
- Only compile if the user asks or if compilation is necessary
- If compilation is unavailable, state that clearly and still complete the text edits
