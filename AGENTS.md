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
- The user's network uses a VPN. If normal `git push origin main` fails with `Recv failure: Connection was reset`, retry with: `git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999999 push origin main`. If HTTPS still resets, use SSH directly: `git push git@github.com:AI4Collaboration/SuperSycophantic.git main:main`; if SSH reports a transient connection abort, retry once.

## File map

- `main.tex`: top-level entry point
- `sections/1-Intro.tex`: abstract + introduction
- `sections/2-RelatedWork.tex`: related work
- `sections/3-Method.tex`: benchmark design, evaluation, ablations
- `sections/4-MechInterp.tex`: mechanistic interpretability
- `sections/appendix.tex`: annotation details, experimental details, extended text
- `tables/BenchScope.tex`: benchmark comparison table
- `Experimental/`: standalone external benchmark runner, tracked panels, and retained result files
- `main.bib`: bibliography

## Windows workflow notes

- Keep repo paths free of spaces when possible. The experimental code lives under `Experimental/` rather than a directory with spaces so PowerShell, `Start-Process`, and Python subprocess calls do not need extra quoting.
- Prefer `rg` first for search. If `rg.exe` fails on Windows with `Access is denied`, falls silent unexpectedly, or otherwise does not return usable output, fall back to PowerShell: `Get-ChildItem -Recurse -File | Select-String -Pattern "..."`.
- For background Python runs on Windows, quote script paths carefully or avoid spaces entirely.

## Figure 1 and Figure 2

- Figure 1 is the robot/settings figure: manuscript uses `images/Figure1.pdf` in `sections/1-Intro.tex`; bitmap source/preview is `images/Figure1.png`
- Figure 2 is the framework overview SVG: source of truth is `images/Figure2.svg`; manuscript uses `images/Figure2.pdf` in `sections/0-main.tex`; do not keep a repo PNG copy for this overview figure
- Current layout is a single-row spectrum: left `Context-induced`, right `Trigger-induced`
- Current card order is `Context -> Verifiability -> Framing -> Trigger -> Temporal -> Sycophancy Scale`
- Trigger layer keeps 8 triggers, with colored flaticon-style icons and black trigger labels
- Keep the middle logo stack centered in the gap between framing and trigger
- Export `Figure2.pdf` from a self-contained/no-scrollbar render and crop tightly; remove extra top/bottom/right whitespace
- Always overwrite old Figure 2 outputs in place: update `images/Figure2.svg`, delete and rewrite `images/Figure2.pdf`, and never keep stale Figure 2 temp previews around
- Correct export flow:
  1. Delete all old temp files matching `Figure2*` under the local temp directory, then delete the existing `images/Figure2.pdf`
  2. Build one temporary self-contained SVG by embedding every external `<image href="...">` asset as a data URI
  3. Render that embedded SVG with headless Edge using `--hide-scrollbars`, overscan the viewport slightly, then crop back to the exact target canvas
  4. Save exactly one current preview file as `Figure2_preview_latest.png`; do not keep timestamped or parallel Figure 2 preview variants
  5. Generate `images/Figure2.pdf` from that final cropped PNG; do not use browser `print-to-pdf` directly for this figure

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
