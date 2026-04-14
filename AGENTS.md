# AGENTS.md

## Repository purpose

This repository is a LaTeX paper project for **SuperSycophantic**, a benchmark and analysis framework for studying LLM sycophancy.

It is a **paper-first repo**, not an application codebase. Most work here is editing manuscript text, tables, citations, and figure references.

## Canonical git workflow

- Canonical remote: `origin` -> `https://github.com/AI4Collaboration/SuperSycophantic.git`
- Default branch: `main`
- Unless the user explicitly asks for a branch or PR workflow, **commit and push directly to `origin/main`**
- Before pushing, make sure the working tree only contains intended changes

## High-level file map

- `main.tex`: top-level entry point
- `sections/0-main.tex`: wires the main body sections together
- `sections/1-Intro.tex`: abstract + introduction
- `sections/2-RelatedWork.tex`: related work
- `sections/3-Method.tex`: benchmark design, evaluation protocol, ablations
- `sections/4-MechInterp.tex`: mechanistic interpretability section
- `sections/appendix.tex`: appendix, annotation details, experimental details, extended tables/text
- `tables/BenchScope.tex`: benchmark comparison table used from related work
- `main.bib`: bibliography
- `images/`: figure assets

## Editing priorities

- Default to **text-only manuscript edits**
- Preserve existing LaTeX structure and macros unless a change is clearly needed
- Avoid editing `neurips_2026.sty` or `neurips_checklist.tex` unless the user explicitly asks
- Avoid introducing multimodal or image-resolution framing unless the user explicitly wants it
- If a methodological decision changes, sync the wording across abstract, intro, method, appendix, and any affected tables

## Current paper conventions

- Core trigger taxonomy: **8 triggers**, including `simple baseline`
- Keep the trigger taxonomy aligned with **Cialdini-style influence**
- Temporal modes: `simple repetition`, `combined-trigger`, `escalation`, `de-escalation`
- `Evidence-bearing` variants are treated as an **ablation**, not as a new trigger family
- In `sections/3-Method.tex`, small internal headings are currently standardized to `\paragraph{...}`
- Tone is currently used as the operational label for interpersonal force, instead of splitting aggressiveness and emotional intensity into separate axes

## Practical guidance for future agents

- Most benchmark-construction edits belong in `sections/3-Method.tex` and `sections/appendix.tex`
- If updating the benchmark summary, check `sections/1-Intro.tex` as well
- If updating comparisons to prior work, check `tables/BenchScope.tex`
- Be alert for legacy or placeholder text in the appendix and tables
- Prefer focused commits with manuscript-oriented commit messages

## Verification expectations

- Do not assume LaTeX tooling is installed locally
- Only compile if the user asks for it or if compilation is necessary for the task
- If compilation is unavailable, state that clearly and still complete the text edits
