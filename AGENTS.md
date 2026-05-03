# AGENTS.md

## Repo

- This is a paper-first LaTeX repository. Default to narrow manuscript edits that follow the existing structure.
- Treat "update Overleaf" as updating this local repo and then pushing to the Overleaf Git remote.
- Do not revert or stage unrelated dirty changes. This repo often has uncommitted experiment/data work in progress.

## Git And Push

- GitHub remote: `origin` -> `https://github.com/AI4Collaboration/SuperSycophantic.git`, branch `main`.
- Overleaf remote: `overleaf` -> `https://git@git.overleaf.com/695e4c8c6d5ea6c90a2bb506`, branch `master`.
- Pull/fetch `origin/main` before new work after a gap. Fetch `overleaf/master` before publishing if Overleaf may have changed.
- When the user says `push`, stage only the intended files, commit, then push the same commit to both remotes:
  - `git push origin main`
  - `git push overleaf main:master`
- After pushing, verify local `HEAD`, `origin/main`, and `overleaf/master` resolve to the same commit. Do not force-push unless the user explicitly asks.
- If GitHub HTTPS push is reset, retry with HTTP/1.1:
  `git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999999 push origin main`

## LaTeX

- Do not run `latexmk`, `pdflatex`, `bibtex`, or other local LaTeX builds unless the user explicitly asks.
- Verify manuscript edits with text checks, targeted diffs, and file references.
- Do not edit `neurips_2026.sty` or `neurips_checklist.tex` unless asked.
- Avoid em-dashes and LaTeX `---` in manuscript prose. Avoid footnote-style source dumps in the main text.

## Main Files

- `main.tex`: top-level entry point.
- `sections/0-main.tex`: main paper assembly and Figure 2.
- `sections/1-Intro.tex`: abstract, Figure 1, and introduction.
- `sections/2-RelatedWork.tex`, `sections/3-Method.tex`, `sections/appendix.tex`: core manuscript text.
- `tables/BenchScope.tex`, `main.bib`: benchmark table and bibliography.
- Figure 1 is `images/Figure1.pdf`; Figure 2 is `images/Figure2.pdf`. Do not recreate Figure 2 SVG or PNG previews unless asked.

## Current Benchmark Contract

- Keep GT and NGT separate in data, templates, scoring, and claims.
- GT: 200 base items across Mathematical, Physical, Chemical, and Biomedical Science. Each domain has 25 MMLU-Pro and 25 HLE-Verified items. GT records must keep source URL, source quote, source-native or documented converted choices, verified answer, and a plausible tracked wrong answer.
- NGT: 100 base decisions, 25 each from policy trade-off, high-stakes moral dilemma, consequential interpersonal relation, and personal choice. Personal choice means long-term career or professional-planning decisions. NGT has two defensible answer states, no hidden correct answer, no item-level source answer, and only domain-level construct grounding.
- Context-framing accounting follows the current prompt-cell schema: GT is 200 base items times neutral/value-relevant/impression-relevant/outcome-relevant; NGT is 100 base decisions times neutral plus value-relevant/impression-relevant/outcome-relevant crossed with A/B direction.
- Trigger taxonomy: seven Cialdini families plus a simple baseline, each with mild/moderate/strong tones. Trigger inputs use neutral first-turn context only.
- Static and adaptive trigger runs are separate modes and should not be pooled. Neutral trigger panels must not contain pressure-target or direction fields. Static triggers are choice-agnostic; adaptive triggers observe the initial answer and are validated by an LM checker before use.
- GT trigger outcomes support truth-preservation/truth-departure claims. NGT trigger outcomes support Flip-Flop or pressure-accommodation claims, never accuracy claims.
- Every assistant utterance should include confidence. Follow-up turns should not ask for a separate rationale line; use the model-visible reasoning text for rationale analysis.
- Main model panel lives in `Experimental/models.py` as `MAIN_MODELS`: OpenAI GPT-5.4/GPT-5.4-mini/GPT-5.4-nano; Claude Opus 4.7/Sonnet 4.6/Haiku 4.5; Gemini 3.1 Pro/3.1 Flash Lite; Mistral Large 2512/Medium 3.1/Small 2603; Cohere Command A/Command R+ 08-2024/Command R 08-2024. Runners accept `--models main`.
- Current runner defaults live in `Experimental/run.py`: adaptive generator/checker `openai/gpt-5.4-mini`, concurrency `200`, request timeout `30`, and max attempts `8`. Keep README examples, manuscript wording, and scripts aligned if these change.
- Before model calls, run `python Experimental/data/helper/audit_supersycophantic_panels.py`; duplicate question text, duplicate IDs, bad source text, invalid choices, prompt-format drift, and trigger direction leakage must fail before any batch starts.

## Current Data Files

- `Experimental/data/supersycophantic_context_gt_200.json`
- `Experimental/data/supersycophantic_context_ngt_100.json`
- `Experimental/data/supersycophantic_trigger_gt_neutral_200.jsonl`
- `Experimental/data/supersycophantic_trigger_ngt_neutral_100.jsonl`
- `Experimental/data/supersycophantic_mixed_gt_200.jsonl`
- `Experimental/data/mmlu_pro_release_gt_100.jsonl`
- `Experimental/data/hle_verified/data/Gold_subset.*.parquet`
- `Experimental/data/hle_verified/data/Revision_subset.*.parquet`

## Bibliography

- Add BibTeX entries only from arXiv or DBLP.
- Verify authors, title, venue or journal, year, identifier, and links before adding bibliography entries.
