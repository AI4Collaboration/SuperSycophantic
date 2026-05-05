# AGENTS.md

## Repo

- GitHub is the codebase remote. Keep it focused on `Experimental/`, data construction, audits, runners, plotting scripts, and code-facing assets.
- Overleaf is the manuscript remote. Paper edits belong on the Overleaf git remote and should not be synced through GitHub.
- Do not revert or stage unrelated dirty changes. This repo often has uncommitted experiment/data work in progress.

## Git And Push

- GitHub remote: `origin` -> `https://github.com/AI4Collaboration/SuperSycophantic.git`, branch `main`.
- Overleaf remote: `overleaf` -> `https://git@git.overleaf.com/695e4c8c6d5ea6c90a2bb506`, branch `master`.
- Pull/fetch `origin/main` before codebase work after a gap. Fetch `overleaf/master` before manuscript work if Overleaf may have changed.
- When the user asks to push code or experimental assets, stage only the intended codebase files, commit, and push to `origin/main`.
- When the user asks to update Overleaf or the paper, stage only manuscript files, commit, and push to `overleaf/master`.
- Do not push one combined commit to both remotes. The two remotes intentionally have different trees.
- Do not force-push unless the user explicitly asks or a just-created mistaken push must be removed; in that case use `--force-with-lease`.
- If GitHub HTTPS push is reset, retry with HTTP/1.1:
  `git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999999 push origin main`

## Manuscript

- Manuscript files are maintained in the Overleaf remote, not in the GitHub codebase tree.
- Do not run `latexmk`, `pdflatex`, `bibtex`, or other local LaTeX builds unless the user explicitly asks.
- Avoid em-dashes and LaTeX `---` in manuscript prose when editing the Overleaf tree.

## Codebase Files

- `Experimental/`: data construction, audits, runners, scoring, plotting, and reporting scripts.
- `Experimental/data/`: current benchmark data and source caches.
- `images/model_logos/`: source logo assets used by plotting scripts.

## Current Benchmark Contract

- Keep GT and NGT separate in data, templates, scoring, and claims.
- GT: 200 base items across Mathematical, Physical, Chemical, and Biomedical Science. Each domain has 25 MMLU-Pro and 25 HLE-Verified items. GT records must keep source URL, source quote, source-native or documented converted choices, verified answer, and a plausible tracked wrong answer.
- NGT: 100 base decisions, 25 each from policy trade-off, high-stakes moral dilemma, consequential interpersonal relation, and personal choice. Personal choice means long-term career or professional-planning decisions. NGT has two defensible answer states, no hidden correct answer, no item-level source answer, and only domain-level construct grounding.
- Context-framing accounting follows the current prompt-cell schema: GT is 200 base items times neutral/value-relevant/impression-relevant/outcome-relevant; NGT is 100 base decisions times neutral plus value-relevant/impression-relevant/outcome-relevant crossed with A/B direction.
- Trigger taxonomy: seven Cialdini families plus a simple baseline, each with mild/moderate/strong tones. Trigger inputs use neutral first-turn context only.
- Static and adaptive trigger runs are separate modes and should not be pooled. Neutral trigger panels must not contain pressure-target or direction fields. Static triggers are choice-agnostic; adaptive triggers observe the initial answer and are validated by an LM checker before use.
- GT trigger outcomes support truth-preservation/truth-departure claims. NGT trigger outcomes support Flip-Flop or pressure-accommodation claims, never accuracy claims.
- Every assistant utterance should include confidence. Follow-up turns should not ask for a separate rationale line; use the model-visible reasoning text for rationale analysis.
- Main model panel lives in `Experimental/models.py` as `MAIN_MODELS`: OpenAI GPT-5.4/GPT-5.4-mini/GPT-5.4-nano; Claude Opus 4.5/Sonnet 4.5/Haiku 4.5; Gemini 3.1 Flash Lite Preview; Mistral Medium 3.1; Cohere Command R 08-2024. Runners accept `--models main`.
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
