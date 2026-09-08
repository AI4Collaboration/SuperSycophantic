# AGENTS.md

## Repo

- This is the GitHub codebase tree for SuperSycophantic experiments.
- Keep GitHub focused on `Experimental/`, benchmark data builders, audits,
  runners, scoring, plotting scripts, and code-facing visual assets.
- Manuscript files, LaTeX sources, bibliography edits, paper tables, and paper
  figures belong only in the Overleaf paper tree, not in the GitHub codebase
  tree.
- Keep release-facing files double-blind. Do not add author names,
  institutions, private remote URLs, local usernames, absolute local paths,
  account names, unpublished acknowledgments, or API secrets.
- Do not revert or stage unrelated dirty changes. This repo often has
  uncommitted experiment/data work in progress.
- Use multi-agent parallelism when work splits cleanly, especially independent
  figure generation, figure-script review, data audits, or manuscript
  consistency checks. Assign agents disjoint files or questions so they
  accelerate execution without duplicating work.

## Git And Push

- Use the configured remotes only when the user explicitly asks to push.
- Codebase changes are submitted only to the configured GitHub remote. Fetch
  after a gap, stage only intended codebase files, commit, push, and verify the
  GitHub remote head.
- Manuscript changes are submitted only from the Overleaf paper tree to
  Overleaf. Do not submit manuscript files, paper figures, LaTeX sources,
  bibliography edits, or paper tables from this GitHub codebase tree.
- Do not mirror code into Overleaf or manuscript assets into GitHub. There is
  no third release target.
- Do not document private remote URLs or account identifiers in repository
  files.
- Do not force-push unless the user explicitly asks.

## Codebase Files

- `Experimental/`: data construction, audits, runners, scoring, plotting, and
  reporting scripts.
- `Experimental/data/`: frozen benchmark panels, source caches, and helper
  data.
- `Experimental/data/helper/`: builders and audits for the frozen benchmark panels.
- `Experimental/IAA/`: small reviewed Human/LLM judge calibration artifacts.
- `images/logos/`: small source logos used by plotting scripts.
- `images/model_logos/`: generated model badge assets used by plotting scripts.
- Paper-only folders such as LaTeX `sections/`, `tables/`, `main.tex`,
  `main.bib`, `images/Figure*.png`, and `images/results/` should not be added
  to GitHub codebase commits.

## Current Benchmark Contract

- Keep OBJ and SUB streams separate in data,
  templates, scoring, and claims.
- Release-facing prose and figures should use OBJ for the factual stream and
  SUB for the opinion/decision stream. Legacy `gt` and `ngt` prefixes may remain
  in data filenames and code paths for compatibility.
- OBJ: 200 base items across Mathematical, Physical, Chemical, and
  Biomedical Science. Each domain has 25 MMLU-Pro and 25 HLE-Verified items.
  Records must keep source URL, source quote, source-native or documented
  converted choices, verified answer, and a plausible tracked wrong answer.
- SUB: 100 base decisions, 25 each from policy trade-off,
  high-stakes moral dilemma, consequential interpersonal relation, and
  professional-planning choice. Professional-planning choice means long-term
  career or professional-planning decisions. SUB records have two defensible
  answer states, no hidden correct answer, no item-level source answer, and
  only domain-level construct grounding.
- Context-framing accounting follows the release prompt-cell schema:
  OBJ is 200 base items times neutral/value-relevant/
  impression-relevant/outcome-relevant; SUB is 100 base
  decisions times neutral plus value-relevant/impression-relevant/
  outcome-relevant crossed with A/B direction.
- Context framing must use a two-sentence structure: first sentence states the
  user's belief or leaning, second sentence supplies only the intended cue. Do
  not use `because` or task-evidence wording in context framing.
- OBJ outcome/stake must be pre-submission user stake such as an
  exam, grade, or application.
- SUB value/belief must not provide option-specific evidence or
  stakeholder consequences; use generic belief wording. SUB
  outcome/stake must describe consequences borne by the user themself, not
  third-party welfare or external stakeholder outcomes.
- Trigger taxonomy: seven Cialdini families plus a simple baseline, each with
  mild/moderate/strong tones. Trigger inputs use neutral first-turn context
  only.
- Static and adaptive trigger runs are separate modes and should not be pooled.
  Neutral trigger panels must not contain pressure-target or direction fields.
  Static triggers are choice-agnostic; adaptive triggers observe the initial
  answer and are validated by an LM checker before use.
- OBJ trigger outcomes support truth-preservation/truth-departure
  claims. SUB trigger outcomes support switching or
  pressure-accommodation claims, never accuracy claims.
- Every assistant utterance should include confidence. Follow-up turns should
  not ask for a separate rationale line; use the model-visible reasoning text
  for rationale analysis.
- Main model panel lives in `Experimental/models.py` as `MAIN_MODELS`: OpenAI
  GPT-5.4/GPT-5.4-mini/GPT-5.4-nano; Opus-4.5/Sonnet-4.5/Haiku-4.5;
  Gemini-3.1 Flash-Lite Preview; Mistral-Medium-3.1; Command-R. The
  `cohere/command-r-08-2024` endpoint id keeps its provider version suffix in
  code, but release-facing labels should display Command-R. Runners accept
  `--models main`.
- When `ANTHROPIC_API_KEY` is present, runners should send `anthropic/...`
  target models directly to Anthropic's Messages API; non-Anthropic targets and
  the default adaptive generator/checker use OpenRouter.
- Runner defaults live in `Experimental/run.py`: adaptive
  generator/checker `openai/gpt-5.4-mini`, concurrency `200`, request timeout
  `30`, and max attempts `8`. Keep README examples, code, and scripts aligned
  if these change.
- Before model calls, run
  `python Experimental/data/helper/audit_supersycophantic_panels.py`;
  duplicate question text, duplicate IDs, bad source text, invalid choices,
  prompt-format drift, and trigger direction leakage must fail before any batch
  starts.

## Current Data Files

- `Experimental/data/supersycophantic_context_gt_200.json`
- `Experimental/data/supersycophantic_context_ngt_100.json`
- `Experimental/data/supersycophantic_context_ngt_swap_control_100.json`
- `Experimental/data/supersycophantic_trigger_gt_neutral_200.jsonl`
- `Experimental/data/supersycophantic_trigger_ngt_neutral_100.jsonl`
- `Experimental/data/supersycophantic_mixed_gt_200.jsonl`
- `Experimental/data/mmlu_pro_release_gt_100.jsonl`
- HLE-Verified source-cache parquet shards are intentionally not tracked; keep
  them local under `Experimental/data/hle_verified/data/` only when rebuilding
  the mixed source panel.
- `Experimental/IAA/Human-LLM-Judge-IAA-100.json`

## Verification

- Prefer targeted text checks, Python syntax checks, and lightweight audits for
  codebase changes.
- Do not run local LaTeX builds from the GitHub codebase tree.
- Keep generated raw outputs in ignored paths unless the user explicitly asks
  to publish them.
- When generating a replacement PNG, remove the obsolete PNG it replaces.
