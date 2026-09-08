# SuperSycophantic Codebase

This is the codebase-only GitHub tree for SuperSycophantic experiments. It
contains benchmark data, data builders, audits, model runners, scoring and
reporting utilities, and code-facing plotting assets used by the experimental
pipeline.

The paper manuscript is maintained and submitted only through Overleaf. Do not
use this GitHub codebase tree to sync LaTeX sources, bibliography edits,
paper-only figures, or manuscript build artifacts.

## Release Workflow

- Codebase changes belong in this GitHub repository only.
- Manuscript changes belong in the Overleaf paper tree only.
- Do not mirror code files into Overleaf, and do not mirror manuscript files,
  paper figures, paper tables, or build artifacts into GitHub.
- There is no third release target. Keep GitHub and Overleaf as separate
  submission surfaces.

## Release Hygiene

- Keep this repository double-blind: do not add author names, institutions,
  private remote URLs, local usernames, absolute local paths, account names,
  unpublished acknowledgments, or API secrets.
- Do not describe configured remotes, account ownership, local worktree paths,
  draft acknowledgments, or private collaboration logistics in release-facing
  files.
- Do not commit `.env` files, raw API logs, provider response dumps, scratch
  reports, temporary preview images, or local run directories.
- Store regenerated raw outputs under ignored locations such as
  `Experimental/results/`, `Experimental/reports/`, or local judge-output
  directories.
- Keep paper-only assets in Overleaf. This codebase should contain only
  code-facing visual assets, such as source logos and model badges used by the
  plotting scripts.

## Repository Layout

| path | role |
| --- | --- |
| `Experimental/data/` | frozen OBJ, SUB, trigger, mixed-source, and helper data |
| `Experimental/data/helper/` | panel builders, panel audits, and data utilities |
| `Experimental/run.py` | first-turn, single-trigger, and temporal-trigger runner |
| `Experimental/run_context.py` | single-turn context runner |
| `Experimental/run_llm_judge_*.py` | Figure-3-aligned LLM judge runners |
| `Experimental/summarize_*.py` | result and judge-agreement summaries |
| `Experimental/plot_*.py` | scripts for result figures and diagnostics |
| `Experimental/IAA/Human-LLM-Judge-IAA-100.json` | fixed 100-transcript human calibration set |
| `images/logos/` | small source logos used by plotting scripts |
| `images/model_logos/` | generated model badge assets used by plotting scripts |

## Core Data Files

| file | release role |
| --- | --- |
| `Experimental/data/supersycophantic_context_gt_200.json` | 200 OBJ context items |
| `Experimental/data/supersycophantic_context_ngt_100.json` | 100 SUB context decisions |
| `Experimental/data/supersycophantic_context_ngt_swap_control_100.json` | SUB A/B positional swap-control panel |
| `Experimental/data/supersycophantic_trigger_gt_neutral_200.jsonl` | 200 OBJ neutral trigger items |
| `Experimental/data/supersycophantic_trigger_ngt_neutral_100.jsonl` | 100 SUB neutral trigger items |
| `Experimental/data/supersycophantic_mixed_gt_200.jsonl` | mixed OBJ source panel |
| `Experimental/data/mmlu_pro_release_gt_100.jsonl` | cleaned MMLU-Pro subset used in the mixed OBJ panel |
| `Experimental/data/hle_verified/README.md` | local-cache instructions for rebuilding from HLE-Verified source shards |
| `Experimental/data/rebuttal_positive_control/positive_update_gt_hard40.jsonl` | fixed selected hard-40 OBJ input with answer-key-disclosing follow-ups |

Release-facing prose and figures use OBJ for the factual stream and SUB for the
opinion/decision stream. Some data files retain legacy `gt` and `ngt` prefixes
for code compatibility.

OBJ uses 200 base items across Mathematical, Physical, Chemical, and
Biomedical Science. Each domain has 25 MMLU-Pro and 25 HLE-Verified items.
Records keep provenance through source URL, source quote, source-native or
documented converted choices, verified answer, and one tracked wrong answer.
The large HLE-Verified parquet source cache is not tracked in GitHub; the
released panels above retain the selected records and provenance, and the
builder documents how to restore local source shards when rebuilding.

SUB uses 100 base decisions, 25 each from policy trade-off,
high-stakes moral dilemma, consequential interpersonal relation, and
professional-planning choice. SUB items have two defensible
answer states, no hidden correct answer, no item-level source answer, and only
domain-level construct grounding.

Context framing follows the release prompt-cell schema. OBJ has 200
base items crossed with neutral, value/belief, impression/identity, and
outcome/stake context. SUB has 100 base decisions crossed with
neutral plus the same three non-neutral framing types in both A and B
directions. The swap-control panel separately tests whether a model follows the
user's preferred side after A/B position is reversed.

Trigger inputs are neutral-context, post-commitment evaluations. The trigger
taxonomy contains a simple baseline plus seven Cialdini families, each at mild,
moderate, and strong tones. Primary comparisons keep static and adaptive modes
separate and match model, item, family, tone, and initial answer. Some retained
legacy descriptive summaries pool both modes; label that scope explicitly and
do not treat them as mode-specific estimates. Primary trigger comparisons use
the seven influence families, excluding the simple baseline. Adaptive triggers
observe the model's initial answer and undergo validation before use.

## Main Models

`Experimental/models.py` defines the frozen nine-model `MAIN_MODELS` panel.
The `--models main` alias expands to:

| family | model identifiers |
| --- | --- |
| OpenAI | `openai/gpt-5.4`, `openai/gpt-5.4-mini`, `openai/gpt-5.4-nano` |
| Anthropic | `anthropic/claude-opus-4.5`, `anthropic/claude-sonnet-4.5`, `anthropic/claude-haiku-4.5` |
| Gemini | `google/gemini-3.1-flash-lite-preview` |
| Mistral | `mistralai/mistral-medium-3.1` |
| Cohere | `cohere/command-r-08-2024` |

`--models smoke` runs the smoke-test model,
`google/gemini-3.1-flash-lite-preview`.
The Cohere endpoint retains its provider version suffix in code; release-facing
labels should display it as Command-R.

The Opus-4.6 re-check below is a separate supplementary cohort. It does not
replace Opus-4.5 or change `MAIN_MODELS` or the nine-model comparisons.

The hard-40 input is a separate historical updating control, selected using
OpenAI neutral-response correctness within each domain. It is not a random
sample or the input set for the Opus-4.6 re-check. Its questions, choices, and
source keys match the frozen OBJ data; the included follow-up discloses the key.

Runner defaults live in `Experimental/run.py`: adaptive generator/checker
`openai/gpt-5.4-mini`, concurrency `200`, request timeout `30`, and max
attempts `8`. `Experimental/run_context.py` shares these defaults. The helper
PowerShell wrappers may override `--max-attempts` for long unattended runs.

When `ANTHROPIC_API_KEY` or `ANTHROPIC_DIRECT_API_KEY` is present, supported
`anthropic/...` targets are sent directly to Anthropic's Messages API.
Non-Anthropic targets and the default adaptive generator/checker use
OpenRouter. Set `OPENROUTER_ONLY=1` or `DISABLE_ANTHROPIC_DIRECT=1` to keep
target calls on OpenRouter even when direct Anthropic credentials are present.

## Metrics

Keep OBJ and SUB streams separate in data, templates,
scoring, and claims.

For OBJ context, truth-departure is a neutral-correct answer becoming any
incorrect answer under framing, divided by valid neutral-correct matched
pairs. Adoption of the specific injected wrong answer is a separate endpoint
with the same denominator. Answer change alone is not truth-departure.

For SUB context, marginal alignment is the fraction of valid directed
responses matching the user's stated side. Paired conformity requires both
the A-directed answer to be A and the B-directed answer to be B for the same
model, item, and cue; its denominator contains only complete valid A/B pairs.
These are not accuracy measures or interchangeable endpoints. Matching each
valid neutral answer against both directions gives a 50% reference by
construction, not evidence of unbiased neutral A/B choices. The swap-control
panel separately checks position sensitivity.

For OBJ triggers, analyses condition on valid initially correct
answers and measure truth departure, persistence, recovery, turn-to-switch, and
confidence trajectories. Revised rates require valid initial and final
answers; paired contrasts additionally require both conditions to be valid
with the same initial answer. Legacy plotting helpers do not consistently
exclude invalid final answers, so their retained summaries are not substitutes
for the revised denominators. For SUB triggers, analyses operate only
over committed A/B states and measure switching or pressure accommodation
without accuracy claims.

Response-annotation metrics follow the illustrated scoring schema (currently
Figure 4 in the manuscript): Social Sycophancy Scale
scores plus binary labels for whether the model redid the question by reasoning
or calculation, rationalized without task-grounded support, or contradicted
itself. LLM judge runs are reliability diagnostics and do not replace human
adjudication.

The revised OpenAI-only `sample@k` diagnostic preserves the original paired
events: OBJ is neutral-correct AND framed-wrong over **all matched samples**,
including neutral-incorrect samples as non-events. This is joint event
incidence, not the main analysis's conditional truth-departure rate. SUB
requires A-directed A AND B-directed B. `any@k` is the fraction of matched
item-cue units with at least one event in the first k samples. Runner-level
success summaries and `est_pass_pct@k` are not substitutes for these events.

Revision intervals use 10,000 base-item cluster bootstrap replicates, retaining
all cues, samples, and matched model/condition responses within each item.
They are percentile 95% intervals for the fixed model panel, not independent
response intervals. The revised @k analysis reports paired model differences
without p-values or q-values.

## Completed Opus-4.6 Re-check

A separate `anthropic/claude-opus-4.6` cohort used OpenRouter with Amazon
Bedrock pinned from the first request and provider fallbacks disabled. It
covered 40 OBJ and 40 SUB items, ten per domain. The completed run submitted
790 calls, all HTTP 200, with zero retries or API errors, yielding 355 valid
condition pairs. One truncated OBJ initial lacked a final answer, so ten
followups were skipped and five pairs excluded from the planned 360 pairs.

Across three certainty-pressure conditions, baseline versus re-check outcomes
were OBJ truth-departure 17/96 (17.71%) versus 8/96 (8.33%), and SUB switching
34/120 (28.33%) versus 13/120 (10.83%). The OBJ denominator covers 32 initially
correct items; the SUB denominator covers 40 items. Domain-stratified paired
item-cluster intervals keep the three conditions together within each item.
Both arms corrected 7/7 initially incorrect OBJ answers after an explicitly
supplied answer key. This is answer-key uptake, not independent reasoning
verification; no SUB positive-updating result is claimed.

This completed cohort is separate from the frozen Opus-4.5 main panel and
earlier stopped or other-model re-check cohorts. Never pool their results.
Raw request/response logs and local audit evidence remain ignored and must
not be published with the codebase.

The re-check configuration uses a 4,096-token response cap, a 90-second request
timeout, at most three attempts for transient failures, and at most six
concurrent requests per model. Temperature and reasoning parameters are not
overridden. Responses retain the standard 1--5 confidence field. Provider or
permission failures stop the affected run without rerouting.

### Reproduce The Supplementary Experiment

Use Python 3.11 or newer, install the tested revision dependencies, and set
`OPENROUTER_API_KEY` locally. Use a new output directory for a new run; do not
overwrite the completed cohort. Preparation verifies the public model/provider
catalog and the benchmark inputs before any inference calls. Smoke must pass
before the full run is admitted.

```powershell
python -m pip install -r Experimental/requirements-revision.txt
python Experimental/revision_recheck_experiment.py prepare --models anthropic/claude-opus-4.6 --provider "Amazon Bedrock" --output Experimental/results/revision_20260908/recheck_opus46_bedrock/reproduction
python Experimental/revision_recheck_experiment.py smoke --models anthropic/claude-opus-4.6 --provider "Amazon Bedrock" --output Experimental/results/revision_20260908/recheck_opus46_bedrock/reproduction
python Experimental/revision_recheck_experiment.py full --models anthropic/claude-opus-4.6 --provider "Amazon Bedrock" --output Experimental/results/revision_20260908/recheck_opus46_bedrock/reproduction
```

## Preflight Checks

Run the panel audit before any model batch:

```powershell
python Experimental/data/helper/audit_supersycophantic_panels.py
```

This audit must fail on duplicate IDs, duplicate question text, bad
OBJ source fields, invalid choices, prompt-format drift, trigger
direction leakage, or missing confidence requirements.

Context-panel rebuild and auxiliary audits:

```powershell
python Experimental/data/helper/build_supersycophantic_context_panels.py --write --audit Experimental/data/context_source_traceability_audit.md
python Experimental/data/helper/audit_context_panel_integrity.py --report Experimental/data/context_panel_integrity_audit.md
python Experimental/data/helper/audit_context_framing_naturalness.py --report Experimental/data/context_framing_naturalness_audit.md
```

The generated audit markdown files are local artifacts and are ignored.

## Running Evaluations

Run from the repository root. Set provider credentials in the local
environment; do not commit them.

Context smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File Experimental/run_context_eval_and_report.ps1 -Models smoke -MaxGt 2 -MaxNgt 2 -DryRun
```

Full context wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File Experimental/run_context_eval_and_report.ps1 -Models main
```

Trigger smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File Experimental/run_trigger_eval_and_report.ps1 -Models smoke -MaxItems 2 -SkipTemporal -DryRun
```

Full trigger wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File Experimental/run_trigger_eval_and_report.ps1 -Models main
```

Subjective-opinion swap-control panel:

```powershell
python Experimental/data/helper/build_ngt_swap_control.py --write
python Experimental/run_context.py --ngt-input data/supersycophantic_context_ngt_swap_control_100.json --max-gt 0 --models main --output results/context_ngt_swap.jsonl.gz --summary results/context_ngt_swap_summary.json
python Experimental/summarize_ngt_swap_control.py --input Experimental/results/context_ngt_swap.jsonl.gz --summary-json Experimental/results/context_ngt_swap_control_summary.json --summary-csv Experimental/results/context_ngt_swap_control_summary.csv
```

OpenAI-only `@k` diagnostics:

```powershell
python Experimental/run_openai_samplek.py --input Experimental/data/supersycophantic_context_gt_200.json --output Experimental/results/samplek/gt.jsonl.gz --summary-json Experimental/results/samplek/gt_summary.json --summary-csv Experimental/results/samplek/gt_summary.csv --variant-set all --rerun-invalid
python Experimental/run_openai_samplek.py --input Experimental/data/supersycophantic_context_ngt_100.json --output Experimental/results/samplek/ngt.jsonl.gz --summary-json Experimental/results/samplek/ngt_summary.json --summary-csv Experimental/results/samplek/ngt_summary.csv --variant-set all --rerun-invalid
```

LLM judge runners write local output directories. Keep those raw directories
out of the release. Only small, reviewed code-facing calibration summaries
belong in GitHub; manuscript tables belong exclusively in Overleaf.

## Offline Revision Analyses

Run these commands from the repository root with the corresponding raw inputs
available locally. They analyze saved responses and make no model calls.
Inputs and outputs remain ignored; do not publish raw logs or provider dumps.

```powershell
python Experimental/revision_paired_analysis.py --original-dir Experimental/results --supplementary-dir Experimental/results --out-dir Experimental/results/revision_20260908
python Experimental/revision_samplek_analysis.py --gt-input Experimental/results/samplek/gt.jsonl.gz --ngt-input Experimental/results/samplek/ngt.jsonl.gz --out-dir Experimental/results/revision_20260908/samplek
python Experimental/revision_trace_audit.py
python Experimental/revision_recheck_experiment.py summarize --models anthropic/claude-opus-4.6 --provider "Amazon Bedrock" --output Experimental/results/revision_20260908/recheck_opus46_bedrock
```

The paired analysis expects the original `context_20260504_184050_context_main`
and `trigger_20260504_070840` raw files in `--original-dir`. Supplementary
inputs use the separate `rebuttal_openrouter_*` filenames. It labels missing
or corrupt originals unavailable rather than substituting supplementary data.
The @k inputs are `gt.jsonl.gz` and `ngt.jsonl.gz` under `results/samplek`.
Both paired analysis scripts validate complete gzip/JSON files and matched
grids, reject corrupt prefixes, and record source hashes and denominators.
The trace audit writes under `Experimental/results/revision_20260908/trace`;
the re-check summarizer requires that cohort's saved manifest and responses.
Its `summarize` phase does not launch or resume an experiment.

Offline revision tests:

```powershell
python -m unittest discover -s Experimental -p "test_revision*.py"
```

## Generated Artifacts

Ignored scratch locations:

```text
Experimental/results/
Experimental/reports/
Experimental/data/*_audit.md
Experimental/IAA/llm_judge_*/
Experimental/IAA/*-8-model/
images/*_latest_*.png
images/model_logos/model_badges/_preview.png
```

Paper-only images and result figures are maintained in Overleaf. Do not add
`images/Figure*.png` or `images/results/` to codebase commits unless the user
explicitly asks for a code-facing release asset.
