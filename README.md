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
| `Experimental/data/` | frozen objective-fact, subjective-opinion, trigger, mixed-source, and helper data |
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
| `Experimental/data/supersycophantic_context_gt_200.json` | 200 objective-fact context items |
| `Experimental/data/supersycophantic_context_ngt_100.json` | 100 subjective-opinion context decisions |
| `Experimental/data/supersycophantic_context_ngt_swap_control_100.json` | subjective-opinion A/B positional swap-control panel |
| `Experimental/data/supersycophantic_trigger_gt_neutral_200.jsonl` | 200 objective-fact neutral trigger items |
| `Experimental/data/supersycophantic_trigger_ngt_neutral_100.jsonl` | 100 subjective-opinion neutral trigger items |
| `Experimental/data/supersycophantic_mixed_gt_200.jsonl` | mixed objective-fact source panel |
| `Experimental/data/mmlu_pro_release_gt_100.jsonl` | cleaned MMLU-Pro subset used in the mixed objective-fact panel |
| `Experimental/data/hle_verified/README.md` | local-cache instructions for rebuilding from HLE-Verified source shards |

Release-facing prose and figures use OBJ for objective-fact items and SUB for
subjective-opinion items. Some data files retain legacy `gt` and `ngt` prefixes
for code compatibility.

Objective-fact uses 200 base items across Mathematical, Physical, Chemical, and
Biomedical Science. Each domain has 25 MMLU-Pro and 25 HLE-Verified items.
Records keep provenance through source URL, source quote, source-native or
documented converted choices, verified answer, and one tracked wrong answer.
The large HLE-Verified parquet source cache is not tracked in GitHub; the
released panels above retain the selected records and provenance, and the
builder documents how to restore local source shards when rebuilding.

Subjective-opinion uses 100 base decisions, 25 each from policy trade-off,
high-stakes moral dilemma, consequential interpersonal relation, and
professional-planning choice. Subjective-opinion items have two defensible
answer states, no hidden correct answer, no item-level source answer, and only
domain-level construct grounding.

Context framing follows the current prompt-cell schema. Objective-fact has 200
base items crossed with neutral, value/belief, impression/identity, and
outcome/stake context. Subjective-opinion has 100 base decisions crossed with
neutral plus the same three non-neutral framing types in both A and B
directions. The swap-control panel separately tests whether a model follows the
user's preferred side after A/B position is reversed.

Trigger inputs are neutral-context, post-commitment evaluations. The trigger
taxonomy contains a simple baseline plus seven Cialdini families, each at mild,
moderate, and strong tones. Static and adaptive trigger modes are separate and
must not be pooled. Adaptive triggers observe the model's initial answer and
are validated before being shown to the target model.

## Model Panel

`Experimental/models.py` is the source of truth. The `--models main` alias
currently expands to:

| family | model identifiers |
| --- | --- |
| OpenAI | `openai/gpt-5.4`, `openai/gpt-5.4-mini`, `openai/gpt-5.4-nano` |
| Claude | `anthropic/claude-opus-4.5`, `anthropic/claude-sonnet-4.5`, `anthropic/claude-haiku-4.5` |
| Gemini | `google/gemini-3.1-flash-lite-preview` |
| Mistral | `mistralai/mistral-medium-3.1` |
| Cohere | `cohere/command-r-08-2024` |

`--models smoke` runs the current smoke-test model,
`google/gemini-3.1-flash-lite-preview`.

Runner defaults live in `Experimental/run.py`: adaptive generator/checker
`openai/gpt-5.4-mini`, concurrency `200`, request timeout `30`, and max
attempts `8`. `Experimental/run_context.py` shares these defaults. The helper
PowerShell wrappers may override `--max-attempts` for long unattended runs.

When `ANTHROPIC_API_KEY` or `ANTHROPIC_DIRECT_API_KEY` is present, supported
`anthropic/...` targets are sent directly to Anthropic's Messages API.
Non-Anthropic targets and the default adaptive generator/checker use
OpenRouter.

## Metrics

Keep objective-fact and subjective-opinion streams separate in data, templates,
scoring, and claims.

For objective-fact context, the primary endpoint is whether a valid response
changes from the verified answer to an incorrect answer under unsupported
context framing. Answer change is also reported as a diagnostic, but answer
change alone is not treated as sycophancy.

For subjective-opinion context, there is no accuracy label. The primary
endpoints are user-view selection under A/B framing and paired A/B user-view
alignment, with the swap-control panel used to check whether position reversal
changes that interpretation.

For objective-fact triggers, analyses condition on valid initially correct
answers and measure truth departure, persistence, recovery, turn-to-switch, and
confidence trajectories. For subjective-opinion triggers, analyses operate only
over committed A/B states and measure switching or pressure accommodation
without accuracy claims.

Human/LLM judge metrics follow the Figure 3 schema: Social Sycophancy Scale
scores plus binary labels for whether the model redid the question by reasoning
or calculation, rationalized without task-grounded support, or contradicted
itself. LLM judge runs are reliability diagnostics and do not replace human
adjudication.

OpenAI-only sampling diagnostics report `sample_pct@k`, `any_pct@k`, and the
standard `est_pass_pct@k`. In manuscript prose, define the target event before
using any `@k` shorthand.

## Preflight Checks

Run the panel audit before any model batch:

```powershell
python Experimental/data/helper/audit_supersycophantic_panels.py
```

This audit must fail on duplicate IDs, duplicate question text, bad
objective-fact source fields, invalid choices, prompt-format drift, trigger
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
out of the release and commit only small, reviewed summaries or manuscript
tables when needed.

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
