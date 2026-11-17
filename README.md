# SuperSycophantic

Code and data for testing language-model sycophancy under user framing and
multi-turn pressure. OBJ questions have verified answers; SUB decisions have
two defensible options.

[NeurIPS 2026 Sydney poster](poster/SuperSycophantic-NeurIPS-2026-Sydney.pdf) · [Print and website files](poster/README.md)

## Installation

Python 3.11+. Run commands from the repository root.

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS, then install dependencies:

```sh
python -m pip install -r requirements.txt
```

Set `OPENROUTER_API_KEY` for live evaluations; see [.env.example](.env.example).
Model calls incur provider charges.

## Quick Start

Check the data and run a small prompt-only evaluation without API calls:

```sh
python Experimental/data/helper/audit_supersycophantic_panels.py
python Experimental/run_context.py --models smoke --max-gt 2 --max-ngt 2 --dry-run
python Experimental/run.py eval --input data/supersycophantic_trigger_gt_neutral_200.jsonl --models smoke --max-items 2 --triggers authority --tones mild --dry-run --output results/smoke/obj_trigger.jsonl.gz
```

Context dry runs print planned counts; trigger dry runs save planned prompts.
Use a new output path for a live trigger run.

Run context evaluations on the nine-model panel:

```sh
python Experimental/run_context.py --models main
```

Run the complete static, adaptive, and multi-turn trigger evaluations using
the PowerShell wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File Experimental/run_trigger_eval_and_report.ps1 -Models main
```

See the [evaluation guide](Experimental/README.md) for scoring, provider routing,
controls, and saved-result analyses. Runners accept `--help`.

## Data

Inputs are ready to use. Source caches are needed only to rebuild OBJ data.

| Files under `Experimental/data/` | Contents |
| --- | --- |
| `supersycophantic_context_gt_200.json` | 200 OBJ items with context variants |
| `supersycophantic_context_ngt_100.json` | 100 SUB decisions with context variants |
| `supersycophantic_context_ngt_swap_control_100.json` | SUB A/B position-swap control |
| `supersycophantic_trigger_gt_neutral_200.jsonl` | Neutral OBJ trigger inputs |
| `supersycophantic_trigger_ngt_neutral_100.jsonl` | Neutral SUB trigger inputs |
| `supersycophantic_mixed_gt_200.jsonl` | Mixed-source OBJ panel |
| `mmlu_pro_release_gt_100.jsonl` | MMLU-Pro subset |
| `rebuttal_positive_control/positive_update_gt_hard40.jsonl` | 40-item OBJ updating control |

OBJ has 50 items per science domain: mathematical, physical, chemical, and
biomedical, each split equally between MMLU-Pro and HLE-Verified. SUB has 25
decisions each in policy, moral dilemmas, interpersonal relations, and
professional planning, with no hidden correct answer.

Filenames use `gt` for OBJ and `ngt` for SUB. Builders and audits are in
`Experimental/data/helper/`; see [source-cache setup](Experimental/data/hle_verified/README.md).

## Models

[`Experimental/models.py`](Experimental/models.py) defines `--models main`:
GPT-5.4, GPT-5.4-mini, GPT-5.4-nano, Claude Opus-4.5, Sonnet-4.5, Haiku-4.5,
Gemini-3.1-Flash-Lite Preview, Mistral-Medium-3.1, and Command-R.
`--models smoke` selects Gemini-3.1-Flash-Lite Preview.

Default adaptive generator: GPT-5.4-mini. Runner defaults: concurrency 200,
timeout 30 seconds, maximum eight attempts.
Anthropic targets use the direct Messages API when an Anthropic key is present;
set `OPENROUTER_ONLY=1` to route all targets through OpenRouter.

## Tests

Offline checks:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest Experimental -q
python Experimental/data/helper/audit_supersycophantic_panels.py
python Experimental/audit_release.py --revision HEAD
```

The release audit checks for credentials, private paths, and raw model outputs.
Use `--include-untracked` to inspect new files. Trajectories, logs, and annotation
exports stay local. Code: [MIT license](LICENSE).
