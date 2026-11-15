# SuperSycophantic

Code and data for testing how user framing and follow-up pressure affect
language-model answers. SuperSycophantic covers objective (OBJ) questions with
verified answers and subjective (SUB) decisions with two defensible options,
from first-turn framing to static and adaptive multi-turn challenges.

## Installation

Requires Python 3.11 or newer. Run commands from the repository root.

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS, then install dependencies:

```sh
python -m pip install -r requirements.txt
```

For live evaluations, set `OPENROUTER_API_KEY` in the environment or in a local
`.env` file using [.env.example](.env.example). Existing environment variables
take precedence. Model calls incur provider charges.

## Quick Start

Check the data and run a small prompt-only evaluation without API calls:

```sh
python Experimental/data/helper/audit_supersycophantic_panels.py
python Experimental/run_context.py --models smoke --max-gt 2 --max-ngt 2 --dry-run
python Experimental/run.py eval --input data/supersycophantic_trigger_gt_neutral_200.jsonl --models smoke --max-items 2 --triggers authority --tones mild --dry-run --output results/smoke/obj_trigger.jsonl.gz
```

Dry runs save planned prompts. Use a new output path for a live run.

Run context evaluations on the nine-model panel:

```sh
python Experimental/run_context.py --models main
```

Run the complete static, adaptive, and multi-turn trigger evaluations using
the PowerShell wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File Experimental/run_trigger_eval_and_report.ps1 -Models main
```

The [evaluation guide](Experimental/README.md) covers scoring, provider routing,
additional experiments, and analysis of saved results. Each runner also accepts
`--help`.

## Data

The included inputs are ready to use; source-cache downloads are only needed
when rebuilding the mixed OBJ panel.

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

OBJ contains 50 items per science domain: mathematical, physical, chemical,
and biomedical. Each domain has 25 MMLU-Pro and 25 HLE-Verified items. SUB
contains 25 decisions per domain: policy, moral dilemmas, interpersonal
relations, and professional planning. SUB has no hidden correct answer.

Data files retain the legacy `gt` and `ngt` names for OBJ and SUB. Builders and
validation scripts are in `Experimental/data/helper/`; see the
[source-cache instructions](Experimental/data/hle_verified/README.md) for
rebuilding the mixed panel.

## Models

[`Experimental/models.py`](Experimental/models.py) defines `--models main`:
GPT-5.4, GPT-5.4-mini, GPT-5.4-nano, Claude Opus-4.5, Sonnet-4.5, Haiku-4.5,
Gemini-3.1-Flash-Lite Preview, Mistral-Medium-3.1, and Command-R.
`--models smoke` selects Gemini-3.1-Flash-Lite Preview.

The default adaptive generator is GPT-5.4-mini. Main runner defaults are 200
concurrent requests, a 30-second timeout, and at most eight attempts.
Anthropic targets use the direct Messages API when an Anthropic key is present;
set `OPENROUTER_ONLY=1` to route all targets through OpenRouter.

## Tests

The tests and data audit run without model API calls:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest Experimental -q
python Experimental/data/helper/audit_supersycophantic_panels.py
python Experimental/audit_release.py --revision HEAD
```

The release check scans committed files for credentials, private paths, and
generated conversations. Use `--include-untracked` to check new files before
staging them.

## Repository Contents

- `Experimental/run*.py`: evaluation runners.
- `Experimental/revision_*.py` and `Experimental/rebuttal_*.py`: paired analyses,
  re-checking experiments, and additional controls.
- `Experimental/plot_*.py` and `Experimental/figures/`: plotting scripts.
- `label_studio/`: annotation-agreement calculation code.
- `images/`: small assets used by plotting scripts.

Run outputs, trajectories, logs, and annotation exports stay local and are
excluded from the repository. The code is available under the [MIT license](LICENSE).
