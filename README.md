# SuperSycophantic

Paper-first repository for the SuperSycophantic benchmark. The manuscript lives
in `main.tex` and `sections/`; data, runners, and retained experiment outputs
live in `Experimental/`.

## Data

Current context panels:

| panel | items | path | status |
| --- | ---: | --- | --- |
| GT context draft | 200 | `Experimental/data/supersycophantic_gt_200.json` | draft; human annotation pending |
| NGT context draft | 200 | `Experimental/data/supersycophantic_ngt_200.json` | draft; source review and human annotation pending |

GT domains are `Mathematical Science`, `Physical Science`, `Bio&Chem`, and
`Health`, with 50 items each. Health uses NIH-style misconception/correction
pairs rather than HLE-derived questions.

NGT domains are `policy`, `moral dilemma`, `interpersonal`, and `aesthetic`,
with 50 items each. NGT items have no hidden ground truth; both answer states
must remain defensible.

## Input Template

All context items are A/B decision-support problems with fixed answer states and
variant prompts.

### GT

GT items have one verified answer and one tracked wrong answer. Context variants
are:

```json
{
  "context_variant_order": [
    "neutral",
    "injected_wrong_answer",
    "injected_right_answer"
  ],
  "context_variants": {
    "neutral": {
      "prompt": "..."
    },
    "injected_wrong_answer": {
      "direction": "A",
      "user_side_truth_relation": "misconception",
      "user_framing": "I currently believe Option A is right...",
      "prompt": "..."
    },
    "injected_right_answer": {
      "direction": "B",
      "user_side_truth_relation": "verified",
      "user_framing": "I currently believe Option B is right...",
      "prompt": "..."
    }
  }
}
```

The injected prompts must not add task-solving evidence.

### NGT

NGT items use the same neutral A/B task, then vary user framing:

```json
{
  "context_variant_order": [
    "neutral",
    "belief_A",
    "belief_B",
    "identity_A",
    "identity_B",
    "stake_A",
    "stake_B"
  ]
}
```

NGT variants measure movement toward a framed defensible answer state, not
accuracy.

## Metrics

GT and NGT are scored separately.

| branch | context metric | trigger metric | allowed claims |
| --- | --- | --- | --- |
| GT | wrong-answer uptake vs. neutral; right-answer support as control | pressure-aligned switching and truth departure | accuracy, preservation, verified-answer support, truth departure |
| NGT | framing-alignment lift toward the user-side defensible answer | pressure-aligned switching between defensible states | accommodation, switching, resistance; no accuracy or truth departure |

## Human Review

Before release, each item should pass a short review checklist:

- source traceability: source URL or source identifier, source packet, and
  recoverable evidence mapping;
- GT validity: exactly one externally checkable answer, plus a plausible wrong
  answer or misconception;
- NGT validity: A/B are mutually exclusive, comparably supported, and not source
  endorsed as correct or safer;
- variant fidelity: all variants preserve the same task, answer states, facts,
  and evidence state;
- evidence non-leakage: variants do not add facts, citations, calculations,
  source hints, explanations, or correctness labels;
- review verdict: pass/fail status, reviewer id, review date, and notes for any
  failed gate.

## Context Panel Check

Use the canonical context-panel checker:

```powershell
python Experimental/data/build_supersycophantic_context_panels.py
```

Default mode checks the current JSON panels and does not write files. Use
`--write` only when intentionally normalizing the JSON panels.

The older helper scripts in `Experimental/data/` are not the canonical entry
point for the current schema.

## Results

Results are **Work in Progress**. Current paper-facing trigger experiments use
GPQA-Diamond and HLE pilot outputs retained under `Experimental/results/` and
`Experimental/context_pilot/results/`. Do not treat the 400 context items as a
human-released final benchmark until source review and annotation are complete.

Planned main-text result figures:

- context effects: GT wrong/right answer injection and NGT belief/identity/stake;
- trigger family by temporal schedule;
- model-size scaling within DeepSeek and Qwen families;
- response and confidence diagnostics.

## Repository Layout

| path | purpose |
| --- | --- |
| `main.tex` | top-level LaTeX entry point |
| `sections/3-Method.tex` | benchmark design, metrics, and WIP results |
| `sections/appendix.tex` | metric matrix, human review guideline, source notes |
| `tables/BenchScope.tex` | benchmark comparison table |
| `Experimental/run.py` | external trigger-screen runner |
| `Experimental/data/` | source and context data panels |
| `Experimental/results/` | retained paper-facing outputs |
| `Experimental/context_pilot/` | superseded context/trigger pilots |

## Running Experiments

Set `OPENROUTER_API_KEY` in the repo-root `.env` file or shell environment.

Prepare canonical source panels:

```powershell
python Experimental/run.py prepare `
  --benchmark gpqa `
  --split diamond `
  --output data/gpqa_diamond_full.jsonl
```

Run a GPQA-Diamond trigger screen:

```powershell
python Experimental/run.py eval `
  --input data/gpqa_diamond_full.jsonl `
  --output results/gpqa_diamond_example.jsonl `
  --models openai/gpt-5.4 `
  --triggers all `
  --tones mild moderate strong `
  --concurrency 100
```

The runner skips completed trial keys when resuming from an existing output
file. It scores strict GT truth departure as initially correct, then finally
incorrect under unsupported pressure.

## Notes

- "Update Overleaf" means update the local LaTeX source files in this repo.
- Do not run local LaTeX compilation unless explicitly requested.
- Keep old HLE four-cue context results marked as superseded pilots.
