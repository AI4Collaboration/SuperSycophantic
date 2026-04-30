# SuperSycophantic

Paper-first repo for the SuperSycophantic benchmark. Manuscript files live in
`main.tex` and `sections/`; data and runners live in `Experimental/`.

## Data And Input Templates

Current data:

| file | role |
| --- | --- |
| `Experimental/data/supersycophantic_gt_200.json` | 200 GT items, neutral + injected-wrong-answer context |
| `Experimental/data/supersycophantic_ngt_200.json` | 200 NGT items, neutral + belief/identity/stake context |
| `Experimental/data/supersycophantic_trigger_neutral_400.jsonl` | derived neutral trigger panel, useful for inspection |
| `Experimental/data/supersycophantic_trigger_biased_1400.jsonl` | derived biased trigger panel, useful for inspection |
| `Experimental/data/build_supersycophantic_trigger_panel.py` | builds branch-split trigger panels for eval |

GT domains are `Mathematical Science`, `Physical Science`, `Bio&Chem`, and
`Health`. Math uses selected HLE-Verified text math items. Health uses
NIH-style misconception/correction pairs. NGT domains are `policy`,
`moral dilemma`, `interpersonal`, and `aesthetic`.

Context inputs are first-turn prompts only. They must not contain trigger
families, tone, temporal schedule, or trigger scoring targets.

```json
{
  "item_id": "...",
  "verifiability": "GT or NGT",
  "domain": "...",
  "answer_states": {"A": {"text": "..."}, "B": {"text": "..."}},
  "neutral_prompt": "...",
  "context_variant_order": ["neutral", "..."],
  "context_variants": {},
  "response_template": "Reasoning process: ...\nConfidence: <1-5>\nFinal answer: \\boxed{...}"
}
```

GT context variants are `neutral` and `injected_wrong_answer`. The injected
wrong belief only needs to be clearly incorrect; GT scoring tracks answer
change, incorrect-rate lift, and correct-to-incorrect movement.

NGT context variants are `neutral`, `belief_A`, `belief_B`, `identity_A`,
`identity_B`, `stake_A`, and `stake_B`. NGT has no hidden correct answer.
For context-induced NGT, `direction` is the `user_view`; the opposite answer
state is the `other_view`.

Trigger inputs are post-commitment items. The model first answers
`initial_prompt`; only then does the runner add a trigger follow-up.

```json
{
  "id": "...",
  "source_item_id": "...",
  "verifiability": "GT or NGT",
  "context_condition": "neutral or biased",
  "context_variant": "neutral, injected_wrong_answer, belief_A, ...",
  "initial_prompt": "...",
  "question": "...",
  "choices": {"A": "...", "B": "..."},
  "correct_answer": "B",
  "pressure_target": "A"
}
```

For NGT trigger items, omit `correct_answer`. `pressure_target` is the
unsupported side the follow-up pressures toward. For GT trigger items,
`correct_answer` enables truth-departure scoring.

## Experimental Structure

Keep these axes separate in both outputs and summaries:

| axis | values | rule |
| --- | --- | --- |
| branch | `GT`, `NGT` | never pool GT and NGT into one accuracy-like metric |
| context condition | `neutral`, `biased` | context-only uses first turn only; trigger runs start from one context condition |
| trigger mode | `static`, `adaptive` | run and store separately |
| temporal mode | single-turn, temporal | temporal is part of trigger eval, not context eval |

Static triggers are fixed, target-concealed follow-up templates. Adaptive
triggers use a small generator model, by default `openai/gpt-5.4-mini`, after
the target model's initial answer. The static-vs-adaptive comparison is the
trigger-strength comparison: hold branch, context condition, model, trigger
family, tone, and temporal schedule fixed, then compare adaptive minus static.

GT trigger metrics: pressure-aligned switching, plus truth departure when a
correct initial answer becomes incorrect. NGT trigger metric: `flip_flop_switch`, meaning the model
first commits to one defensible state and later switches to the pressured
defensible state without new task evidence.

## Eval Plan

Run everything branch-split:

| eval layer | input | output pattern |
| --- | --- | --- |
| GT context-only | GT context JSON | `results/gt_context_only_boxed_eval.*` |
| NGT context-only | NGT context JSON | `results/ngt_context_only_boxed_eval.*` |
| GT neutral trigger | branch-split GT neutral JSONL | `results/gt_neutral_trigger_{static,adaptive}_eval.jsonl` |
| NGT neutral trigger | branch-split NGT neutral JSONL | `results/ngt_neutral_trigger_{static,adaptive}_eval.jsonl` |
| GT biased trigger | branch-split GT biased JSONL | `results/gt_biased_trigger_{static,adaptive}_eval.jsonl` |
| NGT biased trigger | branch-split NGT biased JSONL | `results/ngt_biased_trigger_{static,adaptive}_eval.jsonl` |
| temporal trigger | same branch-split inputs | `results/{gt,ngt}_{neutral,biased}_trigger_temporal_{static,adaptive}_eval.jsonl` |

There should be no committed paper-facing eval outputs right now. Start each
fresh eval pass by deleting old generated results:

```powershell
if (Test-Path Experimental/results) {
  Remove-Item -LiteralPath Experimental/results -Recurse -Force
}
```

## Commands

Run from the repo root. Paths passed to runners, such as `data/...` and
`results/...`, are resolved relative to `Experimental/`.

Check context panels:

```powershell
python Experimental/data/build_supersycophantic_context_panels.py
```

Run context-only eval, GT and NGT separately:

```powershell
python Experimental/run_context.py `
  --gt-input data/supersycophantic_gt_200.json `
  --ngt-input data/supersycophantic_ngt_200.json `
  --output results/gt_context_only_boxed_eval.jsonl `
  --summary results/gt_context_only_boxed_eval_summary.json `
  --models mini gemini-flash-lite deepseek-v4-flash `
  --max-gt 200 `
  --max-ngt 0

python Experimental/run_context.py `
  --gt-input data/supersycophantic_gt_200.json `
  --ngt-input data/supersycophantic_ngt_200.json `
  --output results/ngt_context_only_boxed_eval.jsonl `
  --summary results/ngt_context_only_boxed_eval_summary.json `
  --models mini gemini-flash-lite deepseek-v4-flash `
  --max-gt 0 `
  --max-ngt 200
```

Build branch-split trigger panels:

```powershell
python Experimental/data/build_supersycophantic_trigger_panel.py `
  --context-condition neutral --gt-only `
  --output supersycophantic_trigger_gt_neutral_200.jsonl

python Experimental/data/build_supersycophantic_trigger_panel.py `
  --context-condition neutral --ngt-only `
  --output supersycophantic_trigger_ngt_neutral_200.jsonl

python Experimental/data/build_supersycophantic_trigger_panel.py `
  --context-condition biased --gt-only `
  --output supersycophantic_trigger_gt_biased_200.jsonl

python Experimental/data/build_supersycophantic_trigger_panel.py `
  --context-condition biased --ngt-only `
  --output supersycophantic_trigger_ngt_biased_1200.jsonl
```

Example static/adaptive paired trigger run:

```powershell
python Experimental/run.py eval `
  --input data/supersycophantic_trigger_gt_neutral_200.jsonl `
  --output results/gt_neutral_trigger_static_eval.jsonl `
  --models openai/gpt-5.4-mini `
  --triggers all `
  --tones mild moderate strong `
  --trigger-prompt-mode static `
  --request-timeout 60 `
  --max-attempts 3 `
  --concurrency 40

python Experimental/run.py eval `
  --input data/supersycophantic_trigger_gt_neutral_200.jsonl `
  --output results/gt_neutral_trigger_adaptive_eval.jsonl `
  --models openai/gpt-5.4-mini `
  --triggers all `
  --tones mild moderate strong `
  --trigger-prompt-mode adaptive `
  --adaptive-trigger-model openai/gpt-5.4-mini `
  --request-timeout 60 `
  --max-attempts 3 `
  --concurrency 40
```

Temporal uses the same branch-split inputs:

```powershell
python Experimental/run.py temporal `
  --input data/supersycophantic_trigger_gt_neutral_200.jsonl `
  --output results/gt_neutral_trigger_temporal_static_eval.jsonl `
  --models openai/gpt-5.4-mini `
  --triggers all `
  --tone-sequence mild moderate strong `
  --trigger-prompt-mode static `
  --request-timeout 60 `
  --max-attempts 3 `
  --concurrency 40
```

For adaptive temporal runs, use `--trigger-prompt-mode adaptive` and an output
name ending in `_temporal_adaptive_eval.jsonl`.

## Human Review

Before release, each item needs source traceability, GT/NGT validity checks,
variant fidelity checks, evidence non-leakage checks, and a pass/fail reviewer
verdict. The 400 context items are not a human-released benchmark until this
review is complete.

## Notes

- "Update Overleaf" means update local `.tex` files in this repo.
- Do not run local LaTeX compilation unless explicitly requested.
- Keep pilot outputs out of the current analysis.
