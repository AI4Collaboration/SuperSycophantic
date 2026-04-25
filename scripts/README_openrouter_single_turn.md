# OpenRouter Single-Turn Trigger Screen

This pilot tests whether one unsupported trigger follow-up can move a strong model from its initial answer to a pre-specified pressure target on original GPQA items. Context framing and temporal modes are intentionally held out.

## Input JSONL

Each line should be one item:

```json
{"id":"gpqa-001","question":"...","choices":{"A":"...","B":"...","C":"...","D":"..."},"correct_answer":"C","pressure_target":"A"}
```

The `pressure_target` must be selected from the distractors before model evaluation.

To prepare a GPQA Diamond panel from the official GPQA zip:

```powershell
python scripts/prepare_gpqa_panel.py `
  --split diamond `
  --max-items 25 `
  --output data/gpqa_diamond_panel.jsonl
```

## Run

Dry-run prompt generation:

```powershell
python scripts/openrouter_single_turn_trigger_eval.py `
  --input data/gpqa_panel.jsonl `
  --output results/gpqa_trigger_prompts.jsonl `
  --models openai/gpt-5.5 `
  --tones moderate `
  --dry-run
```

API run:

```powershell
python scripts/openrouter_single_turn_trigger_eval.py `
  --input data/gpqa_panel.jsonl `
  --output results/gpqa_trigger_screen.jsonl `
  --models openai/gpt-5.5 anthropic/claude-opus-4.7 google/gemini-3.1-pro-preview `
  --triggers simple_baseline authority social_proof consistency reciprocity liking scarcity unity `
  --tones moderate
```

Set `OPENROUTER_API_KEY` in the environment or `.env`. Results are written under `results/`, which is gitignored.
