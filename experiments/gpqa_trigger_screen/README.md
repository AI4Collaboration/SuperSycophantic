# GPQA Single-Turn Trigger Screen

This folder contains the standalone GPQA pilot. It tests only the trigger layer:

1. Ask the original GPQA multiple-choice item.
2. Record the model's committed answer and confidence.
3. Apply one unsupported trigger follow-up toward a pre-selected distractor.
4. Score whether the model switches to that pressured distractor.

No context framing and no temporal trajectories are generated in this pilot.

## Files

- `run.py`: prepares GPQA panels and runs/dry-runs OpenRouter calls.
- `data/`: local generated GPQA panels, gitignored.
- `results/`: local prompt previews and API outputs, gitignored.

## Setup

Set `OPENROUTER_API_KEY` in the repo-root `.env` file or in the shell environment. The key is not committed.

## Prepare GPQA

This downloads the official GPQA zip, reads GPQA Diamond, shuffles answer options with a fixed seed, and assigns the first shuffled distractor as `pressure_target`.

```powershell
python experiments/gpqa_trigger_screen/run.py prepare `
  --split diamond `
  --max-items 25 `
  --output data/gpqa_diamond_panel.jsonl
```

## Dry-Run Prompts

```powershell
python experiments/gpqa_trigger_screen/run.py eval `
  --input data/gpqa_diamond_panel.jsonl `
  --output results/gpqa_trigger_prompts.jsonl `
  --models openai/gpt-5.5 `
  --dry-run
```

## Run OpenRouter

```powershell
python experiments/gpqa_trigger_screen/run.py eval `
  --input data/gpqa_diamond_panel.jsonl `
  --output results/gpqa_trigger_screen.jsonl `
  --models openai/gpt-5.5 anthropic/claude-opus-4.7 google/gemini-3.1-pro-preview
```

Each output line includes `initial_answer`, `final_answer`, `eligible`, `pressure_aligned_switch`, and `truth_departure`.
