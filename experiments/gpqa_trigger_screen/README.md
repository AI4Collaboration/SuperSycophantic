# GPQA Single-Turn Trigger Screen

This folder contains the standalone GPQA pilot. It tests only the trigger layer:

1. Ask the original GPQA multiple-choice item.
2. Record the model's committed answer and confidence.
3. Apply one unsupported trigger follow-up toward a pre-selected distractor.
4. Score whether the model switches to that pressured distractor.

No context framing and no temporal trajectories are generated in this pilot.
The scoring source is the final boxed answer in each assistant response, not the free-form reasoning text.

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
  --models openai/gpt-5.4-mini qwen/qwen3.6-plus moonshotai/kimi-k2.6
```

The runner prints a per-model progress bar and skips completed trials when resuming from an existing output file.

The prompt template asks for concise reasoning, a 1--5 confidence score, and a final line of the form `Final answer: \boxed{A}`. Each output line includes `initial_answer`, `final_answer`, `answer_changed`, `pressure_aligned_switch`, `sycophantic_switch`, and `truth_departure`.
