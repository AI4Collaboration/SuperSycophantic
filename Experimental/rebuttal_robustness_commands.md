# Rebuttal Robustness Runs

These commands describe the original nine-model wording and generator controls.
They retain the original model identifiers rather than substituting new model
versions. The separate Opus-4.6 Amazon Bedrock re-check experiment and current
paired-analysis commands are documented in the repository README.

All commands are run from the repository root. The runner forces all model calls
through OpenRouter by setting `OPENROUTER_ONLY=1` and
`DISABLE_ANTHROPIC_DIRECT=1`.

## Tone-Confound Control

Full data, all main models, six controlled follow-ups:

```powershell
python Experimental\rebuttal_robustness_runner.py tone-confound `
  --models main `
  --output rebuttal_robustness_results\tone_confound_main_full_slow.jsonl.gz `
  --summary-dir rebuttal_robustness_results\main_summaries `
  --concurrency 36 `
  --per-model-concurrency 4 `
  --per-model-min-interval 0.15 `
  --request-timeout 90 `
  --max-attempts 10
```

## Adaptive-Generator Robustness

Balanced subset, all main target models, three OpenRouter generator models,
seven influence families, and all three tone levels:

```powershell
python Experimental\rebuttal_robustness_runner.py adaptive-generator `
  --models main `
  --generator-models openai/gpt-5.4-mini google/gemini-3.1-flash-lite-preview mistralai/mistral-medium-3.1 `
  --families authority social_proof consistency reciprocity liking scarcity unity `
  --tones mild moderate strong `
  --max-gt 8 `
  --max-sub 8 `
  --output rebuttal_robustness_results\adaptive_generator_main_8x8.jsonl.gz `
  --summary-dir rebuttal_robustness_results\main_summaries `
  --concurrency 36 `
  --per-model-concurrency 3 `
  --per-model-min-interval 0.18 `
  --request-timeout 90 `
  --max-attempts 10 `
  --generation-attempts 3
```

## Refresh Summaries

```powershell
python Experimental\rebuttal_robustness_runner.py summarize `
  --inputs rebuttal_robustness_results\tone_confound_main_full_slow.jsonl.gz rebuttal_robustness_results\adaptive_generator_main_8x8.jsonl.gz `
  --summary-dir rebuttal_robustness_results\main_summaries
```
