# MMLU-Pro saturated screening selection summary

Screening models: openai/gpt-5.4-mini, google/gemini-3.1-flash-lite-preview
Saturated screening rule: both screening models correct on a source-style neutral first-turn screening prompt; up to 50 items per domain; correct-answer labels are balanced within each domain when possible. Final benchmark-prompt Pass@1 is re-estimated separately and should not be inferred from this screening pass.

| Domain | Candidates | Completed by both | Both correct | Selected |
| --- | ---: | ---: | ---: | ---: |
| Mathematical Science | 150 | 150 | 122 | 50 |
| Physical Science | 150 | 150 | 128 | 50 |
| Bio&Chem | 150 | 150 | 114 | 50 |
| Health | 150 | 150 | 107 | 50 |

## Selected correct-label distribution

| Domain | Label counts |
| --- | --- |
| Mathematical Science | A:5, B:5, C:5, D:5, E:5, F:5, G:5, H:5, I:5, J:5 |
| Physical Science | A:5, B:5, C:5, D:5, E:5, F:5, G:5, H:5, I:5, J:5 |
| Bio&Chem | A:5, B:5, C:5, D:5, E:5, F:5, G:5, H:5, I:5, J:5 |
| Health | A:5, B:5, C:5, D:5, E:5, F:5, G:5, H:5, I:5, J:5 |

## Source-stem quality exclusions

The following otherwise eligible Health candidates were excluded because the source question stem used recommendation/guideline wording that could cue an institutional-answer frame:
- MMLU_PRO-HEALTH-0023

## Per-model accuracy over completed candidate rows

| Model | Domain | Correct / Completed | Accuracy |
| --- | --- | ---: | ---: |
| `openai/gpt-5.4-mini` | Mathematical Science | 126/150 | 84.0% |
| `openai/gpt-5.4-mini` | Physical Science | 131/150 | 87.3% |
| `openai/gpt-5.4-mini` | Bio&Chem | 120/150 | 80.0% |
| `openai/gpt-5.4-mini` | Health | 113/150 | 75.3% |
| `google/gemini-3.1-flash-lite-preview` | Mathematical Science | 138/150 | 92.0% |
| `google/gemini-3.1-flash-lite-preview` | Physical Science | 133/150 | 88.7% |
| `google/gemini-3.1-flash-lite-preview` | Bio&Chem | 124/150 | 82.7% |
| `google/gemini-3.1-flash-lite-preview` | Health | 121/150 | 80.7% |
