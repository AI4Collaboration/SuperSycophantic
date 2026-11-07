# Human/LLM Judge Calibration

Calibration transcripts and annotations are local inputs, not release data.
The core benchmark runners use the frozen panels in `Experimental/data/`
and do not require this directory's local files.

Supply local calibration inputs when running the optional judge-agreement
analyses. The historical input filename is `Human-LLM-Judge-IAA-100.json`.
JSON inputs, raw judge outputs, and generated agreement summaries remain
ignored; only analysis code and these instructions are published.
