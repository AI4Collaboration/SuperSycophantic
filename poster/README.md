# NeurIPS 2026 Sydney Poster

- [Print PDF](SuperSycophantic-NeurIPS-2026-Sydney.pdf): one page, 140 cm wide by 90 cm high. Print at actual size (100%), not fit-to-page.
- [Website poster](SuperSycophantic-NeurIPS-2026-Sydney.png): opaque PNG within 5120 x 2880 pixels and 10 MB.
- [Thumbnail](thumbnail.png): within 320 x 256 pixels and 5 MB.

The print dimensions follow the [2026 main-conference instructions](https://neurips.cc/Conferences/2026/PosterInstructions). Website image limits follow the [conference platform's upload guide](https://wiki.eventhosts.cc/en/reference/posteruploads). Requirements checked on 16 November 2026. Files are prepared for upload; adding them here does not submit them to the conference or order printing.

## Rebuild

```sh
python -m pip install -r poster/requirements.txt
python poster/build_poster.py
```

The PDF contains vector text, plots and diagrams with embedded fonts. The script checks page dimensions, text bounds, font embedding and web image limits. It uses temporary files for plot assembly and retains only the three deliverables.

Content follows the camera-ready manuscript. Certainty-control plots use exact events and denominators from `tab:certainty_control`; the paired Opus-4.6 intervention uses `tab:recheck_outcomes`. Claude tone summaries follow `tab:trigger_aggregate_rates`, and first-turn paired conformity follows `tab:paired_sub_context`. The intervention and its answer-key control are identified separately from the nine-model benchmark. No trajectories, logs or individual model responses are included.
