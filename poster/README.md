# NeurIPS 2026 Sydney Poster

- [Print PDF](SuperSycophantic-NeurIPS-2026-Sydney.pdf): one page, 140 cm wide by 90 cm high. Print at actual size (100%), not fit-to-page.
- [HTML source and preview](poster.html): opens locally, with no server or external assets required.
- [Website poster](SuperSycophantic-NeurIPS-2026-Sydney.png): opaque PNG within 5120 x 2880 pixels and 10 MB.
- [Thumbnail](thumbnail.png): within 320 x 256 pixels and 5 MB.

The print dimensions follow the [2026 main-conference instructions](https://neurips.cc/Conferences/2026/PosterInstructions). Website image limits follow the [conference platform's upload guide](https://wiki.eventhosts.cc/en/reference/posteruploads). Requirements checked on 16 November 2026. Files are prepared for upload; adding them here does not submit them to the conference or order printing.

## Rebuild

Use Python, Node.js 20+, and an installed Arial font. From the repository root, run:

```sh
python -m pip install -r poster/requirements.txt
npm --prefix poster install
python poster/build_poster.py
```

The renderer uses an installed Chrome or Edge, or Playwright's Chromium (`cd poster && npx playwright install chromium`). `CHROMIUM_EXECUTABLE` can select another Chromium executable. Each build uses a fresh headless browser; it does not use existing browser sessions.

[poster.html](poster.html) controls the layout. The Python script regenerates four SVG plots and the QR code, prints through Chromium, and exports the web images. It normalizes Chromium's paper-size rounding to exactly 140 x 90 cm and checks page count, text bounds, embedded Arial fonts, the code link and image limits. The renderer also checks asset loading and layout overflow. The original project and conference logos are retained; no manuscript files are modified.

First-turn results use the manuscript's `tab:context_model_rates`, certainty controls use `tab:certainty_control`, and the paired Opus-4.6 intervention uses `tab:recheck_outcomes`. The Claude plot shows separately recounted Static and Adaptive supplementary runs, not the manuscript's original pooled aggregates. It includes seven families and 100 SUB decisions per tone; one invalid Sonnet-4.5 trial is excluded at moderate tone in each mode. Plot values and denominators are in `build_poster.py`. The intervention and its answer-key control are separate from the nine-model benchmark. No trajectories, logs or individual model responses are included.

## Design References

The centered title, central takeaway and aligned side columns draw on the actual [Measuring What Matters poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/121477.png). The conference logo comes from the template linked in the [2025 official instructions](https://neurips.cc/Conferences/2025/PosterInstructions). This is a custom HTML design adapted to the 2026 Sydney dimensions, not an official conference template.
