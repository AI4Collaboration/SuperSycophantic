# NeurIPS 2026 Sydney Poster

- [Print PDF](SuperSycophantic-NeurIPS-2026-Sydney.pdf): one page, 140 cm wide by 90 cm high. Print at actual size (100%), not fit-to-page.
- [HTML source and preview](poster.html): opens locally, with no server or external assets required.
- [Website poster](SuperSycophantic-NeurIPS-2026-Sydney.png): opaque PNG within 5120 x 2880 pixels and 10 MB.
- [Thumbnail](thumbnail.png): within 320 x 256 pixels and 5 MB.

The print dimensions follow the [2026 main-conference instructions](https://neurips.cc/Conferences/2026/PosterInstructions). Website image limits follow the [conference platform's upload guide](https://wiki.eventhosts.cc/en/reference/posteruploads). Requirements checked on 16 November 2026. Files are prepared for upload; adding them here does not submit them to the conference or order printing.

## Rebuild

Use Python and Node.js 20+. Fonts are included in `assets/fonts/` under the SIL Open Font License. From the repository root, run:

```sh
python -m pip install -r poster/requirements.txt
npm --prefix poster install
python poster/build_poster.py
```

The renderer uses an installed Chrome or Edge, or Playwright's Chromium (`cd poster && npx playwright install chromium`). `CHROMIUM_EXECUTABLE` can select another Chromium executable. Each build uses a fresh headless browser; it does not use existing browser sessions.

[poster.html](poster.html) controls the layout. Raleway is used for headings and Lato for body text and plots. The Python script regenerates four SVG plots and the QR code, prints through Chromium, and exports the web images. Plot text is outlined in the SVGs so it renders consistently without system fonts. Plot dimensions match their HTML slots to preserve physical text sizes. The build normalizes Chromium's paper-size rounding to exactly 140 x 90 cm and checks page count, text bounds, embedded fonts, the code link and image limits. The renderer also checks asset loading and layout overflow. The original project and conference logos are retained; no manuscript files are modified.

First-turn results use the manuscript's `tab:context_model_rates`, certainty controls use `tab:certainty_control`, and the paired Opus-4.6 intervention uses `tab:recheck_outcomes`. The central Claude plot shows separately recounted Static and Adaptive supplementary runs across seven families and 100 SUB decisions per tone. The white comparison line pools valid trials from the other six models in the same runs. Invalid answer pairs are excluded: one Sonnet-4.5 trial at moderate tone in each mode, one Mistral-Medium-3.1 trial at strong tone in Static, and one Command-R trial at each of moderate and strong tones in Adaptive. Plot counts and denominators are in `build_poster.py`. The intervention and its answer-key control are separate from the nine-model benchmark. No raw trajectories or logs are included.

The four-turn example is `NGT-POLICY-001` in `Experimental/data/supersycophantic_context_ngt_100.json`. Short labels retain the clinic closure, one mobile health team, overcrowded urban shelters and isolated rural route. All four replies come from one GPT-5.4 conversation in the local `rebuttal_openrouter_temporal_static_sub_20.jsonl.gz` results, with `trigger=simple_baseline` and neutral initial context. The sequence is B, A, B, A: the initial reply followed by mild, moderate and strong static disagreement. The user and model quotes are verbatim excerpts from their respective turns. Full logs are not distributed.

## Design References

The centered title, central takeaway and aligned side columns draw on the actual [Measuring What Matters poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/121477.png). The conference logo comes from the template linked in the [2025 official instructions](https://neurips.cc/Conferences/2025/PosterInstructions). This is a custom HTML design adapted to the 2026 Sydney dimensions, not an official conference template.
