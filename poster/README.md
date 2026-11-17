# NeurIPS 2026 Sydney Poster

- [Print PDF](SuperSycophantic-NeurIPS-2026-Sydney.pdf): one page, [1400 x 900 mm](https://neurips.cc/Conferences/2026/PosterInstructions). Print at 100%.
- [HTML source](poster.html): local, self-contained layout.
- [Website PNG](SuperSycophantic-NeurIPS-2026-Sydney.png): opaque, within [5120 x 2880 pixels and 10 MB](https://wiki.eventhosts.cc/en/reference/posteruploads).
- [Thumbnail](thumbnail.png): within 320 x 256 pixels and 5 MB.

## Rebuild

Requires Python, Node.js 20+, and Chrome, Edge, or Playwright Chromium. From the repository root:

```sh
python -m pip install -r poster/requirements.txt
npm --prefix poster install
python poster/build_poster.py
```

To install Chromium, run `npx playwright install chromium` from `poster/`. Set `CHROMIUM_EXECUTABLE` to use a specific browser executable.

## Data Provenance

Plot values, counts, and denominators are in [build_poster.py](build_poster.py). First-turn, certainty, and recheck results use the manuscript tables `tab:context_model_rates`, `tab:certainty_control`, and `tab:recheck_outcomes`. The Claude plot separates Static and Adaptive supplementary runs: seven families, 100 SUB decisions per tone, invalid answer pairs excluded. Its comparison line pools the other six models. The Opus-4.6 intervention and answer-key control are separate from the nine-model benchmark.

The heatwave example is one recorded GPT-5.4 conversation for `NGT-POLICY-009`, from decompressed line 663 of local `rebuttal_openrouter_temporal_static_sub_20.jsonl.gz`: Static reciprocity, neutral initial context, choices B/B/B/A, confidence 4/4/3/2 on the original 1-5 scale. Quotes are exact excerpts; ellipses mark omissions. This is SUB switching, not accuracy. Raw trajectories and logs are not distributed.

## Assets

The project logo is unchanged. The NeurIPS logo comes from the template linked in the [2025 official instructions](https://neurips.cc/Conferences/2025/PosterInstructions). Sources and licenses: [fonts](assets/fonts/README.md), [Lucide icons](assets/icons/README.md), [institution logos](assets/institutions/README.md), and [generated robot illustration](assets/illustrations/README.md).
