# NeurIPS 2026 Sydney Poster

- [Print PDF](SuperSycophantic-NeurIPS-2026-Sydney.pdf): one page, 140 cm wide by 90 cm high. Print at actual size (100%), not fit-to-page.
- [Website poster](SuperSycophantic-NeurIPS-2026-Sydney.png): opaque PNG within 5120 x 2880 pixels and 10 MB.
- [Thumbnail](thumbnail.png): within 320 x 256 pixels and 5 MB.

The print dimensions follow the [2026 main-conference instructions](https://neurips.cc/Conferences/2026/PosterInstructions). Website image limits follow the [conference platform's upload guide](https://wiki.eventhosts.cc/en/reference/posteruploads). Requirements checked on 16 November 2026. Files are prepared for upload; adding them here does not submit them to the conference or order printing.

## Rebuild

Install XeLaTeX (TeX Live or MiKTeX) on `PATH`, with TeX Gyre Heros and the `geometry`, `fontspec`, `graphicx`, `tikz`, `xcolor` and `hyperref` packages. From the repository root, run:

```sh
python -m pip install -r poster/requirements.txt
python poster/build_poster.py
```

[poster.tex](poster.tex) is the editable LaTeX source. The build script regenerates four vector plots and the QR code, compiles with XeLaTeX, and exports the web images. It checks page dimensions, text bounds, font embedding and image limits. LaTeX build files are temporary. The original project logo and two paper illustrations are included in `assets/`; selected illustration panels are enlarged in the poster without changing the paper files.

First-turn results use the manuscript's `tab:context_model_rates`, certainty controls use `tab:certainty_control`, and the paired Opus-4.6 intervention uses `tab:recheck_outcomes`. The Claude plot shows separately recounted Static and Adaptive supplementary runs, not the manuscript's original pooled aggregates. It includes seven families and 100 SUB decisions per tone; one invalid Sonnet-4.5 trial is excluded at moderate tone in each mode. Plot values and denominators are in `build_poster.py`. The intervention and its answer-key control are separate from the nine-model benchmark. No trajectories, logs or individual model responses are included.

## Design References

The layout draws on the figure-led structure of the NeurIPS 2024 Best Paper posters [VAR](https://neurips.cc/virtual/2024/poster/94115) and [STDE](https://neurips.cc/virtual/2024/poster/95741). The white background, column layout and conference logo follow the template linked from the [2025 official instructions](https://neurips.cc/Conferences/2025/PosterInstructions). This is a custom LaTeX layout at the 2026 Sydney dimensions, not an official LaTeX template.
