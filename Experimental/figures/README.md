# Figure Regeneration Entrypoints

Each script in this directory is an entrypoint for regenerating a main-text
figure. Scripts call shared plotting code and read raw ignored result files
under `Experimental/results/` or explicitly supplied aggregate tables when the
raw files are not needed. Pass the run `--run-id`, `--summary`, or `--appendix-tex`
explicitly; scripts should not silently fall back to a dated result run.

Main-text figure mapping:

- Figure 1 and Figure 2 are design assets, not result plots.
- Figure 3: `context_neutral_shift.py`
- Figure 4: TeX-native transcript/scoring diagram in the manuscript.
- Figure 5: `trigger_model_quadrant.py`
- Figure 6: `trigger_family_tone_boost.py`
- Figure 7: `appendix_trigger_temporal_strategy.py`
- Figure 8: `Figure8.py`
- Figure 9: `discussion_capability_rank.py`
- Figure 10: `Figure10.py` for the Opus/Sonnet/Haiku tone-detail trajectory plot.

- `trigger_confidence_trajectory.py` is the direct confidence-trajectory helper
  used by the Figure 8 wrapper when publishing under a descriptive
  filename.

Run a script with `--out <path>` to publish reviewed image output. Use
`--appendix-tex <path>` for `discussion_capability_rank.py` when regenerating
that figure from explicit aggregate tables.
