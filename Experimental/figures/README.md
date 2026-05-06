# Figure Regeneration Entrypoints

Each script in this directory is an entrypoint for regenerating a main-text
figure. Scripts call shared plotting code and read raw ignored result files
under `Experimental/results/` or appendix aggregate tables when the raw files
are not needed. The one explicit exception is the temporal-strategy label
patcher, which only rewrites release-facing text on an existing per-model PNG
when the raw temporal trigger files are unavailable.

Main-text figure mapping:

- Figure 1 and Figure 2 are design assets, not result plots.
- Figure 3: `context_neutral_shift.py`
- Figure 4: TeX-native transcript/scoring diagram in the manuscript.
- Figure 5: `trigger_model_quadrant.py`
- Figure 6: `trigger_family_tone_boost.py`
- Figure 7: `appendix_trigger_temporal_strategy.py`
- Figure 8: `Figure8.py`
- Figure 9: `discussion_capability_rank.py`
- Figure 10: `Figure10.py` for the Claude tone-detail trajectory plot.

Additional helper entrypoints:

- `patch_appendix_trigger_temporal_strategy_labels.py` patches only the
  release-facing OBJ/SUB panel labels on an existing temporal strategy PNG when
  the ignored raw result files are unavailable.
- `trigger_confidence_trajectory.py` preserves an older confidence variant.
- `trigger_tone_gradient_opus.py` is the direct Claude tone-detail helper used
  by the current Figure 10 wrapper when publishing under a descriptive filename.

Run a script with `--out <path>` to publish to a manuscript image directory.
Use `--appendix-tex <path>` for `discussion_capability_rank.py` when
regenerating that figure from appendix tables.
