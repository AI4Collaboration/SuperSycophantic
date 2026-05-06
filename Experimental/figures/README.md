# Figure Regeneration Entrypoints

Each script in this directory is named after the figure file it regenerates. The
scripts call shared plotting code and read raw ignored result files under
`Experimental/results/` or appendix aggregate tables when the raw files are not
needed. The one explicit exception is the temporal-strategy label patcher, which
only rewrites release-facing text on an existing per-model PNG when the raw
temporal trigger files are unavailable.

Main-text result figures currently covered here:

- `context_neutral_shift.py`
- `trigger_model_quadrant.py`
- `trigger_family_tone_boost.py`
- `appendix_trigger_temporal_strategy.py`
  regenerates the per-model temporal strategy figure from raw temporal trigger
  result files.
- `patch_appendix_trigger_temporal_strategy_labels.py`
  patches only the release-facing OBJ/SUB panel labels on an existing temporal
  strategy PNG when the ignored raw result files are unavailable.
- `trigger_confidence_trajectory.py`
- `trigger_tone_gradient_opus.py`
- `discussion_capability_rank.py`

Run a script with `--out <path>` to publish to a manuscript image directory.
Use `--appendix-tex <path>` for `trigger_confidence_trajectory.py` and
`discussion_capability_rank.py` when regenerating those figures from appendix
tables.
