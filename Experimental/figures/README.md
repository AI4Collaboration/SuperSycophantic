# Figure Regeneration Entrypoints

Each script in this directory is an entrypoint for regenerating a main-text
figure. Scripts call shared plotting code and read raw ignored result files
under `Experimental/results/` or explicitly supplied aggregate tables when the
raw files are not needed. Pass the run `--run-id`, `--summary`, or `--appendix-tex`
explicitly; scripts should not silently fall back to a dated result run.

Current rendered figure mapping (legacy script names are retained):

- Figure 1 and Figure 2 are design assets, not result plots.
- Figure 3: `context_neutral_shift.py`
- Figure 4: TeX-native transcript/scoring diagram in the manuscript.
- Figure 5: `trigger_model_quadrant.py`
- Figure 6: `appendix_trigger_temporal_strategy.py` (adaptive mode only)
- Figure 7: `Figure8.py` (confidence trajectories)
- Figure 8: `discussion_capability_rank.py`
- Figure 9: `Figure10.py` for the original Opus-4.5/Sonnet/Haiku tone detail.
- The legacy `trigger_family_tone_boost.py` composite is no longer a main-text
  figure. The manuscript instead reports separate OBJ and SUB certainty controls.

- `trigger_confidence_trajectory.py` is the direct confidence-trajectory helper
  used by the Figure 8 wrapper when publishing under a descriptive
  filename. Its optional per-trial intervals are legacy descriptive diagnostics,
  not the paired base-item intervals in the revised manuscript.

Use `Experimental/revision_paired_analysis.py` for paired context and trigger
contrasts, and `Experimental/revision_samplek_analysis.py` for repeated-sampling
intervals. Both retain repeated observations from each base item together.
`Experimental/statistical_analysis.py` is a legacy exporter of response-wise
intervals; those intervals do not replace the revised paired uncertainty estimates.
Original model and tone plots pool static and adaptive modes where identified;
primary paired mode comparisons keep them separate. Regenerating original plots
requires their original raw files or explicitly identified retained summaries.

Run a script with `--out <path>` to publish reviewed image output. Use
`--appendix-tex <path>` for `discussion_capability_rank.py` when regenerating
that figure from explicit aggregate tables.
