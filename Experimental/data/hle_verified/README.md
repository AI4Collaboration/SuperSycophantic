# HLE-Verified Local Cache

This directory is the local cache location for the HLE-Verified parquet shards
needed to rebuild the current SuperSycophantic mixed GT source panel.

The parquet shards are intentionally not tracked in GitHub because they are a
large source cache. To rebuild the mixed source panel, download the required
shards from `skylenage-ai/HLE-Verified` into `data/`:

- `data/Gold_subset.*.parquet`
- `data/Revision_subset.*.parquet`

Do not commit:

- `Uncertain_subset.*.parquet`
- one-off domain-count summaries
- generated full-dataset caches

The mixed GT builder uses these files through
`Experimental/data/helper/build_mixed_gt_panel.py`. It filters image-dependent
rows, excludes audited bad records, keeps source-native multiple-choice rows,
and converts only simple numeric exact-answer rows into five-option
multiple-choice items.

Source dataset: `skylenage-ai/HLE-Verified`.
