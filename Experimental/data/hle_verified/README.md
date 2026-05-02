# HLE-Verified Local Cache

This directory stores only the HLE-Verified parquet shards needed to rebuild the
current SuperSycophantic mixed GT source panel.

Kept locally:

- `data/Gold_subset.*.parquet`
- `data/Revision_subset.*.parquet`

Removed from the repo-local cache:

- `Uncertain_subset.*.parquet`
- one-off domain-count summaries

The mixed GT builder uses these files through
`Experimental/data/helper/build_mixed_gt_panel.py`. It filters image-dependent
rows, excludes audited bad records, keeps source-native multiple-choice rows,
and converts only simple numeric exact-answer rows into five-option
multiple-choice items.

Source dataset: `skylenage-ai/HLE-Verified`.
