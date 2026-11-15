# HLE-Verified Local Cache

This directory is the local cache location for the HLE-Verified parquet shards
needed to rebuild the frozen SuperSycophantic mixed OBJ source panel.

The parquet shards are intentionally not tracked in GitHub because they are a
large source cache. To rebuild the mixed source panel, download the required
shards from `skylenage-ai/HLE-Verified` into
`Experimental/data/hle_verified/data/`:

- `data/Gold_subset.*.parquet`
- `data/Revision_subset.*.parquet`

`source_cache_manifest.tsv` lists the expected shard filenames, byte sizes,
and Git blob SHA-1 identifiers for the frozen panel. Use these to verify the
downloaded cache. The builder uses the Gold and Revision subsets; the
Uncertain subset is not required.

The mixed OBJ builder uses these files through
`Experimental/data/helper/build_mixed_gt_panel.py`. It filters image-dependent
rows, excludes audited bad records, keeps source-native multiple-choice rows,
and converts only simple numeric exact-answer rows into five-option
multiple-choice items.

Source dataset: `skylenage-ai/HLE-Verified`.
