# Upstream source

This project is based on [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole),
version 1.2.1, at commit `7b02be5ec80a88310f2d04a27a82adfcbb5dc211`.
The upstream MIT license and original copyright notices are retained.

The code is imported as a source snapshot. Upstream Git history and bundled
MovieLens datasets are excluded. Small upstream test fixtures under `tests/`
are retained for offline checks. Original documentation is available in
`README_UPSTREAM.md`, `README_UPSTREAM_CN.md`, and `docs/`; references there to
bundled data describe upstream RecBole rather than this snapshot.

## Local changes

- Checkpoint loading across PyTorch versions through `torch_load_checkpoint`.
- Checkpoint names include model and dataset names.
- Training continues when a model does not support the optional FLOPs probe.
- SASRec/SASRecCPR Top-K export and Amazon review/metadata subset extraction.
- Shared Amazon configuration and a JSONL-to-RecBole converter with disk-backed
  last-record deduplication per user/product pair.
- All dataset paths resolve relative to the configured `data_path`, including
  `ml-100k`; this project does not ship the upstream example dataset.
- `use_gpu=False` selects the CPU even on a host with CUDA available.
- Project documentation, artifact exclusion checks, and CI without real datasets
  or automatic source-format commits.

The Python import and distribution names remain `recbole` for compatibility.
The retained upstream model collection is broader than the two models used in
the documented Amazon workflow; optional model dependencies are not installed
by the core requirements.
