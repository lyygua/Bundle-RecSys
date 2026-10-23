# Project validation

Validation date: 2026-10-23.

## Environment

Python 3.11.14, PyTorch 2.10.0, NumPy 1.26.4, Pandas 2.2.3, SciPy 1.17.1,
Ray 2.6.3. Exact direct dependency versions are in `constraints-tested.txt`.
Checks used an isolated virtual environment inheriting the existing training
environment's installed libraries. A completely fresh dependency installation
and the remote GitHub Actions workflow have not been run locally.

Pandas 3.0.1 caused four upstream data tests to fail; dependencies now require
Pandas < 3 and NumPy < 2. Pandas 2.2.3 passes the selected suite, with upstream
deprecation warnings remaining.

## Automated checks

```bash
python -m pytest -q tests/project tests/metrics tests/data tests/config/test_config.py
```

Result: **90 passed**. This covers data processing, ranking/loss metrics,
configuration, conversion with duplicate reviews, extraction of recommended-item
metadata, checkpoint deserialization and CPU selection on a CUDA host.

The four documented command-line entry points load and display `--help`.
Editable installation, building the RecBole source distribution and wheel,
and installation of the built wheel into a separate virtual environment were
checked. The wheel imports both models' configurations and contains their YAML
properties, without bundled datasets or weights. The complete upstream model
suite and optional model dependencies are outside this validation scope.

## Real-data workflow checks

Real datasets and generated artifacts were read or written outside this source
snapshot and are excluded from the repository.

- The four-domain subset extraction command completed using the existing eight
  recommendation files. Every domain reported zero missing reviewed-item
  metadata IDs and zero missing recommended-item metadata IDs.
- Conversion of all four domains matches the existing interaction files'
  user IDs, product IDs, ratings and timestamps after last-record deduplication.
  File order is not identical, so this is not a claim of bitwise experiment
  reproduction. Digital Music reads 130,434 raw reviews and retains 128,763
  unique user/product interactions.
- SASRec and SASRecCPR each completed one CPU epoch on the existing Digital Music
  interactions, followed by checkpoint reload and CLI export for two users
  with Top-10 recommendations. This validates the training and export path;
  it does not reproduce the original multi-epoch evaluation scores.

The shared configuration filters the Digital Music data to 2,782 non-padding
users, 2,389 non-padding items and 7,441 interactions. All data and model artifacts
remain local.

| Converted domain | Unique interactions | Canonical content matches existing data |
| --- | ---: | --- |
| All Beauty | 693,929 | Yes |
| Appliances | 2,103,990 | Yes |
| Digital Music | 128,763 | Yes |
| Health and Personal Care | 488,188 | Yes |

## Source audit

`python scripts/verify_package.py` checks tracked and unignored paths for
datasets outside small test fixtures, model weights, logs, recommendation
exports, files above 10 MiB, machine-specific paths and recognizable credentials.
The upstream MIT license is retained. The snapshot starts a new Git history,
so historical upstream datasets are not included.

Whitespace checks apply to the project-authored files and changed code. Imported
upstream files retain their original formatting, including whitespace in test
fixtures; a full-import whitespace scan is not clean.
