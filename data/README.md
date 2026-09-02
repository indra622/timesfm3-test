# Real benchmark data

Raw files live under `data/raw/` and are ignored by Git. Download and verify
them with:

```bash
uv run python scripts/download_datasets.py
```

## M5 Forecasting

Canonical source: <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>

The downloader uses Zenodo record 10203108 as an unauthenticated transport
mirror and verifies the record's published MD5 checksums:

- `calendar.csv`: `3ffeab2991b0c8e861d008b39ea4c95c`
- `sales_train_evaluation.csv`: `b806dfc9f30a745102b708c09951f6aa`
- `sell_prices.csv`: `08c591caa99e55daf3e0ccac913f7c85`

Use remains subject to the M5 competition rules.

## Beijing Multi-Site Air Quality

Canonical source: <https://archive.ics.uci.edu/dataset/501/beijing>

- DOI: `10.24432/C5RK5G`
- License: CC BY 4.0
- Download archive SHA-256:
  `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`

## METR-LA traffic speed

Canonical code and data link: <https://github.com/liyaguang/DCRNN>

- Official Google Drive file ID: `1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC`
- `metr-la.h5` SHA-256:
  `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`
- DCRNN source/graph revision: `602afd9d767d3aa1c9b3eac51710d6aeee12c227`
- Graph files are downloaded from raw GitHub URLs pinned to that revision and
  verified before use.
- Scope: local non-production evaluation of 16 graph-selected sensors

The raw HDF5 file was created by Pandas 0.15.2. Pandas 3 no longer reads its
byte-valued metadata correctly, so this project pins Pandas below 3 and uses
PyTables only for faithful read access; the raw file is not rewritten.
