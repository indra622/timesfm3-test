# TimesFM 3 local test

This project runs the official **TimesFM 3.0** checkpoint on a deterministic
synthetic daily-demand task. It tests all three of the model's main interfaces:

1. univariate forecasting;
2. joint multivariate forecasting;
3. past-only and known-future covariates.

The benchmark compares TimesFM 3 with a weekly seasonal-naive baseline. Future
promotion days are intentionally irregular, then passed to the multivariate
model as a known-future covariate. A counterfactual run removes those future
promotions to measure the model's response. The script writes exact metrics,
timings, output shapes, and a forecast plot to `artifacts/`.

## Important license boundary

The source repository is Apache-2.0, but the TimesFM 3 pretrained weights use
`timesfm-non-commercial-license-v1.0`. They are restricted to non-commercial,
non-production use. This project is for local evaluation only.

## Reproduce

The official source is pinned in `upstream-timesfm/`. The currently tested
commit is recorded in `artifacts/results.json`.

```bash
git clone https://github.com/indra622/timesfm3-test.git
cd timesfm3-test
scripts/fetch_upstream.sh   # clones the five pinned upstream repos (not committed here)
uv sync
uv run python run_experiment.py
```

The `upstream-*` directories (TimesFM, fev, DCRNN, STAEformer, Torch-MTS) are
excluded from this repository because of their size and separate licenses;
`scripts/fetch_upstream.sh` checks them out at the exact commits used in the
report. Raw datasets under `data/raw/` are also excluded and are downloaded by
`scripts/download_datasets.py`. Result JSON files, figures, and the trained
DCRNN/STAEformer checkpoints under `artifacts/` are committed as the evidence
behind the report.

Device selection defaults to Apple MPS when available, then CUDA, then CPU.
It can be overridden explicitly:

```bash
uv run python run_experiment.py --device mps
uv run python run_experiment.py --device cpu
```

Run the fast, model-free unit tests with:

```bash
uv run pytest -q
```

## Real-data benchmarks

Download the verified inputs and run both small real-data slices:

```bash
uv run python scripts/download_datasets.py
uv run python scripts/download_datasets.py --dataset metr-la
uv run python real_data_benchmarks.py --dataset both
```

The M5 slice uses the top 16 `FOODS_3` items from store `CA_1`, selected using
context-only sales, with a 512-day context and 28-day holdout. The Beijing
slice uses the four stations with the fewest total PM2.5 missing values, a
336-hour context, and a complete 24-hour holdout. Both compare seasonal naive,
univariate TimesFM 3, joint multivariate TimesFM 3, and multivariate TimesFM 3
with covariates.

Real-data outputs are written under `artifacts/real/`. See `data/README.md` for
source, checksum, and license details.

## Multivariate follow-up

The follow-up benchmark isolates useful cross-series information from matched
controls with rolling origins:

```bash
uv run python multivariate_followup.py --dataset m5
uv run python multivariate_followup.py --dataset metr-la
uv run python multivariate_followup.py --dataset both
```

The M5 panel evaluates the same eight `FOODS_3` SKUs across `CA_1`, `CA_2`, and
`CA_3`. Each target is tested alone, jointly with all 24 targets, with the same
SKU in the other stores as past-only covariates, with known-future
calendar/price features, and against a negative control. The control uses a
different selected SKU in the same two companion stores, paired by a fixed
half-list rotation.

The METR-LA panel selects 16 sensors from the lowest-missing 80% using only the
official road graph and missingness markers in the first 80% of time. It
compares nearby road-network sensors against equally sized far-sensor controls
over ten rolling origins.
Outputs are written under `artifacts/multivariate/`.

## FEV-Bench external subset

```bash
uv run python fev_subset_benchmark.py --device mps
```

The FEV command runs three fixed, lightweight multivariate tasks using Google's
pinned official FEV wrapper and compares local metrics with Google's published
100-task result CSV. It is an external-reproduction subset, not a claim to have
reproduced the full benchmark. The output records both local and published
dataset fingerprints because the Hub dataset revision may differ from the one
used for Google's CSV.

## Expanded METR-LA graph check

```bash
uv run python metr_graph_followup.py --device mps
```

To compare foundation models separately from dataset-trained traffic models on
the same leakage-safe METR-LA target panel and 40 rolling origins:

```bash
uv run python metr_model_comparison.py --track all
```

The zero-shot track compares TimesFM 3 with Chronos-2 using the same seven-day
context. The supervised track trains the full Torch-MTS DCRNN and STAEformer
architectures on the first half of the series and keeps their standard
12-step-input/12-step-output setup. Because the training budgets differ, the
artifact reports the tracks separately.

This expands METR-LA to 40 non-overlapping daily forecast origins and adds a
context-only ridge autoregression, a DCRNN-style dual-random-walk diffusion
ridge, and a deterministically shuffled-graph control. The graph ridge is a
lightweight structural baseline, not a reproduction of the full recurrent
DCRNN architecture.

The first experiment downloads roughly 1.3 GB of model weights from
`google/timesfm-3.0-pytorch`. Later runs reuse the Hugging Face cache.

## Outputs

- `artifacts/results.json`: environment, timings, shapes, MAE/RMSE/WAPE, and
  the model's promotion counterfactual response.
- `artifacts/forecast.png`: both target series, point forecasts, p10-p90 band,
  and scheduled promotion days.

This is a synthetic smoke benchmark, not evidence of performance on a real
business dataset. Replace `make_synthetic_dataset()` with a chronological
train/test split from the target domain before drawing deployment conclusions.

## Research report

The complete phase-1 study is available in both Markdown and PDF. The
current version is v11, whose section 9.5 lists insurance use cases where
multivariate forecasting is needed and, for each, why an advantage over
univariate forecasting is expected, mapped to the three gain paths observed in
the experiments. All numeric results are unchanged since v08.

- [`docs/TimesFM 3 다변량 예측 실증 연구리포트 v11.md`](<docs/TimesFM 3 다변량 예측 실증 연구리포트 v11.md>)
- [`docs/TimesFM 3 다변량 예측 실증 연구리포트 v11.pdf`](<docs/TimesFM 3 다변량 예측 실증 연구리포트 v11.pdf>)
- Conference talk slides: [`docs/slides/TimesFM 3 다변량 예측 실증 발표 v2.html`](<docs/slides/TimesFM 3 다변량 예측 실증 발표 v2.html>)

Prior versions (kept for reference): v10, v09 and v08 Markdown and PDF in `docs/`,
slides v1 in `docs/slides/`. See `docs/README.md`.

## Sources

- Official repository: <https://github.com/google-research/timesfm>
- Official model: <https://huggingface.co/google/timesfm-3.0-pytorch>
- Release post: <https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/>
