# EV Adoption Forecasting (Bass Diffusion) — Repro Package

This folder contains the **code + small supporting files** needed to reproduce the current **LIPA EV adoption Bass diffusion model** and the associated figures/outputs, **without** shipping the raw NYSERDA DMV registration dataset.

## What’s included

- `scripts/` — pipeline scripts (aggregation → covariates → Bass fit/forecast)
- `NY_DMV_Snapshots.csv` — DMV_ID → snapshot date mapping (small)
- `data/zip_to_county_ny.csv` — ZIP → county crosswalk (small)
- `data/utility_zip_regions.csv` — ZIP → region overrides (small)
- `prices/gasoline_downstate_ny_monthly.csv` — monthly downstate gasoline prices used for LIPA covariates (small)
- `models/*_monthly2018.*` — the **current** fitted parameters, holdout metrics, forecast table, and plots (monthly-era fit)
- `out_ev_snapshots/` and `out_new_reg/` — snapshot-based stock + “first-seen” (new-to-region) EV counts used by the Bass model

## What’s not included

You must provide the NYSERDA DMV registration dataset locally:

- `split_part_*.csv` (the 9 registration split files)
- `Vehicle Descriptions.csv`

Optional (for policy eligibility from tax-return aggregates):
- `Tax Return LIPA 2022.xlsx` (if available to you; otherwise the script falls back to defaults)

## Quick start (re-run the Bass fit from the precomputed aggregates)

If you only want to reproduce the Bass outputs from the already aggregated inputs in this folder, you can run the fit directly (the folder already includes `covariates/policy_covariates.csv`):

```bash
python scripts/build_and_fit_bass_lipa.py \
  --with-policy \
  --holdout-start 2025-01-01 \
  --min-date 2018-01-01 \
  --output-tag monthly2018 \
  --horizon 24
```

If you want to rebuild policy covariates locally (e.g., after changing assumptions), run:
```bash
python scripts/build_policy_covariates.py
```

Outputs:
- `models/bass_lipa_flow_forecast_monthly2018.png`
- `models/bass_lipa_stock_forecast_monthly2018.png`
- `models/bass_lipa_forecast_monthly2018.csv`
- `models/bass_lipa_*_monthly2018.json`
- `models/bass_lipa_holdout_metrics_monthly2018.json`

## Full rebuild (from raw NYSERDA splits; takes time)

Set `DATA_DIR` to the directory containing `split_part_*.csv` and `Vehicle Descriptions.csv`.

```bash
DATA_DIR="/path/to/NYSERDA_data"

# 1) Snapshot-based on-road stock (EV and ALL) for LIPA by DMV_ID
python scripts/aggregate_ev_trends_by_county.py \
  --inputs-glob "$DATA_DIR/split_part_*.csv" \
  --descriptions "$DATA_DIR/Vehicle Descriptions.csv" \
  --zip-to-county data/zip_to_county_ny.csv \
  --zip-to-region data/utility_zip_regions.csv \
  --regions "LIPA:NASSAU,SUFFOLK" \
  --group-by snapshot \
  --output-dir out_ev_snapshots

# 2) New-to-region (“first-seen”) EV counts by DMV_ID (flow proxy)
python scripts/compute_new_reg_by_snapshot.py \
  --inputs-glob "$DATA_DIR/split_part_*.csv" \
  --descriptions "$DATA_DIR/Vehicle Descriptions.csv" \
  --zip-to-county data/zip_to_county_ny.csv \
  --zip-to-region data/utility_zip_regions.csv \
  --regions "LIPA:NASSAU,SUFFOLK" \
  --snapshot-map NY_DMV_Snapshots.csv \
  --output-dir out_new_reg

# 3) Build policy covariates (optional but recommended for the policy scenario)
python scripts/build_policy_covariates.py

# 4) Fit + holdout validation + forecast (monthly-era snapshots only)
python scripts/build_and_fit_bass_lipa.py \
  --with-policy \
  --holdout-start 2025-01-01 \
  --min-date 2018-01-01 \
  --output-tag monthly2018 \
  --horizon 24
```

## Notes / interpretation

- `flow_ev_t` is **not** true sales. It is “first-seen VINs in region at a snapshot” (a flow proxy).
- `stock_ev_t` is “on-road EV stock” (unique VINs present in that snapshot).
- The Bass model in this package is fit to `flow_ev_t` and uses `stock_ev_t` as the state variable.
- Pre-2018 DMV snapshots are irregular/annual; the current model fit focuses on the monthly-era (2018+).
