#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/NYSERDA_data_dir"
  echo "Expected in that dir: split_part_*.csv and Vehicle Descriptions.csv"
  exit 2
fi

DATA_DIR="$1"

python scripts/aggregate_ev_trends_by_county.py \
  --inputs-glob "${DATA_DIR}/split_part_*.csv" \
  --descriptions "${DATA_DIR}/Vehicle Descriptions.csv" \
  --zip-to-county data/zip_to_county_ny.csv \
  --zip-to-region data/utility_zip_regions.csv \
  --regions "LIPA:NASSAU,SUFFOLK" \
  --group-by snapshot \
  --output-dir out_ev_snapshots

python scripts/compute_new_reg_by_snapshot.py \
  --inputs-glob "${DATA_DIR}/split_part_*.csv" \
  --descriptions "${DATA_DIR}/Vehicle Descriptions.csv" \
  --zip-to-county data/zip_to_county_ny.csv \
  --zip-to-region data/utility_zip_regions.csv \
  --regions "LIPA:NASSAU,SUFFOLK" \
  --snapshot-map NY_DMV_Snapshots.csv \
  --output-dir out_new_reg

python scripts/build_policy_covariates.py

python scripts/build_and_fit_bass_lipa.py \
  --with-policy \
  --holdout-start 2025-01-01 \
  --min-date 2018-01-01 \
  --output-tag monthly2018 \
  --horizon 24

echo "Done. See models/bass_lipa_*_monthly2018.*"

