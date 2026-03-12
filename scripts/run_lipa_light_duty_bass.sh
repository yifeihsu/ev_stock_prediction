#!/usr/bin/env bash
set -euo pipefail

# Separate workflow: Light-duty-only LIPA Bass forecasting.
# Override variables at runtime if needed, e.g.:
#   HOLDOUT_START=2025-01-01 FLOW_LIKELIHOOD=mse bash scripts/run_lipa_light_duty_bass.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PANEL_CSV="${PANEL_CSV:-covariates/panel_LIPA_light_duty_snapshot.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-covariates/panel_LIPA_light_duty_summary.json}"
HOLDOUT_START="${HOLDOUT_START:-2025-01-01}"
HORIZON="${HORIZON:-24}"
FLOW_LIKELIHOOD="${FLOW_LIKELIHOOD:-mse}"
OUTPUT_TAG="${OUTPUT_TAG:-light_duty}"
RESAMPLE_MONTHLY="${RESAMPLE_MONTHLY:-1}"  # 1=yes, 0=no
WITH_EVSE="${WITH_EVSE:-1}"                # 1=yes, 0=no
STATION_INFO_DIR="${STATION_INFO_DIR:-station_info}"
EVSE_RESTRICTED_WEIGHT="${EVSE_RESTRICTED_WEIGHT:-0.0}"  # 0=exclude restricted public
EVSE_WORKPLACE_WEIGHT="${EVSE_WORKPLACE_WEIGHT:-0.0}"    # 0=exclude workplace
EVSE_OUTPUT_CSV="${EVSE_OUTPUT_CSV:-covariates/evse_lipa_monthly.csv}"

python scripts/build_lipa_light_duty_panel.py \
  --inputs-glob "split_part_*.csv" \
  --descriptions "Vehicle Descriptions.csv" \
  --zip-to-county "data/zip_to_county_ny.csv" \
  --zip-to-region "data/utility_zip_regions.csv" \
  --snapshot-map "NY_DMV_Snapshots.csv" \
  --region "LIPA" \
  --counties "NASSAU,SUFFOLK" \
  --output "$PANEL_CSV" \
  --summary "$SUMMARY_JSON"

CMD=(
  python scripts/build_and_fit_bass_lipa.py
  --panel-csv "$PANEL_CSV"
  --with-policy
  --holdout-start "$HOLDOUT_START"
  --horizon "$HORIZON"
  --flow-likelihood "$FLOW_LIKELIHOOD"
  --output-tag "$OUTPUT_TAG"
)

if [[ "$RESAMPLE_MONTHLY" == "1" ]]; then
  CMD+=(--resample-monthly)
fi
if [[ "$WITH_EVSE" == "1" ]]; then
  CMD+=(
    --with-evse
    --station-info-dir "$STATION_INFO_DIR"
    --evse-restricted-weight "$EVSE_RESTRICTED_WEIGHT"
    --evse-workplace-weight "$EVSE_WORKPLACE_WEIGHT"
    --evse-output-csv "$EVSE_OUTPUT_CSV"
  )
fi

"${CMD[@]}"

echo "Done. Outputs tagged with: $OUTPUT_TAG"
