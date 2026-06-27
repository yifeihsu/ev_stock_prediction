# Albany Behavioral Charging Model

This folder has a runnable light-duty behavioral charging model for the updated
Albany County trip dataset:

```bash
python scripts/build_behavioral_charging_model.py \
  --charger-assumptions Albany_Trip_Behaviour/config/charger_assumptions.csv \
  --output-dir out_behavioral_charging
```

The default run reads `Albany_Trip_Behaviour/data/raw/Albany_County_NYU_Team.csv`,
reconstructs vehicle-day trip chains, assigns home ZCTA from home coordinates,
assigns charging H3 cells from destination coordinates, classifies charging
opportunities, allocates expected charging energy, and writes 15-minute and
hourly load curves for unmanaged and managed charging.

## GridUp-Style Charger Availability

The default charger availability model follows the GridUp design note. The
charger assumption file records a peak-demand-hour probability for each charger
category. The model then adjusts that probability by the stopped-vehicle demand
at the charging location and hour:

```text
availability_probability =
    min(1, peak_demand_hour_probability * peak_stopped_count / current_stopped_count)
```

For example, if a charger has a 50% peak-demand-hour probability and the
current hour has half as many stopped vehicles as the peak hour at that
location, the modeled availability becomes 100%. The static peak probabilities
can still be used for debugging with `--availability-choice-model static`.

The current Albany implementation uses the LDV GridUp values that can be mapped
from available fields: single-family home L2, multi-family home L2, workplace
L2, civic L2, long-public L2, and a representative quick-public DCFC 150 kW
case. The Albany file does not expose office/non-office workplace land use or
all quick-public DCFC power-level alternatives, so those remain prototype
approximations.

## Geography Design

The model keeps adoption geography separate from physical charging geography:

- `home_zcta`: ZIP/ZCTA-style home geography used to join ZIP-level adoption
  forecasts.
- `charging_h3`: destination H3 cell used as the primary charging-load
  placement geography. Resolution 8 is the default and is recorded in
  `model_summary.json`.
- `charging_point_id`, `charging_lon`, `charging_lat`: rounded destination
  point outputs kept for QA/debugging.
- `charging_zcta`: optional destination ZCTA reporting geography when
  `--assign-charging-zcta` is set.

The trip file does not directly provide ZIP/ZCTA. By default, the script maps
`HOME_X/HOME_Y` to Census ZCTA polygons using
`data/ny_new_york_zip_codes_geo.min.json` and the Albany ZIP filter in
`data/zip_to_county_ny.csv`. The current project convention is
`usps_zip_mapped_to_zcta`: ZIP-level adoption forecasts are joined through this
ZCTA-like home key. The run summary records both modeled home ZCTAs and adoption
file ZCTAs so coverage can be audited. The model treats `*_X` as longitude and
`*_Y` as latitude because the data values show that orientation, even though
the data dictionary labels them the other way around.

## Core Outputs

- `charging_events.csv`: one row per modeled expected charging event.
- `vehicle_day_energy_summary.csv`: travel energy, delivered energy, and unmet
  energy by scenario and vehicle day.
- `full_ev_load_15min_by_home_zcta_charging_h3.csv`: 100% electrified load
  surface by home ZCTA, charging H3, charger type, scenario, and time bin.
- `full_ev_load_hourly_by_home_zcta_charging_h3.csv`: hourly version of the
  same full-EV load surface.
- `load_15min_by_h3.csv` and `load_hourly_by_h3.csv`: full-EV charging load
  aggregated to charging H3 only.
- `load_15min_by_home_zcta.csv` and `load_hourly_by_home_zcta.csv`: load
  attributed to home ZCTA for adoption-linkage QA.
- `load_15min_by_point.csv` and `load_hourly_by_point.csv`: rounded coordinate
  debug outputs.
- `peak_by_point.csv` and `peak_by_home_zcta.csv`: peak bin summaries.
- `distance_qa.csv`: reported, routed, straight-line, and selected distance
  fields with ratio checks, selected-distance source, route-below-straight-line
  flags, and dwell-window QA fields.
- `scenario_energy_summary.csv`: delivered and unmet expected energy by
  managed/unmanaged scenario.
- `model_summary.json`: run configuration, QA counts, and output row counts.

Optional debug outputs:

```bash
python scripts/build_behavioral_charging_model.py \
  --write-trip-table \
  --write-stop-table \
  --write-point-home-load \
  --charger-assumptions Albany_Trip_Behaviour/config/charger_assumptions.csv \
  --output-dir out_behavioral_charging_debug
```

## Adoption Scaling

Production adoption scaling should use a ZIP/ZCTA adoption forecast file:

```bash
python scripts/build_behavioral_charging_model.py \
  --scale-adoption \
  --adoption-file path/to/zip_adoption_forecast.csv \
  --charger-assumptions Albany_Trip_Behaviour/config/charger_assumptions.csv \
  --output-dir out_behavioral_charging_scaled
```

Expected adoption columns:

```text
adoption_scenario, forecast_year, home_zcta, adoption_fraction, vehicle_growth_factor
```

`vehicle_growth_factor` is optional and defaults to `1.0`. The adoption file may
also use `home_zip` or `zip`; the script normalizes those to `home_zcta`.

The current Albany ZIP Bass forecast can be converted with:

```bash
python scripts/prepare_charging_adoption_file.py \
  --input models/adoption_forecast_albany_zip_bass_central_hudson_covariates_snapshot.csv \
  --output models/adoption_forecast_albany_zip_for_charging.csv
```

Then run:

```bash
python scripts/build_behavioral_charging_model.py \
  --scale-adoption \
  --adoption-file models/adoption_forecast_albany_zip_for_charging.csv \
  --charger-assumptions Albany_Trip_Behaviour/config/charger_assumptions.csv \
  --output-dir out_behavioral_charging_scaled
```

The adapter uses the latest monthly forecast row in each calendar year and
converts scenario stock columns to long-form adoption fractions:

- `stock_ev_t_hat_baseline` -> `baseline`
- `stock_ev_t_hat_tco` -> `tco`
- `stock_ev_t_hat_tco_evse` -> `tco_evse`

By default, EV stock is divided by `total_vehicle_market_proxy`. If that
denominator is zero or missing, the adapter falls back to `market_size` and
records the denominator source in the prepared CSV.

Scaled outputs:

- `scaled_load_15min_by_charging_h3.csv`
- `scaled_load_hourly_by_charging_h3.csv`
- `scaled_load_15min_by_point.csv` and `scaled_load_hourly_by_point.csv` only
  when `--write-point-home-load` is also set.

For software smoke testing only, demo low/base/high adoption curves can be
enabled explicitly:

```bash
python scripts/build_behavioral_charging_model.py \
  --scale-adoption \
  --use-demo-adoption \
  --forecast-years 2030,2035,2040 \
  --charger-assumptions Albany_Trip_Behaviour/config/charger_assumptions.csv \
  --output-dir out_behavioral_charging_demo_scaled
```

## Key Model Fixes Now Reflected

- The default trip input is the updated `Albany_County_NYU_Team.csv`.
- Trip-chain origins are reconstructed by default: first trip starts at home,
  later trips start at the previous destination. Input `START_X/START_Y` is kept
  only for QA.
- Trip times are unwrapped sequentially within each synthetic person/home chain,
  so after-midnight trips can continue the same vehicle-day chain.
- Distance defaults to route-first selection using `SHORTEST_TDIST_miles`, then
  `SHORTEST_PDIST_miles`. Reported `TRPMILES` is used only as a fallback when
  it is positive, below the cap, and plausible against straight-line distance.
  Routed distances are also rejected when materially below straight-line
  distance. Use `--distance-source trip` only for sensitivity checks.
- Invalid-distance drops that create trip-number gaps start a new chain and are
  counted in `trip_qa`. Origins and origin-dependent distance QA fields are
  recomputed after those final chain breaks are assigned.
- Non-final stop dwell windows use inferred next-trip gaps for load timing;
  raw `DWELTIME` is retained for QA and mismatch counts.
- Adoption scaling joins on `home_zcta`, not home block group or synthetic IDs.
  Coverage is checked for every `adoption_scenario` x `forecast_year`, not just
  the union of ZCTAs.
- H3 output is required by default. Use `--skip-h3` only for local QA/debugging
  runs without adoption scaling.
- Charger assumptions are externalized in
  `Albany_Trip_Behaviour/config/charger_assumptions.csv`.
- Charger availability uses the GridUp peak-demand-hour probability plus a
  time-varying inverse-demand adjustment based on stopped vehicles at the
  charging location. The resulting probability then enters conditional
  expected-value energy allocation. Partial-capacity stops retain separate
  available/unavailable probability states.
- Managed charging is applied only to charger types marked
  `managed_eligible_flag=true`; the default assumptions manage home and work
  charging only.
- Charger assumptions fail fast when `rated_power_kw` is not positive or
  `peak_demand_hour_probability` is outside `[0, 1]`.

## Useful Options

- `--origin-mode input`: use input `START_X/START_Y` instead of reconstructed
  origins.
- `--distance-source trip`: prefer plausible `TRPMILES` before route fields for
  sensitivity checks.
- `--availability-choice-model static`: use configured peak-demand-hour
  probabilities directly instead of the default GridUp inverse-demand
  adjustment. This is intended for sensitivity/debugging.
- `--skip-zcta`: skip home ZCTA assignment; this is for debugging only and is
  not suitable for adoption scaling.
- `--assign-charging-zcta`: add destination ZCTA for reporting rollups.
- `--allow-nearest-zcta-fallback`: explicitly allow nearest-centroid ZCTA
  fallback for points outside polygons, subject to `--nearest-zcta-max-miles`.
- `--allow-missing-home-zcta`: explicitly permit adoption scaling with partial
  home-ZCTA coverage.
- `--adoption-geography-type`: records whether adoption inputs are treated as
  Census ZCTA or USPS ZIP mapped to ZCTA. The default is
  `usps_zip_mapped_to_zcta`.
- `--h3-resolution 8`: default H3 resolution; keep resolution 8 for planning
  runs unless a different grid geography has been agreed.

## Current Caveats

- ZIP/ZCTA is assigned from coordinates using local polygon data. Nearest-ZCTA
  fallback is off by default and, when enabled, is thresholded and reported in
  `model_summary.json`.
- Point-level outputs can be large because destination coordinates are granular.
  They are mainly QA artifacts; use H3 outputs for planning analysis.
- The charger assumption file is still a scenario input, not a calibrated local
  parameter set. Review home, workplace, public L2, and DCFC access assumptions
  before treating outputs as planning-ready.
