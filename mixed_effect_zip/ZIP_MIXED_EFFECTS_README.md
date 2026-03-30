# ZIP-Level Mixed-Effects EV Adoption Model

## Why this model

Your current repo already has:
- a regional Bass model for LIPA
- an independent ZIP Bass script (`scripts/build_and_fit_bass_zip.py`) with optional partial pooling toward LIPA parameters

The next logical step is a **single pooled ZIP model** that learns one common structure across ZIPs while allowing each ZIP to deviate from the common trend. That is exactly what a mixed-effects model does.

## Recommended first production model

Model the ZIP-level **adoption flow** (`flow_ev_t`, first-seen EVs) instead of the stock directly.

For ZIP `z` at time `t`:

- Response: `flow_ev_t`
- State: `adopt_ev_cum_prev`
- Remaining market: `M_z - adopt_ev_cum_prev`
- Local imitation share: `adopt_ev_cum_prev / M_z`

Recommended specification:

```text
flow[z,t] ~ NegativeBinomial(mu[z,t], alpha_nb)

log mu[z,t] = log(M[z] - A[z,t-1])
               + log_dt[z,t]
               + alpha
               + u_zip[z]
               + (beta_im + v_zip[z]) * (A[z,t-1] / M[z])
               + beta_tco    * tco_adv_t
               + beta_policy * subsidy_share_t
               + beta_sin    * month_sin
               + beta_cos    * month_cos
```

Interpretation:
- `u_zip[z]` is a ZIP random intercept
- `v_zip[z]` is a ZIP random slope on the imitation term
- fixed effects capture common statewide / utility-wide drivers
- the negative-binomial likelihood is better than plain Poisson when ZIP counts are noisy or overdispersed

## Why this is preferable to fitting each ZIP separately

It solves the biggest ZIP-level problem: **sparsity**.

Small or late-adopting ZIPs do not have enough information to estimate their own `p`, `q`, and `M` reliably. A mixed-effects model partially pools them toward the common pattern while still letting strong ZIPs keep their own shape.

## Practical guidance

### 1) Keep the time axis regular
Use monthly rows, or at least include zero rows for missing months / snapshots. The script adjusts for interval length with `log_dt`, but monthly panels are cleaner.

### 2) Start with the monthly era
Use `2018-01-01+` first. The pre-2018 annual / irregular snapshot era is much harder to pool cleanly across ZIPs.

### 3) Do not start with too many random slopes
Start with:
- ZIP random intercept
- ZIP random slope on imitation share

Only add ZIP-varying TCO or policy slopes after the base model is stable.

### 4) Market size matters
Preferred: provide a ZIP-level `market_size` file.

If you do not have one yet, the script falls back to the same proxy already used in your ZIP Bass code:

```text
ZIP total market ≈ (ZIP EV stock / LIPA EV stock) × LIPA total light-duty stock
EV market potential M_z = market_potential_frac × ZIP total market
```

That fallback is good enough for a first pooled model, but a true ZIP market-size input will improve calibration.

## Files

- `mixed_effect_zip/fit_zip_mixed_effects.py` — implementation module
- `scripts/fit_zip_mixed_effects.py` — integrated entry point that matches the rest of the ZIP workflow

## Requirements

- `pymc`
- `arviz`
- at least **two unique ZIP panels** after filtering

If you use a broad glob like `covariates/panel_zip*.csv`, the integrated loader now
deduplicates by ZIP and prefers the canonical file `panel_zip<ZIP>.csv` over suffixed
variants such as `panel_zip<ZIP>_mse.csv`.

## Basic usage

### Option A — stacked panel

```bash
python scripts/fit_zip_mixed_effects.py \
  --panel-csv covariates/panel_zip_stacked.csv \
  --market-size-csv covariates/zip_market_size.csv \
  --feature-cols tco_adv_t,subsidy_share_t \
  --holdout-start 2025-01-01 \
  --min-date 2018-01-01 \
  --horizon 24 \
  --fit-method advi \
  --output-tag lipa_zip_mixed
```

### Option B — existing cached per-ZIP panels

If you already have files like `covariates/panel_zip11746.csv`, `covariates/panel_zip11743.csv`, etc.:

```bash
python scripts/fit_zip_mixed_effects.py \
  --panel-glob 'covariates/panel_zip*.csv' \
  --feature-cols tco_adv_t,subsidy_share_t \
  --with-policy \
  --holdout-start 2025-01-01 \
  --min-date 2018-01-01 \
  --horizon 24 \
  --fit-method advi \
  --market-potential-frac 0.5 \
  --output-tag lipa_zip_mixed
```

## Suggested validation ladder

1. Compare against independent ZIP Bass fits on the same holdout window.
2. Report overall and by-ZIP metrics.
3. Plot residuals against ZIP size.
4. Check whether the pooled model especially improves the low-volume ZIPs.

## What to expect

A well-specified mixed-effects model should usually beat independent ZIP fits in:
- low-volume ZIPs
- late-adopting ZIPs
- holdout stability of market potential

It may not beat the best large ZIPs one by one, but it should be **more robust overall**.

## Integration notes

- The market-size fallback is aligned with the existing ZIP Bass script and now uses the
  **training window only** when a holdout period is requested, to avoid holdout leakage.
- Future seasonality (`month_sin`, `month_cos`) is now recomputed by future month instead
  of being held flat.
- Stock is reconstructed from predicted ZIP adoption flows using the same retention-curve
  logic already used in the LIPA/ZIP Bass workflow.

## When to move beyond this model

Move to a full nonlinear hierarchical Bass model only after this mixed-effects count model is stable. The nonlinear hierarchical Bass is more structurally faithful, but it is harder to fit and easier to destabilize.
