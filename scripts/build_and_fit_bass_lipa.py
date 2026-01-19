#!/usr/bin/env python3
"""
Build a covariate panel for LIPA and fit Bass diffusion models for EV adoption.

Inputs (already produced by earlier scripts):
  - out_ev_snapshots/counts_by_region_snapshot.csv
      region, group (DMV_ID), metric (EV|ALL), approx_unique_vins
  - out_new_reg/new_reg_by_region_snapshot.csv
      region, snapshot_id (DMV_ID), new_all, new_ev, ev_share_pct
  - NY_DMV_Snapshots.csv
      DMV_ID, DMV_Snapshot_Date, ...
  - prices/Gasoline Retail Prices LIPA.xlsx
      Weekly gas prices for Nassau (used as LIPA proxy):
      sheet 'Gasoline_Retail_Prices_Weekly_A' with columns:
        Date, 'Nassau Average ($/gal)'

What this script does
  1) Build panel_LIPA.csv with:
       DMV_ID, date, stock_ev_t, flow_ev_t, stock_all_t,
       gas_price_t, C_gas_t, C_ev_t, tco_adv_t
     (electricity price is approximated as a constant; adjust as needed).
  2) Fit a baseline Bass diffusion model (p, q, M) on flow_ev_t.
  3) Fit a Bass model where p_t depends on tco_adv_t (simple covariate version).
  4) Write parameter JSONs and a forecast CSV for the next N snapshots.

Usage
  python scripts/build_and_fit_bass_lipa.py \
    --horizon 24 \
    --elec-price 0.22
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BassParams:
    p: float
    q: float
    M: float


@dataclass
class BassCovParams:
    alpha_p: float
    beta_p: float
    q: float
    M: float
    feature_cols: List[str]
    X_mean: List[float]
    X_std: List[float]


def load_lipa_stocks() -> pd.DataFrame:
    """Load snapshot-based EV and ALL stocks for LIPA."""
    counts = pd.read_csv(ROOT / "out_ev_snapshots" / "counts_by_region_snapshot.csv")
    counts = counts[counts["region"] == "LIPA"].copy()
    counts.rename(columns={"group": "DMV_ID"}, inplace=True)
    counts["DMV_ID"] = counts["DMV_ID"].astype(int)

    snap = pd.read_csv(ROOT / "NY_DMV_Snapshots.csv")[["DMV_ID", "DMV_Snapshot_Date"]]
    snap["DMV_ID"] = snap["DMV_ID"].astype(int)
    snap.rename(columns={"DMV_Snapshot_Date": "date"}, inplace=True)

    stocks = (
        counts.merge(snap, on="DMV_ID", how="left")
        .pivot_table(
            index=["DMV_ID", "date"],
            columns="metric",
            values="approx_unique_vins",
            aggfunc="first",
        )
        .reset_index()
    )
    stocks.rename(columns={"EV": "stock_ev_t", "ALL": "stock_all_t"}, inplace=True)
    stocks["date"] = pd.to_datetime(stocks["date"])
    stocks = stocks.sort_values("date").reset_index(drop=True)
    return stocks


def load_lipa_flows() -> pd.DataFrame:
    """Load first-seen (new EV registrations) for LIPA."""
    flows = pd.read_csv(ROOT / "out_new_reg" / "new_reg_by_region_snapshot.csv")
    flows = flows[flows["region"] == "LIPA"].copy()
    flows.rename(columns={"snapshot_id": "DMV_ID", "new_ev": "flow_ev_t"}, inplace=True)
    flows["DMV_ID"] = flows["DMV_ID"].astype(int)
    flows = flows[["DMV_ID", "flow_ev_t"]]
    return flows


def attach_gas_price(panel: pd.DataFrame, gas_series_path: str | None = None) -> pd.DataFrame:
    """Attach gasoline price ($/gal) to the snapshot panel.

    Behavior:
    - If `gas_series_path` is provided, use that CSV (must have columns: `date` + `gas_price_t`
      OR `date` + `gas_price_cents_per_gallon`).
    - Otherwise prefer `prices/gasoline_downstate_ny_monthly.csv` (if present).
    - Otherwise fall back to `prices/Gasoline Retail Prices LIPA.xlsx` (weekly Nassau).
    """
    if gas_series_path:
        p = Path(gas_series_path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"--gas-series not found: {p}")
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        if "gas_price_t" not in df.columns:
            if "gas_price_cents_per_gallon" not in df.columns:
                raise ValueError(f"{p} must contain gas_price_t or gas_price_cents_per_gallon")
            df["gas_price_t"] = df["gas_price_cents_per_gallon"].astype(float) / 100.0
        df = df[["date", "gas_price_t"]].sort_values("date")
    else:
        monthly_path = ROOT / "prices" / "gasoline_downstate_ny_monthly.csv"
        if monthly_path.exists():
            df = pd.read_csv(monthly_path)
            df["date"] = pd.to_datetime(df["date"])
            if "gas_price_t" not in df.columns:
                if "gas_price_cents_per_gallon" not in df.columns:
                    raise ValueError(
                        f"{monthly_path} must contain gas_price_t or gas_price_cents_per_gallon"
                    )
                df["gas_price_t"] = df["gas_price_cents_per_gallon"].astype(float) / 100.0
            df = df[["date", "gas_price_t"]].sort_values("date")
        else:
            excel_path = ROOT / "prices" / "Gasoline Retail Prices LIPA.xlsx"
            xls = pd.ExcelFile(excel_path)
            df = xls.parse(xls.sheet_names[0])
            df = df.rename(columns={"Date": "date", "Nassau Average ($/gal)": "gas_price_t"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

    panel = panel.sort_values("date").reset_index(drop=True)
    panel = pd.merge_asof(
        panel,
        df[["date", "gas_price_t"]],
        on="date",
        direction="backward",
    )
    # Fill missing gas prices (e.g. early years) to prevent NaNs in covariates
    panel["gas_price_t"] = panel["gas_price_t"].bfill().ffill()
    return panel


def build_panel(elec_price_default: float, gas_series_path: str | None = None) -> pd.DataFrame:
    stocks = load_lipa_stocks()
    flows = load_lipa_flows()

    panel = stocks.merge(flows, on="DMV_ID", how="left")
    panel["flow_ev_t"] = panel["flow_ev_t"].fillna(0.0)

    panel = attach_gas_price(panel, gas_series_path=gas_series_path)

    # Approximate electricity price as a constant for now; user can update this column later.
    panel["elec_price_t"] = elec_price_default

    # Construct operating cost advantage
    MPG_AVG = 28.0
    KWH_PER_MI_AVG = 0.30
    panel["C_gas_t"] = panel["gas_price_t"] / MPG_AVG
    panel["C_ev_t"] = panel["elec_price_t"] * KWH_PER_MI_AVG
    panel["tco_adv_t"] = panel["C_gas_t"] - panel["C_ev_t"]

    # Keep periods with any EVs
    panel = panel.sort_values("date").reset_index(drop=True)
    panel = panel[panel["stock_ev_t"] > 0].reset_index(drop=True)
    return panel


def attach_policy(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach monthly policy covariates (e.g., subsidy share) to snapshot panel."""
    policy_path = ROOT / "covariates" / "policy_covariates.csv"
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Missing {policy_path}. Run scripts/build_policy_covariates.py first."
        )
    policy = pd.read_csv(policy_path)
    policy["date"] = pd.to_datetime(policy["date"])
    policy = policy.sort_values("date")

    panel = panel.copy()
    panel = panel.sort_values("date")
    # Merge only the policy columns we need, to avoid name collisions with panel columns.
    keep_cols = [
        "date",
        "subsidy_share_t",
        "total_subsidy_t",
        "fed_credit_avg_t",
        "state_rebate_t",
    ]
    missing = [c for c in keep_cols if c not in policy.columns]
    if missing:
        raise ValueError(f"policy_covariates.csv missing required columns: {missing}")
    merged = pd.merge_asof(panel, policy[keep_cols], on="date", direction="backward")

    return merged


def bass_flow(y_prev: np.ndarray, p: float, q: float, M: float) -> np.ndarray:
    return (p + q * (y_prev / M)) * (M - y_prev)


def fit_bass_baseline(panel: pd.DataFrame) -> BassParams:
    df = panel.copy().sort_values("date").reset_index(drop=True)
    y = df["stock_ev_t"].values.astype(float)
    n = df["flow_ev_t"].values.astype(float)
    y_prev = np.concatenate(([0.0], y[:-1]))

    M0 = 1.5 * y.max()
    p0 = 0.01
    q0 = 0.3

    def objective(theta: np.ndarray) -> float:
        p, q, M = theta
        if M <= y.max() or p <= 0 or q <= 0:
            return 1e12
        n_hat = bass_flow(y_prev, p, q, M)
        return float(np.mean((n - n_hat) ** 2))

    bounds = [(1e-6, 0.1), (1e-3, 2.0), (y.max() * 1.01, y.max() * 10)]
    res = minimize(objective, x0=np.array([p0, q0, M0]), bounds=bounds)
    p_hat, q_hat, M_hat = res.x
    return BassParams(p=p_hat, q=q_hat, M=M_hat)


def fit_bass_with_tco(panel: pd.DataFrame, feature_cols: List[str]) -> BassCovParams:
    df = panel.copy().sort_values("date").reset_index(drop=True)
    y = df["stock_ev_t"].values.astype(float)
    n = df["flow_ev_t"].values.astype(float)
    y_prev = np.concatenate(([0.0], y[:-1]))
    X = df[feature_cols].values.astype(float)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    Xs = (X - X_mean) / X_std
    k = Xs.shape[1]

    alpha0 = -4.0
    beta0 = np.zeros(k)
    q0 = 0.3
    M0 = 1.5 * y.max()

    def objective(theta: np.ndarray) -> float:
        alpha_p = theta[0]
        beta_p = theta[1 : 1 + k]
        q = theta[1 + k]
        M = theta[2 + k]
        if M <= y.max() or q <= 0:
            return 1e12

        log_p_t = alpha_p + Xs @ beta_p
        p_t = np.exp(log_p_t)
        n_hat = (p_t + q * (y_prev / M)) * (M - y_prev)
        return float(np.mean((n - n_hat) ** 2))

    x0 = np.concatenate(([alpha0], beta0, [q0, M0]))
    bounds = [(-10, 0)] + [(-5, 5)] * k + [(1e-3, 2.0), (y.max() * 1.01, y.max() * 10)]
    res = minimize(objective, x0=x0, bounds=bounds)
    alpha_p, *rest = res.x
    beta_p = np.array(rest[:-2])
    q_hat, M_hat = rest[-2], rest[-1]

    return BassCovParams(
        alpha_p=float(alpha_p),
        beta_p=beta_p.tolist(),
        q=float(q_hat),
        M=float(M_hat),
        feature_cols=feature_cols,
        X_mean=X_mean.tolist(),
        X_std=X_std.tolist(),
    )


def forecast_bass(panel: pd.DataFrame, params: BassParams, horizon: int) -> pd.DataFrame:
    df = panel.copy().sort_values("date").reset_index(drop=True)
    y = df["stock_ev_t"].values.astype(float)
    n_obs = df["flow_ev_t"].values.astype(float)
    dates = df["date"].tolist()
    T = len(df)

    p, q, M = params.p, params.q, params.M

    # Simulate full history + forecast
    # Start simulation from y[0] to match the dynamic nature of diffusion
    y_hat = np.zeros(T + horizon, dtype=float)
    n_hat = np.zeros(T + horizon, dtype=float)

    # Initial condition
    y_hat[0] = y[0]
    n_hat[0] = np.nan

    future_dates = [dates[-1] + pd.DateOffset(months=h) for h in range(1, horizon + 1)]
    all_dates = dates + future_dates
    dmv_ids = list(df["DMV_ID"]) + [np.nan] * horizon

    for t in range(1, T + horizon):
        y_prev = y_hat[t - 1]
        n_t = float(bass_flow(np.array([y_prev]), p, q, M)[0])
        y_hat[t] = y_prev + n_t
        n_hat[t] = n_t

    # Build DataFrame with observed + forecast
    y_obs_ext = np.concatenate([y, [np.nan] * horizon])
    n_obs_ext = np.concatenate([n_obs, [np.nan] * horizon])

    return pd.DataFrame(
        {
            "date": all_dates,
            "DMV_ID": dmv_ids,
            "stock_ev_t_obs": y_obs_ext,
            "flow_ev_t_obs": n_obs_ext,
            "stock_ev_t_hat": y_hat,
            "flow_ev_t_hat": n_hat,
        }
    )


def forecast_bass_anchor(panel: pd.DataFrame, params: BassParams, horizon: int) -> pd.DataFrame:
    """One-step fitted flows on history + forward simulation from the last observed stock.

    This avoids compounding in-sample simulation error when plotting fitted flows:
      - History: n_hat_t uses observed stock_{t-1}
      - Forecast: simulate forward starting from last observed stock
    """
    df = panel.copy().sort_values("date").reset_index(drop=True)
    y_obs = df["stock_ev_t"].values.astype(float)
    n_obs = df["flow_ev_t"].values.astype(float)
    dates = df["date"].tolist()
    T = len(df)

    p, q, M = params.p, params.q, params.M

    y_prev_obs = np.concatenate(([0.0], y_obs[:-1]))
    n_hat_hist = bass_flow(y_prev_obs, p, q, M)

    # Forecast horizon: simulate from last observed stock.
    y_prev = float(y_obs[-1]) if T else 0.0
    y_future = np.zeros(horizon, dtype=float)
    n_future = np.zeros(horizon, dtype=float)
    for h in range(horizon):
        n_t = float(bass_flow(np.array([y_prev]), p, q, M)[0])
        y_prev = y_prev + n_t
        n_future[h] = n_t
        y_future[h] = y_prev

    future_dates = [dates[-1] + pd.DateOffset(months=h) for h in range(1, horizon + 1)]
    all_dates = dates + future_dates
    dmv_ids = list(df["DMV_ID"]) + [np.nan] * horizon

    # Stock: use observed history (to keep the series anchored); forecast thereafter.
    y_hat = np.concatenate([y_obs, y_future])
    n_hat = np.concatenate([n_hat_hist, n_future])

    y_obs_ext = np.concatenate([y_obs, [np.nan] * horizon])
    n_obs_ext = np.concatenate([n_obs, [np.nan] * horizon])

    return pd.DataFrame(
        {
            "date": all_dates,
            "DMV_ID": dmv_ids,
            "stock_ev_t_obs": y_obs_ext,
            "flow_ev_t_obs": n_obs_ext,
            "stock_ev_t_hat_anchor": y_hat,
            "flow_ev_t_hat_anchor": n_hat,
        }
    )


def forecast_bass_with_tco(
    panel: pd.DataFrame, cov_params: BassCovParams, horizon: int
) -> pd.DataFrame:
    """Forecast Bass with time-varying p_t driven by tco_adv_t (or other features).

    For now we:
      - match the historical period to panel dates
      - hold covariates (and thus p_t) constant at the last observed value
        for the forecast horizon.
    """
    df = panel.copy().sort_values("date").reset_index(drop=True)
    y = df["stock_ev_t"].values.astype(float)
    dates = df["date"].tolist()
    T = len(df)

    # Prepare p_t from covariates
    X = df[cov_params.feature_cols].values.astype(float)
    Xs = (X - np.array(cov_params.X_mean)) / np.array(cov_params.X_std)
    beta = np.array(cov_params.beta_p)
    alpha = cov_params.alpha_p
    p_hist = np.exp(alpha + Xs @ beta)  # length T

    # Extend p_t into the future by holding constant the last value
    p_future = np.repeat(p_hist[-1], horizon)
    p_all = np.concatenate([p_hist, p_future])

    # Date sequence: historical + horizon months forward
    future_dates = [dates[-1] + pd.DateOffset(months=h) for h in range(1, horizon + 1)]
    all_dates = dates + future_dates

    # DMV_ID: keep historical, NaN for forecast
    dmv_ids = list(df["DMV_ID"]) + [np.nan] * horizon

    # Iterative stock/flow forecast
    M = cov_params.M
    q = cov_params.q
    y_hat = np.zeros(T + horizon, dtype=float)
    n_hat = np.zeros(T + horizon, dtype=float)
    y_hat[0] = y[0]
    n_hat[0] = np.nan
    for t in range(1, T + horizon):
        y_prev = y_hat[t - 1]
        p_t = p_all[t]
        n_t = float(bass_flow(np.array([y_prev]), p_t, q, M)[0])
        n_hat[t] = n_t
        y_hat[t] = y_prev + n_t

    return pd.DataFrame(
        {
            "date": all_dates,
            "DMV_ID": dmv_ids,
            "stock_ev_t_hat_cov": y_hat,
            "flow_ev_t_hat_cov": n_hat,
        }
    )


def forecast_bass_with_tco_anchor(panel: pd.DataFrame, cov_params: BassCovParams, horizon: int) -> pd.DataFrame:
    """Anchored version of forecast_bass_with_tco (one-step fit + forward simulation).

    - History: one-step fitted flows use observed stock_{t-1}
    - Forecast: simulate forward from last observed stock, holding covariates (p_t) constant at last value
    """
    df = panel.copy().sort_values("date").reset_index(drop=True)
    y_obs = df["stock_ev_t"].values.astype(float)
    n_obs = df["flow_ev_t"].values.astype(float)
    dates = df["date"].tolist()
    T = len(df)

    X = df[cov_params.feature_cols].values.astype(float)
    Xs = (X - np.array(cov_params.X_mean)) / np.array(cov_params.X_std)
    beta = np.array(cov_params.beta_p)
    alpha = cov_params.alpha_p
    p_hist = np.exp(alpha + Xs @ beta)

    M = cov_params.M
    q = cov_params.q

    y_prev_obs = np.concatenate(([0.0], y_obs[:-1]))
    n_hat_hist = (p_hist + q * (y_prev_obs / M)) * (M - y_prev_obs)

    # Forecast horizon: simulate from last observed stock, hold p_t constant at last observed value.
    p_last = float(p_hist[-1]) if T else float(np.exp(alpha))
    y_prev = float(y_obs[-1]) if T else 0.0
    y_future = np.zeros(horizon, dtype=float)
    n_future = np.zeros(horizon, dtype=float)
    for h in range(horizon):
        n_t = float(bass_flow(np.array([y_prev]), p_last, q, M)[0])
        y_prev = y_prev + n_t
        n_future[h] = n_t
        y_future[h] = y_prev

    future_dates = [dates[-1] + pd.DateOffset(months=h) for h in range(1, horizon + 1)]
    all_dates = dates + future_dates
    dmv_ids = list(df["DMV_ID"]) + [np.nan] * horizon

    y_hat = np.concatenate([y_obs, y_future])
    n_hat = np.concatenate([n_hat_hist, n_future])

    return pd.DataFrame(
        {
            "date": all_dates,
            "DMV_ID": dmv_ids,
            "stock_ev_t_hat_cov_anchor": y_hat,
            "flow_ev_t_hat_cov_anchor": n_hat,
        }
    )


def extend_panel_with_future(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Append horizon months after the last observed row, forward-filling covariates."""
    df = panel.copy().sort_values("date").reset_index(drop=True)
    if horizon <= 0 or df.empty:
        return df

    last = df.iloc[-1]
    future_dates = [last["date"] + pd.DateOffset(months=h) for h in range(1, horizon + 1)]
    future = pd.DataFrame({"date": future_dates})
    future["DMV_ID"] = np.nan
    future["stock_ev_t"] = np.nan
    future["flow_ev_t"] = np.nan

    for c in df.columns:
        if c in ("date", "DMV_ID", "stock_ev_t", "flow_ev_t"):
            continue
        future[c] = last[c]

    return pd.concat([df, future], ignore_index=True)


def forecast_train_test_baseline(
    panel_full: pd.DataFrame, params: BassParams, holdout_start: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    """Fit-on-train, forecast-on-test using discrete-time Bass recursion.

    - Train period is used only for parameter estimation (done outside).
    - Forecast is produced for the holdout period (>= holdout_start) and optional future horizon.
    - For plotting clarity, we return NaNs for predicted flows/stocks in the training period,
      except that the last training stock is set for continuity.
    """
    df = extend_panel_with_future(panel_full, horizon=horizon).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty panel provided to forecast_train_test_baseline")

    dates = pd.to_datetime(df["date"])
    is_train = dates < holdout_start
    if is_train.sum() < 2:
        raise ValueError("Train set too small for holdout split")
    if (~is_train & df["stock_ev_t"].notna()).sum() < 1:
        raise ValueError("Test set empty for holdout split")

    # Observations (NaN for future rows)
    y_obs = pd.to_numeric(df["stock_ev_t"], errors="coerce").to_numpy(float)
    n_obs = pd.to_numeric(df["flow_ev_t"], errors="coerce").to_numpy(float)

    test_start_idx = int(np.where(~is_train.to_numpy(bool))[0][0])
    train_end_idx = test_start_idx - 1

    # In-sample fitted curve (train only): one-step using observed stock_{t-1}.
    y_fit = np.full(len(df), np.nan, dtype=float)
    n_fit = np.full(len(df), np.nan, dtype=float)
    if np.isfinite(y_obs[0]):
        y_fit[0] = y_obs[0]
    p, q, M = params.p, params.q, params.M
    for t in range(1, test_start_idx):
        y_prev_obs = y_obs[t - 1]
        if not np.isfinite(y_prev_obs):
            continue
        n_t = float(bass_flow(np.array([y_prev_obs]), p, q, M)[0])
        n_fit[t] = n_t
        y_fit[t] = y_prev_obs + n_t

    y_prev = float(y_obs[train_end_idx])
    y_hat = np.full(len(df), np.nan, dtype=float)
    n_hat = np.full(len(df), np.nan, dtype=float)
    y_hat[train_end_idx] = y_prev  # continuity point

    for t in range(test_start_idx, len(df)):
        n_t = float(bass_flow(np.array([y_prev]), p, q, M)[0])
        y_prev = y_prev + n_t
        n_hat[t] = n_t
        y_hat[t] = y_prev

    return pd.DataFrame(
        {
            "date": df["date"].tolist(),
            "DMV_ID": df["DMV_ID"].tolist(),
            "is_train": is_train.astype(int).tolist(),
            "stock_ev_t_obs": y_obs,
            "flow_ev_t_obs": n_obs,
            "stock_ev_t_hat_fit": y_fit,
            "flow_ev_t_hat_fit": n_fit,
            "stock_ev_t_hat_anchor": y_hat,
            "flow_ev_t_hat_anchor": n_hat,
        }
    )


def forecast_train_test_covariate(
    panel_full: pd.DataFrame, cov_params: BassCovParams, holdout_start: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    """Fit-on-train, forecast-on-test for covariate Bass (time-varying p_t)."""
    df = extend_panel_with_future(panel_full, horizon=horizon).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty panel provided to forecast_train_test_covariate")

    dates = pd.to_datetime(df["date"])
    is_train = dates < holdout_start
    if is_train.sum() < 2:
        raise ValueError("Train set too small for holdout split")
    if (~is_train & df["stock_ev_t"].notna()).sum() < 1:
        raise ValueError("Test set empty for holdout split")

    y_obs = pd.to_numeric(df["stock_ev_t"], errors="coerce").to_numpy(float)
    n_obs = pd.to_numeric(df["flow_ev_t"], errors="coerce").to_numpy(float)

    test_start_idx = int(np.where(~is_train.to_numpy(bool))[0][0])
    train_end_idx = test_start_idx - 1

    # p_t series (uses training standardization stored in cov_params)
    X = df[cov_params.feature_cols].values.astype(float)
    Xs = (X - np.array(cov_params.X_mean)) / np.array(cov_params.X_std)
    beta = np.array(cov_params.beta_p)
    alpha = cov_params.alpha_p
    p_t = np.exp(alpha + Xs @ beta)

    M = cov_params.M
    q = cov_params.q

    # In-sample fitted curve (train only): one-step using observed stock_{t-1}.
    y_fit = np.full(len(df), np.nan, dtype=float)
    n_fit = np.full(len(df), np.nan, dtype=float)
    if np.isfinite(y_obs[0]):
        y_fit[0] = y_obs[0]
    for t in range(1, test_start_idx):
        y_prev_obs = y_obs[t - 1]
        if not np.isfinite(y_prev_obs):
            continue
        p_now = float(p_t[t])
        n_t = float(bass_flow(np.array([y_prev_obs]), p_now, q, M)[0])
        n_fit[t] = n_t
        y_fit[t] = y_prev_obs + n_t

    y_prev = float(y_obs[train_end_idx])
    y_hat = np.full(len(df), np.nan, dtype=float)
    n_hat = np.full(len(df), np.nan, dtype=float)
    y_hat[train_end_idx] = y_prev

    for t in range(test_start_idx, len(df)):
        n_t = float(bass_flow(np.array([y_prev]), float(p_t[t]), q, M)[0])
        y_prev = y_prev + n_t
        n_hat[t] = n_t
        y_hat[t] = y_prev

    return pd.DataFrame(
        {
            "date": df["date"].tolist(),
            "DMV_ID": df["DMV_ID"].tolist(),
            "is_train": is_train.astype(int).tolist(),
            "stock_ev_t_obs": y_obs,
            "flow_ev_t_obs": n_obs,
            "stock_ev_t_hat_cov_fit": y_fit,
            "flow_ev_t_hat_cov_fit": n_fit,
            "stock_ev_t_hat_cov_anchor": y_hat,
            "flow_ev_t_hat_cov_anchor": n_hat,
        }
    )


def main():
    ap = argparse.ArgumentParser(description="Build LIPA panel and fit Bass models")
    ap.add_argument("--elec-price", type=float, default=0.22, help="Assumed LIPA electricity price ($/kWh)")
    ap.add_argument("--horizon", type=int, default=24, help="Forecast horizon in snapshots (approx. months)")
    ap.add_argument(
        "--min-date",
        type=str,
        default=None,
        help=(
            "If set (e.g., 2018-01-01), restrict the panel to snapshots on/after this date. "
            "Useful to exclude early irregular/annual snapshots."
        ),
    )
    ap.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help=(
            "Optional tag appended to output filenames in models/ (e.g., 'monthly' -> *_monthly.json/png/csv). "
            "If omitted, outputs use the default filenames."
        ),
    )
    ap.add_argument(
        "--gas-series",
        type=str,
        default=None,
        help="Optional path to a monthly gas price CSV with columns date + gas_price_t (or gas_price_cents_per_gallon).",
    )
    ap.add_argument(
        "--with-policy",
        action="store_true",
        help="Also fit a covariate Bass model using policy effective subsidy (subsidy_share_t).",
    )
    ap.add_argument(
        "--holdout-start",
        type=str,
        default=None,
        help=(
            "If set (e.g., 2025-01-01), fit parameters on data strictly before this date, "
            "and forecast/evaluate on data from this date onward."
        ),
    )
    args = ap.parse_args()

    tag = (args.output_tag or "").strip()
    suffix = f"_{tag}" if tag else ""

    def _tagged(filename: str) -> str:
        if not suffix:
            return filename
        p = Path(filename)
        return f"{p.stem}{suffix}{p.suffix}"

    panel = build_panel(elec_price_default=args.elec_price, gas_series_path=args.gas_series)

    if args.min_date:
        min_ts = pd.to_datetime(args.min_date)
        panel = panel[panel["date"] >= min_ts].copy()
        if panel.empty:
            raise ValueError(f"--min-date {args.min_date} filters the panel to zero rows")

    out_dir = ROOT / "covariates"
    out_dir.mkdir(exist_ok=True)
    panel.to_csv(out_dir / _tagged("panel_LIPA.csv"), index=False)

    panel_policy = attach_policy(panel) if args.with_policy else None

    holdout_ts: Optional[pd.Timestamp] = (
        pd.to_datetime(args.holdout_start) if args.holdout_start else None
    )

    if holdout_ts is None:
        # Full-sample fit
        bass_params = fit_bass_baseline(panel)
        cov_params = fit_bass_with_tco(panel, feature_cols=["tco_adv_t"])
        cov_params_policy = (
            fit_bass_with_tco(panel_policy, feature_cols=["tco_adv_t", "subsidy_share_t"])
            if panel_policy is not None
            else None
        )

        forecast_df = forecast_bass(panel, bass_params, horizon=args.horizon)
        forecast_cov = forecast_bass_with_tco(panel, cov_params, horizon=args.horizon)
        forecast_anchor = forecast_bass_anchor(panel, bass_params, horizon=args.horizon)
        forecast_cov_anchor = forecast_bass_with_tco_anchor(panel, cov_params, horizon=args.horizon)

        forecast_df = forecast_df.merge(
            forecast_cov[["date", "stock_ev_t_hat_cov", "flow_ev_t_hat_cov"]],
            on="date",
            how="left",
        )
        forecast_df = forecast_df.merge(
            forecast_anchor[["date", "stock_ev_t_hat_anchor", "flow_ev_t_hat_anchor"]],
            on="date",
            how="left",
        )
        forecast_df = forecast_df.merge(
            forecast_cov_anchor[["date", "stock_ev_t_hat_cov_anchor", "flow_ev_t_hat_cov_anchor"]],
            on="date",
            how="left",
        )
        if cov_params_policy is not None:
            forecast_cov_policy_anchor = forecast_bass_with_tco_anchor(
                panel_policy, cov_params_policy, horizon=args.horizon
            ).rename(
                columns={
                    "stock_ev_t_hat_cov_anchor": "stock_ev_t_hat_cov_policy_anchor",
                    "flow_ev_t_hat_cov_anchor": "flow_ev_t_hat_cov_policy_anchor",
                }
            )
            forecast_df = forecast_df.merge(
                forecast_cov_policy_anchor[
                    ["date", "stock_ev_t_hat_cov_policy_anchor", "flow_ev_t_hat_cov_policy_anchor"]
                ],
                on="date",
                how="left",
            )
    else:
        # Train/Test split fit: fit params only on pre-holdout data, forecast on holdout.
        train = panel[panel["date"] < holdout_ts].copy()
        if train.empty:
            raise ValueError(f"Holdout start {holdout_ts.date()} leaves an empty training set")

        bass_params = fit_bass_baseline(train)
        cov_params = fit_bass_with_tco(train, feature_cols=["tco_adv_t"])

        forecast_df = forecast_train_test_baseline(panel, bass_params, holdout_start=holdout_ts, horizon=args.horizon)
        forecast_cov_tt = forecast_train_test_covariate(panel, cov_params, holdout_start=holdout_ts, horizon=args.horizon)
        forecast_df = forecast_df.merge(
            forecast_cov_tt[
                [
                    "date",
                    "stock_ev_t_hat_cov_fit",
                    "flow_ev_t_hat_cov_fit",
                    "stock_ev_t_hat_cov_anchor",
                    "flow_ev_t_hat_cov_anchor",
                ]
            ],
            on="date",
            how="left",
        )

        cov_params_policy = None
        if panel_policy is not None:
            train_policy = panel_policy[panel_policy["date"] < holdout_ts].copy()
            cov_params_policy = fit_bass_with_tco(
                train_policy, feature_cols=["tco_adv_t", "subsidy_share_t"]
            )
            forecast_policy_tt = forecast_train_test_covariate(
                panel_policy, cov_params_policy, holdout_start=holdout_ts, horizon=args.horizon
            ).rename(
                columns={
                    "stock_ev_t_hat_cov_fit": "stock_ev_t_hat_cov_policy_fit",
                    "flow_ev_t_hat_cov_fit": "flow_ev_t_hat_cov_policy_fit",
                    "stock_ev_t_hat_cov_anchor": "stock_ev_t_hat_cov_policy_anchor",
                    "flow_ev_t_hat_cov_anchor": "flow_ev_t_hat_cov_policy_anchor",
                }
            )
            forecast_df = forecast_df.merge(
                forecast_policy_tt[
                    [
                        "date",
                        "stock_ev_t_hat_cov_policy_fit",
                        "flow_ev_t_hat_cov_policy_fit",
                        "stock_ev_t_hat_cov_policy_anchor",
                        "flow_ev_t_hat_cov_policy_anchor",
                    ]
                ],
                on="date",
                how="left",
            )

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # Save parameters
    with open(models_dir / _tagged("bass_lipa_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(bass_params), f, indent=2)
    with open(models_dir / _tagged("bass_lipa_with_tco.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cov_params), f, indent=2)
    if cov_params_policy is not None:
        with open(models_dir / _tagged("bass_lipa_with_tco_policy.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cov_params_policy), f, indent=2)

    # Save forecast table (now includes both baseline and covariate forecasts)
    forecast_df.to_csv(models_dir / _tagged("bass_lipa_forecast.csv"), index=False)

    # Optional holdout metrics (flows + stock) when a holdout split is used.
    if holdout_ts is not None and "is_train" in forecast_df.columns:
        test_mask = (forecast_df["is_train"] == 0) & (~forecast_df["stock_ev_t_obs"].isna())

        def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            m = np.isfinite(y_true) & np.isfinite(y_pred)
            if not m.any():
                return float("nan")
            return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))

        def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-9)
            if not m.any():
                return float("nan")
            return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100.0)

        obs_flow = forecast_df.loc[test_mask, "flow_ev_t_obs"].to_numpy(float)
        obs_stock = forecast_df.loc[test_mask, "stock_ev_t_obs"].to_numpy(float)

        metrics = {
            "holdout_start": str(holdout_ts.date()),
            "n_test": int(test_mask.sum()),
            "date_min_test": str(pd.to_datetime(forecast_df.loc[test_mask, "date"]).min().date()),
            "date_max_test": str(pd.to_datetime(forecast_df.loc[test_mask, "date"]).max().date()),
            "models": {},
        }

        candidates = [
            ("baseline", "flow_ev_t_hat_anchor", "stock_ev_t_hat_anchor"),
            ("tco_adv", "flow_ev_t_hat_cov_anchor", "stock_ev_t_hat_cov_anchor"),
            ("tco_adv_policy", "flow_ev_t_hat_cov_policy_anchor", "stock_ev_t_hat_cov_policy_anchor"),
        ]
        for name, flow_col, stock_col in candidates:
            if flow_col not in forecast_df.columns or stock_col not in forecast_df.columns:
                continue
            pred_flow = forecast_df.loc[test_mask, flow_col].to_numpy(float)
            pred_stock = forecast_df.loc[test_mask, stock_col].to_numpy(float)
            metrics["models"][name] = {
                "rmse_flow": _rmse(obs_flow, pred_flow),
                "mape_flow_pct": _mape(obs_flow, pred_flow),
                "rmse_stock": _rmse(obs_stock, pred_stock),
                "mape_stock_pct": _mape(obs_stock, pred_stock),
            }

        with open(models_dir / _tagged("bass_lipa_holdout_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Holdout metrics written to {models_dir / _tagged('bass_lipa_holdout_metrics.json')}")

    # Quick plots: stock and flows, baseline vs covariate
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Stock plot
        fig, ax = plt.subplots(figsize=(10, 5))
        hist = forecast_df[~forecast_df["stock_ev_t_obs"].isna()]
        if holdout_ts is None:
            last_hist_date = hist["date"].max()
            future = forecast_df[forecast_df["date"] >= last_hist_date]
        else:
            # show forecast from the last training snapshot onward
            train_end_date = forecast_df.loc[forecast_df["is_train"] == 1, "date"].max()
            fit_seg = forecast_df[forecast_df["is_train"] == 1]
            future = forecast_df[forecast_df["date"] >= train_end_date]
        ax.plot(hist["date"], hist["stock_ev_t_obs"], label="Observed stock", color="C0")
        ax.plot(
            future["date"],
            future["stock_ev_t_hat_anchor"],
            label="Bass forecast (baseline)" if holdout_ts is None else "Bass forecast (baseline, train<2025)",
            color="C1",
            linestyle="--",
        )
        if holdout_ts is not None and "stock_ev_t_hat_fit" in forecast_df.columns:
            ax.plot(
                fit_seg["date"],
                fit_seg["stock_ev_t_hat_fit"],
                label="Bass fit (baseline, train<2025)",
                color="C1",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if "stock_ev_t_hat_cov_anchor" in forecast_df.columns:
            ax.plot(
                future["date"],
                future["stock_ev_t_hat_cov_anchor"],
                label="Bass forecast (with tco_adv)" if holdout_ts is None else "Bass forecast (tco_adv, train<2025)",
                color="C2",
                linestyle=":",
            )
        if holdout_ts is not None and "stock_ev_t_hat_cov_fit" in forecast_df.columns:
            ax.plot(
                fit_seg["date"],
                fit_seg["stock_ev_t_hat_cov_fit"],
                label="Bass fit (tco_adv, train<2025)",
                color="C2",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if "stock_ev_t_hat_cov_policy_anchor" in forecast_df.columns:
            ax.plot(
                future["date"],
                future["stock_ev_t_hat_cov_policy_anchor"],
                label="Bass forecast (tco_adv + policy)" if holdout_ts is None else "Bass forecast (tco_adv + policy, train<2025)",
                color="C3",
                linestyle="-.",
            )
        if holdout_ts is not None and "stock_ev_t_hat_cov_policy_fit" in forecast_df.columns:
            ax.plot(
                fit_seg["date"],
                fit_seg["stock_ev_t_hat_cov_policy_fit"],
                label="Bass fit (tco_adv + policy, train<2025)",
                color="C3",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if holdout_ts is None:
            ax.set_title("LIPA EV Stock: Observed and Bass Forecasts")
        else:
            ax.axvline(holdout_ts, color="k", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_title("LIPA EV Stock: Train (<2025) and Holdout Forecast (>=2025-01)")
        ax.set_xlabel("Snapshot date")
        ax.set_ylabel("EV stock (unique VINs)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        fig.savefig(models_dir / _tagged("bass_lipa_stock_forecast.png"), dpi=150)
        plt.close(fig)

        # Flow plot
        fig, ax = plt.subplots(figsize=(10, 5))
        hist = forecast_df[~forecast_df["flow_ev_t_obs"].isna()]
        ax.plot(
            hist["date"],
            hist["flow_ev_t_obs"],
            label="Observed new EV registrations",
            color="C0",
        )
        ax.plot(
            forecast_df["date"],
            forecast_df["flow_ev_t_hat_anchor"],
            label="Bass flow (baseline)" if holdout_ts is None else "Bass flow (baseline, holdout forecast)",
            color="C1",
            linestyle="--",
        )
        if holdout_ts is not None and "flow_ev_t_hat_fit" in forecast_df.columns:
            ax.plot(
                forecast_df.loc[forecast_df["is_train"] == 1, "date"],
                forecast_df.loc[forecast_df["is_train"] == 1, "flow_ev_t_hat_fit"],
                label="Bass flow (baseline, fit)",
                color="C1",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if "flow_ev_t_hat_cov_anchor" in forecast_df.columns:
            ax.plot(
                forecast_df["date"],
                forecast_df["flow_ev_t_hat_cov_anchor"],
                label="Bass flow (with tco_adv)" if holdout_ts is None else "Bass flow (tco_adv, holdout forecast)",
                color="C2",
                linestyle=":",
            )
        if holdout_ts is not None and "flow_ev_t_hat_cov_fit" in forecast_df.columns:
            ax.plot(
                forecast_df.loc[forecast_df["is_train"] == 1, "date"],
                forecast_df.loc[forecast_df["is_train"] == 1, "flow_ev_t_hat_cov_fit"],
                label="Bass flow (tco_adv, fit)",
                color="C2",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if "flow_ev_t_hat_cov_policy_anchor" in forecast_df.columns:
            ax.plot(
                forecast_df["date"],
                forecast_df["flow_ev_t_hat_cov_policy_anchor"],
                label="Bass flow (tco_adv + policy)" if holdout_ts is None else "Bass flow (tco_adv + policy, holdout forecast)",
                color="C3",
                linestyle="-.",
            )
        if holdout_ts is not None and "flow_ev_t_hat_cov_policy_fit" in forecast_df.columns:
            ax.plot(
                forecast_df.loc[forecast_df["is_train"] == 1, "date"],
                forecast_df.loc[forecast_df["is_train"] == 1, "flow_ev_t_hat_cov_policy_fit"],
                label="Bass flow (tco_adv + policy, fit)",
                color="C3",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if holdout_ts is None:
            ax.set_title("LIPA New EV Registrations: Observed vs Bass Forecasts")
        else:
            ax.axvline(holdout_ts, color="k", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_title("LIPA New EV Registrations: Holdout Forecast (train<2025, test>=2025-01)")
        ax.set_xlabel("Snapshot date")
        ax.set_ylabel("New EVs per snapshot")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        fig.savefig(models_dir / _tagged("bass_lipa_flow_forecast.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("Plotting skipped:", e)

    print(f"Panel written to covariates/{_tagged('panel_LIPA.csv')}")
    print(f"Baseline Bass params written to models/{_tagged('bass_lipa_baseline.json')}")
    print(f"Covariate Bass params written to models/{_tagged('bass_lipa_with_tco.json')}")
    print(f"Forecast written to models/{_tagged('bass_lipa_forecast.csv')}")


if __name__ == "__main__":
    main()
