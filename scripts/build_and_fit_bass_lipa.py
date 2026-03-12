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
       DMV_ID, date,
       stock_ev_t (on-road stock),
       flow_ev_t (first-seen EVs; adoption inflow proxy),
       adopt_ev_cum_t (cumulative adopters / ever-arrived),
       gas_price_t, C_gas_t, C_ev_t, tco_adv_t, ...
  2) Fit a baseline Bass diffusion model (p, q, M) on the *adoption process*:
       a_t = flow_ev_t
       A_t = sum_{k<=t} a_k
       â_t = (p + q * A_{t-1} / M) * (M - A_{t-1})
     i.e., the Bass state is cumulative adopters A_{t-1}, not on-road stock.
  3) Fit a covariate Bass model where p_t depends on covariates (e.g. tco_adv_t,
     optionally subsidy_share_t).
  4) Convert predicted adoption flows to on-road stock via a retention curve:
       Ŝ_t = Σ_{k<=t} â_k · Pr(still on-road at t | adopted at k)
     where Pr(.) is estimated separately from last-seen VIN spans (see
     scripts/estimate_ev_retention_curve.py) and passed in via --retention-curve.
  5) Write parameter JSONs, a forecast CSV, and summary figures.

Usage
  python scripts/build_and_fit_bass_lipa.py \
    --horizon 24 \
    --elec-price 0.22
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


ROOT = Path(__file__).resolve().parents[1]


def month_index(ts: pd.Timestamp) -> int:
    """Map a timestamp to a monotonically increasing month index."""
    return int(ts.year) * 12 + int(ts.month)


def load_retention_curve(path: Path, region: str = "LIPA") -> Tuple[np.ndarray, int]:
    """Load a discrete monthly survival curve.

    Expects columns:
      - lag_months (int, >=0)
      - survival_prob (float in [0,1])
    Optionally includes:
      - region

    Returns:
      (survival_by_lag, max_lag)
    where survival_by_lag[m] = Pr(still on-road at lag m months).
    """
    df = pd.read_csv(path)
    if "region" in df.columns:
        df = df[df["region"].astype(str).str.upper() == region.upper()].copy()
    if df.empty:
        raise ValueError(f"Retention curve {path} has no rows for region={region}")
    if "lag_months" not in df.columns or "survival_prob" not in df.columns:
        raise ValueError(f"Retention curve {path} must have lag_months and survival_prob columns")
    df["lag_months"] = pd.to_numeric(df["lag_months"], errors="coerce").astype("Int64")
    df["survival_prob"] = pd.to_numeric(df["survival_prob"], errors="coerce")
    df = df.dropna(subset=["lag_months", "survival_prob"]).sort_values("lag_months")
    if df.empty:
        raise ValueError(f"Retention curve {path} has no usable numeric rows")

    lags = df["lag_months"].astype(int).to_numpy()
    probs = df["survival_prob"].astype(float).to_numpy()
    if lags[0] != 0:
        raise ValueError(f"Retention curve {path} must include lag_months=0 (survival=1)")
    max_lag = int(lags.max())
    survival = np.full(max_lag + 1, np.nan, dtype=float)
    for m, s in zip(lags, probs):
        if m < 0:
            continue
        survival[int(m)] = float(s)
    # Forward-fill any gaps in the curve.
    for m in range(1, len(survival)):
        if np.isnan(survival[m]):
            survival[m] = survival[m - 1]
    # Back-fill any leading NaNs (should not happen if lag 0 present).
    if np.isnan(survival[0]):
        survival[0] = 1.0
    survival[0] = 1.0
    return survival, max_lag


def survival_at_lag(survival_by_lag: np.ndarray, lag_months: int) -> float:
    if lag_months <= 0:
        return 1.0
    if lag_months >= len(survival_by_lag):
        return float(survival_by_lag[-1])
    return float(survival_by_lag[int(lag_months)])


def stock_from_adoptions(
    dates: np.ndarray, adoptions: np.ndarray, survival_by_lag: np.ndarray
) -> np.ndarray:
    """Compute on-road stock as adoption inflow convolved with retention."""
    if len(dates) != len(adoptions):
        raise ValueError("dates and adoptions length mismatch")
    month_ids = np.array([month_index(pd.Timestamp(d)) for d in dates], dtype=int)
    a = np.nan_to_num(adoptions.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(len(a), dtype=float)
    for t in range(len(a)):
        lags = month_ids[t] - month_ids[: t + 1]
        surv = np.array([survival_at_lag(survival_by_lag, int(l)) for l in lags], dtype=float)
        out[t] = float(np.sum(a[: t + 1] * surv))
    return out


def estimate_exponential_retention_from_aggregates(
    dates: np.ndarray,
    adoption_flows: np.ndarray,
    stock_obs: np.ndarray,
    *,
    eval_start: pd.Timestamp | None,
    max_lag_needed: int,
) -> tuple[np.ndarray, float]:
    """Estimate an exponential survival curve S(lag)=exp(-lambda*lag) from aggregates.

    This is a pragmatic fallback when VIN-level last-seen retention is not available.
    We choose lambda to best match observed on-road stock using observed adoption flows:
      stock_t ≈ Σ_{k<=t} a_k · exp(-lambda · lag_months(t,k))
    """
    mask = np.isfinite(stock_obs)
    if eval_start is not None:
        mask = mask & np.asarray(pd.to_datetime(dates) >= eval_start, dtype=bool)
    if int(mask.sum()) < 5:
        raise ValueError("Not enough observations to estimate retention from aggregates")

    dates_ts = pd.to_datetime(dates).to_numpy()
    a = np.nan_to_num(adoption_flows.astype(float), nan=0.0)
    s = stock_obs.astype(float)

    def build_curve(lmbd: float) -> np.ndarray:
        lmbd = float(lmbd)
        lags = np.arange(max_lag_needed + 1, dtype=float)
        curve = np.exp(-lmbd * lags)
        curve[0] = 1.0
        return curve

    def objective(theta: np.ndarray) -> float:
        lmbd = float(theta[0])
        if lmbd < 0.0 or lmbd > 1.0:
            return 1e12
        curve = build_curve(lmbd)
        pred = stock_from_adoptions(dates=dates_ts, adoptions=a, survival_by_lag=curve)
        err = s[mask] - pred[mask]
        return float(np.mean(err**2))

    # Reasonable starting point: ~2% monthly attrition => lambda ≈ -ln(0.98) ≈ 0.0202
    res = minimize(objective, x0=np.array([0.02]), bounds=[(0.0, 1.0)])
    lmbd_hat = float(res.x[0])
    return build_curve(lmbd_hat), lmbd_hat


@dataclass
class BassParams:
    p: float
    q: float
    M: float


@dataclass
class BassCovParams:
    alpha_p: float
    beta_p: List[float]
    alpha_q: float
    beta_q: List[float]
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


def _month_name_to_num(name: str) -> int | None:
    s = str(name or "").strip().lower()
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    return months.get(s)


def load_electricity_prices_from_xlsx(
    xlsx_path: Path, *, rate_class: str, utility: str
) -> pd.DataFrame:
    """Load a monthly electricity price series from the provided EIA-style workbook.

    The workbook `CH LIPA Electricity Residential Retail Price.xlsx` contains:
      - Utility annual average prices (cents/kWh) for LIPA and Central Hudson
      - A NY statewide monthly table (months x years) in cents/kWh

    We produce a monthly series in $/kWh. If utility is LIPA or CENTRAL_HUDSON,
    we scale the statewide monthly series by the ratio:
      utility_annual_avg(year) / statewide_annual_avg(year)
    to approximate utility-level monthly variation.
    """
    sheet = rate_class.strip().title()
    xl = pd.ExcelFile(xlsx_path)
    if sheet not in xl.sheet_names:
        raise ValueError(f"Sheet '{sheet}' not found in {xlsx_path}. Available: {xl.sheet_names}")
    df = xl.parse(sheet)
    if df.empty:
        raise ValueError(f"Empty sheet '{sheet}' in {xlsx_path}")

    first_col = df.columns[0]

    # 1) Utility annual averages (first rows with numeric years)
    annual = df[[first_col] + [c for c in df.columns if "Average Price" in str(c)]].copy()
    annual["year"] = pd.to_numeric(annual[first_col], errors="coerce")
    annual = annual.dropna(subset=["year"])
    annual["year"] = annual["year"].astype(int)
    annual = annual.set_index("year")

    # 2) Statewide monthly table: find the row that contains "Statewide Monthly Average Retail Price"
    header_idx = None
    for i, v in enumerate(df[first_col].astype(str).tolist()):
        if "statewide monthly average retail price" in v.strip().lower():
            header_idx = i
            break
    if header_idx is None or header_idx + 2 >= len(df):
        raise ValueError(f"Could not locate statewide monthly table in {xlsx_path} sheet '{sheet}'")

    years_row = df.iloc[header_idx + 1]
    col_year: Dict[str, int] = {}
    for col, val in years_row.items():
        if col == first_col:
            continue
        if isinstance(val, (int, float)) and pd.notna(val):
            col_year[col] = int(val)
    if not col_year:
        raise ValueError(f"Could not parse year headers for statewide monthly table in {xlsx_path}")

    # Monthly values start after the year header row. Only use the first contiguous
    # Jan..Dec block to avoid accidentally parsing other tables/footnotes below.
    month_rows: list[int] = []
    seen_months: set[int] = set()
    for i in range(header_idx + 2, len(df)):
        m = _month_name_to_num(df.at[i, first_col])
        if m is None:
            # Allow blank spacer rows while collecting the first Jan..Dec block.
            if month_rows and pd.isna(df.at[i, first_col]):
                continue
            # If we've already started collecting and hit a non-month label, keep scanning
            # until we complete the first full Jan..Dec set.
            continue
        month_rows.append(i)
        seen_months.add(m)
        if len(seen_months) >= 12:
            break
    if len(seen_months) < 12:
        raise ValueError(
            f"Could not find a full Jan..Dec monthly block after statewide header in {xlsx_path} sheet '{sheet}'"
        )
    block = df.loc[month_rows].copy()
    records: list[dict] = []
    for _, r in block.iterrows():
        m = _month_name_to_num(r.get(first_col))
        if m is None:
            continue
        for col, year in col_year.items():
            v = r.get(col)
            if pd.isna(v):
                continue
            records.append({"date": pd.Timestamp(year=year, month=m, day=1), "cents_per_kwh": float(v)})
    monthly = pd.DataFrame(records)
    if monthly.empty:
        raise ValueError(f"No monthly values parsed from {xlsx_path} sheet '{sheet}'")
    monthly = monthly.sort_values("date")

    # Optionally scale to utility using annual ratio (utility annual / statewide annual)
    util = utility.strip().upper()
    if util in ("LIPA", "CENTRAL_HUDSON"):
        util_col = None
        for c in annual.columns:
            lc = str(c).lower()
            if util == "LIPA" and "lipa" in lc:
                util_col = c
                break
            if util == "CENTRAL_HUDSON" and "central hudson" in lc:
                util_col = c
                break
        if util_col is None:
            raise ValueError(f"Utility '{utility}' not found in annual average table in {xlsx_path}")

        state_annual = monthly.groupby(monthly["date"].dt.year)["cents_per_kwh"].mean()
        util_annual = pd.to_numeric(annual[util_col], errors="coerce")
        ratio = (util_annual / state_annual).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            raise ValueError(f"Could not compute utility/state ratio for utility={utility}")

        monthly["year"] = monthly["date"].dt.year
        monthly["scale"] = monthly["year"].map(ratio).astype(float)
        # Fill missing years (if any) with last known ratio
        monthly["scale"] = monthly["scale"].ffill().bfill()
        monthly["cents_per_kwh"] = monthly["cents_per_kwh"] * monthly["scale"]
        monthly = monthly.drop(columns=["year", "scale"])

    monthly["elec_price_t"] = monthly["cents_per_kwh"] / 100.0
    return monthly[["date", "elec_price_t"]]


def attach_elec_price(
    panel: pd.DataFrame,
    *,
    elec_price_default: float,
    elec_series_path: str | None = None,
    elec_rate_class: str = "Residential",
    elec_utility: str = "LIPA",
) -> pd.DataFrame:
    """Attach electricity price ($/kWh) to the snapshot panel."""
    df: pd.DataFrame | None = None

    if elec_series_path:
        p = Path(elec_series_path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"--elec-series not found: {p}")
        if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            df = load_electricity_prices_from_xlsx(
                p, rate_class=elec_rate_class, utility=elec_utility
            )
        else:
            tmp = pd.read_csv(p)
            tmp["date"] = pd.to_datetime(tmp["date"])
            if "elec_price_t" not in tmp.columns:
                if "elec_price_cents_per_kwh" in tmp.columns:
                    tmp["elec_price_t"] = tmp["elec_price_cents_per_kwh"].astype(float) / 100.0
                else:
                    raise ValueError(
                        f"{p} must contain elec_price_t ($/kWh) or elec_price_cents_per_kwh"
                    )
            df = tmp[["date", "elec_price_t"]].sort_values("date")
    else:
        # Default to the locally provided workbook(s) if present.
        #
        # Prefer `updated_retail_price.xlsx` when available because it contains a clean monthly
        # retail price table used for current TCO covariates.
        preferred_xlsx = ROOT / "updated_retail_price.xlsx"
        legacy_xlsx = ROOT / "CH LIPA Electricity Residential Retail Price.xlsx"
        default_xlsx = preferred_xlsx if preferred_xlsx.exists() else legacy_xlsx
        if default_xlsx.exists():
            df = load_electricity_prices_from_xlsx(
                default_xlsx, rate_class=elec_rate_class, utility=elec_utility
            )

    panel = panel.sort_values("date").reset_index(drop=True)
    if df is not None and not df.empty:
        panel = pd.merge_asof(panel, df.sort_values("date"), on="date", direction="backward")

    # Fill missing values with the configured default so downstream covariates have no NaNs.
    if "elec_price_t" not in panel.columns:
        panel["elec_price_t"] = elec_price_default
    else:
        panel["elec_price_t"] = panel["elec_price_t"].astype(float).bfill().ffill()
        panel["elec_price_t"] = panel["elec_price_t"].fillna(float(elec_price_default))

    return panel


def normalize_zip5(z: str) -> str:
    s = str(z or "").strip()
    m = re.search(r"(\d{5})", s)
    return m.group(1) if m else ""


def parse_bool_like(v: object) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n", ""}:
        return False
    return False


def parse_station_snapshot_date(path: Path) -> pd.Timestamp:
    # Example filename: alt_fuel_stations_historical_day (Apr 1 2025).csv
    m = re.search(r"\(([^)]+)\)", path.name)
    if not m:
        raise ValueError(f"Cannot parse snapshot date from station file name: {path.name}")
    ts = pd.to_datetime(m.group(1).strip(), errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot parse snapshot date token '{m.group(1)}' from {path.name}")
    return pd.Timestamp(ts).to_period("M").to_timestamp(how="start")


def load_lipa_zip_set(zip_to_county_path: Path, zip_overrides_path: Path, *, region: str = "LIPA") -> set[str]:
    """Build LIPA ZIP set from Nassau/Suffolk + utility ZIP overrides."""
    lipa_zips: set[str] = set()

    with open(zip_to_county_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        lower = {c.lower(): c for c in fields}
        zcol = lower.get("zip")
        ccol = lower.get("county_name")
        if not zcol or not ccol:
            raise ValueError(f"{zip_to_county_path} must contain 'zip' and 'county_name'")
        for row in r:
            z = normalize_zip5(row.get(zcol, ""))
            county = str(row.get(ccol, "")).strip().upper()
            if z and county in {"NASSAU", "SUFFOLK"}:
                lipa_zips.add(z)

    if zip_overrides_path.exists():
        with open(zip_overrides_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            fields = r.fieldnames or []
            lower = {c.lower(): c for c in fields}
            zcol = lower.get("zip")
            rcol = lower.get("region")
            if zcol and rcol:
                target = region.strip().upper()
                for row in r:
                    if str(row.get(rcol, "")).strip().upper() != target:
                        continue
                    z = normalize_zip5(row.get(zcol, ""))
                    if z:
                        lipa_zips.add(z)

    if not lipa_zips:
        raise ValueError("LIPA ZIP set is empty; check ZIP crosswalk inputs")
    return lipa_zips


def build_evse_monthly_covariates(
    station_info_dir: Path,
    *,
    zip_to_county_path: Path,
    zip_overrides_path: Path,
    region: str = "LIPA",
    restricted_weight: float = 0.0,
    workplace_weight: float = 0.0,
) -> pd.DataFrame:
    """Build monthly EVSE capacity series for LIPA using AFDC snapshots.

    Baseline access filters:
      - Fuel Type Code == ELEC
      - State == NY
      - Status Code == E
      - Access Code == public
      - Restricted Access down-weighted by `restricted_weight` (default 0 = exclude)
      - EV Workplace Charging down-weighted by `workplace_weight` (default 0 = exclude)

    Returns monthly rows with:
      date,
      evse_l1_ports_t, evse_l2_ports_t, evse_dcfc_ports_t,
      evse_total_ports_t,
      x_evse_total_t (log1p total ports),
      evse_dcfc_share_t (DCFC / (L2 + DCFC))
    """
    if restricted_weight < 0 or workplace_weight < 0:
        raise ValueError("EVSE restricted/workplace weights must be >= 0")
    if not station_info_dir.exists():
        raise FileNotFoundError(f"EVSE station folder not found: {station_info_dir}")

    lipa_zips = load_lipa_zip_set(
        zip_to_county_path=zip_to_county_path,
        zip_overrides_path=zip_overrides_path,
        region=region,
    )
    files = sorted(station_info_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in EVSE station folder: {station_info_dir}")

    needed = [
        "Fuel Type Code",
        "State",
        "ZIP",
        "Status Code",
        "Access Code",
        "Restricted Access",
        "EV Workplace Charging",
        "EV Level1 EVSE Num",
        "EV Level2 EVSE Num",
        "EV DC Fast Count",
    ]
    rows: list[dict] = []

    for p in files:
        snap_date = parse_station_snapshot_date(p)
        df = pd.read_csv(p, usecols=lambda c: c in needed, low_memory=False)
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan

        zip5 = df["ZIP"].map(normalize_zip5)
        mask = (
            df["Fuel Type Code"].astype(str).str.upper().eq("ELEC")
            & df["State"].astype(str).str.upper().eq("NY")
            & df["Status Code"].astype(str).str.upper().eq("E")
            & df["Access Code"].astype(str).str.lower().eq("public")
            & zip5.isin(lipa_zips)
        )
        if not bool(mask.any()):
            rows.append(
                {
                    "date": snap_date,
                    "evse_l1_ports_t": 0.0,
                    "evse_l2_ports_t": 0.0,
                    "evse_dcfc_ports_t": 0.0,
                }
            )
            continue

        sub = df.loc[mask].copy()
        restricted = sub["Restricted Access"].map(parse_bool_like).astype(float)
        workplace = sub["EV Workplace Charging"].map(parse_bool_like).astype(float)

        # Exclude or down-weight restricted/workplace ports.
        weights = (1.0 - restricted) + restricted * float(restricted_weight)
        weights *= (1.0 - workplace) + workplace * float(workplace_weight)

        l1 = pd.to_numeric(sub["EV Level1 EVSE Num"], errors="coerce").fillna(0.0).clip(lower=0.0)
        l2 = pd.to_numeric(sub["EV Level2 EVSE Num"], errors="coerce").fillna(0.0).clip(lower=0.0)
        dc = pd.to_numeric(sub["EV DC Fast Count"], errors="coerce").fillna(0.0).clip(lower=0.0)

        rows.append(
            {
                "date": snap_date,
                "evse_l1_ports_t": float(np.sum(l1 * weights)),
                "evse_l2_ports_t": float(np.sum(l2 * weights)),
                "evse_dcfc_ports_t": float(np.sum(dc * weights)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.groupby("date", as_index=False)[["evse_l1_ports_t", "evse_l2_ports_t", "evse_dcfc_ports_t"]].sum()
    out = out.sort_values("date").reset_index(drop=True)
    out["evse_total_ports_t"] = out["evse_l2_ports_t"].astype(float) + out["evse_dcfc_ports_t"].astype(float)
    out["x_evse_total_t"] = np.log1p(out["evse_total_ports_t"].astype(float))
    out["evse_dcfc_share_t"] = np.where(
        out["evse_total_ports_t"].astype(float) > 0.0,
        out["evse_dcfc_ports_t"].astype(float) / out["evse_total_ports_t"].astype(float),
        0.0,
    )
    out["evse_dcfc_share_t"] = out["evse_dcfc_share_t"].clip(lower=0.0, upper=1.0)
    return out


def attach_evse_covariates(
    panel: pd.DataFrame,
    *,
    station_info_dir: str = "station_info",
    zip_to_county_path: str = "data/zip_to_county_ny.csv",
    zip_overrides_path: str = "data/utility_zip_regions.csv",
    region: str = "LIPA",
    restricted_weight: float = 0.0,
    workplace_weight: float = 0.0,
    lag_months: int = 3,
    output_csv: str | None = "covariates/evse_lipa_monthly.csv",
) -> pd.DataFrame:
    """Attach EVSE infrastructure covariates to panel by snapshot date (as-of merge)."""
    if lag_months < 0:
        raise ValueError("EVSE lag_months must be >= 0")
    station_dir = Path(station_info_dir)
    if not station_dir.is_absolute():
        station_dir = ROOT / station_dir
    county_path = Path(zip_to_county_path)
    if not county_path.is_absolute():
        county_path = ROOT / county_path
    override_path = Path(zip_overrides_path)
    if not override_path.is_absolute():
        override_path = ROOT / override_path

    evse = build_evse_monthly_covariates(
        station_dir,
        zip_to_county_path=county_path,
        zip_overrides_path=override_path,
        region=region,
        restricted_weight=restricted_weight,
        workplace_weight=workplace_weight,
    )
    # Keep only the re-parameterized series used in modeling (+ raw port counts for inspection).
    keep_cols = [
        "date",
        "evse_l1_ports_t",
        "evse_l2_ports_t",
        "evse_dcfc_ports_t",
        "evse_total_ports_t",
        "x_evse_total_t",
        "evse_dcfc_share_t",
    ]
    evse = evse[keep_cols].copy()

    if output_csv:
        outp = Path(output_csv)
        if not outp.is_absolute():
            outp = ROOT / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        evse.to_csv(outp, index=False)

    out = panel.sort_values("date").reset_index(drop=True).copy()
    out["evse_month"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp(how="start")
    evse_sorted = evse.sort_values("date").reset_index(drop=True)
    out = pd.merge_asof(
        out.sort_values("evse_month"),
        evse_sorted.rename(columns={"date": "evse_month"}),
        on="evse_month",
        direction="backward",
    ).sort_values("date").reset_index(drop=True)

    # Lagged EVSE series (calendar months), shifted forward so merge_asof matches t-lag.
    if lag_months > 0:
        lagged = evse_sorted.copy()
        lagged["evse_month"] = lagged["date"] + pd.DateOffset(months=int(lag_months))
        lagged = lagged.drop(columns=["date"])
        rename = {
            "x_evse_total_t": f"x_evse_total_lag{lag_months}_t",
            "evse_dcfc_share_t": f"evse_dcfc_share_lag{lag_months}_t",
        }
        lagged = lagged.rename(columns=rename)
        out = pd.merge_asof(
            out.sort_values("evse_month"),
            lagged.sort_values("evse_month")[["evse_month", *rename.values()]],
            on="evse_month",
            direction="backward",
        ).sort_values("date").reset_index(drop=True)

    for c in [
        "evse_l1_ports_t",
        "evse_l2_ports_t",
        "evse_dcfc_ports_t",
        "evse_total_ports_t",
        "x_evse_total_t",
        "evse_dcfc_share_t",
    ]:
        if c not in out.columns:
            out[c] = 0.0
        # Keep pre-series periods at 0 when EVSE snapshots are unavailable, then forward-fill.
        out[c] = pd.to_numeric(out[c], errors="coerce").ffill().fillna(0.0)

    if lag_months > 0:
        lag_x = f"x_evse_total_lag{lag_months}_t"
        lag_s = f"evse_dcfc_share_lag{lag_months}_t"
        for c in (lag_x, lag_s):
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(out[c], errors="coerce").ffill().fillna(0.0)
        out[lag_s] = out[lag_s].clip(lower=0.0, upper=1.0)

    # Store EVSE coverage start (month-start) for downstream fitting filters.
    out["evse_series_start"] = evse_sorted["date"].min()
    return out


def resample_snapshot_panel_to_monthly(panel_snap: pd.DataFrame) -> pd.DataFrame:
    """Convert irregular snapshot rows into a regular month-start (MS) panel.

    - Flows: disaggregate each snapshot's first-seen count across the months between the
      previous snapshot date and the current snapshot date, proportional to day overlap.
      Counts are allocated as integers and preserve the per-snapshot totals.
    - Stock: kept only on months that contain a snapshot (the last snapshot in that month).
      Intermediate months have NaN stock (not observed).
    """
    df = panel_snap.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    # Convert to a calendar month-start timestamp.
    # (Period -> timestamp supports how='start'; 'MS' is not a valid Period freq.)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")
    months = pd.date_range(start=df["month"].min(), end=df["month"].max(), freq="MS")
    out = pd.DataFrame({"date": months})
    out["flow_ev_t"] = 0

    # Stock: keep last snapshot in each month.
    stock_month = (
        df.sort_values("date")
        .groupby("month", as_index=False)
        .tail(1)[["month", "DMV_ID", "stock_ev_t", "stock_all_t"]]
        .rename(columns={"month": "date"})
    )
    out = out.merge(stock_month, on="date", how="left")

    # Flows: distribute each snapshot's total across the interval months.
    for i in range(len(df)):
        end = pd.Timestamp(df.at[i, "date"])
        total = int(pd.to_numeric(df.at[i, "flow_ev_t"], errors="coerce") or 0)
        if total <= 0:
            continue

        if i == 0:
            m_end = end.to_period("M").to_timestamp(how="start")
            out.loc[out["date"] == m_end, "flow_ev_t"] += total
            continue

        start = pd.Timestamp(df.at[i - 1, "date"])
        if end <= start:
            m_end = end.to_period("M").to_timestamp(how="start")
            out.loc[out["date"] == m_end, "flow_ev_t"] += total
            continue

        m_start = start.to_period("M").to_timestamp(how="start")
        m_end = end.to_period("M").to_timestamp(how="start")
        mlist = list(pd.date_range(start=m_start, end=m_end, freq="MS"))
        if not mlist:
            out.loc[out["date"] == m_end, "flow_ev_t"] += total
            continue

        # Compute day-overlap weights for each month in the (start, end] interval.
        overlaps = []
        for ms in mlist:
            me = ms + pd.DateOffset(months=1)
            overlap_start = max(start, ms)
            overlap_end = min(end, me)
            days = int((overlap_end - overlap_start).days)
            overlaps.append(max(days, 0))
        total_days = int(sum(overlaps))

        if total_days <= 0:
            out.loc[out["date"] == m_end, "flow_ev_t"] += total
            continue

        weights = np.array(overlaps, dtype=float) / float(total_days)
        expected = weights * float(total)
        floors = np.floor(expected).astype(int)
        remainder = int(total - int(floors.sum()))
        if remainder > 0:
            frac = expected - floors
            order = np.argsort(-frac)
            floors[order[:remainder]] += 1

        for ms, a in zip(mlist, floors):
            if a:
                out.loc[out["date"] == ms, "flow_ev_t"] += int(a)

    out["flow_ev_t"] = out["flow_ev_t"].astype(float)

    # The final calendar month is often incomplete because we only observe data through the
    # last snapshot date (e.g., 2025-09-08), not through the month end. Drop the final month
    # row entirely (do not roll partial-month flow into the previous month) to avoid a misleading
    # last-point artifact.
    last_snapshot_date = pd.Timestamp(df["date"].iloc[-1])
    last_month_start = last_snapshot_date.to_period("M").to_timestamp(how="start")
    last_month_end = (last_month_start + pd.offsets.MonthEnd(0)).normalize()
    if last_snapshot_date.normalize() < last_month_end:
        out = out[out["date"] != last_month_start].reset_index(drop=True)
    return out


def build_panel(
    elec_price_default: float,
    gas_series_path: str | None = None,
    elec_series_path: str | None = None,
    elec_rate_class: str = "Residential",
    elec_utility: str = "LIPA",
    resample_monthly: bool = False,
    with_evse: bool = False,
    station_info_dir: str = "station_info",
    evse_zip_source: str = "data/zip_to_county_ny.csv",
    evse_zip_overrides: str = "data/utility_zip_regions.csv",
    evse_restricted_weight: float = 0.0,
    evse_workplace_weight: float = 0.0,
    evse_lag_months: int = 3,
    evse_output_csv: str | None = "covariates/evse_lipa_monthly.csv",
) -> pd.DataFrame:
    stocks = load_lipa_stocks()
    flows = load_lipa_flows()

    panel = stocks.merge(flows, on="DMV_ID", how="left")
    panel["flow_ev_t"] = panel["flow_ev_t"].fillna(0.0)

    if resample_monthly:
        panel = resample_snapshot_panel_to_monthly(panel)

    panel = attach_gas_price(panel, gas_series_path=gas_series_path)
    panel = attach_elec_price(
        panel,
        elec_price_default=elec_price_default,
        elec_series_path=elec_series_path,
        elec_rate_class=elec_rate_class,
        elec_utility=elec_utility,
    )
    if with_evse:
        panel = attach_evse_covariates(
            panel,
            station_info_dir=station_info_dir,
            zip_to_county_path=evse_zip_source,
            zip_overrides_path=evse_zip_overrides,
            region="LIPA",
            restricted_weight=evse_restricted_weight,
            workplace_weight=evse_workplace_weight,
            lag_months=evse_lag_months,
            output_csv=evse_output_csv,
        )

    # Construct operating cost advantage
    MPG_AVG = 28.0
    KWH_PER_MI_AVG = 0.30
    panel["C_gas_t"] = panel["gas_price_t"] / MPG_AVG
    panel["C_ev_t"] = panel["elec_price_t"] * KWH_PER_MI_AVG
    panel["tco_adv_t"] = panel["C_gas_t"] - panel["C_ev_t"]

    # Keep periods after adoption starts (works for both snapshot and resampled-monthly panels).
    panel = panel.sort_values("date").reset_index(drop=True)

    # Adoption process (cumulative first-seen EVs). This is the Bass "state" we fit on.
    # NOTE: This is intentionally *not* the same as on-road stock (stock_ev_t) because
    # vehicles can exit the on-road inventory over time.
    panel["adopt_ev_t"] = panel["flow_ev_t"].astype(float)
    panel["adopt_ev_cum_t"] = panel["adopt_ev_t"].cumsum()
    panel["adopt_ev_cum_prev"] = panel["adopt_ev_cum_t"].shift(1).fillna(0.0)
    panel = panel[panel["adopt_ev_cum_t"] > 0].reset_index(drop=True)
    return panel


def prepare_panel_with_covariates(
    panel: pd.DataFrame,
    *,
    elec_price_default: float,
    gas_series_path: str | None = None,
    elec_series_path: str | None = None,
    elec_rate_class: str = "Residential",
    elec_utility: str = "LIPA",
    resample_monthly: bool = False,
    with_evse: bool = False,
    station_info_dir: str = "station_info",
    evse_zip_source: str = "data/zip_to_county_ny.csv",
    evse_zip_overrides: str = "data/utility_zip_regions.csv",
    evse_restricted_weight: float = 0.0,
    evse_workplace_weight: float = 0.0,
    evse_lag_months: int = 3,
    evse_output_csv: str | None = "covariates/evse_lipa_monthly.csv",
) -> pd.DataFrame:
    """Prepare an externally supplied panel with covariates + adoption state."""
    out = panel.copy()
    if "date" not in out.columns:
        raise ValueError("--panel-csv must include a 'date' column")
    out["date"] = pd.to_datetime(out["date"])
    if "flow_ev_t" not in out.columns:
        raise ValueError("--panel-csv must include a 'flow_ev_t' column")
    if "stock_ev_t" not in out.columns:
        raise ValueError("--panel-csv must include a 'stock_ev_t' column")
    if "DMV_ID" not in out.columns:
        out["DMV_ID"] = np.nan
    if "stock_all_t" not in out.columns:
        out["stock_all_t"] = np.nan
    out["flow_ev_t"] = pd.to_numeric(out["flow_ev_t"], errors="coerce").fillna(0.0)
    out["stock_ev_t"] = pd.to_numeric(out["stock_ev_t"], errors="coerce")
    out["stock_all_t"] = pd.to_numeric(out["stock_all_t"], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)

    if resample_monthly:
        out = resample_snapshot_panel_to_monthly(out)

    # If cached covariates already exist and no override series is provided, keep them.
    if "gas_price_t" in out.columns and gas_series_path is None:
        pass
    else:
        if "gas_price_t" in out.columns:
            out = out.drop(columns=["gas_price_t"])
        out = attach_gas_price(out, gas_series_path=gas_series_path)

    if "elec_price_t" in out.columns and elec_series_path is None:
        pass
    else:
        if "elec_price_t" in out.columns:
            out = out.drop(columns=["elec_price_t"])
        out = attach_elec_price(
            out,
            elec_price_default=elec_price_default,
            elec_series_path=elec_series_path,
            elec_rate_class=elec_rate_class,
            elec_utility=elec_utility,
        )
    if with_evse:
        out = attach_evse_covariates(
            out,
            station_info_dir=station_info_dir,
            zip_to_county_path=evse_zip_source,
            zip_overrides_path=evse_zip_overrides,
            region="LIPA",
            restricted_weight=evse_restricted_weight,
            workplace_weight=evse_workplace_weight,
            lag_months=evse_lag_months,
            output_csv=evse_output_csv,
        )

    MPG_AVG = 28.0
    KWH_PER_MI_AVG = 0.30
    out["C_gas_t"] = out["gas_price_t"] / MPG_AVG
    out["C_ev_t"] = out["elec_price_t"] * KWH_PER_MI_AVG
    out["tco_adv_t"] = out["C_gas_t"] - out["C_ev_t"]

    out["adopt_ev_t"] = out["flow_ev_t"].astype(float)
    out["adopt_ev_cum_t"] = out["adopt_ev_t"].cumsum()
    out["adopt_ev_cum_prev"] = out["adopt_ev_cum_t"].shift(1).fillna(0.0)
    out = out[out["adopt_ev_cum_t"] > 0].reset_index(drop=True)
    return out


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


def poisson_nll(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson negative log-likelihood (includes log(y!) constant).

    This is preferred over MSE for count data because variance scales with the mean.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    m = np.isfinite(y) & np.isfinite(mu)
    if not m.any():
        return float("nan")
    y = np.maximum(y[m], 0.0)
    mu = np.maximum(mu[m], 1e-9)
    nll = mu - y * np.log(mu) + gammaln(y + 1.0)
    return float(np.mean(nll))


def mse_loss(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if not m.any():
        return float("nan")
    return float(np.mean((y[m] - yhat[m]) ** 2))


def fit_bass_baseline(
    panel: pd.DataFrame,
    *,
    flow_likelihood: str = "poisson",
    fit_mode: str = "one_step",
    M_fixed: float | None = None,
) -> BassParams:
    df = panel.copy().sort_values("date").reset_index(drop=True)
    # Fit Bass on cumulative adopters (A_{t-1}), not on-road stock.
    # Observed adoption inflow proxy is first-seen EVs (flow_ev_t == adopt_ev_t).
    A_prev = df["adopt_ev_cum_prev"].values.astype(float)
    A = df["adopt_ev_cum_t"].values.astype(float)
    n = df["adopt_ev_t"].values.astype(float)

    p0 = 1e-4
    q0 = 0.03

    if M_fixed is not None:
        M_fixed = float(M_fixed)
        if M_fixed <= A.max():
            raise ValueError(f"M_fixed must exceed observed A.max(); got {M_fixed} <= {A.max()}")

        def objective(theta: np.ndarray) -> float:
            p, q = theta
            if p <= 0 or q <= 0:
                return 1e12
            if fit_mode == "one_step":
                n_hat = bass_flow(A_prev, p, q, M_fixed)
            elif fit_mode == "simulate":
                A_sim = float(A_prev[0])
                n_hat = np.zeros_like(n, dtype=float)
                for t in range(len(n_hat)):
                    if A_sim >= M_fixed:
                        return 1e12
                    n_hat[t] = float((p + q * (A_sim / M_fixed)) * (M_fixed - A_sim))
                    A_sim += float(n_hat[t])
            else:
                raise ValueError(f"Unknown fit_mode: {fit_mode} (expected 'one_step' or 'simulate')")
            if flow_likelihood == "mse":
                return mse_loss(n, n_hat)
            if flow_likelihood == "poisson":
                return poisson_nll(n, n_hat)
            raise ValueError(f"Unknown --flow-likelihood: {flow_likelihood}")

        bounds = [(1e-8, 0.1), (1e-4, 2.0)]
        res = minimize(objective, x0=np.array([p0, q0]), bounds=bounds)
        p_hat, q_hat = res.x
        return BassParams(p=float(p_hat), q=float(q_hat), M=float(M_fixed))

    M0 = 1.5 * A.max()

    def objective(theta: np.ndarray) -> float:
        p, q, M = theta
        if M <= A.max() or p <= 0 or q <= 0:
            return 1e12
        if fit_mode == "one_step":
            n_hat = bass_flow(A_prev, p, q, M)
        elif fit_mode == "simulate":
            A_sim = float(A_prev[0])
            n_hat = np.zeros_like(n, dtype=float)
            for t in range(len(n_hat)):
                if A_sim >= M:
                    return 1e12
                n_hat[t] = float((p + q * (A_sim / M)) * (M - A_sim))
                A_sim += float(n_hat[t])
        else:
            raise ValueError(f"Unknown fit_mode: {fit_mode} (expected 'one_step' or 'simulate')")
        if flow_likelihood == "mse":
            return mse_loss(n, n_hat)
        if flow_likelihood == "poisson":
            return poisson_nll(n, n_hat)
        raise ValueError(f"Unknown --flow-likelihood: {flow_likelihood}")

    bounds = [(1e-8, 0.1), (1e-4, 2.0), (A.max() * 1.01, A.max() * 10)]
    res = minimize(objective, x0=np.array([p0, q0, M0]), bounds=bounds)
    p_hat, q_hat, M_hat = res.x
    return BassParams(p=p_hat, q=q_hat, M=M_hat)


def fit_bass_with_tco(
    panel: pd.DataFrame,
    feature_cols: List[str],
    *,
    flow_likelihood: str = "poisson",
    fit_mode: str = "one_step",
    ridge_lambda: float = 0.0,
    M_fixed: float | None = None,
) -> BassCovParams:
    df = panel.copy().sort_values("date").reset_index(drop=True)
    A_prev = df["adopt_ev_cum_prev"].values.astype(float)
    A = df["adopt_ev_cum_t"].values.astype(float)
    n = df["adopt_ev_t"].values.astype(float)
    X = df[feature_cols].values.astype(float)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    Xs = (X - X_mean) / X_std
    k = Xs.shape[1]

    # Reasonable starting values for monthly diffusion in a large market:
    # p is typically small (1e-6..1e-3); q is often ~0.01..0.5 depending on time step.
    alpha_p0 = float(np.log(1e-4))
    beta_p0 = np.zeros(k)
    alpha_q0 = float(np.log(0.03))
    beta_q0 = np.zeros(k)
    M0 = 1.5 * A.max()

    def objective(theta: np.ndarray) -> float:
        alpha_p = float(theta[0])
        beta_p = theta[1 : 1 + k]
        alpha_q = float(theta[1 + k])
        beta_q = theta[2 + k : 2 + 2 * k]
        if M_fixed is None:
            M = float(theta[-1])
        else:
            M = float(M_fixed)
        if M <= A.max():
            return 1e12

        eta_p = alpha_p + Xs @ beta_p
        eta_q = alpha_q + Xs @ beta_q
        # Prevent overflow in exp() for extreme combinations (still penalized by bounds/regularization).
        eta_p = np.clip(eta_p, -50.0, 50.0)
        eta_q = np.clip(eta_q, -50.0, 50.0)
        p_t = np.exp(eta_p)
        q_t = np.exp(eta_q)

        if fit_mode == "one_step":
            n_hat = (p_t + q_t * (A_prev / M)) * (M - A_prev)
        elif fit_mode == "simulate":
            A_sim = float(A_prev[0])
            n_hat = np.zeros_like(n, dtype=float)
            for t in range(len(n_hat)):
                if A_sim >= M:
                    return 1e12
                n_hat[t] = float((p_t[t] + q_t[t] * (A_sim / M)) * (M - A_sim))
                A_sim += float(n_hat[t])
        else:
            raise ValueError(f"Unknown fit_mode: {fit_mode} (expected 'one_step' or 'simulate')")

        if flow_likelihood == "mse":
            loss = mse_loss(n, n_hat)
        elif flow_likelihood == "poisson":
            loss = poisson_nll(n, n_hat)
        else:
            raise ValueError(f"Unknown --flow-likelihood: {flow_likelihood}")

        ridge_lambda_f = float(ridge_lambda or 0.0)
        if ridge_lambda_f > 0.0:
            # Scale penalty to the likelihood so ridge_lambda remains roughly interpretable.
            # For MSE, counts are O(1e3) so the raw loss is O(1e6); scale by mean(y^2).
            scale = 1.0
            if flow_likelihood == "mse":
                scale = float(np.mean(np.square(n))) if np.isfinite(n).any() else 1.0
                if not np.isfinite(scale) or scale <= 0.0:
                    scale = 1.0
            loss = float(loss + ridge_lambda_f * scale * (np.sum(beta_p**2) + np.sum(beta_q**2)))
        return float(loss)

    if M_fixed is not None:
        M_fixed = float(M_fixed)
        if M_fixed <= A.max():
            raise ValueError(f"M_fixed must exceed observed A.max(); got {M_fixed} <= {A.max()}")
        x0 = np.concatenate(([alpha_p0], beta_p0, [alpha_q0], beta_q0))
        bounds = [(-10, 0)] + [(-5, 5)] * k + [(-10, 1)] + [(-5, 5)] * k
        res = minimize(objective, x0=x0, bounds=bounds)
        theta = res.x
        alpha_p = float(theta[0])
        beta_p = np.array(theta[1 : 1 + k], dtype=float)
        alpha_q = float(theta[1 + k])
        beta_q = np.array(theta[2 + k : 2 + 2 * k], dtype=float)
        M_hat = float(M_fixed)
    else:
        x0 = np.concatenate(([alpha_p0], beta_p0, [alpha_q0], beta_q0, [M0]))
        bounds = [(-10, 0)] + [(-5, 5)] * k + [(-10, 1)] + [(-5, 5)] * k + [
            (A.max() * 1.01, A.max() * 10)
        ]
        res = minimize(objective, x0=x0, bounds=bounds)
        theta = res.x
        alpha_p = float(theta[0])
        beta_p = np.array(theta[1 : 1 + k], dtype=float)
        alpha_q = float(theta[1 + k])
        beta_q = np.array(theta[2 + k : 2 + 2 * k], dtype=float)
        M_hat = float(theta[-1])

    return BassCovParams(
        alpha_p=float(alpha_p),
        beta_p=beta_p.tolist(),
        alpha_q=float(alpha_q),
        beta_q=beta_q.tolist(),
        M=float(M_hat),
        feature_cols=feature_cols,
        X_mean=X_mean.tolist(),
        X_std=X_std.tolist(),
    )


def _compute_pq_t_from_covariates(df: pd.DataFrame, cov_params: BassCovParams) -> tuple[np.ndarray, np.ndarray]:
    X = df[cov_params.feature_cols].values.astype(float)
    Xs = (X - np.array(cov_params.X_mean)) / np.array(cov_params.X_std)
    beta_p = np.array(cov_params.beta_p, dtype=float)
    beta_q = np.array(cov_params.beta_q, dtype=float)
    alpha_p = float(cov_params.alpha_p)
    alpha_q = float(cov_params.alpha_q)
    p_t = np.exp(alpha_p + Xs @ beta_p)
    q_t = np.exp(alpha_q + Xs @ beta_q)
    return p_t, q_t


def forecast_fullsample_with_retention(
    panel_full: pd.DataFrame,
    *,
    p_const: float | None = None,
    q_const: float | None = None,
    cov_params: BassCovParams | None = None,
    M: float,
    horizon: int,
    survival_by_lag: np.ndarray,
) -> pd.DataFrame:
    """Full-sample one-step fit + forward simulation, then stock via retention convolution."""
    df = extend_panel_with_future(panel_full, horizon=horizon).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty panel provided to forecast_fullsample_with_retention")

    if cov_params is not None:
        p_t, q_t = _compute_pq_t_from_covariates(df, cov_params=cov_params)
    elif p_const is not None and q_const is not None:
        p_t = np.full(len(df), float(p_const), dtype=float)
        q_t = np.full(len(df), float(q_const), dtype=float)
    else:
        raise ValueError("Provide either (p_const, q_const) or cov_params")

    dates = pd.to_datetime(df["date"]).to_numpy()
    dmv_ids = df["DMV_ID"].to_list()

    flow_obs = pd.to_numeric(df["flow_ev_t"], errors="coerce").to_numpy(float)
    stock_obs = pd.to_numeric(df["stock_ev_t"], errors="coerce").to_numpy(float)

    # Observed cumulative adopters (treat missing future flows as 0 for state initialization).
    flow_obs_filled = np.nan_to_num(flow_obs, nan=0.0)
    A_obs = np.cumsum(flow_obs_filled)
    A_prev_obs = np.concatenate(([0.0], A_obs[:-1]))

    # One-step fitted flows on the observed history (uses observed A_{t-1}).
    flow_fit = (p_t + q_t * (A_prev_obs / M)) * (M - A_prev_obs)

    # Forward simulate adoption from the last observed snapshot.
    last_obs_idx = int(np.where(np.isfinite(stock_obs))[0][-1])
    A_sim = float(A_obs[last_obs_idx])
    flow_forecast = np.full(len(df), np.nan, dtype=float)
    for t in range(last_obs_idx + 1, len(df)):
        p_now = float(p_t[t])
        q_now = float(q_t[t])
        a_t = float((p_now + q_now * (A_sim / M)) * (M - A_sim))
        A_sim += a_t
        flow_forecast[t] = a_t

    # Anchor series combines in-sample one-step fit (history) + simulated forecast (future).
    flow_hat_anchor = flow_fit.copy()
    flow_hat_anchor[last_obs_idx + 1 :] = flow_forecast[last_obs_idx + 1 :]

    # Stock implied by model adoption + retention.
    stock_hat_anchor = stock_from_adoptions(dates=dates, adoptions=flow_hat_anchor, survival_by_lag=survival_by_lag)

    # Return observed + predicted.
    return pd.DataFrame(
        {
            "date": df["date"].tolist(),
            "DMV_ID": dmv_ids,
            "stock_ev_t_obs": stock_obs,
            "flow_ev_t_obs": flow_obs,
            "adopt_ev_cum_t_obs": A_obs,
            "stock_ev_t_hat_anchor": stock_hat_anchor,
            "flow_ev_t_hat_anchor": flow_hat_anchor,
        }
    )


def forecast_holdout_with_retention(
    panel_full: pd.DataFrame,
    *,
    p_const: float | None = None,
    q_const: float | None = None,
    cov_params: BassCovParams | None = None,
    M: float,
    holdout_start: pd.Timestamp,
    horizon: int,
    survival_by_lag: np.ndarray,
) -> pd.DataFrame:
    """Train/Test split: one-step fit on train + forward simulation on holdout, stock via retention."""
    df = extend_panel_with_future(panel_full, horizon=horizon).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty panel provided to forecast_holdout_with_retention")

    if cov_params is not None:
        p_t, q_t = _compute_pq_t_from_covariates(df, cov_params=cov_params)
    elif p_const is not None and q_const is not None:
        p_t = np.full(len(df), float(p_const), dtype=float)
        q_t = np.full(len(df), float(q_const), dtype=float)
    else:
        raise ValueError("Provide either (p_const, q_const) or cov_params")

    dates = pd.to_datetime(df["date"])
    is_train = dates < holdout_start
    if int(is_train.sum()) < 2:
        raise ValueError("Train set too small for holdout split")
    if int((~is_train & df["stock_ev_t"].notna()).sum()) < 1:
        raise ValueError("Test set empty for holdout split")

    dates_np = dates.to_numpy()
    dmv_ids = df["DMV_ID"].to_list()

    flow_obs = pd.to_numeric(df["flow_ev_t"], errors="coerce").to_numpy(float)
    stock_obs = pd.to_numeric(df["stock_ev_t"], errors="coerce").to_numpy(float)
    flow_obs_filled = np.nan_to_num(flow_obs, nan=0.0)
    A_obs = np.cumsum(flow_obs_filled)
    A_prev_obs = np.concatenate(([0.0], A_obs[:-1]))

    test_start_idx = int(np.where(~is_train.to_numpy(bool))[0][0])
    train_end_idx = test_start_idx - 1

    # One-step fitted flows on train (uses observed A_{t-1} and observed p_t).
    flow_fit = np.full(len(df), np.nan, dtype=float)
    for t in range(0, test_start_idx):
        p_now = float(p_t[t])
        q_now = float(q_t[t])
        a_t = float((p_now + q_now * (A_prev_obs[t] / M)) * (M - A_prev_obs[t]))
        flow_fit[t] = a_t

    # Holdout + future forecast via simulation (state is predicted A).
    flow_anchor = np.full(len(df), np.nan, dtype=float)
    A_sim = float(A_obs[train_end_idx])
    for t in range(test_start_idx, len(df)):
        p_now = float(p_t[t])
        q_now = float(q_t[t])
        a_t = float((p_now + q_now * (A_sim / M)) * (M - A_sim))
        A_sim += a_t
        flow_anchor[t] = a_t

    # Stock fit: use model flows on train + model forecast thereafter; keep observed pre-train as-is (if any).
    adoptions_for_stock_fit = flow_obs_filled.copy()
    train_mask = np.arange(len(df)) < test_start_idx
    adoptions_for_stock_fit[train_mask] = np.nan_to_num(flow_fit[train_mask], nan=0.0)
    adoptions_for_stock_fit[test_start_idx:] = np.nan_to_num(flow_anchor[test_start_idx:], nan=0.0)
    stock_fit_all = stock_from_adoptions(dates=dates_np, adoptions=adoptions_for_stock_fit, survival_by_lag=survival_by_lag)

    # Stock anchor: use observed adoptions up to train end, then model forecast.
    adoptions_for_stock_anchor = flow_obs_filled.copy()
    adoptions_for_stock_anchor[test_start_idx:] = np.nan_to_num(flow_anchor[test_start_idx:], nan=0.0)
    stock_anchor_all = stock_from_adoptions(dates=dates_np, adoptions=adoptions_for_stock_anchor, survival_by_lag=survival_by_lag)

    stock_fit = np.full(len(df), np.nan, dtype=float)
    stock_fit[train_mask] = stock_fit_all[train_mask]

    stock_anchor = np.full(len(df), np.nan, dtype=float)
    stock_anchor[train_end_idx] = stock_anchor_all[train_end_idx]
    stock_anchor[test_start_idx:] = stock_anchor_all[test_start_idx:]

    return pd.DataFrame(
        {
            "date": df["date"].tolist(),
            "DMV_ID": dmv_ids,
            "is_train": is_train.astype(int).to_list(),
            "stock_ev_t_obs": stock_obs,
            "flow_ev_t_obs": flow_obs,
            "adopt_ev_cum_t_obs": A_obs,
            "stock_ev_t_hat_fit": stock_fit,
            "flow_ev_t_hat_fit": flow_fit,
            "stock_ev_t_hat_anchor": stock_anchor,
            "flow_ev_t_hat_anchor": flow_anchor,
        }
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

    # Prepare (p_t, q_t) from covariates
    p_hist, q_hist = _compute_pq_t_from_covariates(df, cov_params=cov_params)  # length T

    # Extend p_t into the future by holding constant the last value
    p_future = np.repeat(p_hist[-1], horizon)
    q_future = np.repeat(q_hist[-1], horizon)
    p_all = np.concatenate([p_hist, p_future])
    q_all = np.concatenate([q_hist, q_future])

    # Date sequence: historical + horizon months forward
    future_dates = [dates[-1] + pd.DateOffset(months=h) for h in range(1, horizon + 1)]
    all_dates = dates + future_dates

    # DMV_ID: keep historical, NaN for forecast
    dmv_ids = list(df["DMV_ID"]) + [np.nan] * horizon

    # Iterative stock/flow forecast
    M = cov_params.M
    y_hat = np.zeros(T + horizon, dtype=float)
    n_hat = np.zeros(T + horizon, dtype=float)
    y_hat[0] = y[0]
    n_hat[0] = np.nan
    for t in range(1, T + horizon):
        y_prev = y_hat[t - 1]
        p_t = p_all[t]
        q_t = q_all[t]
        n_t = float(bass_flow(np.array([y_prev]), p_t, q_t, M)[0])
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

    p_hist, q_hist = _compute_pq_t_from_covariates(df, cov_params=cov_params)

    M = cov_params.M

    y_prev_obs = np.concatenate(([0.0], y_obs[:-1]))
    n_hat_hist = (p_hist + q_hist * (y_prev_obs / M)) * (M - y_prev_obs)

    # Forecast horizon: simulate from last observed stock, hold p_t constant at last observed value.
    p_last = float(p_hist[-1]) if T else float(np.exp(float(cov_params.alpha_p)))
    q_last = float(q_hist[-1]) if T else float(np.exp(float(cov_params.alpha_q)))
    y_prev = float(y_obs[-1]) if T else 0.0
    y_future = np.zeros(horizon, dtype=float)
    n_future = np.zeros(horizon, dtype=float)
    for h in range(horizon):
        n_t = float(bass_flow(np.array([y_prev]), p_last, q_last, M)[0])
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
    if "adopt_ev_t" in df.columns:
        future["adopt_ev_t"] = np.nan
    if "adopt_ev_cum_t" in df.columns:
        future["adopt_ev_cum_t"] = np.nan
    if "adopt_ev_cum_prev" in df.columns:
        future["adopt_ev_cum_prev"] = np.nan

    for c in df.columns:
        if c in (
            "date",
            "DMV_ID",
            "stock_ev_t",
            "flow_ev_t",
            "adopt_ev_t",
            "adopt_ev_cum_t",
            "adopt_ev_cum_prev",
        ):
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

    # (p_t, q_t) series (uses training standardization stored in cov_params)
    p_t, q_t = _compute_pq_t_from_covariates(df, cov_params=cov_params)

    M = cov_params.M

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
        q_now = float(q_t[t])
        n_t = float(bass_flow(np.array([y_prev_obs]), p_now, q_now, M)[0])
        n_fit[t] = n_t
        y_fit[t] = y_prev_obs + n_t

    y_prev = float(y_obs[train_end_idx])
    y_hat = np.full(len(df), np.nan, dtype=float)
    n_hat = np.full(len(df), np.nan, dtype=float)
    y_hat[train_end_idx] = y_prev

    for t in range(test_start_idx, len(df)):
        n_t = float(bass_flow(np.array([y_prev]), float(p_t[t]), float(q_t[t]), M)[0])
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
    ap.add_argument(
        "--flow-likelihood",
        type=str,
        default="poisson",
        choices=["poisson", "mse"],
        help="Objective for fitting flow counts (recommended: poisson).",
    )
    ap.add_argument(
        "--cov-fit-mode",
        type=str,
        default="one_step",
        choices=["one_step", "simulate"],
        help=(
            "How to fit Bass models (baseline and covariate). "
            "'one_step' matches flows using the observed adoption state A_{t-1}; "
            "'simulate' fits by forward-simulating A_t, which can reduce exaggerated multi-step forecasts."
        ),
    )
    ap.add_argument(
        "--cov-ridge-lambda",
        type=float,
        default=0.0,
        help=(
            "L2 regularization strength for covariate coefficients (beta_p and beta_q). "
            "For MSE objectives the penalty is scaled by mean(flow^2) so values like 0.01–0.2 are reasonable."
        ),
    )
    ap.add_argument(
        "--evse-fit-mode",
        type=str,
        default=None,
        choices=["one_step", "simulate"],
        help=(
            "Optional override for --cov-fit-mode applied to the EVSE pipeline only. "
            "Useful when EVSE models exhibit exaggerated multi-step holdout forecasts."
        ),
    )
    ap.add_argument(
        "--evse-ridge-lambda",
        type=float,
        default=None,
        help="Optional override for --cov-ridge-lambda applied to the EVSE pipeline only.",
    )
    ap.add_argument(
        "--panel-csv",
        type=str,
        default=None,
        help=(
            "Optional prebuilt panel CSV. If set, the script loads this panel instead of reading "
            "out_ev_snapshots/out_new_reg. Required columns: date, stock_ev_t, flow_ev_t "
            "(optional: DMV_ID, stock_all_t)."
        ),
    )
    ap.add_argument(
        "--resample-monthly",
        action="store_true",
        help=(
            "Convert irregular/annual DMV snapshots into a regular month-start (MS) panel by distributing each "
            "snapshot's first-seen EV counts across calendar months between snapshots (day-overlap weighted). "
            "This is recommended to smooth pre-2018 annual/irregular snapshots for monthly-time-step fitting."
        ),
    )
    ap.add_argument(
        "--elec-series",
        type=str,
        default=None,
        help=(
            "Optional path to an electricity price series (CSV or XLSX). "
            "CSV must have columns date + elec_price_t ($/kWh) or elec_price_cents_per_kwh. "
            "If omitted, the script will prefer 'updated_retail_price.xlsx' if present, otherwise "
            "fall back to 'CH LIPA Electricity Residential Retail Price.xlsx' (if present)."
        ),
    )
    ap.add_argument(
        "--elec-rate-class",
        type=str,
        default="Residential",
        choices=["Residential", "Commercial", "Industrial"],
        help="Which retail electricity price class to use (for XLSX sources).",
    )
    ap.add_argument(
        "--elec-utility",
        type=str,
        default="LIPA",
        choices=["LIPA", "CENTRAL_HUDSON", "STATEWIDE"],
        help=(
            "For the provided workbook, optionally scale NY statewide monthly prices to a utility using annual ratios. "
            "Use STATEWIDE to skip scaling."
        ),
    )
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
        "--retention-curve",
        type=str,
        default=str(Path("covariates") / "retention_LIPA_ev_km.csv"),
        help=(
            "Path to a retention curve CSV (lag_months, survival_prob) estimated from last-seen VINs. "
            "If missing, the script falls back to survival=1 (no attrition)."
        ),
    )
    ap.add_argument(
        "--with-policy",
        action="store_true",
        help="Also fit a covariate Bass model using policy effective subsidy (subsidy_share_t).",
    )
    ap.add_argument(
        "--with-evse",
        action="store_true",
        help=(
            "Also fit a covariate Bass model that includes EVSE infrastructure covariates: "
            "x_evse_total_lagK_t=log1p(public-open (L2+DCFC) ports), "
            "evse_dcfc_share_lagK_t=DCFC/(L2+DCFC) with a K-month lag."
        ),
    )
    ap.add_argument(
        "--station-info-dir",
        type=str,
        default="station_info",
        help="Directory with AFDC monthly snapshot CSVs (alt_fuel_stations_historical_day ...).",
    )
    ap.add_argument(
        "--evse-zip-source",
        type=str,
        default="data/zip_to_county_ny.csv",
        help="ZIP-to-county CSV used for Nassau/Suffolk LIPA filtering.",
    )
    ap.add_argument(
        "--evse-zip-overrides",
        type=str,
        default="data/utility_zip_regions.csv",
        help="ZIP overrides CSV with region=LIPA rows for partial-service ZIPs.",
    )
    ap.add_argument(
        "--evse-restricted-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for restricted-public stations when aggregating EVSE ports "
            "(0=exclude, 1=fully include; default 0)."
        ),
    )
    ap.add_argument(
        "--evse-workplace-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for workplace stations when aggregating EVSE ports "
            "(0=exclude, 1=fully include; default 0)."
        ),
    )
    ap.add_argument(
        "--evse-output-csv",
        type=str,
        default="covariates/evse_lipa_monthly.csv",
        help="Where to write the monthly EVSE covariate table used in fitting.",
    )
    ap.add_argument(
        "--evse-lag-months",
        type=int,
        default=3,
        help="Calendar-month lag K applied to EVSE covariates (default 3).",
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
    ap.add_argument(
        "--market-potential-frac",
        type=float,
        default=0.5,
        help=(
            "Fix Bass market potential M to this fraction of the region's total on-road vehicle stock "
            "(ICE + EV, i.e., stock_all_t). Set to 0 to disable and estimate M from the data."
        ),
    )
    ap.add_argument(
        "--search-m-grid",
        action="store_true",
        help=(
            "Grid-search plausible fixed M values and pick the M that minimizes holdout error "
            "for a target pipeline. Requires --holdout-start."
        ),
    )
    ap.add_argument(
        "--m-grid-fracs",
        type=str,
        default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help=(
            "Comma-separated fractions applied to the region's total on-road vehicle stock (stock_all_t) "
            "to generate candidate M values for --search-m-grid."
        ),
    )
    ap.add_argument(
        "--m-grid-min-alast-mult",
        type=float,
        default=2.0,
        help=(
            "Lower bound for candidate M values as a multiple of last cumulative adopters A_last "
            "in the training window (e.g., 2.0 => min M is 2*A_last)."
        ),
    )
    ap.add_argument(
        "--m-grid-target",
        type=str,
        default="auto",
        choices=["auto", "baseline", "tco_adv", "tco_adv_policy", "tco_adv_evse"],
        help=(
            "Which pipeline's holdout error to optimize when --search-m-grid is enabled. "
            "Use 'auto' to pick tco_adv_evse if --with-evse, else tco_adv_policy if --with-policy, else tco_adv."
        ),
    )
    ap.add_argument(
        "--m-grid-metric",
        type=str,
        default="auto",
        choices=[
            "auto",
            "rmse_flow",
            "poisson_nll_flow",
            "mape_flow_pct",
            "rmse_stock",
            "mape_stock_pct",
        ],
        help=(
            "Holdout metric minimized to choose M during --search-m-grid. "
            "Use 'auto' => rmse_flow if --flow-likelihood=mse else poisson_nll_flow."
        ),
    )
    args = ap.parse_args()
    if not (0.0 <= args.evse_restricted_weight <= 1.0):
        raise ValueError("--evse-restricted-weight must be in [0,1]")
    if not (0.0 <= args.evse_workplace_weight <= 1.0):
        raise ValueError("--evse-workplace-weight must be in [0,1]")
    if args.evse_lag_months < 0:
        raise ValueError("--evse-lag-months must be >= 0")

    tag = (args.output_tag or "").strip()
    suffix = f"_{tag}" if tag else ""

    def _tagged(filename: str) -> str:
        if not suffix:
            return filename
        p = Path(filename)
        return f"{p.stem}{suffix}{p.suffix}"

    if args.panel_csv:
        panel_path = Path(args.panel_csv)
        if not panel_path.is_absolute():
            panel_path = ROOT / panel_path
        if not panel_path.exists():
            raise FileNotFoundError(f"--panel-csv not found: {panel_path}")
        panel_raw = pd.read_csv(panel_path)
        panel_all = prepare_panel_with_covariates(
            panel_raw,
            elec_price_default=args.elec_price,
            gas_series_path=args.gas_series,
            elec_series_path=args.elec_series,
            elec_rate_class=args.elec_rate_class,
            elec_utility=args.elec_utility,
            resample_monthly=args.resample_monthly,
            with_evse=args.with_evse,
            station_info_dir=args.station_info_dir,
            evse_zip_source=args.evse_zip_source,
            evse_zip_overrides=args.evse_zip_overrides,
            evse_restricted_weight=args.evse_restricted_weight,
            evse_workplace_weight=args.evse_workplace_weight,
            evse_lag_months=args.evse_lag_months,
            evse_output_csv=args.evse_output_csv,
        )
    else:
        panel_all = build_panel(
            elec_price_default=args.elec_price,
            gas_series_path=args.gas_series,
            elec_series_path=args.elec_series,
            elec_rate_class=args.elec_rate_class,
            elec_utility=args.elec_utility,
            resample_monthly=args.resample_monthly,
            with_evse=args.with_evse,
            station_info_dir=args.station_info_dir,
            evse_zip_source=args.evse_zip_source,
            evse_zip_overrides=args.evse_zip_overrides,
            evse_restricted_weight=args.evse_restricted_weight,
            evse_workplace_weight=args.evse_workplace_weight,
            evse_lag_months=args.evse_lag_months,
            evse_output_csv=args.evse_output_csv,
        )

    min_ts: Optional[pd.Timestamp] = pd.to_datetime(args.min_date) if args.min_date else None
    panel_fit = panel_all
    if min_ts is not None:
        panel_fit = panel_all[panel_all["date"] >= min_ts].copy()
        if panel_fit.empty:
            raise ValueError(f"--min-date {args.min_date} filters the panel to zero rows")

    out_dir = ROOT / "covariates"
    out_dir.mkdir(exist_ok=True)
    panel_fit.to_csv(out_dir / _tagged("panel_LIPA.csv"), index=False)

    panel_policy_all = attach_policy(panel_all) if args.with_policy else None
    panel_policy_fit = (
        panel_policy_all[panel_policy_all["date"] >= min_ts].copy()
        if (panel_policy_all is not None and min_ts is not None)
        else panel_policy_all
    )

    holdout_ts: Optional[pd.Timestamp] = (
        pd.to_datetime(args.holdout_start) if args.holdout_start else None
    )

    # Market potential constraint: fix M to a fraction of total (ICE+EV) stock in-region.
    # This prevents unstable/implausible M estimates (e.g., collapsing M) when adding covariates.
    M_fixed: float | None = None
    if not args.search_m_grid:
        frac = float(args.market_potential_frac or 0.0)
        if frac > 0.0:
            stock_ref = (
                panel_fit.loc[panel_fit["date"] < holdout_ts, "stock_all_t"].dropna()
                if holdout_ts is not None
                else panel_fit["stock_all_t"].dropna()
            )
            if not stock_ref.empty:
                total_market = float(stock_ref.iloc[-1])
                M_fixed = frac * total_market
                A_max = float(pd.to_numeric(panel_fit["adopt_ev_cum_t"], errors="coerce").max())
                if not np.isfinite(A_max) or A_max <= 0:
                    M_fixed = None
                elif M_fixed <= A_max:
                    raise ValueError(
                        f"market_potential_frac implies M_fixed={M_fixed:.0f} which is <= observed adopters A.max={A_max:.0f}. "
                        "Increase --market-potential-frac or disable the constraint with --market-potential-frac 0."
                    )
            else:
                print(
                    "Warning: market potential constraint requested but stock_all_t is missing; "
                    "falling back to estimating M from adoption data."
                )

    # Load retention curve (or fall back to an exponential survival fitted from aggregates).
    retention_path = Path(args.retention_curve)
    if not retention_path.is_absolute():
        retention_path = ROOT / retention_path
    if retention_path.exists():
        survival_by_lag, _max_lag = load_retention_curve(retention_path, region="LIPA")
        print(f"Loaded retention curve: {retention_path} (max_lag={_max_lag})")
    else:
        # Estimate a simple exponential retention curve from aggregate series so that
        # stock is not forced to equal cumulative adoption.
        min_date_for_retention = min_ts
        max_date_needed = pd.to_datetime(panel_all["date"]).max() + pd.DateOffset(months=args.horizon)
        max_lag_needed = month_index(max_date_needed) - month_index(pd.to_datetime(panel_all["date"]).min())
        survival_by_lag, lambda_hat = estimate_exponential_retention_from_aggregates(
            dates=pd.to_datetime(panel_all["date"]).to_numpy(),
            adoption_flows=panel_all["flow_ev_t"].to_numpy(float),
            stock_obs=panel_all["stock_ev_t"].to_numpy(float),
            eval_start=min_date_for_retention,
            max_lag_needed=max_lag_needed,
        )
        monthly_retention = float(np.exp(-lambda_hat))
        print(
            "Warning: retention curve not found at "
            f"{retention_path}; using exponential retention fitted from aggregates: "
            f"lambda={lambda_hat:.4f} (S(1)={monthly_retention:.4f}). "
            "For VIN-level KM retention, run scripts/estimate_ev_retention_curve.py to generate the curve."
        )

    # Optional grid search over plausible M values, chosen by holdout performance.
    if args.search_m_grid:
        if holdout_ts is None:
            raise ValueError("--search-m-grid requires --holdout-start")

        # Resolve target pipeline.
        target = args.m_grid_target
        if target == "auto":
            target = "tco_adv_evse" if args.with_evse else ("tco_adv_policy" if args.with_policy else "tco_adv")
        if target == "tco_adv_policy" and not args.with_policy:
            raise ValueError("--m-grid-target tco_adv_policy requires --with-policy")
        if target == "tco_adv_evse" and not args.with_evse:
            raise ValueError("--m-grid-target tco_adv_evse requires --with-evse")

        metric_name = args.m_grid_metric
        if metric_name == "auto":
            metric_name = "rmse_flow" if args.flow_likelihood == "mse" else "poisson_nll_flow"

        # Anchors for the candidate grid: last train total market and last cumulative adopters.
        train_anchor = panel_fit[panel_fit["date"] < holdout_ts].copy()
        if train_anchor.empty:
            raise ValueError(f"Holdout start {holdout_ts.date()} leaves an empty training set for M grid search")

        stock_ref = pd.to_numeric(train_anchor["stock_all_t"], errors="coerce").dropna()
        if stock_ref.empty:
            raise ValueError("Cannot build M grid: training data is missing stock_all_t (total market)")
        total_market = float(stock_ref.iloc[-1])

        A_last = float(pd.to_numeric(train_anchor["adopt_ev_cum_t"], errors="coerce").iloc[-1])
        if not np.isfinite(A_last) or A_last <= 0:
            raise ValueError("Cannot build M grid: invalid A_last from adopt_ev_cum_t")

        min_M = max(float(args.m_grid_min_alast_mult) * A_last, 1.01 * A_last)
        if total_market <= min_M:
            raise ValueError(
                f"Total market stock_all_t={total_market:.0f} is <= min_M={min_M:.0f} (from A_last={A_last:.0f}). "
                "Lower --m-grid-min-alast-mult or provide a larger market anchor."
            )

        # Candidate grid (fractions of total market), filtered to [min_M, total_market].
        fracs: list[float] = []
        for part in str(args.m_grid_fracs or "").split(","):
            part = part.strip()
            if not part:
                continue
            fracs.append(float(part))
        if not fracs:
            raise ValueError("--m-grid-fracs produced an empty list; provide fractions like '0.3,0.4,0.5'")

        M_candidates = sorted({float(f) * total_market for f in fracs})
        M_candidates = [m for m in M_candidates if (m >= min_M and m <= total_market)]
        if not M_candidates:
            # Fallback to a small linear grid if the fraction list doesn't overlap [min_M, total_market].
            M_candidates = list(np.linspace(min_M, total_market, num=6))

        # Build training sets used for parameter estimation.
        train_fit = panel_fit[panel_fit["date"] < holdout_ts].copy()
        train_policy_fit = (
            panel_policy_fit[panel_policy_fit["date"] < holdout_ts].copy() if panel_policy_fit is not None else None
        )

        # Convenience: holdout metric extraction for a single forecast df.
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

        def _metrics_from_forecast(f: pd.DataFrame) -> dict[str, float | None]:
            test_mask_flow = (f["is_train"] == 0) & (~f["flow_ev_t_obs"].isna())
            test_mask_stock = (f["is_train"] == 0) & (~f["stock_ev_t_obs"].isna())
            obs_flow = f.loc[test_mask_flow, "flow_ev_t_obs"].to_numpy(float)
            pred_flow = f.loc[test_mask_flow, "flow_ev_t_hat_anchor"].to_numpy(float)
            obs_stock = f.loc[test_mask_stock, "stock_ev_t_obs"].to_numpy(float)
            pred_stock = f.loc[test_mask_stock, "stock_ev_t_hat_anchor"].to_numpy(float)
            return {
                "rmse_flow": _rmse(obs_flow, pred_flow) if bool(test_mask_flow.any()) else None,
                "mape_flow_pct": _mape(obs_flow, pred_flow) if bool(test_mask_flow.any()) else None,
                "poisson_nll_flow": (
                    poisson_nll(obs_flow, pred_flow) if bool(test_mask_flow.any()) else None
                ),
                "rmse_stock": _rmse(obs_stock, pred_stock) if bool(test_mask_stock.any()) else None,
                "mape_stock_pct": _mape(obs_stock, pred_stock) if bool(test_mask_stock.any()) else None,
            }

        rows: list[dict[str, float | str | None]] = []
        best_M: float | None = None
        best_val: float | None = None

        for M_cand in M_candidates:
            M_cand = float(M_cand)
            # Fit parameters on training data for this fixed M.
            bass_params_c = fit_bass_baseline(
                train_fit,
                flow_likelihood=args.flow_likelihood,
                fit_mode=args.cov_fit_mode,
                M_fixed=M_cand,
            )
            cov_params_c = fit_bass_with_tco(
                train_fit,
                feature_cols=["tco_adv_t"],
                flow_likelihood=args.flow_likelihood,
                fit_mode=args.cov_fit_mode,
                ridge_lambda=args.cov_ridge_lambda,
                M_fixed=M_cand,
            )

            cov_params_policy_c = None
            if args.with_policy and train_policy_fit is not None and not train_policy_fit.empty:
                cov_params_policy_c = fit_bass_with_tco(
                    train_policy_fit,
                    feature_cols=["tco_adv_t", "subsidy_share_t"],
                    flow_likelihood=args.flow_likelihood,
                    fit_mode=args.cov_fit_mode,
                    ridge_lambda=args.cov_ridge_lambda,
                    M_fixed=M_cand,
                )

            cov_params_evse_c = None
            if args.with_evse:
                lag = int(args.evse_lag_months)
                if lag > 0:
                    x_total = f"x_evse_total_lag{lag}_t"
                    dcfc_share = f"evse_dcfc_share_lag{lag}_t"
                else:
                    x_total = "x_evse_total_t"
                    dcfc_share = "evse_dcfc_share_t"
                evse_feature_cols = ["tco_adv_t", x_total, dcfc_share]
                train_evse_fit = train_fit
                panel_evse_all = panel_all
                if args.with_policy:
                    if train_policy_fit is None or panel_policy_all is None:
                        raise ValueError("EVSE+policy requested but policy covariates are unavailable")
                    train_evse_fit = train_policy_fit
                    panel_evse_all = panel_policy_all
                    evse_feature_cols = ["tco_adv_t", "subsidy_share_t", x_total, dcfc_share]
                if "evse_series_start" in train_evse_fit.columns:
                    evse_start = pd.to_datetime(train_evse_fit["evse_series_start"].iloc[0])
                    evse_fit_start = evse_start + pd.DateOffset(months=lag)
                    if min_ts is not None and evse_fit_start < min_ts:
                        evse_fit_start = min_ts
                    train_evse_fit = train_evse_fit[train_evse_fit["date"] >= evse_fit_start].copy()
                if train_evse_fit.empty:
                    raise ValueError("EVSE covariate training set is empty (check station_info coverage and lag)")
                cov_params_evse_c = fit_bass_with_tco(
                    train_evse_fit,
                    feature_cols=evse_feature_cols,
                    flow_likelihood=args.flow_likelihood,
                    fit_mode=(args.evse_fit_mode or args.cov_fit_mode),
                    ridge_lambda=(
                        args.evse_ridge_lambda
                        if args.evse_ridge_lambda is not None
                        else args.cov_ridge_lambda
                    ),
                    M_fixed=M_cand,
                )

            # Holdout forecasts and metrics for each pipeline.
            m_baseline = _metrics_from_forecast(
                forecast_holdout_with_retention(
                    panel_all,
                    p_const=bass_params_c.p,
                    q_const=bass_params_c.q,
                    M=bass_params_c.M,
                    holdout_start=holdout_ts,
                    horizon=args.horizon,
                    survival_by_lag=survival_by_lag,
                )
            )
            m_tco = _metrics_from_forecast(
                forecast_holdout_with_retention(
                    panel_all,
                    cov_params=cov_params_c,
                    M=cov_params_c.M,
                    holdout_start=holdout_ts,
                    horizon=args.horizon,
                    survival_by_lag=survival_by_lag,
                )
            )
            m_policy = None
            if cov_params_policy_c is not None and panel_policy_all is not None:
                m_policy = _metrics_from_forecast(
                    forecast_holdout_with_retention(
                        panel_policy_all,
                        cov_params=cov_params_policy_c,
                        M=cov_params_policy_c.M,
                        holdout_start=holdout_ts,
                        horizon=args.horizon,
                        survival_by_lag=survival_by_lag,
                    )
                )
            m_evse = None
            if cov_params_evse_c is not None:
                panel_evse_all = panel_policy_all if args.with_policy else panel_all
                m_evse = _metrics_from_forecast(
                    forecast_holdout_with_retention(
                        panel_evse_all,
                        cov_params=cov_params_evse_c,
                        M=cov_params_evse_c.M,
                        holdout_start=holdout_ts,
                        horizon=args.horizon,
                        survival_by_lag=survival_by_lag,
                    )
                )

            # Target metric selection.
            metric_map = {
                "baseline": m_baseline,
                "tco_adv": m_tco,
                "tco_adv_policy": m_policy,
                "tco_adv_evse": m_evse,
            }
            sel = metric_map.get(target)
            if sel is None or sel.get(metric_name) is None:
                continue
            sel_val = float(sel[metric_name])  # type: ignore[arg-type]

            row = {
                "M": M_cand,
                "M_frac_of_total": M_cand / total_market if total_market > 0 else None,
                "target": target,
                "metric": metric_name,
                "target_value": sel_val,
                "baseline_rmse_flow": m_baseline["rmse_flow"],
                "baseline_poisson_nll_flow": m_baseline["poisson_nll_flow"],
                "baseline_rmse_stock": m_baseline["rmse_stock"],
                "tco_rmse_flow": m_tco["rmse_flow"],
                "tco_poisson_nll_flow": m_tco["poisson_nll_flow"],
                "tco_rmse_stock": m_tco["rmse_stock"],
                "policy_rmse_flow": (m_policy or {}).get("rmse_flow") if m_policy is not None else None,
                "policy_poisson_nll_flow": (m_policy or {}).get("poisson_nll_flow") if m_policy is not None else None,
                "policy_rmse_stock": (m_policy or {}).get("rmse_stock") if m_policy is not None else None,
                "evse_rmse_flow": (m_evse or {}).get("rmse_flow") if m_evse is not None else None,
                "evse_poisson_nll_flow": (m_evse or {}).get("poisson_nll_flow") if m_evse is not None else None,
                "evse_rmse_stock": (m_evse or {}).get("rmse_stock") if m_evse is not None else None,
            }
            rows.append(row)
            if best_val is None or sel_val < best_val:
                best_val = sel_val
                best_M = M_cand

        if not rows or best_M is None:
            raise ValueError("M grid search produced no valid candidates; check target pipeline and metric settings.")

        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        grid_df = pd.DataFrame(rows).sort_values("target_value").reset_index(drop=True)
        grid_path = models_dir / _tagged("bass_lipa_M_grid_search.csv")
        grid_df.to_csv(grid_path, index=False)
        print(
            f"M grid search written to {grid_path}; selected M={best_M:.0f} "
            f"(target={target}, metric={metric_name}, value={best_val:.4g})"
        )
        M_fixed = float(best_M)

    cov_params_policy = None
    cov_params_evse = None

    if holdout_ts is None:
        # Full-sample fit (parameters estimated on panel_fit), forecast on full panel_all.
        bass_params = fit_bass_baseline(
            panel_fit,
            flow_likelihood=args.flow_likelihood,
            fit_mode=args.cov_fit_mode,
            M_fixed=M_fixed,
        )
        cov_params = fit_bass_with_tco(
            panel_fit,
            feature_cols=["tco_adv_t"],
            flow_likelihood=args.flow_likelihood,
            fit_mode=args.cov_fit_mode,
            ridge_lambda=args.cov_ridge_lambda,
            M_fixed=M_fixed,
        )
        cov_params_policy = (
            fit_bass_with_tco(
                panel_policy_fit,
                feature_cols=["tco_adv_t", "subsidy_share_t"],
                flow_likelihood=args.flow_likelihood,
                fit_mode=args.cov_fit_mode,
                ridge_lambda=args.cov_ridge_lambda,
                M_fixed=M_fixed,
            )
            if panel_policy_fit is not None
            else None
        )
        if args.with_evse:
            lag = int(args.evse_lag_months)
            if lag > 0:
                x_total = f"x_evse_total_lag{lag}_t"
                dcfc_share = f"evse_dcfc_share_lag{lag}_t"
            else:
                x_total = "x_evse_total_t"
                dcfc_share = "evse_dcfc_share_t"
            evse_feature_cols = ["tco_adv_t", x_total, dcfc_share]
            panel_evse_fit = panel_fit
            panel_evse_all = panel_all
            if args.with_policy:
                if panel_policy_fit is None or panel_policy_all is None:
                    raise ValueError("EVSE+policy requested but panel policy covariates are unavailable")
                evse_feature_cols = ["tco_adv_t", "subsidy_share_t", x_total, dcfc_share]
                panel_evse_fit = panel_policy_fit
                panel_evse_all = panel_policy_all
            # Fit only on the EVSE-coverage window to avoid treating 2011-2019 as true zeros.
            if "evse_series_start" in panel_evse_fit.columns:
                evse_start = pd.to_datetime(panel_evse_fit["evse_series_start"].iloc[0])
                evse_fit_start = evse_start + pd.DateOffset(months=lag)
                if min_ts is not None and evse_fit_start < min_ts:
                    evse_fit_start = min_ts
                panel_evse_fit = panel_evse_fit[panel_evse_fit["date"] >= evse_fit_start].copy()
            if len(panel_evse_fit) < 3:
                raise ValueError("EVSE fit window too small; check station_info coverage and lag settings")
            cov_params_evse = fit_bass_with_tco(
                panel_evse_fit,
                feature_cols=evse_feature_cols,
                flow_likelihood=args.flow_likelihood,
                fit_mode=(args.evse_fit_mode or args.cov_fit_mode),
                ridge_lambda=(
                    args.evse_ridge_lambda
                    if args.evse_ridge_lambda is not None
                    else args.cov_ridge_lambda
                ),
                M_fixed=M_fixed,
            )

        forecast_df = forecast_fullsample_with_retention(
            panel_all,
            p_const=bass_params.p,
            q_const=bass_params.q,
            M=bass_params.M,
            horizon=args.horizon,
            survival_by_lag=survival_by_lag,
        )

        forecast_cov = forecast_fullsample_with_retention(
            panel_all,
            cov_params=cov_params,
            M=cov_params.M,
            horizon=args.horizon,
            survival_by_lag=survival_by_lag,
        ).rename(
            columns={
                "stock_ev_t_hat_anchor": "stock_ev_t_hat_cov_anchor",
                "flow_ev_t_hat_anchor": "flow_ev_t_hat_cov_anchor",
            }
        )
        forecast_df = forecast_df.merge(
            forecast_cov[["date", "stock_ev_t_hat_cov_anchor", "flow_ev_t_hat_cov_anchor"]],
            on="date",
            how="left",
        )

        if cov_params_policy is not None and panel_policy_all is not None:
            forecast_cov_policy = forecast_fullsample_with_retention(
                panel_policy_all,
                cov_params=cov_params_policy,
                M=cov_params_policy.M,
                horizon=args.horizon,
                survival_by_lag=survival_by_lag,
            ).rename(
                columns={
                    "stock_ev_t_hat_anchor": "stock_ev_t_hat_cov_policy_anchor",
                    "flow_ev_t_hat_anchor": "flow_ev_t_hat_cov_policy_anchor",
                }
            )
            forecast_df = forecast_df.merge(
                forecast_cov_policy[
                    ["date", "stock_ev_t_hat_cov_policy_anchor", "flow_ev_t_hat_cov_policy_anchor"]
                ],
                on="date",
                how="left",
            )
        if cov_params_evse is not None:
            panel_evse_all = panel_policy_all if args.with_policy else panel_all
            forecast_cov_evse = forecast_fullsample_with_retention(
                panel_evse_all,
                cov_params=cov_params_evse,
                M=cov_params_evse.M,
                horizon=args.horizon,
                survival_by_lag=survival_by_lag,
            ).rename(
                columns={
                    "stock_ev_t_hat_anchor": "stock_ev_t_hat_cov_evse_anchor",
                    "flow_ev_t_hat_anchor": "flow_ev_t_hat_cov_evse_anchor",
                }
            )
            forecast_df = forecast_df.merge(
                forecast_cov_evse[["date", "stock_ev_t_hat_cov_evse_anchor", "flow_ev_t_hat_cov_evse_anchor"]],
                on="date",
                how="left",
            )
    else:
        # Train/Test split: estimate parameters on pre-holdout rows in panel_fit, forecast on full panel_all.
        train_fit = panel_fit[panel_fit["date"] < holdout_ts].copy()
        if train_fit.empty:
            raise ValueError(f"Holdout start {holdout_ts.date()} leaves an empty training set")

        bass_params = fit_bass_baseline(
            train_fit,
            flow_likelihood=args.flow_likelihood,
            fit_mode=args.cov_fit_mode,
            M_fixed=M_fixed,
        )
        cov_params = fit_bass_with_tco(
            train_fit,
            feature_cols=["tco_adv_t"],
            flow_likelihood=args.flow_likelihood,
            fit_mode=args.cov_fit_mode,
            ridge_lambda=args.cov_ridge_lambda,
            M_fixed=M_fixed,
        )
        if args.with_evse:
            lag = int(args.evse_lag_months)
            if lag > 0:
                x_total = f"x_evse_total_lag{lag}_t"
                dcfc_share = f"evse_dcfc_share_lag{lag}_t"
            else:
                x_total = "x_evse_total_t"
                dcfc_share = "evse_dcfc_share_t"
            evse_feature_cols = ["tco_adv_t", x_total, dcfc_share]
            panel_evse_all = panel_all
            train_evse_fit = train_fit
            if args.with_policy:
                if panel_policy_fit is None or panel_policy_all is None:
                    raise ValueError("EVSE+policy requested but panel policy covariates are unavailable")
                evse_feature_cols = ["tco_adv_t", "subsidy_share_t", x_total, dcfc_share]
                panel_evse_all = panel_policy_all
                train_evse_fit = panel_policy_fit[panel_policy_fit["date"] < holdout_ts].copy()
            if "evse_series_start" in train_evse_fit.columns:
                evse_start = pd.to_datetime(train_evse_fit["evse_series_start"].iloc[0])
                evse_fit_start = evse_start + pd.DateOffset(months=lag)
                if min_ts is not None and evse_fit_start < min_ts:
                    evse_fit_start = min_ts
                train_evse_fit = train_evse_fit[train_evse_fit["date"] >= evse_fit_start].copy()
            if train_evse_fit.empty:
                raise ValueError("EVSE covariate training set is empty")
            cov_params_evse = fit_bass_with_tco(
                train_evse_fit,
                feature_cols=evse_feature_cols,
                flow_likelihood=args.flow_likelihood,
                fit_mode=(args.evse_fit_mode or args.cov_fit_mode),
                ridge_lambda=(
                    args.evse_ridge_lambda
                    if args.evse_ridge_lambda is not None
                    else args.cov_ridge_lambda
                ),
                M_fixed=M_fixed,
            )

        forecast_df = forecast_holdout_with_retention(
            panel_all,
            p_const=bass_params.p,
            q_const=bass_params.q,
            M=bass_params.M,
            holdout_start=holdout_ts,
            horizon=args.horizon,
            survival_by_lag=survival_by_lag,
        )

        forecast_cov_tt = forecast_holdout_with_retention(
            panel_all,
            cov_params=cov_params,
            M=cov_params.M,
            holdout_start=holdout_ts,
            horizon=args.horizon,
            survival_by_lag=survival_by_lag,
        ).rename(
            columns={
                "stock_ev_t_hat_fit": "stock_ev_t_hat_cov_fit",
                "flow_ev_t_hat_fit": "flow_ev_t_hat_cov_fit",
                "stock_ev_t_hat_anchor": "stock_ev_t_hat_cov_anchor",
                "flow_ev_t_hat_anchor": "flow_ev_t_hat_cov_anchor",
            }
        )
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

        if panel_policy_fit is not None and panel_policy_all is not None:
            train_policy_fit = panel_policy_fit[panel_policy_fit["date"] < holdout_ts].copy()
            if not train_policy_fit.empty:
                cov_params_policy = fit_bass_with_tco(
                    train_policy_fit,
                    feature_cols=["tco_adv_t", "subsidy_share_t"],
                    flow_likelihood=args.flow_likelihood,
                    fit_mode=args.cov_fit_mode,
                    ridge_lambda=args.cov_ridge_lambda,
                    M_fixed=M_fixed,
                )
                forecast_policy_tt = forecast_holdout_with_retention(
                    panel_policy_all,
                    cov_params=cov_params_policy,
                    M=cov_params_policy.M,
                    holdout_start=holdout_ts,
                    horizon=args.horizon,
                    survival_by_lag=survival_by_lag,
                ).rename(
                    columns={
                        "stock_ev_t_hat_fit": "stock_ev_t_hat_cov_policy_fit",
                        "flow_ev_t_hat_fit": "flow_ev_t_hat_cov_policy_fit",
                        "stock_ev_t_hat_anchor": "stock_ev_t_hat_cov_policy_anchor",
                        "flow_ev_t_hat_anchor": "flow_ev_t_hat_cov_policy_anchor",
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
        if cov_params_evse is not None:
            panel_evse_all = panel_policy_all if args.with_policy else panel_all
            forecast_evse_tt = forecast_holdout_with_retention(
                panel_evse_all,
                cov_params=cov_params_evse,
                M=cov_params_evse.M,
                holdout_start=holdout_ts,
                horizon=args.horizon,
                survival_by_lag=survival_by_lag,
            ).rename(
                columns={
                    "stock_ev_t_hat_fit": "stock_ev_t_hat_cov_evse_fit",
                    "flow_ev_t_hat_fit": "flow_ev_t_hat_cov_evse_fit",
                    "stock_ev_t_hat_anchor": "stock_ev_t_hat_cov_evse_anchor",
                    "flow_ev_t_hat_anchor": "flow_ev_t_hat_cov_evse_anchor",
                }
            )
            forecast_df = forecast_df.merge(
                forecast_evse_tt[
                    [
                        "date",
                        "stock_ev_t_hat_cov_evse_fit",
                        "flow_ev_t_hat_cov_evse_fit",
                        "stock_ev_t_hat_cov_evse_anchor",
                        "flow_ev_t_hat_cov_evse_anchor",
                    ]
                ],
                on="date",
                how="left",
            )

    # For readability, trim outputs to --min-date if set (retains correct cumulative state internally).
    if min_ts is not None:
        forecast_df = forecast_df[pd.to_datetime(forecast_df["date"]) >= min_ts].reset_index(drop=True)

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
    if cov_params_evse is not None:
        evse_name = "bass_lipa_with_tco_policy_evse.json" if args.with_policy else "bass_lipa_with_tco_evse.json"
        with open(models_dir / _tagged(evse_name), "w", encoding="utf-8") as f:
            json.dump(asdict(cov_params_evse), f, indent=2)

    # Save forecast table (now includes both baseline and covariate forecasts)
    forecast_df.to_csv(models_dir / _tagged("bass_lipa_forecast.csv"), index=False)

    # Optional holdout metrics (flows + stock) when a holdout split is used.
    if holdout_ts is not None and "is_train" in forecast_df.columns:
        test_mask_flow = (forecast_df["is_train"] == 0) & (~forecast_df["flow_ev_t_obs"].isna())
        test_mask_stock = (forecast_df["is_train"] == 0) & (~forecast_df["stock_ev_t_obs"].isna())

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

        metrics = {
            "holdout_start": str(holdout_ts.date()),
            "n_test_flow": int(test_mask_flow.sum()),
            "n_test_stock": int(test_mask_stock.sum()),
            "date_min_flow": (
                str(pd.to_datetime(forecast_df.loc[test_mask_flow, "date"]).min().date())
                if bool(test_mask_flow.any())
                else None
            ),
            "date_max_flow": (
                str(pd.to_datetime(forecast_df.loc[test_mask_flow, "date"]).max().date())
                if bool(test_mask_flow.any())
                else None
            ),
            "date_min_stock": (
                str(pd.to_datetime(forecast_df.loc[test_mask_stock, "date"]).min().date())
                if bool(test_mask_stock.any())
                else None
            ),
            "date_max_stock": (
                str(pd.to_datetime(forecast_df.loc[test_mask_stock, "date"]).max().date())
                if bool(test_mask_stock.any())
                else None
            ),
            "models": {},
        }

        candidates = [
            ("baseline", "flow_ev_t_hat_anchor", "stock_ev_t_hat_anchor"),
            ("tco_adv", "flow_ev_t_hat_cov_anchor", "stock_ev_t_hat_cov_anchor"),
            ("tco_adv_policy", "flow_ev_t_hat_cov_policy_anchor", "stock_ev_t_hat_cov_policy_anchor"),
            ("tco_adv_evse", "flow_ev_t_hat_cov_evse_anchor", "stock_ev_t_hat_cov_evse_anchor"),
        ]
        for name, flow_col, stock_col in candidates:
            if flow_col not in forecast_df.columns or stock_col not in forecast_df.columns:
                continue
            obs_flow = forecast_df.loc[test_mask_flow, "flow_ev_t_obs"].to_numpy(float)
            pred_flow = forecast_df.loc[test_mask_flow, flow_col].to_numpy(float)
            obs_stock = forecast_df.loc[test_mask_stock, "stock_ev_t_obs"].to_numpy(float)
            pred_stock = forecast_df.loc[test_mask_stock, stock_col].to_numpy(float)
            metrics["models"][name] = {
                "rmse_flow": _rmse(obs_flow, pred_flow) if bool(test_mask_flow.any()) else None,
                "mape_flow_pct": _mape(obs_flow, pred_flow) if bool(test_mask_flow.any()) else None,
                "poisson_nll_flow": (
                    poisson_nll(obs_flow, pred_flow) if bool(test_mask_flow.any()) else None
                ),
                "rmse_stock": _rmse(obs_stock, pred_stock) if bool(test_mask_stock.any()) else None,
                "mape_stock_pct": _mape(obs_stock, pred_stock) if bool(test_mask_stock.any()) else None,
            }

        with open(models_dir / _tagged("bass_lipa_holdout_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Holdout metrics written to {models_dir / _tagged('bass_lipa_holdout_metrics.json')}")

    # EVSE "fit" lines should only be shown where EVSE covariates exist (AFDC coverage window + lag).
    # Forecast tables intentionally don't carry panel metadata columns, so compute this from the
    # underlying panel(s) used to build covariates.
    evse_fit_start_plot: pd.Timestamp | None = None
    if args.with_evse:
        panel_meta = panel_policy_all if (args.with_policy and panel_policy_all is not None) else panel_all
        if "evse_series_start" in panel_meta.columns:
            evse_start = pd.to_datetime(panel_meta["evse_series_start"]).max()
            if pd.notna(evse_start):
                evse_fit_start_plot = pd.Timestamp(evse_start) + pd.DateOffset(
                    months=int(args.evse_lag_months)
                )

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
        if "stock_ev_t_hat_cov_evse_anchor" in forecast_df.columns:
            ax.plot(
                future["date"],
                future["stock_ev_t_hat_cov_evse_anchor"],
                label=(
                    "Bass forecast (tco_adv + EVSE)"
                    if holdout_ts is None and not args.with_policy
                    else (
                        "Bass forecast (tco_adv + policy + EVSE)"
                        if holdout_ts is None
                        else (
                            "Bass forecast (tco_adv + EVSE, train<2025)"
                            if not args.with_policy
                            else "Bass forecast (tco_adv + policy + EVSE, train<2025)"
                        )
                    )
                ),
                color="C4",
                linestyle=(0, (3, 1, 1, 1)),
            )
        if holdout_ts is not None and "stock_ev_t_hat_cov_evse_fit" in forecast_df.columns:
            evse_fit_mask = forecast_df["is_train"] == 1
            if evse_fit_start_plot is not None:
                evse_fit_mask = evse_fit_mask & (
                    pd.to_datetime(forecast_df["date"]) >= evse_fit_start_plot
                )
            ax.plot(
                forecast_df.loc[evse_fit_mask, "date"],
                forecast_df.loc[evse_fit_mask, "stock_ev_t_hat_cov_evse_fit"],
                label=(
                    "Bass fit (tco_adv + EVSE, train<2025)"
                    if not args.with_policy
                    else "Bass fit (tco_adv + policy + EVSE, train<2025)"
                ),
                color="C4",
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
        if holdout_ts is None:
            flow_future = forecast_df
        else:
            # Holdout forecast is only meaningful in the test window; plotting the anchored
            # forecast across the full history is confusing (especially for EVSE which is
            # fit only on the EVSE-coverage window).
            flow_future = forecast_df[pd.to_datetime(forecast_df["date"]) >= holdout_ts]
        ax.plot(
            flow_future["date"],
            flow_future["flow_ev_t_hat_anchor"],
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
                flow_future["date"],
                flow_future["flow_ev_t_hat_cov_anchor"],
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
                flow_future["date"],
                flow_future["flow_ev_t_hat_cov_policy_anchor"],
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
        if "flow_ev_t_hat_cov_evse_anchor" in forecast_df.columns:
            ax.plot(
                flow_future["date"],
                flow_future["flow_ev_t_hat_cov_evse_anchor"],
                label=(
                    "Bass flow (tco_adv + EVSE)"
                    if holdout_ts is None and not args.with_policy
                    else (
                        "Bass flow (tco_adv + policy + EVSE)"
                        if holdout_ts is None
                        else (
                            "Bass flow (tco_adv + EVSE, holdout forecast)"
                            if not args.with_policy
                            else "Bass flow (tco_adv + policy + EVSE, holdout forecast)"
                        )
                    )
                ),
                color="C4",
                linestyle=(0, (3, 1, 1, 1)),
            )
        if holdout_ts is not None and "flow_ev_t_hat_cov_evse_fit" in forecast_df.columns:
            # EVSE covariates only exist from the AFDC snapshot start; the EVSE model is fit
            # only on that coverage window (+lag). Hide the pre-coverage "fit" line to avoid
            # implying we estimated EVSE effects before EVSE data exists.
            evse_fit_mask = forecast_df["is_train"] == 1
            if evse_fit_start_plot is not None:
                evse_fit_mask = evse_fit_mask & (
                    pd.to_datetime(forecast_df["date"]) >= evse_fit_start_plot
                )
            ax.plot(
                forecast_df.loc[evse_fit_mask, "date"],
                forecast_df.loc[evse_fit_mask, "flow_ev_t_hat_cov_evse_fit"],
                label=(
                    "Bass flow (tco_adv + EVSE, fit)"
                    if not args.with_policy
                    else "Bass flow (tco_adv + policy + EVSE, fit)"
                ),
                color="C4",
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

        # Cumulative adoption plot (Bass is fit on cumulative adopters, stock is derived via retention).
        fig, ax = plt.subplots(figsize=(10, 5))
        hist = forecast_df[~forecast_df["flow_ev_t_obs"].isna()]
        ax.plot(
            hist["date"],
            hist["adopt_ev_cum_t_obs"],
            label="Observed cumulative adopters (A_t)",
            color="C0",
        )

        def _cum_holdout(flow_col: str) -> np.ndarray:
            flow = pd.to_numeric(forecast_df[flow_col], errors="coerce").to_numpy(float)
            A_obs = pd.to_numeric(forecast_df["adopt_ev_cum_t_obs"], errors="coerce").to_numpy(float)
            if holdout_ts is None or "is_train" not in forecast_df.columns:
                return np.cumsum(np.nan_to_num(flow, nan=0.0))
            is_train_arr = forecast_df["is_train"].to_numpy(int) == 1
            test_start_idx = int(np.where(~is_train_arr)[0][0])
            train_end_idx = test_start_idx - 1
            A_hat = np.full(len(flow), np.nan, dtype=float)
            A_hat[train_end_idx] = A_obs[train_end_idx]
            for t in range(test_start_idx, len(flow)):
                prev = A_hat[t - 1] if np.isfinite(A_hat[t - 1]) else A_obs[train_end_idx]
                A_hat[t] = prev + (flow[t] if np.isfinite(flow[t]) else 0.0)
            return A_hat

        ax.plot(
            forecast_df["date"],
            _cum_holdout("flow_ev_t_hat_anchor"),
            label="Bass adopters (baseline)",
            color="C1",
            linestyle="--",
        )
        if "flow_ev_t_hat_cov_anchor" in forecast_df.columns:
            ax.plot(
                forecast_df["date"],
                _cum_holdout("flow_ev_t_hat_cov_anchor"),
                label="Bass adopters (tco_adv)",
                color="C2",
                linestyle=":",
            )
        if "flow_ev_t_hat_cov_policy_anchor" in forecast_df.columns:
            ax.plot(
                forecast_df["date"],
                _cum_holdout("flow_ev_t_hat_cov_policy_anchor"),
                label="Bass adopters (tco_adv + policy)",
                color="C3",
                linestyle="-.",
            )
        if "flow_ev_t_hat_cov_evse_anchor" in forecast_df.columns:
            ax.plot(
                forecast_df["date"],
                _cum_holdout("flow_ev_t_hat_cov_evse_anchor"),
                label=(
                    "Bass adopters (tco_adv + EVSE)"
                    if not args.with_policy
                    else "Bass adopters (tco_adv + policy + EVSE)"
                ),
                color="C4",
                linestyle=(0, (3, 1, 1, 1)),
            )

        if holdout_ts is None:
            ax.set_title("LIPA EV Cumulative Adopters: Observed vs Bass")
        else:
            ax.axvline(holdout_ts, color="k", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_title("LIPA EV Cumulative Adopters: Holdout Forecast (train<2025, test>=2025-01)")
        ax.set_xlabel("Snapshot date")
        ax.set_ylabel("Cumulative adopters (unique VINs ever first-seen)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        fig.savefig(models_dir / _tagged("bass_lipa_adopters_forecast.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("Plotting skipped:", e)

    print(f"Panel written to covariates/{_tagged('panel_LIPA.csv')}")
    print(f"Baseline Bass params written to models/{_tagged('bass_lipa_baseline.json')}")
    print(f"Covariate Bass params written to models/{_tagged('bass_lipa_with_tco.json')}")
    if cov_params_policy is not None:
        print(f"Policy covariate Bass params written to models/{_tagged('bass_lipa_with_tco_policy.json')}")
    if cov_params_evse is not None:
        evse_name = "bass_lipa_with_tco_policy_evse.json" if args.with_policy else "bass_lipa_with_tco_evse.json"
        print(f"EVSE covariate Bass params written to models/{_tagged(evse_name)}")
    print(f"Forecast written to models/{_tagged('bass_lipa_forecast.csv')}")


if __name__ == "__main__":
    main()
