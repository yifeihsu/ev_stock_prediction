#!/usr/bin/env python3
"""
Fit Bass diffusion models for EV adoption in a single ZIP code.

This script:
  1) Streams the NYSERDA split CSVs to build a ZIP-level snapshot panel:
       - stock_ev_t: unique EV VINs on-road in each DMV snapshot (DMV_ID)
       - flow_ev_t: first-seen EV VINs in that ZIP (new to the ZIP) by DMV_ID
  2) Fits an independent Bass baseline on the ZIP adoption process (cumulative first-seen EVs).
  3) Optionally fits a hierarchical (partial-pooling) Bass baseline for the ZIP using
     penalized least squares that shrinks ZIP (p,q) toward global (p̄,q̄) and ties ZIP market
     potential M_z to an observable scale.
  4) Converts adoption to on-road stock via a retention (survival) curve.

Notes
  - EV classification uses Vehicle Descriptions.csv: Drivetrain_Type in {BEV, PHEV}.
  - Stocks/flows are ZIP-based: a VIN moving into the ZIP later is counted as
    "new to the ZIP" at its first appearance in that ZIP.
  - Optional --resample-monthly can smooth pre-2018 annual/irregular snapshots
    into a regular month-start series (as in the LIPA pipeline).

Example
  python scripts/build_and_fit_bass_zip.py \\
    --zip 11746 \\
    --inputs-glob "split_part_*.csv" \\
    --descriptions "Vehicle Descriptions.csv" \\
    --hierarchical \\
    --resample-monthly \\
    --holdout-start 2025-01-01 \\
    --horizon 24
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Reuse the model + covariate logic from the LIPA script.
import build_and_fit_bass_lipa as bass


ROOT = Path(__file__).resolve().parents[1]


def normalize_zip(z: str) -> str:
    z = (z or "").strip()
    if z.isdigit() and len(z) <= 5:
        return z.zfill(5)
    return z


def load_ev_vin_keys(descriptions_path: Path) -> set[str]:
    """Return VIN_Key values that are BEV/PHEV."""
    ev_keys: set[str] = set()
    with open(descriptions_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "VIN_Key" not in r.fieldnames or "Drivetrain_Type" not in r.fieldnames:
            raise ValueError("Vehicle Descriptions.csv must have VIN_Key and Drivetrain_Type columns")
        for row in r:
            k = (row.get("VIN_Key") or "").strip()
            if not k:
                continue
            d = (row.get("Drivetrain_Type") or "").strip().upper()
            if d in ("BEV", "PHEV"):
                ev_keys.add(k)
    return ev_keys


def iter_input_paths(inputs_glob: str) -> list[Path]:
    paths = sorted(Path(p).resolve() for p in ROOT.glob(inputs_glob))
    if not paths:
        raise FileNotFoundError(f"No input files matched: {inputs_glob}")
    return paths


def load_snapshot_map(snapshot_map_path: Path) -> pd.DataFrame:
    snap = pd.read_csv(snapshot_map_path)
    if "DMV_ID" not in snap.columns or "DMV_Snapshot_Date" not in snap.columns:
        raise ValueError("NY_DMV_Snapshots.csv must have DMV_ID and DMV_Snapshot_Date")
    snap = snap[["DMV_ID", "DMV_Snapshot_Date"]].copy()
    snap["DMV_ID"] = pd.to_numeric(snap["DMV_ID"], errors="coerce").astype("Int64")
    snap = snap.dropna(subset=["DMV_ID"]).copy()
    snap["DMV_ID"] = snap["DMV_ID"].astype(int)
    snap.rename(columns={"DMV_Snapshot_Date": "date"}, inplace=True)
    snap["date"] = pd.to_datetime(snap["date"])
    snap = snap.sort_values("DMV_ID").reset_index(drop=True)
    return snap


def drop_partial_last_month_from_monthly_panel(
    panel_monthly: pd.DataFrame, snapshot_map: pd.DataFrame
) -> pd.DataFrame:
    """Drop the last month if it is only partially observed.

    When we map DMV snapshots to month-start timestamps, the final month can be partial
    if the last snapshot date is not at month-end (e.g., last snapshot is 2025-09-08).
    For resampled-monthly flows, this can create an artificial last-point drop.

    Behavior:
      - If panel dates are not month-start, returns unchanged.
      - If the last month corresponds to a snapshot date before that month-end:
          drop the last month row (no rolling), to avoid a misleading last-point artifact.
    """
    if panel_monthly.empty or "date" not in panel_monthly.columns:
        return panel_monthly

    df = panel_monthly.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    last_month_start = pd.Timestamp(df["date"].max())
    if int(last_month_start.day) != 1:
        return df

    # Need a snapshot ID to map to the true snapshot date.
    if "DMV_ID" not in df.columns:
        return df
    last_rows = df[df["date"] == last_month_start].copy()
    if last_rows.empty:
        return df
    last_dmv_ids = pd.to_numeric(last_rows["DMV_ID"], errors="coerce").dropna()
    if last_dmv_ids.empty:
        return df
    last_dmv_id = int(last_dmv_ids.max())

    snap_row = snapshot_map[snapshot_map["DMV_ID"] == last_dmv_id]
    if snap_row.empty:
        return df
    last_snapshot_date = pd.Timestamp(snap_row["date"].iloc[0]).normalize()

    last_month_end = (last_month_start + pd.offsets.MonthEnd(0)).normalize()
    if last_snapshot_date >= last_month_end:
        return df

    # Drop the last month (do not roll).
    df = df[df["date"] != last_month_start].reset_index(drop=True)
    return df


def _default_global_params_path() -> Path | None:
    """Best-effort default for global p,q anchoring the hierarchical ZIP fit."""
    candidates = [
        ROOT / "models" / "bass_lipa_baseline_light_duty_poisson_evse_suppEVSE_allSim_ridge01_evseR5000.json",
        ROOT / "models" / "bass_lipa_baseline_light_duty_poisson_evse_fix_pq_Mgrid_allSim_ridge01.json",
        ROOT / "models" / "bass_lipa_baseline_light_duty_mse_evse_fix_pq_Mgrid_allSim_ridge01.json",
        ROOT / "models" / "bass_lipa_baseline_monthly_resampled.json",
        ROOT / "models" / "bass_lipa_baseline_monthly2018.json",
        ROOT / "models" / "bass_lipa_baseline.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _default_lipa_panel_path() -> Path | None:
    """Best-effort default for LIPA stock scaling (stock_ev_t, stock_all_t)."""
    candidates = [
        ROOT / "covariates" / "panel_LIPA_light_duty_poisson_evse_suppEVSE_allSim_ridge01_evseR5000.csv",
        ROOT / "covariates" / "panel_LIPA_light_duty_poisson_evse_fix_pq_Mgrid_allSim_ridge01.csv",
        ROOT / "covariates" / "panel_LIPA_light_duty_snapshot.csv",
        ROOT / "covariates" / "panel_LIPA_monthly_resampled.csv",
        ROOT / "covariates" / "panel_LIPA_monthly2018.csv",
        ROOT / "covariates" / "panel_LIPA.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_global_pq(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "p" not in data or "q" not in data:
        raise ValueError(f"Global params JSON must contain p and q; got keys={list(data)[:10]}")
    return float(data["p"]), float(data["q"])


def estimate_zip_total_market_from_lipa_scale(
    *,
    zip_panel: pd.DataFrame,
    lipa_panel: pd.DataFrame,
    scale_date: pd.Timestamp,
) -> float | None:
    """Estimate ZIP total vehicle market size using ZIP EV stock share × LIPA total stock.

    This avoids scanning all-vehicle VINs at the ZIP level while still tying M_z to
    an observable scale (vehicles). It assumes the ZIP's share of EV stock is a
    reasonable proxy for its share of total vehicles in LIPA.
    """
    z = zip_panel.copy().sort_values("date")
    l = lipa_panel.copy().sort_values("date")
    z["date"] = pd.to_datetime(z["date"])
    l["date"] = pd.to_datetime(l["date"])

    # ZIP EV stock at scale_date (as-of).
    z_row = pd.merge_asof(
        pd.DataFrame({"date": [scale_date]}),
        z[["date", "stock_ev_t"]].dropna(subset=["stock_ev_t"]),
        on="date",
        direction="backward",
    )
    if z_row.empty or not np.isfinite(z_row["stock_ev_t"].iloc[0]):
        return None
    zip_ev = float(z_row["stock_ev_t"].iloc[0])

    # LIPA EV stock + total stock at scale_date (as-of).
    need_cols = {"stock_ev_t", "stock_all_t"}
    if not need_cols.issubset(set(l.columns)):
        return None
    l_row = pd.merge_asof(
        pd.DataFrame({"date": [scale_date]}),
        l[["date", "stock_ev_t", "stock_all_t"]].dropna(subset=["stock_ev_t", "stock_all_t"]),
        on="date",
        direction="backward",
    )
    if l_row.empty:
        return None
    lipa_ev = float(l_row["stock_ev_t"].iloc[0])
    lipa_all = float(l_row["stock_all_t"].iloc[0])
    if not np.isfinite(lipa_ev) or lipa_ev <= 0 or not np.isfinite(lipa_all) or lipa_all <= 0:
        return None

    share = zip_ev / lipa_ev
    if not np.isfinite(share) or share <= 0:
        return None
    return float(share * lipa_all)


def fit_bass_baseline_hierarchical(
    panel: pd.DataFrame,
    *,
    p_bar: float,
    q_bar: float,
    M_prior: float,
    M_cap: float | None = None,
    lambda_p: float,
    lambda_q: float,
    lambda_M: float,
    fit_mode: str = "simulate",
) -> bass.BassParams:
    """Penalized least-squares Bass fit for one ZIP.

    Objective (scaled, relative-deviation form):
      SSE(flow) + scale * [ λp*((p-p̄)/p̄)^2 + λq*((q-q̄)/q̄)^2 + λM*(log M - log M_prior)^2 ]
    """
    df = panel.copy().sort_values("date").reset_index(drop=True)
    A_prev = df["adopt_ev_cum_prev"].values.astype(float)
    n = df["adopt_ev_t"].values.astype(float)
    if len(n) < 5:
        raise ValueError("Not enough observations to fit a ZIP Bass model (need >= 5 rows)")

    p_bar = float(p_bar)
    q_bar = float(q_bar)
    if not np.isfinite(p_bar) or p_bar <= 0:
        p_bar = 1e-4
    if not np.isfinite(q_bar) or q_bar <= 0:
        q_bar = 0.03

    A_max = float(np.nanmax(df["adopt_ev_cum_t"].values.astype(float)))
    if not np.isfinite(M_prior) or M_prior <= 1.01 * A_max:
        M_prior = 1.1 * A_max

    scale = float(np.mean(np.square(n))) * float(len(n))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    log_p_bar = float(np.log(p_bar))
    log_q_bar = float(np.log(q_bar))
    log_M_prior = float(np.log(M_prior))
    log_M_cap = float(np.log(M_cap)) if (M_cap is not None and np.isfinite(M_cap) and M_cap > 0) else None

    def simulate_flow(p: float, q: float, M: float) -> np.ndarray:
        A_sim = float(A_prev[0])
        out = np.zeros_like(n, dtype=float)
        for t in range(len(out)):
            if A_sim >= M:
                out[t] = 0.0
                continue
            out[t] = float((p + q * (A_sim / M)) * (M - A_sim))
            A_sim += float(out[t])
        return out

    def one_step_flow(p: float, q: float, M: float) -> np.ndarray:
        return (p + q * (A_prev / M)) * (M - A_prev)

    def objective(theta: np.ndarray) -> float:
        logp, logq, logM = map(float, theta)
        p = float(np.exp(logp))
        q = float(np.exp(logq))
        M = float(np.exp(logM))
        if M <= A_max * 1.001:
            return 1e50

        if fit_mode == "one_step":
            n_hat = one_step_flow(p, q, M)
        elif fit_mode == "simulate":
            n_hat = simulate_flow(p, q, M)
        else:
            raise ValueError(f"Unknown fit_mode: {fit_mode} (expected 'one_step' or 'simulate')")

        err = n - n_hat
        sse = float(np.sum(np.square(err[np.isfinite(err)])))

        pen_p = float(((p - p_bar) / p_bar) ** 2)
        pen_q = float(((q - q_bar) / q_bar) ** 2)
        pen_M = float((np.log(M) - log_M_prior) ** 2)
        penalty = scale * (
            float(lambda_p) * pen_p + float(lambda_q) * pen_q + float(lambda_M) * pen_M
        )
        return sse + penalty

    # Initialize near global values, but M at prior scale.
    x0 = np.array([log_p_bar, log_q_bar, log_M_prior], dtype=float)

    # Bounds: p,q positive and within reasonable ranges; M bounded above to avoid runaway fits.
    logM_lo = float(np.log(max(1.01 * A_max, 2.0)))
    if log_M_cap is not None:
        logM_hi = float(log_M_cap)
    else:
        logM_hi = float(np.log(max(M_prior * 5.0, A_max * 20.0)))
    bounds = [(-18.0, -2.0), (-12.0, 1.0), (logM_lo, logM_hi)]

    res = minimize(objective, x0=x0, bounds=bounds)
    logp_hat, logq_hat, logM_hat = map(float, res.x)
    return bass.BassParams(p=float(np.exp(logp_hat)), q=float(np.exp(logq_hat)), M=float(np.exp(logM_hat)))


def build_zip_snapshot_panel(
    *,
    zip_code: str,
    input_paths: Iterable[Path],
    ev_vin_keys: set[str],
    snapshot_map: pd.DataFrame,
    log_every: int = 2_000_000,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build (DMV_ID,date,stock_ev_t,flow_ev_t) for one ZIP by streaming splits."""
    target_zip = normalize_zip(zip_code)

    # ZIP-level EV stock: per DMV_ID set of VINs (exact unique VINs per snapshot).
    ev_stock_by_id: Dict[int, set[str]] = {}

    # ZIP-level adoption flow: first-seen snapshot per EV VIN within this ZIP.
    first_seen_ev: Dict[str, int] = {}

    counters: Dict[str, int] = {
        "rows_total": 0,
        "rows_bad": 0,
        "rows_zip_match": 0,
        "rows_ev": 0,
        "unique_ev_vins": 0,
        "unique_ev_vins_by_snapshot_sets": 0,  # sum of per-snapshot set sizes (debug)
    }

    for path in input_paths:
        local_rows = 0
        with open(path, "r", encoding="utf-8", newline="") as f:
            for line in f:
                counters["rows_total"] += 1
                local_rows += 1
                if log_every and (local_rows % log_every == 0):
                    print(f"[{path.name}] scanned {local_rows:,} lines", file=sys.stderr)

                parts = line.rstrip("\n").split(",")
                if len(parts) < 8:
                    counters["rows_bad"] += 1
                    continue

                z = normalize_zip(parts[3])
                if z != target_zip:
                    continue
                counters["rows_zip_match"] += 1

                try:
                    dmv_id = int(parts[5])
                except Exception:
                    counters["rows_bad"] += 1
                    continue

                vin_key = (parts[7] or "").strip()
                if vin_key not in ev_vin_keys:
                    continue

                vin = (parts[0] or "").strip()
                if not vin:
                    counters["rows_bad"] += 1
                    continue
                vin = sys.intern(vin)

                counters["rows_ev"] += 1

                # Update first-seen within ZIP.
                prev = first_seen_ev.get(vin)
                if prev is None or dmv_id < prev:
                    first_seen_ev[vin] = dmv_id

                # Update snapshot stock (unique VINs in this snapshot).
                s = ev_stock_by_id.get(dmv_id)
                if s is None:
                    s = set()
                    ev_stock_by_id[dmv_id] = s
                s.add(vin)

    counters["unique_ev_vins"] = len(first_seen_ev)
    counters["unique_ev_vins_by_snapshot_sets"] = sum(len(s) for s in ev_stock_by_id.values())

    flows = Counter(first_seen_ev.values())

    # Build a full snapshot-indexed panel (missing snapshots => 0 stock/flow).
    rows = []
    for dmv_id, date in snapshot_map[["DMV_ID", "date"]].itertuples(index=False):
        rows.append(
            {
                "DMV_ID": int(dmv_id),
                "date": pd.Timestamp(date),
                "stock_ev_t": float(len(ev_stock_by_id.get(int(dmv_id), set()))),
                "stock_all_t": np.nan,  # optional; not computed here
                "flow_ev_t": float(flows.get(int(dmv_id), 0)),
            }
        )
    panel = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return panel, counters


def main():
    ap = argparse.ArgumentParser(description="ZIP-level Bass diffusion fit (EV adoption)")
    ap.add_argument("--zip", type=str, default="11746", help="Target ZIP code (default: 11746)")
    ap.add_argument(
        "--reuse-panel",
        action="store_true",
        help=(
            "Skip streaming the large split CSVs and reuse the previously extracted ZIP panel at "
            "covariates/panel_zip<ZIP>.csv (must exist). Useful for re-fitting with different objectives."
        ),
    )
    ap.add_argument(
        "--inputs-glob",
        type=str,
        default="split_part_*.csv",
        help="Glob for the split NYSERDA registration CSVs (relative to repo root).",
    )
    ap.add_argument(
        "--descriptions",
        type=str,
        default="Vehicle Descriptions.csv",
        help="Path to Vehicle Descriptions.csv (VIN_Key -> Drivetrain_Type).",
    )
    ap.add_argument(
        "--snapshot-map",
        type=str,
        default="NY_DMV_Snapshots.csv",
        help="Path to NY_DMV_Snapshots.csv (DMV_ID -> snapshot date).",
    )
    ap.add_argument(
        "--flow-likelihood",
        type=str,
        default="mse",
        choices=["poisson", "mse"],
        help="Objective for fitting flow counts (default: mse).",
    )
    ap.add_argument(
        "--fit-mode",
        type=str,
        default="one_step",
        choices=["one_step", "simulate"],
        help=(
            "How to fit Bass parameters on the training data. "
            "'one_step' uses observed adoption state A_{t-1}; "
            "'simulate' fits by forward-simulating A_t, which can reduce exaggerated holdout forecasts."
        ),
    )
    ap.add_argument(
        "--hierarchical",
        action="store_true",
        help=(
            "Also fit a hierarchical (partial-pooling) ZIP Bass baseline via penalized least squares: "
            "shrink (p,q) toward a global (p̄,q̄) and log(M) toward an observable-scale prior."
        ),
    )
    ap.add_argument(
        "--global-params",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with global 'p' and 'q' used to anchor the hierarchical ZIP fit. "
            "Defaults to the latest LIPA baseline params if present in models/."
        ),
    )
    ap.add_argument(
        "--lipa-panel",
        type=str,
        default=None,
        help=(
            "Path to a LIPA panel CSV (must include date, stock_ev_t, stock_all_t) used to estimate ZIP "
            "total market size for the hierarchical M prior. Defaults to the latest LIPA panel in covariates/."
        ),
    )
    ap.add_argument(
        "--market-potential-frac",
        type=float,
        default=0.5,
        help=(
            "Hierarchical M prior is set to this fraction of the estimated ZIP total vehicle stock "
            "(ZIP EV stock share × LIPA total stock). Default: 0.5."
        ),
    )
    ap.add_argument("--lambda-p", type=float, default=1.0, help="Hierarchical shrink penalty for p (default: 1.0)")
    ap.add_argument("--lambda-q", type=float, default=1.0, help="Hierarchical shrink penalty for q (default: 1.0)")
    ap.add_argument("--lambda-M", type=float, default=1.0, help="Hierarchical shrink penalty for log(M) (default: 1.0)")
    ap.add_argument(
        "--resample-monthly",
        action="store_true",
        help="Resample irregular snapshot rows into a regular month-start panel (smooth pre-2018).",
    )
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--min-date", type=str, default=None)
    ap.add_argument("--holdout-start", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=2_000_000, help="Progress log frequency (lines). 0 disables.")
    ap.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional tag appended to output filenames (in addition to the zip code).",
    )
    args = ap.parse_args()

    zip_code = normalize_zip(args.zip)
    tag_bits = [f"zip{zip_code}"]
    if args.output_tag:
        tag_bits.append(str(args.output_tag).strip())
    suffix = "_" + "_".join([t for t in tag_bits if t])

    def out_name(stem: str, ext: str) -> str:
        return f"{stem}{suffix}.{ext}"

    desc_path = (ROOT / args.descriptions) if not Path(args.descriptions).is_absolute() else Path(args.descriptions)
    snap_path = (ROOT / args.snapshot_map) if not Path(args.snapshot_map).is_absolute() else Path(args.snapshot_map)
    snapshot_map = load_snapshot_map(snap_path)

    counters: Dict[str, int]
    if args.reuse_panel:
        cached_panel = ROOT / "covariates" / f"panel_zip{zip_code}.csv"
        if not cached_panel.exists():
            raise FileNotFoundError(
                f"--reuse-panel requested but cached panel not found: {cached_panel}"
            )
        panel = pd.read_csv(cached_panel)
        if "date" not in panel.columns:
            raise ValueError(f"Cached panel missing required column 'date': {cached_panel}")
        panel["date"] = pd.to_datetime(panel["date"])
        counters = {"reused_panel": 1}
    else:
        input_paths = iter_input_paths(args.inputs_glob)
        ev_vin_keys = load_ev_vin_keys(desc_path)
        panel, counters = build_zip_snapshot_panel(
            zip_code=zip_code,
            input_paths=input_paths,
            ev_vin_keys=ev_vin_keys,
            snapshot_map=snapshot_map,
            log_every=int(args.log_every),
        )

    if args.resample_monthly and not args.reuse_panel:
        panel = bass.resample_snapshot_panel_to_monthly(panel)

    # If we are working with a month-start panel (resampled), drop the final partial month
    # to avoid an artificial last-point drop in flows.
    panel = drop_partial_last_month_from_monthly_panel(panel, snapshot_map=snapshot_map)

    panel = panel.sort_values("date").reset_index(drop=True)

    # Adoption process
    panel["adopt_ev_t"] = pd.to_numeric(panel["flow_ev_t"], errors="coerce").fillna(0.0).astype(float)
    panel["adopt_ev_cum_t"] = panel["adopt_ev_t"].cumsum()
    panel["adopt_ev_cum_prev"] = panel["adopt_ev_cum_t"].shift(1).fillna(0.0)
    panel = panel[panel["adopt_ev_cum_t"] > 0].reset_index(drop=True)

    min_ts = pd.to_datetime(args.min_date) if args.min_date else None
    if min_ts is not None:
        panel_fit = panel[panel["date"] >= min_ts].copy()
        if panel_fit.empty:
            raise ValueError(f"--min-date {args.min_date} filters the panel to zero rows")
    else:
        panel_fit = panel

    holdout_ts = pd.to_datetime(args.holdout_start) if args.holdout_start else None

    # Retention curve: use provided file if available; else estimate exponential from aggregates.
    retention_path = ROOT / "covariates" / "retention_LIPA_ev_km.csv"
    if retention_path.exists():
        survival_by_lag, _max_lag = bass.load_retention_curve(retention_path, region="LIPA")
        print(f"Loaded retention curve: {retention_path} (max_lag={_max_lag})")
    else:
        max_date_needed = pd.to_datetime(panel["date"]).max() + pd.DateOffset(months=args.horizon)
        max_lag_needed = bass.month_index(max_date_needed) - bass.month_index(pd.to_datetime(panel["date"]).min())
        survival_by_lag, lambda_hat = bass.estimate_exponential_retention_from_aggregates(
            dates=pd.to_datetime(panel["date"]).to_numpy(),
            adoption_flows=panel["flow_ev_t"].to_numpy(float),
            stock_obs=panel["stock_ev_t"].to_numpy(float),
            eval_start=min_ts,
            max_lag_needed=int(max_lag_needed),
        )
        print(
            f"Warning: retention curve not found at {retention_path}; "
            f"using exponential retention fitted from aggregates: lambda={lambda_hat:.4f} (S(1)={float(np.exp(-lambda_hat)):.4f})."
        )

    hier_params = None
    hier_meta: dict | None = None

    # Fit models
    if holdout_ts is None:
        bass_params = bass.fit_bass_baseline(
            panel_fit, flow_likelihood=args.flow_likelihood, fit_mode=args.fit_mode
        )

        forecast_df = bass.forecast_fullsample_with_retention(
            panel,
            p_const=bass_params.p,
            q_const=bass_params.q,
            M=bass_params.M,
            horizon=args.horizon,
            survival_by_lag=survival_by_lag,
        )

        # Optional: hierarchical ZIP baseline (partial pooling).
        if args.hierarchical:
            global_params_path = (
                Path(args.global_params).resolve()
                if args.global_params
                else _default_global_params_path()
            )
            lipa_panel_path = (
                Path(args.lipa_panel).resolve()
                if args.lipa_panel
                else _default_lipa_panel_path()
            )
            if global_params_path is None or not global_params_path.exists():
                raise FileNotFoundError(
                    "--hierarchical requested but global params JSON not found. "
                    "Provide --global-params or ensure a LIPA baseline JSON exists in models/."
                )
            if lipa_panel_path is None or not lipa_panel_path.exists():
                raise FileNotFoundError(
                    "--hierarchical requested but LIPA panel CSV not found. "
                    "Provide --lipa-panel or ensure a LIPA panel exists in covariates/."
                )
            p_bar, q_bar = load_global_pq(global_params_path)
            lipa_panel = pd.read_csv(lipa_panel_path)
            lipa_panel["date"] = pd.to_datetime(lipa_panel["date"])

            scale_date = pd.to_datetime(panel_fit["date"]).max()
            zip_total_market_est = estimate_zip_total_market_from_lipa_scale(
                zip_panel=panel_fit, lipa_panel=lipa_panel, scale_date=scale_date
            )
            M_cap = (
                float(args.market_potential_frac) * float(zip_total_market_est)
                if zip_total_market_est is not None and np.isfinite(zip_total_market_est)
                else None
            )
            A_max = float(np.nanmax(panel_fit["adopt_ev_cum_t"].values.astype(float)))
            M_prior = M_cap if (M_cap is not None and M_cap > A_max * 1.01) else max(1.1 * A_max, A_max + 1.0)

            hier_params = fit_bass_baseline_hierarchical(
                panel_fit,
                p_bar=p_bar,
                q_bar=q_bar,
                M_prior=float(M_prior),
                M_cap=M_cap,
                lambda_p=float(args.lambda_p),
                lambda_q=float(args.lambda_q),
                lambda_M=float(args.lambda_M),
                fit_mode=str(args.fit_mode),
            )
            hier_forecast = bass.forecast_fullsample_with_retention(
                panel,
                p_const=hier_params.p,
                q_const=hier_params.q,
                M=hier_params.M,
                horizon=args.horizon,
                survival_by_lag=survival_by_lag,
            ).rename(
                columns={
                    "stock_ev_t_hat_anchor": "stock_ev_t_hat_hier_anchor",
                    "flow_ev_t_hat_anchor": "flow_ev_t_hat_hier_anchor",
                }
            )
            forecast_df = forecast_df.merge(
                hier_forecast[["date", "stock_ev_t_hat_hier_anchor", "flow_ev_t_hat_hier_anchor"]],
                on="date",
                how="left",
            )

            hier_meta = {
                "zip": zip_code,
                "p": float(hier_params.p),
                "q": float(hier_params.q),
                "M": float(hier_params.M),
                "p_bar": float(p_bar),
                "q_bar": float(q_bar),
                "M_prior": float(M_prior),
                "M_cap": float(M_cap) if M_cap is not None else None,
                "zip_total_market_est": float(zip_total_market_est)
                if zip_total_market_est is not None
                else None,
                "market_potential_frac": float(args.market_potential_frac),
                "lambda_p": float(args.lambda_p),
                "lambda_q": float(args.lambda_q),
                "lambda_M": float(args.lambda_M),
                "fit_mode": str(args.fit_mode),
                "global_params_path": str(global_params_path),
                "lipa_panel_path": str(lipa_panel_path),
                "scale_date": str(pd.Timestamp(scale_date).date()),
            }
    else:
        train_fit = panel_fit[panel_fit["date"] < holdout_ts].copy()
        if train_fit.empty:
            raise ValueError(f"Holdout start {holdout_ts.date()} leaves an empty training set")
        bass_params = bass.fit_bass_baseline(
            train_fit, flow_likelihood=args.flow_likelihood, fit_mode=args.fit_mode
        )

        forecast_df = bass.forecast_holdout_with_retention(
            panel,
            p_const=bass_params.p,
            q_const=bass_params.q,
            M=bass_params.M,
            holdout_start=holdout_ts,
            horizon=args.horizon,
            survival_by_lag=survival_by_lag,
        )

        # Optional: hierarchical ZIP baseline (partial pooling).
        if args.hierarchical:
            global_params_path = (
                Path(args.global_params).resolve()
                if args.global_params
                else _default_global_params_path()
            )
            lipa_panel_path = (
                Path(args.lipa_panel).resolve()
                if args.lipa_panel
                else _default_lipa_panel_path()
            )
            if global_params_path is None or not global_params_path.exists():
                raise FileNotFoundError(
                    "--hierarchical requested but global params JSON not found. "
                    "Provide --global-params or ensure a LIPA baseline JSON exists in models/."
                )
            if lipa_panel_path is None or not lipa_panel_path.exists():
                raise FileNotFoundError(
                    "--hierarchical requested but LIPA panel CSV not found. "
                    "Provide --lipa-panel or ensure a LIPA panel exists in covariates/."
                )
            p_bar, q_bar = load_global_pq(global_params_path)
            lipa_panel = pd.read_csv(lipa_panel_path)
            lipa_panel["date"] = pd.to_datetime(lipa_panel["date"])

            scale_date = pd.to_datetime(train_fit["date"]).max()
            zip_total_market_est = estimate_zip_total_market_from_lipa_scale(
                zip_panel=panel_fit, lipa_panel=lipa_panel, scale_date=scale_date
            )
            M_cap = (
                float(args.market_potential_frac) * float(zip_total_market_est)
                if zip_total_market_est is not None and np.isfinite(zip_total_market_est)
                else None
            )
            A_max = float(np.nanmax(train_fit["adopt_ev_cum_t"].values.astype(float)))
            M_prior = M_cap if (M_cap is not None and M_cap > A_max * 1.01) else max(1.1 * A_max, A_max + 1.0)

            hier_params = fit_bass_baseline_hierarchical(
                train_fit,
                p_bar=p_bar,
                q_bar=q_bar,
                M_prior=float(M_prior),
                M_cap=M_cap,
                lambda_p=float(args.lambda_p),
                lambda_q=float(args.lambda_q),
                lambda_M=float(args.lambda_M),
                fit_mode=str(args.fit_mode),
            )
            hier_forecast = bass.forecast_holdout_with_retention(
                panel,
                p_const=hier_params.p,
                q_const=hier_params.q,
                M=hier_params.M,
                holdout_start=holdout_ts,
                horizon=args.horizon,
                survival_by_lag=survival_by_lag,
            ).rename(
                columns={
                    "stock_ev_t_hat_fit": "stock_ev_t_hat_hier_fit",
                    "flow_ev_t_hat_fit": "flow_ev_t_hat_hier_fit",
                    "stock_ev_t_hat_anchor": "stock_ev_t_hat_hier_anchor",
                    "flow_ev_t_hat_anchor": "flow_ev_t_hat_hier_anchor",
                }
            )
            forecast_df = forecast_df.merge(
                hier_forecast[
                    [
                        "date",
                        "stock_ev_t_hat_hier_fit",
                        "flow_ev_t_hat_hier_fit",
                        "stock_ev_t_hat_hier_anchor",
                        "flow_ev_t_hat_hier_anchor",
                    ]
                ],
                on="date",
                how="left",
            )

            hier_meta = {
                "zip": zip_code,
                "p": float(hier_params.p),
                "q": float(hier_params.q),
                "M": float(hier_params.M),
                "p_bar": float(p_bar),
                "q_bar": float(q_bar),
                "M_prior": float(M_prior),
                "M_cap": float(M_cap) if M_cap is not None else None,
                "zip_total_market_est": float(zip_total_market_est)
                if zip_total_market_est is not None
                else None,
                "market_potential_frac": float(args.market_potential_frac),
                "lambda_p": float(args.lambda_p),
                "lambda_q": float(args.lambda_q),
                "lambda_M": float(args.lambda_M),
                "fit_mode": str(args.fit_mode),
                "global_params_path": str(global_params_path),
                "lipa_panel_path": str(lipa_panel_path),
                "scale_date": str(pd.Timestamp(scale_date).date()),
            }

    # Persist outputs
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    cov_dir = ROOT / "covariates"
    cov_dir.mkdir(exist_ok=True)

    panel_stem = "panel_fit" if args.reuse_panel else "panel"
    panel.to_csv(cov_dir / out_name(panel_stem, "csv"), index=False)
    with open(models_dir / out_name("bass_baseline", "json"), "w", encoding="utf-8") as f:
        json.dump(bass.asdict(bass_params), f, indent=2)
    if args.hierarchical and hier_meta is not None:
        with open(models_dir / out_name("bass_baseline_hierarchical", "json"), "w", encoding="utf-8") as f:
            json.dump(hier_meta, f, indent=2)

    forecast_df.to_csv(models_dir / out_name("bass_forecast", "csv"), index=False)
    with open(models_dir / out_name("zip_extract_counters", "json"), "w", encoding="utf-8") as f:
        json.dump(counters, f, indent=2)

    # Holdout metrics (flows + stock)
    if holdout_ts is not None and "is_train" in forecast_df.columns:
        test_mask_flow = (forecast_df["is_train"] == 0) & (~forecast_df["flow_ev_t_obs"].isna())
        test_mask_stock = (forecast_df["is_train"] == 0) & (~forecast_df["stock_ev_t_obs"].isna())

        def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            m = np.isfinite(y_true) & np.isfinite(y_pred)
            return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))) if m.any() else float("nan")

        def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-9)
            return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100.0) if m.any() else float("nan")

        metrics = {
            "zip": zip_code,
            "holdout_start": str(holdout_ts.date()),
            "n_test_flow": int(test_mask_flow.sum()),
            "n_test_stock": int(test_mask_stock.sum()),
            "models": {},
        }
        candidates = [
            ("baseline", "flow_ev_t_hat_anchor", "stock_ev_t_hat_anchor"),
            ("baseline_hier", "flow_ev_t_hat_hier_anchor", "stock_ev_t_hat_hier_anchor"),
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
                    bass.poisson_nll(obs_flow, pred_flow) if bool(test_mask_flow.any()) else None
                ),
                "rmse_stock": _rmse(obs_stock, pred_stock) if bool(test_mask_stock.any()) else None,
                "mape_stock_pct": _mape(obs_stock, pred_stock) if bool(test_mask_stock.any()) else None,
            }
        with open(models_dir / out_name("bass_holdout_metrics", "json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Plots (best-effort)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        title_suffix = f"ZIP {zip_code}"

        # Stock plot
        fig, ax = plt.subplots(figsize=(10, 5))
        hist = forecast_df[~forecast_df["stock_ev_t_obs"].isna()]
        if holdout_ts is None:
            last_hist_date = hist["date"].max()
            future = forecast_df[forecast_df["date"] >= last_hist_date]
        else:
            train_end_date = forecast_df.loc[forecast_df["is_train"] == 1, "date"].max()
            fit_seg = forecast_df[forecast_df["is_train"] == 1]
            future = forecast_df[forecast_df["date"] >= train_end_date]
        ax.plot(hist["date"], hist["stock_ev_t_obs"], label="Observed stock", color="C0")
        ax.plot(
            future["date"],
            future["stock_ev_t_hat_anchor"],
            label="Bass forecast (baseline)" if holdout_ts is None else "Bass forecast (baseline, holdout)",
            color="C1",
            linestyle="--",
        )
        if holdout_ts is not None and "stock_ev_t_hat_fit" in forecast_df.columns:
            ax.plot(
                fit_seg["date"],
                fit_seg["stock_ev_t_hat_fit"],
                label="Bass fit (baseline)",
                color="C1",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if "stock_ev_t_hat_hier_anchor" in forecast_df.columns:
            ax.plot(
                future["date"],
                future["stock_ev_t_hat_hier_anchor"],
                label="Bass forecast (hierarchical ZIP)",
                color="C4",
                linestyle="--",
                linewidth=1.2,
            )
        if holdout_ts is not None and "stock_ev_t_hat_hier_fit" in forecast_df.columns:
            ax.plot(
                fit_seg["date"],
                fit_seg["stock_ev_t_hat_hier_fit"],
                label="Bass fit (hierarchical ZIP)",
                color="C4",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if holdout_ts is not None:
            ax.axvline(holdout_ts, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(f"EV Stock: {title_suffix}")
        ax.set_xlabel("Snapshot date")
        ax.set_ylabel("EV stock (unique VINs)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        fig.savefig(models_dir / out_name("bass_stock_forecast", "png"), dpi=150)
        plt.close(fig)

        # Flow plot
        fig, ax = plt.subplots(figsize=(10, 5))
        hist = forecast_df[~forecast_df["flow_ev_t_obs"].isna()]
        ax.plot(hist["date"], hist["flow_ev_t_obs"], label="Observed first-seen EVs", color="C0")
        ax.plot(
            forecast_df["date"],
            forecast_df["flow_ev_t_hat_anchor"],
            label="Bass flow (baseline)",
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
        if "flow_ev_t_hat_hier_anchor" in forecast_df.columns:
            ax.plot(
                forecast_df["date"],
                forecast_df["flow_ev_t_hat_hier_anchor"],
                label="Bass flow (hierarchical ZIP)",
                color="C4",
                linestyle="--",
                linewidth=1.2,
            )
        if holdout_ts is not None and "flow_ev_t_hat_hier_fit" in forecast_df.columns:
            ax.plot(
                forecast_df.loc[forecast_df["is_train"] == 1, "date"],
                forecast_df.loc[forecast_df["is_train"] == 1, "flow_ev_t_hat_hier_fit"],
                label="Bass flow (hierarchical ZIP, fit)",
                color="C4",
                linestyle="-",
                alpha=0.6,
                linewidth=1.0,
            )
        if holdout_ts is not None:
            ax.axvline(holdout_ts, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(f"New EVs (first-seen): {title_suffix}")
        ax.set_xlabel("Snapshot date")
        ax.set_ylabel("New EVs per snapshot")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()
        fig.savefig(models_dir / out_name("bass_flow_forecast", "png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("Plotting skipped:", e)

    # Console summary
    if "rows_zip_match" in counters and "unique_ev_vins" in counters:
        print(
            f"ZIP={zip_code} rows_zip_match={counters['rows_zip_match']:,} "
            f"unique_ev_vins={counters['unique_ev_vins']:,}"
        )
    else:
        print(f"ZIP={zip_code} (reused extracted panel)")
    print("Baseline params:", bass_params)
    if args.hierarchical and hier_meta is not None:
        print("Hierarchical baseline params:", hier_params)
    print("Wrote outputs to:", models_dir)


if __name__ == "__main__":
    main()
