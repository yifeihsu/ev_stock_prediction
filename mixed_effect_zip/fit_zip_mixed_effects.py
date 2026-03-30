#!/usr/bin/env python3
"""
Fit one mixed-effects EV adoption model across many ZIP codes.

This script is designed as the next step after the current independent ZIP Bass fits.
It pools information across ZIPs by modeling ZIP-level EV adoption flows as a
hierarchical count process.

Recommended use case
--------------------
Use this on a *stacked ZIP panel* with one row per (zip, date), or on a glob of
cached per-ZIP panels emitted by `build_and_fit_bass_zip.py` (e.g. `covariates/panel_zip*.csv`).

Model
-----
For ZIP z and period t, the pooled model now uses the exact Bass flow equation:

    a_hat[z,t] = (p[z,t] + q[z,t] * A[z,t-1] / M[z]) * (M[z] - A[z,t-1]) * dt_scale[z,t]

    log p[z,t] = alpha_p + u_p[z] + X[z,t] beta_p
    log q[z,t] = alpha_q + u_q[z] + X[z,t] beta_q

    y[z,t] ~ NegativeBinomial(mu=a_hat[z,t], alpha_nb)

where
  - y[z,t] is the first-seen EV count (flow_ev_t)
  - A[z,t-1] is cumulative adopters through t-1
  - M[z] is ZIP market potential / eventual EV-capable market size
  - u_p[z], u_q[z] are ZIP random intercepts on log p / log q
  - X[z,t] are optional covariates such as TCO advantage, policy, or seasonality
  - dt_scale[z,t] is an interval-length adjustment, equal to 1.0 on monthly-resampled panels

This keeps the pooled model structurally aligned with the independent ZIP Bass
baseline: same diffusion state, same recursive simulate vs one_step distinction,
and stock reconstructed afterward from retention.

Input expectations
------------------
Minimum required columns:
  - zip (or inferable from filenames like panel_zip11746.csv)
  - date
  - flow_ev_t
  - stock_ev_t

Optional covariates can already be present. If common covariates are missing, this
script can attach the repo's gas/electricity/TCO series and optional policy series.

Market size (M[z])
------------------
Preferred:
  1) provide `--market-size-csv` with columns [zip, market_size]
  2) or provide `--market-col` in the stacked panel

Fallback:
  If neither is provided, estimate each ZIP's total vehicle stock using the same
  proxy already used by `build_and_fit_bass_zip.py` for hierarchical ZIP Bass:
      ZIP total market ~ (ZIP EV stock / LIPA EV stock) * LIPA total light-duty stock
  then convert it to EV market potential with `--market-potential-frac`.

Outputs
-------
  - models/zip_mixed_effects_forecast_<tag>.csv
  - models/zip_mixed_effects_summary_<tag>.json
  - models/zip_mixed_effects_posterior_<tag>.csv
  - models/zip_mixed_effects_holdout_metrics_<tag>.json (if holdout is requested)

Notes
-----
  - Default fit method is ADVI for speed on large ZIP panels.
  - If PyMC compilation fails on a machine without Python headers, rerun with
        PYTENSOR_FLAGS='linker=py,cxx='
    or use this script's `--disable-c-compiler` flag.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

# Optional runtime fallback for environments without Python headers.
# We only set this when explicitly requested by CLI.

ROOT = Path(__file__).resolve().parents[1]


def normalize_zip(z: object) -> str:
    s = str(z or "").strip()
    m = re.search(r"(\d{5})", s)
    return m.group(1) if m else ""


@dataclass
class Standardization:
    feature_cols: list[str]
    mean: list[float]
    std: list[float]


@dataclass
class ModelConfig:
    feature_cols: list[str]
    seasonality: bool
    resample_monthly: bool
    fit_mode: str
    beta_prior_sd: float
    tco_prior_sd: float
    pre_period_cutoff: str | None
    pre_period_weight: float
    fit_method: str
    draws: int
    tune: int
    chains: int
    target_accept: float
    advi_iters: int
    holdout_start: str | None
    min_date: str | None
    market_potential_frac: float
    min_total_flow: float
    min_obs: int
    family: str


@dataclass
class PosteriorPoint:
    alpha_p: float
    alpha_q: float
    beta_p: np.ndarray
    beta_q: np.ndarray
    zip_re_p: np.ndarray
    zip_re_q: np.ndarray
    alpha_nb: float


def _import_repo_modules(disable_c_compiler: bool = False):
    if disable_c_compiler:
        flags = os.environ.get("PYTENSOR_FLAGS", "")
        extra = "linker=py,cxx="
        if extra not in flags:
            os.environ["PYTENSOR_FLAGS"] = f"{flags},{extra}" if flags else extra

    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        import pymc as pm  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pymc is required for the ZIP mixed-effects model. "
            "Install pymc and arviz in this environment before fitting."
        ) from exc

    try:
        import build_and_fit_bass_lipa as bass  # noqa: WPS433
        import build_and_fit_bass_zip as zipbass  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the existing ZIP/LIPA Bass modules from scripts/. "
            "Run the integrated entry point scripts/fit_zip_mixed_effects.py or ensure "
            "the repository root is on PYTHONPATH."
        ) from exc

    return pm, bass, zipbass


def parse_feature_cols(s: str | None) -> list[str]:
    if s is None:
        return []
    vals = [v.strip() for v in str(s).split(",")]
    return [v for v in vals if v]


def infer_zip_from_path(path: Path) -> str:
    m = re.search(r"zip(\d{5})", path.stem, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _panel_file_priority(path: Path) -> tuple[int, int, str]:
    """Prefer canonical per-ZIP panels over suffixed refit variants.

    Example:
      panel_zip11746.csv      -> preferred
      panel_zip11746_mse.csv  -> lower priority
    """
    stem = path.stem.lower()
    canonical = bool(re.fullmatch(r"panel_zip\d{5}", stem))
    return (0 if canonical else 1, len(stem), path.name)


def load_stacked_panel(panel_csv: Path, *, zip_col: str, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(panel_csv)
    if zip_col not in df.columns:
        raise ValueError(f"{panel_csv} must contain zip column '{zip_col}'")
    if date_col not in df.columns:
        raise ValueError(f"{panel_csv} must contain date column '{date_col}'")
    df = df.copy()
    df[zip_col] = df[zip_col].map(normalize_zip)
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def load_panel_glob(panel_glob: str, *, zip_col: str, date_col: str) -> pd.DataFrame:
    raw_paths = sorted(ROOT.glob(panel_glob))
    if not raw_paths:
        raise FileNotFoundError(f"No panel files matched: {panel_glob}")
    # Deduplicate by ZIP so a broad glob like covariates/panel_zip*.csv does not
    # accidentally stack panel_zip11746.csv and panel_zip11746_mse.csv together.
    selected_by_zip: dict[str, Path] = {}
    passthrough: list[Path] = []
    for path in raw_paths:
        z = infer_zip_from_path(path)
        if not z:
            passthrough.append(path)
            continue
        cur = selected_by_zip.get(z)
        if cur is None or _panel_file_priority(path) < _panel_file_priority(cur):
            selected_by_zip[z] = path
    paths = sorted(passthrough + list(selected_by_zip.values()))
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        if date_col not in df.columns:
            raise ValueError(f"{path} missing required date column '{date_col}'")
        z = infer_zip_from_path(path)
        if zip_col in df.columns:
            df[zip_col] = df[zip_col].map(normalize_zip)
        else:
            if not z:
                raise ValueError(
                    f"Could not infer ZIP from filename {path.name}; add a '{zip_col}' column or use panel_zip<ZIP>.csv naming."
                )
            df[zip_col] = z
        df[date_col] = pd.to_datetime(df[date_col])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out[zip_col] = out[zip_col].map(normalize_zip)
    return out


def resample_monthly_by_zip(
    panel: pd.DataFrame,
    *,
    bass,
    zip_col: str,
    date_col: str,
    flow_col: str,
    stock_col: str,
) -> pd.DataFrame:
    """Apply the repository's monthly resampling logic ZIP by ZIP.

    This keeps the mixed-effects workflow aligned with the independent ZIP Bass script:
    irregular pre-2018 snapshots are converted to month-start rows with explicit zero-flow
    months, while stock is only observed in months containing a snapshot.
    """
    frames: list[pd.DataFrame] = []
    for z, g in panel.groupby(zip_col, sort=False):
        gg = g.copy().sort_values(date_col).reset_index(drop=True)
        cols = []
        if "DMV_ID" in gg.columns:
            cols.append("DMV_ID")
        cols.extend([date_col, stock_col, flow_col])
        work = gg[cols].rename(columns={date_col: "date", stock_col: "stock_ev_t", flow_col: "flow_ev_t"})
        if "stock_all_t" in gg.columns:
            work["stock_all_t"] = pd.to_numeric(gg["stock_all_t"], errors="coerce")
        else:
            work["stock_all_t"] = np.nan
        if "DMV_ID" not in work.columns:
            work["DMV_ID"] = np.nan
        res = bass.resample_snapshot_panel_to_monthly(work)
        res[zip_col] = z
        frames.append(
            res.rename(columns={"date": date_col, "stock_ev_t": stock_col, "flow_ev_t": flow_col})
        )
    out = pd.concat(frames, ignore_index=True)
    out[date_col] = pd.to_datetime(out[date_col])
    return out


def attach_common_covariates(
    panel: pd.DataFrame,
    *,
    bass,
    elec_price_default: float,
    gas_series_path: str | None,
    elec_series_path: str | None,
    elec_rate_class: str,
    elec_utility: str,
    with_policy: bool,
) -> pd.DataFrame:
    out = panel.copy().sort_values("date").reset_index(drop=True)

    need_tco = any(c not in out.columns for c in ["gas_price_t", "elec_price_t", "tco_adv_t", "C_gas_t", "C_ev_t"])
    if need_tco:
        out = bass.attach_gas_price(out, gas_series_path=gas_series_path)
        out = bass.attach_elec_price(
            out,
            elec_price_default=elec_price_default,
            elec_series_path=elec_series_path,
            elec_rate_class=elec_rate_class,
            elec_utility=elec_utility,
        )
        MPG_AVG = 28.0
        KWH_PER_MI_AVG = 0.30
        out["C_gas_t"] = out["gas_price_t"] / MPG_AVG
        out["C_ev_t"] = out["elec_price_t"] * KWH_PER_MI_AVG
        out["tco_adv_t"] = out["C_gas_t"] - out["C_ev_t"]

    if with_policy and "subsidy_share_t" not in out.columns:
        out = bass.attach_policy(out)

    return out


def build_market_size_map(
    panel: pd.DataFrame,
    *,
    zip_col: str,
    market_col: str | None,
    market_size_csv: Path | None,
    stock_col: str,
    lipa_panel_path: Path | None,
    market_potential_frac: float,
    zipbass,
) -> dict[str, float]:
    if market_potential_frac <= 0:
        raise ValueError("--market-potential-frac must be > 0")

    out: dict[str, float] = {}

    if market_size_csv is not None:
        df = pd.read_csv(market_size_csv)
        required = {zip_col, "market_size"}
        if not required.issubset(df.columns):
            raise ValueError(f"{market_size_csv} must contain columns {sorted(required)}")
        df = df[[zip_col, "market_size"]].copy()
        df[zip_col] = df[zip_col].map(normalize_zip)
        df["market_size"] = pd.to_numeric(df["market_size"], errors="coerce")
        df = df.dropna(subset=[zip_col, "market_size"])
        out.update({str(z): float(m) for z, m in df.itertuples(index=False) if float(m) > 0})

    if market_col and market_col in panel.columns:
        tmp = panel[[zip_col, market_col]].copy()
        tmp[market_col] = pd.to_numeric(tmp[market_col], errors="coerce")
        tmp = tmp.dropna(subset=[market_col]).groupby(zip_col, as_index=False)[market_col].last()
        for z, m in tmp.itertuples(index=False):
            if z and np.isfinite(m) and float(m) > 0:
                out[str(z)] = float(m)

    need = [z for z in sorted(panel[zip_col].dropna().unique()) if z not in out]
    if not need:
        return out

    if lipa_panel_path is None:
        lipa_panel_path = zipbass._default_lipa_panel_path()  # type: ignore[attr-defined]
    if lipa_panel_path is None or not Path(lipa_panel_path).exists():
        missing = ", ".join(need[:10]) + ("..." if len(need) > 10 else "")
        raise FileNotFoundError(
            "Missing market sizes for ZIPs and no LIPA panel available for fallback estimation. "
            f"Provide --market-size-csv or --market-col. Missing ZIPs include: {missing}"
        )

    lipa_panel = pd.read_csv(lipa_panel_path)
    lipa_panel["date"] = pd.to_datetime(lipa_panel["date"])

    for z in need:
        zp = panel.loc[panel[zip_col] == z, ["date", stock_col]].copy()
        if zp.empty:
            continue
        zp[stock_col] = pd.to_numeric(zp[stock_col], errors="coerce")
        zp = zp.dropna(subset=[stock_col]).sort_values("date")
        if zp.empty:
            continue
        scale_date = pd.Timestamp(zp["date"].max())
        # Reuse the same scaling idea already present in build_and_fit_bass_zip.py
        total_vehicle_market = zipbass.estimate_zip_total_market_from_lipa_scale(  # type: ignore[attr-defined]
            zip_panel=zp.rename(columns={stock_col: "stock_ev_t"}),
            lipa_panel=lipa_panel,
            scale_date=scale_date,
        )
        if total_vehicle_market is not None and np.isfinite(total_vehicle_market) and float(total_vehicle_market) > 0:
            out[z] = float(total_vehicle_market) * float(market_potential_frac)

    return out


def add_seasonality_columns(df: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    out = df.copy()
    month = pd.to_datetime(out[date_col]).dt.month.astype(int)
    angle = 2.0 * np.pi * (month - 1) / 12.0
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    return out


def prepare_panel(
    panel: pd.DataFrame,
    *,
    zip_col: str,
    date_col: str,
    flow_col: str,
    stock_col: str,
    feature_cols: Sequence[str],
    market_size_map: dict[str, float],
    min_date: pd.Timestamp | None,
    holdout_start: pd.Timestamp | None,
    min_total_flow: float,
    min_obs: int,
    seasonality: bool,
    resample_monthly: bool,
    pre_period_cutoff: pd.Timestamp | None,
    pre_period_weight: float,
) -> tuple[pd.DataFrame, Standardization, list[str]]:
    out = panel.copy()
    out[zip_col] = out[zip_col].map(normalize_zip)
    out[date_col] = pd.to_datetime(out[date_col])
    out[flow_col] = pd.to_numeric(out[flow_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out[stock_col] = pd.to_numeric(out[stock_col], errors="coerce")
    out = out.sort_values([zip_col, date_col]).reset_index(drop=True)

    # Carry in the diffusion state from the full ZIP history before min_date filtering.
    out["adopt_ev_t_full"] = out[flow_col].astype(float)
    out["adopt_ev_cum_t_full"] = out.groupby(zip_col)["adopt_ev_t_full"].cumsum()
    out["adopt_ev_cum_prev_full"] = out.groupby(zip_col)["adopt_ev_cum_t_full"].shift(1).fillna(0.0)

    if min_date is not None:
        out = out[out[date_col] >= min_date].copy()
    out = out.sort_values([zip_col, date_col]).reset_index(drop=True)
    out["sample_weight"] = 1.0
    if pre_period_cutoff is not None:
        out.loc[out[date_col] < pre_period_cutoff, "sample_weight"] = float(pre_period_weight)

    if seasonality:
        out = add_seasonality_columns(out, date_col=date_col)

    # Keep only ZIPs with enough history and some signal.
    zip_summary = (
        out.groupby(zip_col)
        .agg(n_obs=(date_col, "size"), total_flow=(flow_col, "sum"))
        .reset_index()
    )
    keep_zips = set(
        zip_summary.loc[
            (zip_summary["n_obs"] >= int(min_obs)) & (zip_summary["total_flow"] >= float(min_total_flow)),
            zip_col,
        ].astype(str)
    )
    out = out[out[zip_col].isin(keep_zips)].copy()
    if out.empty:
        raise ValueError(
            "No ZIPs remain after filters. Reduce --min-obs or --min-total-flow."
        )
    n_zip = int(out[zip_col].nunique())
    if n_zip < 2:
        raise ValueError(
            "Mixed-effects fitting requires at least 2 ZIPs after filtering. "
            "Build more panel_zip<ZIP>.csv files or lower the ZIP filters."
        )

    out["market_size"] = out[zip_col].map(market_size_map).astype(float)
    # Backstop: ensure M[z] always exceeds observed cumulative adoptions.
    max_cum = out.groupby(zip_col)["adopt_ev_cum_t_full"].transform("max")
    out["market_size"] = np.maximum(out["market_size"], 1.05 * max_cum)

    # Adoption state by ZIP.
    out["adopt_ev_t"] = out[flow_col].astype(float)
    out["adopt_ev_cum_t"] = out["adopt_ev_cum_t_full"].astype(float)
    out["adopt_ev_cum_prev"] = out["adopt_ev_cum_prev_full"].astype(float)
    out["remaining_market"] = (out["market_size"] - out["adopt_ev_cum_prev"]).clip(lower=1e-6)
    out["imitation_share"] = (out["adopt_ev_cum_prev"] / out["market_size"]).clip(lower=0.0, upper=0.999999)

    # Interval-length adjustment is only useful when the panel keeps irregular
    # snapshot spacing. For monthly-resampled panels it just reintroduces
    # calendar-day wiggles, so keep the monthly step size fixed at 1.
    dt_days = out.groupby(zip_col)[date_col].diff().dt.days.astype(float)
    fallback_days = np.nanmedian(dt_days.values)
    if not np.isfinite(fallback_days) or fallback_days <= 0:
        fallback_days = 30.4375
    out["interval_days"] = dt_days.fillna(fallback_days).clip(lower=1.0)
    if resample_monthly:
        out["log_interval_scale"] = 0.0
    else:
        out["log_interval_scale"] = np.log(out["interval_days"] / 30.4375)
    out["log_remaining_market"] = np.log(out["remaining_market"])

    feature_cols_final = list(feature_cols)
    if seasonality:
        for c in ["month_sin", "month_cos"]:
            if c not in feature_cols_final:
                feature_cols_final.append(c)

    missing = [c for c in feature_cols_final if c not in out.columns]
    if missing:
        raise ValueError(f"Panel missing required feature columns: {missing}")

    train_mask = (
        out[date_col] < holdout_start if holdout_start is not None else np.ones(len(out), dtype=bool)
    )
    if int(np.sum(train_mask)) == 0:
        raise ValueError("Holdout start leaves zero training rows")

    means: list[float] = []
    stds: list[float] = []
    for c in feature_cols_final:
        vals = pd.to_numeric(out.loc[train_mask, c], errors="coerce").astype(float)
        mu = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
        sd = float(np.nanstd(vals)) if np.isfinite(vals).any() else 1.0
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        means.append(mu)
        stds.append(sd)
        out[f"{c}__z"] = ((pd.to_numeric(out[c], errors="coerce").astype(float) - mu) / sd).astype(float)

    out = out.reset_index(drop=True)
    std = Standardization(feature_cols=feature_cols_final, mean=means, std=stds)
    return out, std, feature_cols_final


def build_indexes(df: pd.DataFrame, *, zip_col: str, date_col: str) -> tuple[pd.DataFrame, list[str], list[pd.Timestamp]]:
    out = df.copy()
    zip_levels = sorted(out[zip_col].astype(str).unique())
    date_levels = sorted(pd.to_datetime(out[date_col]).unique())
    zip_to_i = {z: i for i, z in enumerate(zip_levels)}
    date_to_i = {pd.Timestamp(d): i for i, d in enumerate(date_levels)}
    out["zip_idx"] = out[zip_col].map(zip_to_i).astype(int)
    out["date_idx"] = pd.to_datetime(out[date_col]).map(lambda x: date_to_i[pd.Timestamp(x)]).astype(int)
    return out, zip_levels, [pd.Timestamp(d) for d in date_levels]


def fit_hierarchical_count_model(
    train: pd.DataFrame,
    *,
    feature_cols_final: Sequence[str],
    zip_levels: Sequence[str],
    family: str,
    config: ModelConfig,
    disable_c_compiler: bool,
):
    pm, _bass, _zipbass = _import_repo_modules(disable_c_compiler=disable_c_compiler)
    import pytensor.tensor as pt  # noqa: WPS433
    from pytensor.scan import scan  # noqa: WPS433

    train = train.sort_values(["zip", "date"]).reset_index(drop=True)
    X_cols = [f"{c}__z" for c in feature_cols_final]
    y = np.asarray(np.round(train["adopt_ev_t"].astype(float).to_numpy()), dtype="int64")
    X = train[X_cols].to_numpy(float)
    zip_idx = train["zip_idx"].to_numpy(int)
    market_size = train["market_size"].to_numpy(float)
    A_prev_obs = train["adopt_ev_cum_prev"].to_numpy(float)
    interval_scale = np.exp(train["log_interval_scale"].to_numpy(float))
    sample_weight = train.get("sample_weight", pd.Series(1.0, index=train.index)).to_numpy(float)
    is_new_zip = np.r_[True, zip_idx[1:] != zip_idx[:-1]]
    carry_in = np.where(is_new_zip, A_prev_obs, 0.0).astype(float)

    coords = {
        "obs": np.arange(len(train)),
        "zip": list(zip_levels),
        "feature": list(feature_cols_final),
    }
    beta_sigma = np.full(len(feature_cols_final), float(config.beta_prior_sd), dtype=float)
    if "tco_adv_t" in feature_cols_final:
        beta_sigma[list(feature_cols_final).index("tco_adv_t")] = float(config.tco_prior_sd)

    with pm.Model(coords=coords) as model:
        x_data = pm.Data("X", X, dims=("obs", "feature"))
        zip_data = pm.Data("zip_idx", zip_idx, dims="obs")
        market_size_data = pm.Data("market_size", market_size, dims="obs")
        a_prev_obs_data = pm.Data("adopt_ev_cum_prev", A_prev_obs, dims="obs")
        interval_scale_data = pm.Data("interval_scale", interval_scale, dims="obs")
        weight_data = pm.Data("sample_weight", sample_weight, dims="obs")
        new_zip_data = pm.Data("is_new_zip", is_new_zip.astype("int8"), dims="obs")
        carry_in_data = pm.Data("carry_in", carry_in, dims="obs")

        alpha_p = pm.Normal("alpha_p", mu=-9.5, sigma=2.0)
        alpha_q = pm.Normal("alpha_q", mu=-3.5, sigma=1.5)
        beta_p = pm.Normal("beta_p", mu=0.0, sigma=beta_sigma, dims="feature")
        beta_q = pm.Normal("beta_q", mu=0.0, sigma=beta_sigma, dims="feature")

        sigma_p = pm.HalfNormal("sigma_p", sigma=1.0)
        zip_re_p = pm.Normal("zip_re_p", mu=0.0, sigma=sigma_p, dims="zip")

        sigma_q = pm.HalfNormal("sigma_q", sigma=1.0)
        zip_re_q = pm.Normal("zip_re_q", mu=0.0, sigma=sigma_q, dims="zip")

        log_p_row = alpha_p + zip_re_p[zip_data] + pm.math.dot(x_data, beta_p)
        log_q_row = alpha_q + zip_re_q[zip_data] + pm.math.dot(x_data, beta_q)
        p_row = pm.Deterministic("p_t", pt.exp(log_p_row), dims="obs")
        q_row = pm.Deterministic("q_t", pt.exp(log_q_row), dims="obs")

        if str(config.fit_mode) == "one_step":
            adoption_share = pt.clip(a_prev_obs_data / market_size_data, 0.0, 0.999999)
            remaining_market = pt.maximum(market_size_data - a_prev_obs_data, 1e-6)
            mu = pm.Deterministic(
                "mu",
                (p_row + q_row * adoption_share) * remaining_market * interval_scale_data,
                dims="obs",
            )
        elif str(config.fit_mode) == "simulate":
            def step(
                new_zip_t,
                carry_in_t,
                p_t,
                q_t,
                M_t,
                dt_scale_t,
                prev_a_after,
            ):
                a_prev = pt.switch(pt.neq(new_zip_t, 0), carry_in_t, prev_a_after)
                remaining = pt.maximum(M_t - a_prev, 1e-6)
                adoption_share = pt.clip(a_prev / M_t, 0.0, 0.999999)
                a_hat = (p_t + q_t * adoption_share) * remaining * dt_scale_t
                a_after = a_prev + a_hat
                return a_after, a_hat

            (a_after_seq, mu_seq), _ = scan(
                fn=step,
                sequences=[new_zip_data, carry_in_data, p_row, q_row, market_size_data, interval_scale_data],
                outputs_info=[pt.as_tensor_variable(np.array(0.0, dtype="float64")), None],
            )
            pm.Deterministic("a_after", a_after_seq, dims="obs")
            mu = pm.Deterministic("mu", mu_seq, dims="obs")
        else:
            raise ValueError("fit_mode must be one of {'simulate','one_step'}")

        if family == "poisson":
            obs_dist = pm.Poisson.dist(mu=mu)
            pm.Potential("weighted_logp", pm.math.sum(weight_data * pm.logp(obs_dist, y)))
        elif family == "nb":
            alpha_nb = pm.HalfNormal("alpha_nb", sigma=10.0)
            obs_dist = pm.NegativeBinomial.dist(mu=mu, alpha=alpha_nb)
            pm.Potential("weighted_logp", pm.math.sum(weight_data * pm.logp(obs_dist, y)))
        else:
            raise ValueError(f"Unsupported family: {family}")

        if config.fit_method == "nuts":
            idata = pm.sample(
                draws=int(config.draws),
                tune=int(config.tune),
                chains=int(config.chains),
                target_accept=float(config.target_accept),
                return_inferencedata=True,
                progressbar=True,
                idata_kwargs={"log_likelihood": True},
            )
        elif config.fit_method == "advi":
            approx = pm.fit(int(config.advi_iters), progressbar=True)
            idata = approx.sample(int(config.draws))
        else:
            raise ValueError("fit_method must be one of {'advi','nuts'}")

    return model, idata


def posterior_point_from_idata(idata, *, family: str) -> PosteriorPoint:
    post = idata.posterior
    dims = ("chain", "draw")
    alpha_nb = 1.0
    if family == "nb" and "alpha_nb" in post:
        alpha_nb = float(post["alpha_nb"].mean(dim=dims).values)
    return PosteriorPoint(
        alpha_p=float(post["alpha_p"].mean(dim=dims).values),
        alpha_q=float(post["alpha_q"].mean(dim=dims).values),
        beta_p=np.asarray(post["beta_p"].mean(dim=dims).values, dtype=float),
        beta_q=np.asarray(post["beta_q"].mean(dim=dims).values, dtype=float),
        zip_re_p=np.asarray(post["zip_re_p"].mean(dim=dims).values, dtype=float),
        zip_re_q=np.asarray(post["zip_re_q"].mean(dim=dims).values, dtype=float),
        alpha_nb=float(alpha_nb),
    )


def compute_mu_row(
    row: pd.Series,
    *,
    point: PosteriorPoint,
    feature_cols_final: Sequence[str],
    a_prev_override: float | None = None,
) -> float:
    zip_idx = int(row["zip_idx"])
    x = np.array([float(row[f"{c}__z"]) for c in feature_cols_final], dtype=float)
    M = float(row["market_size"])
    a_prev = float(a_prev_override) if a_prev_override is not None else float(row["adopt_ev_cum_prev"])
    remaining = max(M - a_prev, 1e-6)
    adoption_share = min(max(a_prev / M, 0.0), 0.999999) if M > 0 else 0.0
    interval_scale = float(np.exp(float(row["log_interval_scale"])))
    log_p = float(point.alpha_p) + float(point.zip_re_p[zip_idx]) + float(np.dot(x, point.beta_p))
    log_q = float(point.alpha_q) + float(point.zip_re_q[zip_idx]) + float(np.dot(x, point.beta_q))
    p_t = float(np.exp(log_p))
    q_t = float(np.exp(log_q))
    mu = (p_t + q_t * adoption_share) * remaining * interval_scale
    if not np.isfinite(mu):
        return 0.0
    return max(mu, 0.0)


def posterior_mean_fit(df: pd.DataFrame, *, point: PosteriorPoint, feature_cols_final: Sequence[str]) -> np.ndarray:
    return np.array([compute_mu_row(r, point=point, feature_cols_final=feature_cols_final) for _, r in df.iterrows()], dtype=float)


def extend_future_rows(
    df_zip: pd.DataFrame,
    *,
    horizon: int,
    feature_cols_final: Sequence[str],
    std: Standardization,
    holdout_start: pd.Timestamp | None,
) -> pd.DataFrame:
    out = df_zip.copy().sort_values("date").reset_index(drop=True)
    if horizon <= 0:
        return out

    last_row = out.iloc[-1].copy()
    last_date = pd.Timestamp(last_row["date"])
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    mean_map = dict(zip(std.feature_cols, std.mean))
    std_map = dict(zip(std.feature_cols, std.std))
    rows = []
    for d in future_dates:
        r = last_row.copy()
        r["date"] = pd.Timestamp(d)
        r["adopt_ev_t"] = np.nan
        r["stock_ev_t"] = np.nan
        r["flow_ev_t"] = np.nan
        r["adopt_ev_cum_t"] = np.nan
        r["adopt_ev_cum_prev"] = np.nan
        r["imitation_share"] = np.nan
        r["remaining_market"] = np.nan
        r["log_remaining_market"] = np.nan
        r["interval_days"] = 30.4375
        r["log_interval_scale"] = 0.0
        for c in feature_cols_final:
            # Hold economic/policy covariates flat by default, but recompute
            # deterministic calendar seasonality from the future month.
            if c == "month_sin":
                angle = 2.0 * np.pi * (pd.Timestamp(d).month - 1) / 12.0
                val = float(np.sin(angle))
            elif c == "month_cos":
                angle = 2.0 * np.pi * (pd.Timestamp(d).month - 1) / 12.0
                val = float(np.cos(angle))
            else:
                val = float(last_row[c])
            r[c] = val
            r[f"{c}__z"] = (val - float(mean_map.get(c, 0.0))) / float(std_map.get(c, 1.0))
        rows.append(r)
    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)

    if holdout_start is not None:
        out["is_train"] = (pd.to_datetime(out["date"]) < holdout_start).astype(int)
    else:
        out["is_train"] = (~out["flow_ev_t"].isna()).astype(int)
    return out


def simulate_anchor_forecast_for_zip(
    df_zip: pd.DataFrame,
    *,
    point: PosteriorPoint,
    feature_cols_final: Sequence[str],
    holdout_start: pd.Timestamp | None,
    fit_mode: str,
) -> pd.DataFrame:
    out = df_zip.copy().sort_values("date").reset_index(drop=True)

    obs_mask = out["flow_ev_t"].notna()
    if holdout_start is None:
        forecast_start = int(obs_mask.sum())
    else:
        idx = np.where(pd.to_datetime(out["date"]) >= holdout_start)[0]
        forecast_start = int(idx[0]) if len(idx) else int(obs_mask.sum())

    # One-step fit on observed rows using observed state.
    fit_mask = obs_mask.copy()
    out["flow_ev_t_hat_one_step"] = np.nan
    out.loc[fit_mask, "flow_ev_t_hat_one_step"] = posterior_mean_fit(
        out.loc[fit_mask],
        point=point,
        feature_cols_final=feature_cols_final,
    )

    out["flow_ev_t_hat_fit"] = np.nan
    M = float(out["market_size"].iloc[0])
    carry_in = float(out["adopt_ev_cum_prev"].iloc[0])
    if str(fit_mode) == "simulate":
        A_prev_fit = carry_in
        for i in range(min(forecast_start, len(out))):
            r = out.loc[i].copy()
            remaining = max(M - A_prev_fit, 1e-6)
            imitation = min(max(A_prev_fit / M, 0.0), 0.999999)
            r["adopt_ev_cum_prev"] = A_prev_fit
            r["remaining_market"] = remaining
            r["imitation_share"] = imitation
            r["log_remaining_market"] = np.log(remaining)
            mu = compute_mu_row(r, point=point, feature_cols_final=feature_cols_final, a_prev_override=A_prev_fit)
            out.at[i, "flow_ev_t_hat_fit"] = mu
            A_prev_fit += mu
    else:
        out.loc[fit_mask, "flow_ev_t_hat_fit"] = out.loc[fit_mask, "flow_ev_t_hat_one_step"]

    # Recursive anchored forecast uses actual history before forecast start,
    # then predicted flows afterward.
    out["flow_ev_t_hat_anchor"] = np.nan
    actual_hist = pd.to_numeric(out.loc[: max(forecast_start - 1, -1), "flow_ev_t"], errors="coerce").fillna(0.0)
    A_prev = float(carry_in + actual_hist.sum())

    for i in range(forecast_start, len(out)):
        out.at[i, "adopt_ev_cum_prev"] = A_prev
        out.at[i, "remaining_market"] = max(M - A_prev, 1e-6)
        out.at[i, "imitation_share"] = min(max(A_prev / M, 0.0), 0.999999)
        out.at[i, "log_remaining_market"] = np.log(float(out.at[i, "remaining_market"]))
        mu = compute_mu_row(out.loc[i], point=point, feature_cols_final=feature_cols_final, a_prev_override=A_prev)
        out.at[i, "flow_ev_t_hat_anchor"] = mu
        A_prev += mu

    return out


def build_stock_paths(
    df: pd.DataFrame,
    *,
    bass,
    retention_curve_path: Path | None,
    retention_region: str,
) -> pd.DataFrame:
    out = df.copy().sort_values(["zip", "date"]).reset_index(drop=True)

    survival_by_lag = None
    if retention_curve_path is not None:
        rp = retention_curve_path
        if not rp.is_absolute():
            rp = ROOT / rp
        if rp.exists():
            survival_by_lag, _ = bass.load_retention_curve(rp, region=retention_region)

    if survival_by_lag is None:
        # Fall back to no attrition.
        survival_by_lag = np.ones(1, dtype=float)

    frames: list[pd.DataFrame] = []
    for z, g in out.groupby("zip", sort=False):
        gg = g.copy().sort_values("date")
        dates = pd.to_datetime(gg["date"]).to_numpy()
        actual_flow = pd.to_numeric(gg["flow_ev_t"], errors="coerce").fillna(0.0).to_numpy(float)
        fit_flow = pd.to_numeric(gg["flow_ev_t_hat_fit"], errors="coerce").fillna(0.0).to_numpy(float)
        one_step_flow = pd.to_numeric(gg.get("flow_ev_t_hat_one_step"), errors="coerce").fillna(0.0).to_numpy(float)
        anchor_flow = actual_flow.copy()
        pred_anchor = pd.to_numeric(gg["flow_ev_t_hat_anchor"], errors="coerce")
        replace_mask = pred_anchor.notna().to_numpy()
        anchor_flow[replace_mask] = pred_anchor.to_numpy(float)[replace_mask]

        gg["stock_ev_t_hat_fit"] = bass.stock_from_adoptions(dates, fit_flow, survival_by_lag)
        gg["stock_ev_t_hat_one_step"] = bass.stock_from_adoptions(dates, one_step_flow, survival_by_lag)
        gg["stock_ev_t_hat_anchor"] = bass.stock_from_adoptions(dates, anchor_flow, survival_by_lag)
        frames.append(gg)

    return pd.concat(frames, ignore_index=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    denom = float(np.sum(np.abs(y_true[m])))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(np.abs(y_true[m] - y_pred[m])) / denom * 100.0)


def collect_metrics(df: pd.DataFrame, *, bass, holdout_start: pd.Timestamp) -> dict:
    out: dict = {
        "holdout_start": str(pd.Timestamp(holdout_start).date()),
        "overall": {},
        "by_zip": {},
    }

    test = df[(pd.to_datetime(df["date"]) >= holdout_start) & df["flow_ev_t"].notna()].copy()
    if test.empty:
        return out

    y_flow = pd.to_numeric(test["flow_ev_t"], errors="coerce").to_numpy(float)
    yhat_flow = pd.to_numeric(test["flow_ev_t_hat_anchor"], errors="coerce").to_numpy(float)
    y_stock = pd.to_numeric(test["stock_ev_t"], errors="coerce").to_numpy(float)
    yhat_stock = pd.to_numeric(test["stock_ev_t_hat_anchor"], errors="coerce").to_numpy(float)

    out["overall"] = {
        "n_test_rows": int(len(test)),
        "rmse_flow": rmse(y_flow, yhat_flow),
        "mae_flow": mae(y_flow, yhat_flow),
        "wape_flow_pct": wape(y_flow, yhat_flow),
        "poisson_nll_flow": bass.poisson_nll(y_flow, yhat_flow),
        "rmse_stock": rmse(y_stock, yhat_stock),
        "mae_stock": mae(y_stock, yhat_stock),
        "wape_stock_pct": wape(y_stock, yhat_stock),
    }

    for z, g in test.groupby("zip"):
        yf = pd.to_numeric(g["flow_ev_t"], errors="coerce").to_numpy(float)
        ph = pd.to_numeric(g["flow_ev_t_hat_anchor"], errors="coerce").to_numpy(float)
        ys = pd.to_numeric(g["stock_ev_t"], errors="coerce").to_numpy(float)
        ps = pd.to_numeric(g["stock_ev_t_hat_anchor"], errors="coerce").to_numpy(float)
        out["by_zip"][str(z)] = {
            "n_test_rows": int(len(g)),
            "rmse_flow": rmse(yf, ph),
            "wape_flow_pct": wape(yf, ph),
            "poisson_nll_flow": bass.poisson_nll(yf, ph),
            "rmse_stock": rmse(ys, ps),
            "wape_stock_pct": wape(ys, ps),
        }

    return out


def posterior_summary_table(idata) -> pd.DataFrame:
    import arviz as az  # noqa: WPS433

    vars_keep = [
        v
        for v in ["alpha_p", "alpha_q", "sigma_p", "sigma_q", "alpha_nb", "beta_p", "beta_q"]
        if v in idata.posterior
    ]
    if not vars_keep:
        return pd.DataFrame()
    summ = az.summary(idata, var_names=vars_keep, kind="stats")
    return summ.reset_index().rename(columns={"index": "parameter"})


def main():
    ap = argparse.ArgumentParser(description="ZIP-level mixed-effects EV adoption model")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--panel-csv", type=str, default=None, help="Path to a stacked multi-ZIP panel CSV")
    src.add_argument(
        "--panel-glob",
        type=str,
        default=None,
        help="Glob for cached per-ZIP panels, e.g. 'covariates/panel_zip*.csv'",
    )

    ap.add_argument("--zip-col", type=str, default="zip")
    ap.add_argument("--date-col", type=str, default="date")
    ap.add_argument("--flow-col", type=str, default="flow_ev_t")
    ap.add_argument("--stock-col", type=str, default="stock_ev_t")

    ap.add_argument("--market-col", type=str, default=None, help="Optional row-level market size column")
    ap.add_argument(
        "--market-size-csv",
        type=str,
        default=None,
        help="Optional CSV with columns [zip, market_size]",
    )
    ap.add_argument(
        "--lipa-panel",
        type=str,
        default=None,
        help="Optional LIPA panel used for fallback ZIP market-size estimation",
    )
    ap.add_argument(
        "--market-potential-frac",
        type=float,
        default=0.5,
        help=(
            "When market size is inferred from ZIP EV share × LIPA total stock, convert total vehicle stock "
            "to eventual EV market potential using this fraction (default 0.5)."
        ),
    )

    ap.add_argument(
        "--feature-cols",
        type=str,
        default="",
        help="Comma-separated fixed-effect features. Default: none (seasonality only unless --no-seasonality is set).",
    )
    ap.add_argument("--with-policy", action="store_true", help="Attach subsidy_share_t if missing")
    ap.add_argument("--no-seasonality", action="store_true", help="Disable month_sin/month_cos covariates")
    ap.add_argument(
        "--resample-monthly",
        action="store_true",
        help="Resample irregular snapshot ZIP panels to month-start rows with explicit zero-flow months.",
    )

    ap.add_argument("--min-date", type=str, default="2018-01-01")
    ap.add_argument(
        "--pre-period-cutoff",
        type=str,
        default="2021-01-01",
        help="Training rows before this date receive --pre-period-weight in the likelihood. Default: 2021-01-01.",
    )
    ap.add_argument(
        "--pre-period-weight",
        type=float,
        default=0.35,
        help="Likelihood weight for rows before --pre-period-cutoff. Default: 0.35.",
    )
    ap.add_argument("--holdout-start", type=str, default=None)
    ap.add_argument("--horizon", type=int, default=24, help="Additional months beyond the last observed row")
    ap.add_argument("--min-total-flow", type=float, default=10.0)
    ap.add_argument("--min-obs", type=int, default=24)

    ap.add_argument("--gas-series", type=str, default=None)
    ap.add_argument("--elec-series", type=str, default=None)
    ap.add_argument("--elec-rate-class", type=str, default="Residential")
    ap.add_argument("--elec-utility", type=str, default="LIPA")
    ap.add_argument("--elec-price-default", type=float, default=0.22)

    ap.add_argument("--retention-curve", type=str, default="covariates/retention_LIPA_ev_km.csv")
    ap.add_argument("--retention-region", type=str, default="LIPA")

    ap.add_argument("--family", type=str, default="nb", choices=["nb", "poisson"])
    ap.add_argument(
        "--fit-mode",
        type=str,
        default="simulate",
        choices=["simulate", "one_step"],
        help=(
            "How to fit the pooled generalized-Bass model on the training data. "
            "'simulate' recursively propagates the Bass state; "
            "'one_step' conditions on observed A_{t-1}. Default: simulate."
        ),
    )
    ap.add_argument("--beta-prior-sd", type=float, default=1.0)
    ap.add_argument(
        "--tco-prior-sd",
        type=float,
        default=0.02,
        help="Prior standard deviation for the standardized TCO coefficient (default: 0.02).",
    )
    ap.add_argument("--fit-method", type=str, default="advi", choices=["advi", "nuts"])
    ap.add_argument("--advi-iters", type=int, default=20000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument(
        "--disable-c-compiler",
        action="store_true",
        help="Set PYTENSOR_FLAGS='linker=py,cxx=' for environments without Python headers",
    )

    ap.add_argument("--output-tag", type=str, default="zip_mixed_effects")
    args = ap.parse_args()

    pm, bass, zipbass = _import_repo_modules(disable_c_compiler=bool(args.disable_c_compiler))
    del pm  # imported for side effects / later fit call

    min_date = pd.to_datetime(args.min_date) if args.min_date else None
    pre_period_cutoff = pd.to_datetime(args.pre_period_cutoff) if args.pre_period_cutoff else None
    holdout_start = pd.to_datetime(args.holdout_start) if args.holdout_start else None

    if args.panel_csv:
        panel_path = Path(args.panel_csv)
        if not panel_path.is_absolute():
            panel_path = ROOT / panel_path
        panel = load_stacked_panel(panel_path, zip_col=args.zip_col, date_col=args.date_col)
    else:
        panel = load_panel_glob(args.panel_glob, zip_col=args.zip_col, date_col=args.date_col)

    # Normalize the expected internal names so downstream code stays simple.
    rename_map = {
        args.zip_col: "zip",
        args.date_col: "date",
        args.flow_col: "flow_ev_t",
        args.stock_col: "stock_ev_t",
    }
    panel = panel.rename(columns=rename_map).copy()
    panel["zip"] = panel["zip"].map(normalize_zip)
    panel["date"] = pd.to_datetime(panel["date"])

    if args.resample_monthly:
        panel = resample_monthly_by_zip(
            panel,
            bass=bass,
            zip_col="zip",
            date_col="date",
            flow_col="flow_ev_t",
            stock_col="stock_ev_t",
        )

    panel = attach_common_covariates(
        panel,
        bass=bass,
        elec_price_default=float(args.elec_price_default),
        gas_series_path=args.gas_series,
        elec_series_path=args.elec_series,
        elec_rate_class=args.elec_rate_class,
        elec_utility=args.elec_utility,
        with_policy=bool(args.with_policy),
    )

    feature_cols = parse_feature_cols(args.feature_cols)
    if args.with_policy and "subsidy_share_t" not in feature_cols:
        feature_cols.append("subsidy_share_t")

    market_size_csv = None
    if args.market_size_csv:
        market_size_csv = Path(args.market_size_csv)
        if not market_size_csv.is_absolute():
            market_size_csv = ROOT / market_size_csv

    lipa_panel_path = None
    if args.lipa_panel:
        lipa_panel_path = Path(args.lipa_panel)
        if not lipa_panel_path.is_absolute():
            lipa_panel_path = ROOT / lipa_panel_path

    market_panel = panel[pd.to_datetime(panel["date"]) < holdout_start].copy() if holdout_start is not None else panel
    market_size_map = build_market_size_map(
        market_panel,
        zip_col="zip",
        market_col=args.market_col,
        market_size_csv=market_size_csv,
        stock_col="stock_ev_t",
        lipa_panel_path=lipa_panel_path,
        market_potential_frac=float(args.market_potential_frac),
        zipbass=zipbass,
    )

    prepared, std, feature_cols_final = prepare_panel(
        panel,
        zip_col="zip",
        date_col="date",
        flow_col="flow_ev_t",
        stock_col="stock_ev_t",
        feature_cols=feature_cols,
        market_size_map=market_size_map,
        min_date=min_date,
        holdout_start=holdout_start,
        min_total_flow=float(args.min_total_flow),
        min_obs=int(args.min_obs),
        seasonality=not bool(args.no_seasonality),
        resample_monthly=bool(args.resample_monthly),
        pre_period_cutoff=pre_period_cutoff,
        pre_period_weight=float(args.pre_period_weight),
    )

    prepared, zip_levels, date_levels = build_indexes(prepared, zip_col="zip", date_col="date")
    train = prepared[prepared["date"] < holdout_start].copy() if holdout_start is not None else prepared.copy()

    config = ModelConfig(
        feature_cols=list(feature_cols_final),
        seasonality=not bool(args.no_seasonality),
        resample_monthly=bool(args.resample_monthly),
        fit_mode=str(args.fit_mode),
        beta_prior_sd=float(args.beta_prior_sd),
        tco_prior_sd=float(args.tco_prior_sd),
        pre_period_cutoff=str(args.pre_period_cutoff) if args.pre_period_cutoff else None,
        pre_period_weight=float(args.pre_period_weight),
        fit_method=str(args.fit_method),
        draws=int(args.draws),
        tune=int(args.tune),
        chains=int(args.chains),
        target_accept=float(args.target_accept),
        advi_iters=int(args.advi_iters),
        holdout_start=str(args.holdout_start) if args.holdout_start else None,
        min_date=str(args.min_date) if args.min_date else None,
        market_potential_frac=float(args.market_potential_frac),
        min_total_flow=float(args.min_total_flow),
        min_obs=int(args.min_obs),
        family=str(args.family),
    )

    model, idata = fit_hierarchical_count_model(
        train,
        feature_cols_final=feature_cols_final,
        zip_levels=zip_levels,
        family=str(args.family),
        config=config,
        disable_c_compiler=bool(args.disable_c_compiler),
    )
    del model

    point = posterior_point_from_idata(idata, family=str(args.family))

    # Forecast by ZIP: holdout rows use recursive forecast; future rows hold covariates flat.
    frames: list[pd.DataFrame] = []
    for z, g in prepared.groupby("zip", sort=False):
        gg = g.copy().sort_values("date").reset_index(drop=True)
        gg = extend_future_rows(
            gg,
            horizon=int(args.horizon),
            feature_cols_final=feature_cols_final,
            std=std,
            holdout_start=holdout_start,
        )
        gg["zip_idx"] = int(gg["zip_idx"].iloc[0])
        gg = simulate_anchor_forecast_for_zip(
            gg,
            point=point,
            feature_cols_final=feature_cols_final,
            holdout_start=holdout_start,
            fit_mode=str(args.fit_mode),
        )
        frames.append(gg)
    forecast = pd.concat(frames, ignore_index=True)

    retention_curve_path = Path(args.retention_curve)
    forecast = build_stock_paths(
        forecast,
        bass=bass,
        retention_curve_path=retention_curve_path,
        retention_region=str(args.retention_region),
    )

    # Mark training rows after potential future extension.
    if holdout_start is not None:
        forecast["is_train"] = (pd.to_datetime(forecast["date"]) < holdout_start).astype(int)
    else:
        forecast["is_train"] = forecast["flow_ev_t"].notna().astype(int)

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(args.output_tag).strip())

    forecast_out = models_dir / f"zip_mixed_effects_forecast_{tag}.csv"
    posterior_out = models_dir / f"zip_mixed_effects_posterior_{tag}.csv"
    summary_out = models_dir / f"zip_mixed_effects_summary_{tag}.json"
    forecast.to_csv(forecast_out, index=False)

    posterior_tbl = posterior_summary_table(idata)
    if not posterior_tbl.empty:
        posterior_tbl.to_csv(posterior_out, index=False)

    summary = {
        "n_rows_total": int(len(prepared)),
        "n_rows_train": int(len(train)),
        "n_zip": int(len(zip_levels)),
        "zip_sample": list(zip_levels[:20]),
        "date_min": str(pd.Timestamp(prepared["date"].min()).date()),
        "date_max": str(pd.Timestamp(prepared["date"].max()).date()),
        "feature_cols": list(feature_cols_final),
        "standardization": asdict(std),
        "config": asdict(config),
        "posterior_point": {
            "alpha_p": float(point.alpha_p),
            "alpha_q": float(point.alpha_q),
            "beta_p": {c: float(v) for c, v in zip(feature_cols_final, point.beta_p)},
            "beta_q": {c: float(v) for c, v in zip(feature_cols_final, point.beta_q)},
            "alpha_nb": float(point.alpha_nb),
        },
        "market_size": {
            "n_zip_with_market": int(len(market_size_map)),
            "min": float(np.nanmin(list(market_size_map.values()))) if market_size_map else None,
            "median": float(np.nanmedian(list(market_size_map.values()))) if market_size_map else None,
            "max": float(np.nanmax(list(market_size_map.values()))) if market_size_map else None,
        },
        "outputs": {
            "forecast_csv": str(forecast_out),
            "posterior_csv": str(posterior_out),
        },
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if holdout_start is not None:
        metrics = collect_metrics(forecast, bass=bass, holdout_start=holdout_start)
        metrics_out = models_dir / f"zip_mixed_effects_holdout_metrics_{tag}.json"
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics["overall"], indent=2))
        print(f"Wrote holdout metrics: {metrics_out}")

    print(f"Wrote forecast table: {forecast_out}")
    if posterior_tbl is not None and not posterior_tbl.empty:
        print(f"Wrote posterior summary: {posterior_out}")
    print(f"Wrote run summary: {summary_out}")


if __name__ == "__main__":
    main()
