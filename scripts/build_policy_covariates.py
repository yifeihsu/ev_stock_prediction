#!/usr/bin/env python3
"""
Build Detailed Policy Covariates for Generalized Bass Model (GBM).

Implements manufacturer-specific Federal Credit logic (Tesla/GM phase-outs),
NY State Rebate phases, and Point-of-Sale discount factors.

Outputs:
  - covariates/policy_covariates.csv
"""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_gas_prices(gas_series_path: str | None = None):
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
        df = df[["date", "gas_price_t"]].sort_values("date").set_index("date")
        df = df.resample("D").ffill()
        return df

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
        df = df[["date", "gas_price_t"]].sort_values("date").set_index("date")
        # Resample to daily for stable merge_asof behavior.
        df = df.resample("D").ffill()
        return df

    lipa_gas_path = ROOT / "prices" / "Gasoline Retail Prices LIPA.xlsx"
    try:
        lipa_df = pd.read_excel(lipa_gas_path)
        lipa_df = lipa_df.rename(columns={"Date": "date", "Nassau Average ($/gal)": "gas_price_t"})
        lipa_df["date"] = pd.to_datetime(lipa_df["date"])
        lipa_df = lipa_df.sort_values("date").set_index("date")
        lipa_df = lipa_df.resample('D').ffill()
        return lipa_df
    except Exception as e:
        print(f"Warning: Could not load LIPA gas prices: {e}")
        return pd.DataFrame(columns=["gas_price_t"])

def get_fed_credit_tesla(date):
    """
    Tesla Federal Credit History:
    - Pre-2019: $7500
    - H1 2019: $3750
    - H2 2019: $1875
    - 2020-2022: $0 (Cap hit)
    - 2023-Sep 2025: $7500 (IRA restoration)
    - Post-Sep 2025: $0 (OBBB)
    """
    ts = pd.Timestamp(date)
    if ts < pd.Timestamp("2019-01-01"):
        return 7500.0
    elif ts < pd.Timestamp("2019-07-01"):
        return 3750.0
    elif ts < pd.Timestamp("2020-01-01"):
        return 1875.0
    elif ts < pd.Timestamp("2023-01-01"):
        return 0.0 # Valley of Death
    elif ts <= pd.Timestamp("2025-09-30"):
        return 7500.0 # IRA restored
    else:
        return 0.0 # OBBB Sunset

def get_fed_credit_gm(date):
    """
    GM Federal Credit History:
    - Pre-Apr 2019: $7500
    - Apr-Sep 2019: $3750
    - Oct 2019-Mar 2020: $1875
    - Apr 2020-2022: $0 (Cap hit)
    - 2023-Sep 2025: $7500 (IRA restoration)
    - Post-Sep 2025: $0 (OBBB)
    """
    ts = pd.Timestamp(date)
    if ts < pd.Timestamp("2019-04-01"):
        return 7500.0
    elif ts < pd.Timestamp("2019-10-01"):
        return 3750.0
    elif ts < pd.Timestamp("2020-04-01"):
        return 1875.0
    elif ts < pd.Timestamp("2023-01-01"):
        return 0.0 # Valley of Death
    elif ts <= pd.Timestamp("2025-09-30"):
        return 7500.0
    else:
        return 0.0

def get_fed_credit_other(date):
    """
    Other OEMs (Ford, Hyundai, etc.):
    - Generally kept $7500 until IRA changes.
    - Post-IRA (2023+): Assumed 50% eligibility blend due to sourcing rules.
    """
    ts = pd.Timestamp(date)
    if ts < pd.Timestamp("2023-01-01"):
        return 7500.0
    elif ts <= pd.Timestamp("2025-09-30"):
        # IRA constraints (North American assembly, battery minerals)
        # Assume 70% weighted average eligibility
        return 7500.0 * 0.7 
    else:
        return 0.0

def get_nys_rebate(date):
    """
    NY Drive Clean Rebate:
    - Phase I (Apr 2017 - Jun 2021): Max $2000
    - Phase II (Jul 2021 - Present): Max $2000 (lower caps), Avg ~1500
    """
    ts = pd.Timestamp(date)
    start_date = pd.Timestamp("2017-03-21")
    phase2_date = pd.Timestamp("2021-07-01")
    
    if ts < start_date:
        return 0.0
    elif ts < phase2_date:
        return 2000.0 # Phase I Avg
    else:
        return 1500.0 # Phase II Avg (stricter caps)

def get_pos_discount_factor(date):
    """
    Delta(t): Value of a tax credit.
    - Pre-2024: Deferred (File next year). Value ~ 0.85
    - Post-2024: Point of Sale (Immediate cash). Value ~ 1.0
    """
    if pd.Timestamp(date) < pd.Timestamp("2024-01-01"):
        return 0.85
    else:
        return 1.0

def calculate_urgency(date):
    # OBBB Signing (July 4, 2025) to Sunset (Sept 30, 2025)
    start = pd.Timestamp("2025-07-04")
    end = pd.Timestamp("2025-09-30")
    if date >= start and date <= end:
        return 1.0
    return 0.0

def _parse_lipa_tax_returns(path: Path) -> pd.DataFrame:
    """
    Parse `Tax Return LIPA 2022.xlsx` into a long income-bin table.

    Expected sheet structure (as provided by the user):
      - A header row containing "Number of returns"
      - County name rows (e.g., "Suffolk County") followed by income bracket rows:
          Under $1
          $1 under $10,000
          ...
          $200,000 or more

    Returns columns:
      county, bracket, lo, hi, single, joint, hoh
    where hi is NaN for open-ended top bin.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None)

    header_row = None
    for i, row in raw.iterrows():
        if row.astype(str).str.contains("Number of returns", case=False, na=False).any():
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find header row in {path}")

    header = raw.iloc[header_row].tolist()
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = header
    if len(header) < 5:
        raise ValueError(f"Unexpected header width in {path}: {header}")

    col0, col1, col2, col3, col4 = header[:5]
    data = data.rename(
        columns={
            col0: "County",
            col1: "total",
            col2: "single",
            col3: "joint",
            col4: "hoh",
        }
    ).dropna(how="all")

    def bracket_bounds(label: str) -> tuple[float, float]:
        s = str(label).replace(",", "").strip()
        if s.lower().startswith("under"):
            # "Under $1"
            return 0.0, float(s.split("$", 1)[1])
        if "or more" in s.lower():
            # "$200,000 or more"
            lo = float(s.split("$", 1)[1].split()[0])
            return lo, float("nan")
        if "under" in s.lower():
            # "$1 under $10,000"
            left, right = s.split("under", 1)
            lo = float(left.split("$", 1)[1])
            hi = float(right.split("$", 1)[1])
            return lo, hi
        raise ValueError(f"Unrecognized income bracket label: {label}")

    rows = []
    current_county: str | None = None
    for _, r in data.iterrows():
        label = r.get("County")
        if (
            isinstance(label, str)
            and label.strip().endswith("County")
            and (pd.isna(r.get("total")) or str(r.get("total")).strip().lower() == "nan")
        ):
            current_county = label.strip()
            continue
        if (
            isinstance(label, str)
            and label.strip().endswith("County")
            and not pd.isna(r.get("total"))
        ):
            # Some sheets include a totals row for the county. Keep the county context, but skip totals row.
            current_county = label.strip()
            continue
        if current_county is None:
            continue
        lo, hi = bracket_bounds(label)
        rows.append(
            {
                "county": current_county,
                "bracket": str(label).strip(),
                "lo": float(lo),
                "hi": float(hi),
                "single": float(r.get("single") or 0.0),
                "joint": float(r.get("joint") or 0.0),
                "hoh": float(r.get("hoh") or 0.0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No income bracket rows parsed from {path}")
    return out


def _share_under_threshold(
    bins: pd.DataFrame, thresholds: dict[str, float], pareto_alpha: float
) -> float:
    """
    Estimate the share of filers under a set of filing-status thresholds.

    For closed bins we assume uniform density within the bin.
    For the open-ended top bin (hi is NaN) we assume a Pareto tail with shape `pareto_alpha`
    anchored at the lower bound of the bin.
    """
    eligible = 0.0
    total = 0.0
    for status, thr in thresholds.items():
        if status not in ("single", "joint", "hoh"):
            raise ValueError(f"Unexpected filing status: {status}")
        for _, r in bins.iterrows():
            cnt = float(r[status])
            total += cnt
            lo = float(r["lo"])
            hi = r["hi"]

            if pd.isna(hi):
                # Open-ended: approximate with Pareto tail above lo.
                if thr <= lo:
                    frac = 0.0
                else:
                    frac = 1.0 - (lo / thr) ** pareto_alpha
                    frac = max(0.0, min(1.0, frac))
            else:
                hi = float(hi)
                if thr <= lo:
                    frac = 0.0
                elif thr >= hi:
                    frac = 1.0
                else:
                    frac = (thr - lo) / (hi - lo)
            eligible += cnt * frac
    return eligible / total if total > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Build policy covariates for GBM (LIPA-focused).")
    ap.add_argument(
        "--gas-series",
        type=str,
        default=None,
        help="Optional path to a monthly gas price CSV (date + gas_price_t or gas_price_cents_per_gallon).",
    )
    ap.add_argument(
        "--msrp-series",
        type=str,
        default=None,
        help=(
            "Optional path to an MSRP series CSV (e.g., covariates/msrp_series_LIPA.csv) with columns "
            "'date' and 'avg_msrp_ev_stock_t'. If omitted, the script will auto-detect "
            "covariates/msrp_series_LIPA.csv if present; otherwise it falls back to a constant MSRP."
        ),
    )
    ap.add_argument(
        "--tax-returns",
        type=str,
        default=str(ROOT / "Tax Return LIPA 2022.xlsx"),
        help="Path to LIPA-area tax return summary (used for IRA income-cap eligibility).",
    )
    ap.add_argument(
        "--pareto-alpha",
        type=float,
        default=2.5,
        help="Pareto tail shape for the top income bin when threshold exceeds $200k.",
    )
    ap.add_argument(
        "--post-2023-credit-face",
        type=float,
        default=7500.0,
        help="Face value assumed available after 2023 for qualifying transactions.",
    )
    ap.add_argument(
        "--lease-share",
        type=float,
        default=0.30,
        help="Share of EV transactions treated as leases (modeled via 45W, no household income cap).",
    )
    ap.add_argument(
        "--purchase-used-share",
        type=float,
        default=0.0,
        help="Share of ALL EV transactions that are used-EV purchases (25E).",
    )
    ap.add_argument(
        "--used-credit-face",
        type=float,
        default=4000.0,
        help="Face value assumed for used EV credit (25E) when applicable.",
    )
    ap.add_argument(
        "--veh-elig-share",
        type=float,
        default=1.0,
        help="Share of vehicles eligible for the face credit (MSRP/assembly/battery rules); default=1.0.",
    )
    ap.add_argument(
        "--takeup",
        type=float,
        default=1.0,
        help="Take-up among eligible transactions (nonrefundability / awareness).",
    )
    args = ap.parse_args()

    print("Building Detailed GBM Policy Covariates...")
    tax_path = Path(args.tax_returns)
    if tax_path.exists():
        bins = _parse_lipa_tax_returns(tax_path)
        thr_30d = {"single": 150_000.0, "joint": 300_000.0, "hoh": 225_000.0}
        thr_25e = {"single": 75_000.0, "joint": 150_000.0, "hoh": 112_500.0}
        s_inc_30d = _share_under_threshold(bins, thr_30d, pareto_alpha=args.pareto_alpha)
        s_inc_25e = _share_under_threshold(bins, thr_25e, pareto_alpha=args.pareto_alpha)
        print(
            f"Income eligibility (from {tax_path.name}): s_inc_30D={s_inc_30d:.3f}, s_inc_25E={s_inc_25e:.3f}"
        )
    else:
        # Fall back to a conservative default if the tax file is missing.
        s_inc_30d = 0.85
        s_inc_25e = 0.65
        print(f"Warning: tax return file not found at {tax_path}; using defaults.")
    
    dates = pd.date_range(start="2011-01-01", end="2030-12-31", freq='MS')
    df = pd.DataFrame({"date": dates})
    
    # Gas Prices
    gas_df = load_gas_prices(gas_series_path=args.gas_series)
    if not gas_df.empty:
        df = pd.merge_asof(df, gas_df, on="date", direction="backward")
        df["gas_price_t"] = df["gas_price_t"].bfill().ffill()
    else:
        df["gas_price_t"] = 3.50
        
    # --- Calculate Manufacturer-Specific Credits ---
    df["credit_tsla"] = df["date"].apply(get_fed_credit_tesla)
    df["credit_gm"] = df["date"].apply(get_fed_credit_gm)
    df["credit_other"] = df["date"].apply(get_fed_credit_other)
    
    # --- Weighted Average Federal Credit ---
    # Based on LIPA snapshot 102 (Sep 2025) counts:
    # Tesla ~39%, GM ~4-5%, Other ~56%
    w_tsla, w_gm, w_other = 0.39, 0.05, 0.56
    df["fed_credit_avg_t"] = (w_tsla * df["credit_tsla"] + 
                              w_gm * df["credit_gm"] + 
                              w_other * df["credit_other"])

    # --- Override post-2023 credits with IRA-style income caps (effective subsidy series) ---
    # Assumption requested by the user:
    #   - Full face credit is available after 2023
    #   - Household income caps bind purchases (30D/25E) but not leases (45W pass-through)
    #   - We approximate buyer-income eligibility using 2022 county tax-return distributions (Nassau + Suffolk).
    lease_share = float(args.lease_share)
    if not (0.0 <= lease_share <= 1.0):
        raise ValueError("--lease-share must be between 0 and 1")
    used_share = float(args.purchase_used_share)
    if not (0.0 <= used_share <= 1.0):
        raise ValueError("--purchase-used-share must be between 0 and 1")
    if lease_share + used_share > 1.0:
        raise ValueError("--lease-share + --purchase-used-share must be <= 1")

    used_purchase_share = used_share
    new_purchase_share = 1.0 - lease_share - used_purchase_share

    face_new = float(args.post_2023_credit_face)
    face_lease = float(args.post_2023_credit_face)
    face_used = float(args.used_credit_face)

    veh_elig = float(args.veh_elig_share)
    takeup = float(args.takeup)

    post_2023_mask = df["date"] >= pd.Timestamp("2023-01-01")
    df.loc[post_2023_mask, "fed_credit_avg_t"] = takeup * veh_elig * (
        (lease_share * face_lease)
        + (new_purchase_share * face_new * s_inc_30d)
        + (used_purchase_share * face_used * s_inc_25e)
    )

    # Expose the implied income-eligibility shares for transparency (constant series).
    df["s_inc_30d_t"] = float(s_inc_30d)
    df["s_inc_25e_t"] = float(s_inc_25e)
    df["lease_share_t"] = float(lease_share)
    df["used_purchase_share_t"] = float(used_purchase_share)
    
    # --- NY State Rebate ---
    df["state_rebate_t"] = df["date"].apply(get_nys_rebate)
    
    # --- Net Effective Subsidy ---
    # Subsidy(t) = State + Delta(t) * Fed_Avg
    df["delta_t"] = df["date"].apply(get_pos_discount_factor)
    df["total_subsidy_t"] = df["state_rebate_t"] + (df["delta_t"] * df["fed_credit_avg_t"])
    
    # --- Subsidy Share of Price ---
    # u(t) = Subsidy / Price
    DEFAULT_AVG_MSRP = 45000.0
    msrp_series_path = (
        Path(args.msrp_series)
        if args.msrp_series
        else (ROOT / "covariates" / "msrp_series_LIPA.csv")
    )
    if msrp_series_path.exists():
        msrp = pd.read_csv(msrp_series_path)
        if "date" not in msrp.columns:
            raise ValueError(f"{msrp_series_path} must include a 'date' column")
        if "avg_msrp_ev_stock_t" not in msrp.columns:
            raise ValueError(f"{msrp_series_path} must include 'avg_msrp_ev_stock_t'")
        msrp["date"] = pd.to_datetime(msrp["date"], errors="coerce")
        msrp = msrp.dropna(subset=["date"]).sort_values("date")
        msrp["avg_msrp_ev_stock_t"] = pd.to_numeric(msrp["avg_msrp_ev_stock_t"], errors="coerce")

        df = df.sort_values("date")
        df = pd.merge_asof(
            df,
            msrp[["date", "avg_msrp_ev_stock_t"]],
            on="date",
            direction="backward",
        )
        df["avg_msrp_t"] = (
            df["avg_msrp_ev_stock_t"].ffill().bfill().fillna(DEFAULT_AVG_MSRP)
        )
    else:
        df["avg_msrp_t"] = DEFAULT_AVG_MSRP

    df["subsidy_share_t"] = df["total_subsidy_t"] / df["avg_msrp_t"]
    
    # --- Valley of Death Dummy ---
    # Period where Tesla/GM had $0 credit: ~2020 to 2022
    # This helps explain why adoption might have slowed relative to trend
    df["valley_dummy_t"] = ((df["date"] >= "2020-01-01") & (df["date"] < "2023-01-01")).astype(int)
    
    # --- Urgency & ZEV ---
    df["urgency_t"] = df["date"].apply(calculate_urgency)
    
    # ZEV Index (Simple ramp)
    def get_zev(d):
        if d < pd.Timestamp("2022-12-01"): return 0.0
        elif d < pd.Timestamp("2026-01-01"): return 0.2
        else: return 0.35 + (d.year - 2026) * (0.65/9)
    df["zev_index_t"] = df["date"].apply(get_zev)
    
    # Fed Policy Index (Climate)
    def get_fed_policy_index(d):
        ira_start = pd.Timestamp("2022-08-16")
        obbb_start = pd.Timestamp("2025-01-20")
        if d < ira_start: return 0.1
        elif d < obbb_start: return 0.5
        else: return 0.2
    df["fed_policy_index_t"] = df["date"].apply(get_fed_policy_index)
    
    # TCO
    MPG, KWH_MI, ELEC = 28.0, 0.30, 0.22
    df["tco_adv_t"] = (df["gas_price_t"]/MPG) - (ELEC*KWH_MI)

    # Save
    out_path = ROOT / "covariates" / "policy_covariates.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(
        df[
            [
                "date",
                "fed_credit_avg_t",
                "total_subsidy_t",
                "avg_msrp_t",
                "subsidy_share_t",
                "valley_dummy_t",
            ]
        ]
        .sample(10)
        .sort_values("date")
    )

if __name__ == "__main__":
    main()
