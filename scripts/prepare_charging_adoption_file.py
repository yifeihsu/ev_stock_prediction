#!/usr/bin/env python3
"""Prepare ZIP/ZCTA adoption forecasts for the charging model.

The behavioral charging model expects one row per:

    adoption_scenario, forecast_year, home_zcta

The Albany ZIP Bass forecast is monthly and wide, with one stock column per
scenario. This adapter keeps the latest forecast snapshot in each year, reshapes
the scenario columns to long form, and converts EV stock to a fleet adoption
fraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_COLUMNS = {
    "baseline": "stock_ev_t_hat_baseline",
    "tco": "stock_ev_t_hat_tco",
    "tco_evse": "stock_ev_t_hat_tco_evse",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def normalize_zip(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(5)


def prepare_adoption(
    source: pd.DataFrame,
    *,
    denominator_column: str,
    fallback_denominator_column: str | None,
    clip: bool,
) -> tuple[pd.DataFrame, dict]:
    required = {"zip", "date", "model_period", denominator_column, *SCENARIO_COLUMNS.values()}
    if fallback_denominator_column:
        required.add(fallback_denominator_column)
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Source adoption forecast missing columns: {missing}")

    df = source.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["home_zcta"] = df["zip"].map(normalize_zip).astype("string")
    df["forecast_year"] = df["date"].dt.year.astype(int)

    rows = []
    for scenario, stock_col in SCENARIO_COLUMNS.items():
        part = df[
            [
                "home_zcta",
                "forecast_year",
                "date",
                "model_period",
                stock_col,
                denominator_column,
            ]
            + ([fallback_denominator_column] if fallback_denominator_column else [])
        ].copy()
        part = part.sort_values(["home_zcta", "forecast_year", "date"])
        part = part.groupby(["home_zcta", "forecast_year"], as_index=False).tail(1)
        denom = pd.to_numeric(part[denominator_column], errors="coerce")
        denom_source = pd.Series(denominator_column, index=part.index, dtype="string")
        if fallback_denominator_column:
            fallback = pd.to_numeric(part[fallback_denominator_column], errors="coerce")
            use_fallback = denom.le(0) | denom.isna()
            denom = denom.mask(use_fallback, fallback)
            denom_source = denom_source.mask(use_fallback, fallback_denominator_column)

        stock = pd.to_numeric(part[stock_col], errors="coerce")
        adoption_fraction = stock / denom.replace(0, np.nan)
        invalid_before_clip = adoption_fraction.isna() | ~np.isfinite(adoption_fraction)
        if clip:
            adoption_fraction = adoption_fraction.clip(lower=0.0, upper=1.0)

        out = pd.DataFrame(
            {
                "adoption_scenario": scenario,
                "forecast_year": part["forecast_year"].astype(int),
                "home_zcta": part["home_zcta"],
                "adoption_fraction": adoption_fraction,
                "vehicle_growth_factor": 1.0,
                "source_date": part["date"].dt.strftime("%Y-%m-%d"),
                "source_model_period": part["model_period"],
                "source_stock_ev": stock,
                "source_denominator": denom,
                "source_denominator_column": denom_source,
                "invalid_fraction_before_clip": invalid_before_clip,
            }
        )
        rows.append(out)

    prepared = pd.concat(rows, ignore_index=True)
    prepared = prepared.dropna(subset=["home_zcta", "adoption_fraction"])
    prepared = prepared[prepared["adoption_fraction"].between(0.0, 1.0)]
    prepared = prepared.sort_values(["adoption_scenario", "forecast_year", "home_zcta"])

    key_cols = ["adoption_scenario", "forecast_year", "home_zcta"]
    duplicate_keys = prepared.duplicated(key_cols, keep=False)
    if duplicate_keys.any():
        examples = prepared.loc[duplicate_keys, key_cols].drop_duplicates().head(10)
        raise ValueError(
            "Prepared adoption file has duplicate keys:\n"
            f"{examples.to_string(index=False)}"
        )

    fallback_count = int(
        prepared["source_denominator_column"].eq(fallback_denominator_column).sum()
        if fallback_denominator_column
        else 0
    )
    summary = {
        "rows": int(len(prepared)),
        "scenarios": sorted(prepared["adoption_scenario"].unique().tolist()),
        "forecast_year_min": int(prepared["forecast_year"].min()),
        "forecast_year_max": int(prepared["forecast_year"].max()),
        "home_zcta_count": int(prepared["home_zcta"].nunique()),
        "fallback_denominator_rows": fallback_count,
        "invalid_fraction_rows_before_clip": int(prepared["invalid_fraction_before_clip"].sum()),
        "max_adoption_fraction": float(prepared["adoption_fraction"].max()),
    }
    return prepared, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Albany ZIP adoption forecast to charging-model schema"
    )
    parser.add_argument(
        "--input",
        default="models/adoption_forecast_albany_zip_bass_central_hudson_covariates_snapshot.csv",
    )
    parser.add_argument(
        "--output",
        default="models/adoption_forecast_albany_zip_for_charging.csv",
    )
    parser.add_argument(
        "--denominator-column",
        default="total_vehicle_market_proxy",
        help="Column used to convert EV stock to adoption fraction.",
    )
    parser.add_argument(
        "--fallback-denominator-column",
        default="market_size",
        help="Fallback denominator for ZIPs with zero or missing primary denominator.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip adoption fractions to [0, 1].",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    source = pd.read_csv(input_path)
    prepared, summary = prepare_adoption(
        source,
        denominator_column=args.denominator_column,
        fallback_denominator_column=args.fallback_denominator_column or None,
        clip=not args.no_clip,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps({"output": str(output_path), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
