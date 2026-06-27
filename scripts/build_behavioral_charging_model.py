#!/usr/bin/env python3
"""
Build a light-duty behavioral EV charging model from Albany trip chains.

The model turns household/person trip records into vehicle-day chains, stop-level
charging opportunities, expected charging events, and load curves. It keeps home
geography separate from charging geography: ZIP/ZCTA is used for adoption
attribution, while charging load is located by destination H3, destination ZIP,
or destination point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = {
    "SYN_RECORD_ID",
    "PERSONID",
    "HHS",
    "VEH",
    "HHI",
    "URBRUR",
    "HOMEOWN",
    "TDTRPNUM",
    "STRTTIME",
    "ENDTIME",
    "TRVLCMIN",
    "TRPMILES",
    "DWELTIME",
    "WHYTRP1S",
    "HHSTFIPS",
    "PUMA",
    "CITYTOWN_NAME",
    "HOME_X",
    "HOME_Y",
    "ACTIVITY_X",
    "ACTIVITY_Y",
    "ACTIVITY_TYPE",
    "START_X",
    "START_Y",
    "SHORTEST_TDIST_miles",
    "SHORTEST_PDIST_miles",
}

OLD_TO_CANONICAL_COLUMNS = {
    "GEO_ID": "home_geoid",
    "HOME_CITYTOWN": "CITYTOWN_NAME",
    "SHORTEST_TIME_DISTANCE_MILES": "SHORTEST_TDIST_miles",
    "SHORTEST_PATH_DISTANCE_MILES": "SHORTEST_PDIST_miles",
}

IDENTITY_COLUMNS = ["SYN_RECORD_ID", "PERSONID"]
CHAIN_GROUP_COLUMNS = IDENTITY_COLUMNS + ["HOME_X", "HOME_Y"]

STOP_PRIORITY = {
    "home": 1,
    "work": 2,
    "long_public": 3,
    "civic": 4,
    "quick_public": 5,
    "no_charging": 99,
}

STOP_TO_CHARGER = {
    "home": "home_l2",
    "work": "work_l2",
    "civic": "civic_l2",
    "long_public": "public_l2",
    "quick_public": "dcfc_low",
}

CHARGER_POWER_KW = {
    "home_l2": 7.2,
    "work_l2": 7.2,
    "civic_l2": 7.2,
    "public_l2": 11.2,
    "dcfc_low": 50.0,
    "dcfc_150": 136.5,
    "dcfc_250": 162.5,
    "dcfc_350": 227.5,
    "dcfc_high": 150.0,
}

BASE_AVAILABILITY = {
    "home_sfh_l2": 1.00,
    "home_mfh_l2": 0.50,
    "work_l2": 0.50,
    "civic_l2": 0.50,
    "public_l2": 1.00,
    "dcfc_150": 0.35,
    "dcfc_high": 0.10,
}

SINGLE_FAMILY_HOME_PROP_CLASSES = {210, 215, 240, 241, 250, 260, 270, 283, 483}

AVAILABILITY_MULTIPLIER = {
    "low": 0.65,
    "base": 1.0,
    "high": 1.25,
}

DEFAULT_CHARGER_ASSUMPTIONS = [
    {
        "assumption_key": "home_sfh",
        "charge_location_type": "home",
        "charger_type": "home_l2",
        "rated_power_kw": 7.2,
        "peak_demand_hour_probability": 1.00,
        "managed_eligible_flag": True,
    },
    {
        "assumption_key": "home_mfh",
        "charge_location_type": "home",
        "charger_type": "home_l2",
        "rated_power_kw": 7.2,
        "peak_demand_hour_probability": 0.50,
        "managed_eligible_flag": True,
    },
    {
        "assumption_key": "work",
        "charge_location_type": "work",
        "charger_type": "work_l2",
        "rated_power_kw": 7.2,
        "peak_demand_hour_probability": 0.50,
        "managed_eligible_flag": True,
    },
    {
        "assumption_key": "civic",
        "charge_location_type": "civic",
        "charger_type": "civic_l2",
        "rated_power_kw": 7.2,
        "peak_demand_hour_probability": 0.50,
        "managed_eligible_flag": False,
    },
    {
        "assumption_key": "long_public",
        "charge_location_type": "long_public",
        "charger_type": "public_l2",
        "rated_power_kw": 11.2,
        "peak_demand_hour_probability": 1.00,
        "managed_eligible_flag": False,
    },
    {
        "assumption_key": "quick_public",
        "charge_location_type": "quick_public",
        "charger_type": "dcfc_150",
        "rated_power_kw": 136.5,
        "peak_demand_hour_probability": 0.35,
        "managed_eligible_flag": False,
    },
]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_hhmm_to_minutes(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        n = int(value)
    except (TypeError, ValueError):
        return np.nan
    hour = n // 100
    minute = n % 100
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return np.nan
    return float(hour * 60 + minute)


def hhmm_series_to_minutes(values: pd.Series) -> pd.Series:
    n = pd.to_numeric(values, errors="coerce")
    hour = np.floor(n / 100.0)
    minute = n - hour * 100.0
    valid = n.notna() & hour.between(0, 23) & minute.between(0, 59)
    out = hour * 60.0 + minute
    return out.where(valid, np.nan)


def haversine_miles(
    lon1: pd.Series,
    lat1: pd.Series,
    lon2: pd.Series,
    lat2: pd.Series,
) -> pd.Series:
    lon1_rad = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat1_rad = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon2_rad = np.radians(pd.to_numeric(lon2, errors="coerce"))
    lat2_rad = np.radians(pd.to_numeric(lat2, errors="coerce"))
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return pd.Series(3958.7613 * 2.0 * np.arcsin(np.sqrt(a)), index=lon1.index)


def choose_network_miles(
    df: pd.DataFrame,
    *,
    max_fallback_miles: float,
    distance_source: str,
) -> pd.Series:
    time_dist = pd.to_numeric(df["SHORTEST_TDIST_miles"], errors="coerce")
    path_dist = pd.to_numeric(df["SHORTEST_PDIST_miles"], errors="coerce")
    trip_dist = pd.to_numeric(df["TRPMILES"], errors="coerce")
    straight_line = (
        pd.to_numeric(df["straight_line_miles"], errors="coerce")
        if "straight_line_miles" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    route_time_ok = (time_dist > 0) & (
        straight_line.isna() | (time_dist >= 0.95 * straight_line)
    )
    route_path_ok = (path_dist > 0) & (
        straight_line.isna() | (path_dist >= 0.95 * straight_line)
    )
    plausible_trip = trip_dist.where(
        (trip_dist > 0)
        & (trip_dist <= max_fallback_miles)
        & (straight_line.isna() | (trip_dist >= 0.5 * straight_line))
    )
    plausible_time = time_dist.where(route_time_ok)
    plausible_path = path_dist.where(route_path_ok)

    if distance_source == "trip":
        return plausible_trip.fillna(plausible_time).fillna(plausible_path)
    if distance_source == "route":
        return plausible_time.fillna(plausible_path).fillna(plausible_trip)
    raise ValueError("distance_source must be 'trip' or 'route'")


def unwrap_group_times(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(["_td_num", "source_row_id"], kind="mergesort").copy()
    starts = []
    ends = []
    offset = 0.0
    prev_end = -np.inf

    for base_start, base_end in g[["_base_start_time_min", "_base_end_time_min"]].itertuples(
        index=False, name=None
    ):
        start = float(base_start)
        end = float(base_end)
        while start + offset < prev_end - 1e-9:
            offset += 1440.0
        starts.append(start + offset)
        ends.append(end + offset)
        prev_end = end + offset

    g["start_time_min"] = starts
    g["end_time_min"] = ends
    return g


def stable_hash(parts: Iterable[object], *, prefix: str = "") -> str:
    text = "|".join("" if pd.isna(p) else str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=10).hexdigest()
    return f"{prefix}{digest}"


def normalize_activity_type(value: object) -> str:
    s = str(value or "").strip().upper()
    replacements = {
        "SOCIAL / RECREATIONAL": "SOCIAL/REC",
        "SHOPPING / ERRANDS": "SHOPPING/ERRANDS",
        "SCHOOL / DAYCARE / RELIGIOUS": "SCL/CLG/UNIV/CHURCH",
        "MEDICAL / DENTAL": "MEDICAL/DENTAL",
        "TRANSPORT (DROP-OFF / PICK-UP)": "TRANSPORT OTHER",
        "OTHER": "SOMETHING ELSE",
    }
    return replacements.get(s, s)


def normalize_trip_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    rename_map = {old: new for old, new in OLD_TO_CANONICAL_COLUMNS.items() if old in out.columns}
    if rename_map:
        out = out.rename(columns=rename_map)

    # Backward compatibility for the older file that had no synthetic record ID.
    if "SYN_RECORD_ID" not in out.columns:
        legacy_parts = []
        for col in ["home_geoid", "SERIALNO", "HOUSEID"]:
            if col in out.columns:
                legacy_parts.append(out[col].astype("string").fillna(""))
        if legacy_parts:
            legacy = legacy_parts[0]
            for part in legacy_parts[1:]:
                legacy = legacy + "_" + part
            out["SYN_RECORD_ID"] = legacy

    if "home_geoid" not in out.columns:
        out["home_geoid"] = pd.NA

    if "CITYTOWN_NAME" not in out.columns:
        out["CITYTOWN_NAME"] = pd.NA

    if "ACTIVITY_ADDR" not in out.columns:
        out["ACTIVITY_ADDR"] = pd.NA

    missing = sorted(REQUIRED_COLUMNS - set(out.columns))
    if missing:
        raise ValueError(f"Trip input is missing required columns after schema normalization: {missing}")

    out["ACTIVITY_TYPE"] = out["ACTIVITY_TYPE"].map(normalize_activity_type).astype("string")
    out["SYN_RECORD_ID"] = out["SYN_RECORD_ID"].astype("string")
    out["home_geoid"] = out["home_geoid"].astype("string")

    summary = {
        "schema_columns_renamed": rename_map,
        "has_home_geoid": bool(out["home_geoid"].notna().any()),
        "activity_type_normalized": True,
    }
    return out, summary


def load_charger_assumptions(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        df = pd.DataFrame(DEFAULT_CHARGER_ASSUMPTIONS)
    else:
        df = pd.read_csv(path)
    required = {
        "charge_location_type",
        "charger_type",
        "rated_power_kw",
        "managed_eligible_flag",
    }
    if "peak_demand_hour_probability" not in df.columns:
        required.add("availability_probability")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Charger assumptions missing required columns: {missing}")
    out = df.copy()
    if "assumption_key" not in out.columns:
        out["assumption_key"] = out["charge_location_type"]
    if "peak_demand_hour_probability" not in out.columns:
        out["peak_demand_hour_probability"] = out["availability_probability"]
    out["charge_location_type"] = out["charge_location_type"].astype(str).str.strip()
    out["assumption_key"] = out["assumption_key"].astype(str).str.strip()
    out["charger_type"] = out["charger_type"].astype(str).str.strip()
    out["rated_power_kw"] = pd.to_numeric(out["rated_power_kw"], errors="coerce")
    out["peak_demand_hour_probability"] = pd.to_numeric(
        out["peak_demand_hour_probability"], errors="coerce"
    )
    out["managed_eligible_flag"] = out["managed_eligible_flag"].map(
        lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"}
    )
    if out["rated_power_kw"].isna().any() or (out["rated_power_kw"] <= 0).any():
        examples = out.loc[
            out["rated_power_kw"].isna() | (out["rated_power_kw"] <= 0),
            ["charge_location_type", "charger_type", "rated_power_kw"],
        ].head(5)
        raise ValueError(
            "rated_power_kw must be positive for all charger assumptions. "
            f"Examples:\n{examples.to_string(index=False)}"
        )
    invalid_probability = out["peak_demand_hour_probability"].isna() | ~out[
        "peak_demand_hour_probability"
    ].between(0.0, 1.0)
    if invalid_probability.any():
        examples = out.loc[
            invalid_probability,
            ["assumption_key", "charge_location_type", "charger_type", "peak_demand_hour_probability"],
        ].head(5)
        raise ValueError(
            "peak_demand_hour_probability must be between 0 and 1 for all charger assumptions. "
            f"Examples:\n{examples.to_string(index=False)}"
        )
    if out["assumption_key"].duplicated().any():
        dupes = sorted(out.loc[out["assumption_key"].duplicated(), "assumption_key"].unique())
        raise ValueError(
            "Each charger assumption_key must have one default charger row; "
            f"duplicates found: {dupes}"
        )
    out["availability_probability"] = out["peak_demand_hour_probability"]
    return out


def gridup_time_varying_availability(
    stops: pd.DataFrame,
    *,
    peak_probability_col: str,
    location_precision: int = 4,
) -> pd.Series:
    """Apply GridUp's inverse-demand availability adjustment.

    The configured probability is interpreted as the probability of finding a
    charger during the peak stopped-vehicle hour for that charging location.
    At lower-demand hours the probability scales up in inverse proportion to
    stopped vehicles and is clipped at 1.0.
    """
    peak_probability = pd.to_numeric(stops[peak_probability_col], errors="coerce").fillna(0.0)
    out = peak_probability.copy()
    eligible = (
        stops["charge_location_type"].ne("no_charging")
        & peak_probability.gt(0.0)
        & pd.to_numeric(stops["rated_power_kw"], errors="coerce").gt(0.0)
        & pd.to_numeric(stops["dwell_min"], errors="coerce").gt(0.0)
        & pd.to_numeric(stops["arrival_time_min"], errors="coerce").notna()
        & pd.to_numeric(stops["departure_time_min"], errors="coerce").notna()
        & pd.to_numeric(stops["destination_lon"], errors="coerce").notna()
        & pd.to_numeric(stops["destination_lat"], errors="coerce").notna()
    )
    if not eligible.any():
        return out.clip(0.0, 1.0)

    work = stops.loc[
        eligible,
        [
            "charger_assumption_key",
            "charger_type",
            "destination_lon",
            "destination_lat",
            "arrival_time_min",
            "departure_time_min",
        ],
    ].copy()
    work["_row_index"] = work.index
    work["_availability_lon"] = pd.to_numeric(work["destination_lon"], errors="coerce").round(
        location_precision
    )
    work["_availability_lat"] = pd.to_numeric(work["destination_lat"], errors="coerce").round(
        location_precision
    )
    work["_arrival_hour"] = (
        np.floor(pd.to_numeric(work["arrival_time_min"], errors="coerce").mod(1440.0) / 60.0)
        .astype("Int64")
    )

    group_cols = [
        "charger_assumption_key",
        "charger_type",
        "_availability_lon",
        "_availability_lat",
    ]
    occupancy_parts = []
    start = pd.to_numeric(work["arrival_time_min"], errors="coerce")
    end = pd.to_numeric(work["departure_time_min"], errors="coerce")
    for hour in range(24):
        hour_start = float(hour * 60)
        hour_end = float((hour + 1) * 60)
        in_hour = ((start < hour_end) & (end > hour_start)) | (
            (start < hour_end + 1440.0) & (end > hour_start + 1440.0)
        )
        if in_hour.any():
            occupancy_parts.append(work.loc[in_hour, group_cols].assign(_availability_hour=hour))

    if not occupancy_parts:
        return out.clip(0.0, 1.0)

    occupancy = (
        pd.concat(occupancy_parts, ignore_index=True)
        .groupby(group_cols + ["_availability_hour"], dropna=False)
        .size()
        .rename("gridup_stopped_count")
        .reset_index()
    )
    occupancy["gridup_peak_stopped_count"] = occupancy.groupby(group_cols, dropna=False)[
        "gridup_stopped_count"
    ].transform("max")

    row_counts = work.merge(
        occupancy,
        left_on=group_cols + ["_arrival_hour"],
        right_on=group_cols + ["_availability_hour"],
        how="left",
    ).set_index("_row_index")
    current_count = row_counts["gridup_stopped_count"].reindex(stops.index)
    peak_count = row_counts["gridup_peak_stopped_count"].reindex(stops.index)
    adjusted = peak_probability * peak_count / current_count
    out.loc[eligible] = adjusted.loc[eligible].fillna(peak_probability.loc[eligible])
    return out.clip(0.0, 1.0)


def read_trip_file(path: Path, *, sample_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    dtype = {
        "SYN_RECORD_ID": "string",
        "GEO_ID": "string",
        "SERIALNO": "string",
        "HHI": "string",
        "CITYTOWN_NAME": "string",
        "ACTIVITY_ADDR": "string",
        "ACTIVITY_TYPE": "string",
    }
    df = pd.read_csv(path, dtype=dtype, nrows=sample_rows)
    df, schema_summary = normalize_trip_schema(df)

    rows_read = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df["source_row_id"] = np.arange(len(df), dtype=np.int64)
    return df, {
        "input_path": str(path),
        "rows_read": rows_read,
        "exact_duplicate_rows_removed": duplicate_rows,
        "rows_after_exact_deduplication": len(df),
        **schema_summary,
    }


def reconstruct_vehicle_day_trips(
    raw: pd.DataFrame,
    *,
    max_fallback_miles: float,
    distance_source: str,
    origin_mode: str,
    coordinate_tolerance: float,
) -> tuple[pd.DataFrame, dict]:
    if origin_mode not in {"reconstructed", "input"}:
        raise ValueError("origin_mode must be 'reconstructed' or 'input'")

    df = raw.copy()
    df["_td_num"] = pd.to_numeric(df["TDTRPNUM"], errors="coerce")
    df["_base_start_time_min"] = hhmm_series_to_minutes(df["STRTTIME"])
    df["_base_end_time_min"] = hhmm_series_to_minutes(df["ENDTIME"])
    df["_base_end_time_min"] = df["_base_end_time_min"].where(
        df["_base_end_time_min"] >= df["_base_start_time_min"],
        df["_base_end_time_min"] + 1440.0,
    )
    invalid_time = df["_base_start_time_min"].isna() | df["_base_end_time_min"].isna()
    df = df.loc[~invalid_time].copy()

    sort_cols = CHAIN_GROUP_COLUMNS + [
        "_td_num",
        "_base_start_time_min",
        "_base_end_time_min",
        "source_row_id",
    ]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df = pd.concat(
        [
            unwrap_group_times(group)
            for _, group in df.groupby(CHAIN_GROUP_COLUMNS, sort=False, dropna=False)
        ],
        ignore_index=True,
    )

    def assign_origin_and_distance(
        frame: pd.DataFrame,
        *,
        within_vehicle_day: bool = False,
    ) -> pd.DataFrame:
        out = frame.copy()
        if within_vehicle_day:
            group = out.groupby("vehicle_day_id", sort=False, dropna=False)
            prev_dest_x = group["ACTIVITY_X"].shift()
            prev_dest_y = group["ACTIVITY_Y"].shift()
        else:
            group = out.groupby(CHAIN_GROUP_COLUMNS, sort=False, dropna=False)
            prev_dest_x = group["ACTIVITY_X"].shift()
            prev_dest_y = group["ACTIVITY_Y"].shift()
        input_start_x = pd.to_numeric(out["START_X"], errors="coerce")
        input_start_y = pd.to_numeric(out["START_Y"], errors="coerce")
        home_x = pd.to_numeric(out["HOME_X"], errors="coerce")
        home_y = pd.to_numeric(out["HOME_Y"], errors="coerce")
        if within_vehicle_day:
            chain_first = prev_dest_x.isna()
            first_origin_x = input_start_x.where(out["_td_num"].ne(1), home_x).fillna(home_x)
            first_origin_y = input_start_y.where(out["_td_num"].ne(1), home_y).fillna(home_y)
            reconstructed_start_x = prev_dest_x.where(~chain_first, first_origin_x)
            reconstructed_start_y = prev_dest_y.where(~chain_first, first_origin_y)
        else:
            reconstructed_start_x = prev_dest_x.fillna(home_x)
            reconstructed_start_y = prev_dest_y.fillna(home_y)

        out["destination_lon"] = pd.to_numeric(out["ACTIVITY_X"], errors="coerce")
        out["destination_lat"] = pd.to_numeric(out["ACTIVITY_Y"], errors="coerce")
        if origin_mode == "reconstructed":
            out["origin_lon"] = reconstructed_start_x
            out["origin_lat"] = reconstructed_start_y
        else:
            out["origin_lon"] = input_start_x
            out["origin_lat"] = input_start_y
        out["input_origin_lon"] = input_start_x
        out["input_origin_lat"] = input_start_y
        out["home_lon"] = pd.to_numeric(out["HOME_X"], errors="coerce")
        out["home_lat"] = pd.to_numeric(out["HOME_Y"], errors="coerce")
        out["route_time_miles"] = pd.to_numeric(out["SHORTEST_TDIST_miles"], errors="coerce")
        out["route_path_miles"] = pd.to_numeric(out["SHORTEST_PDIST_miles"], errors="coerce")
        out["reported_trip_miles"] = pd.to_numeric(out["TRPMILES"], errors="coerce")
        out["straight_line_miles"] = haversine_miles(
            out["origin_lon"],
            out["origin_lat"],
            out["destination_lon"],
            out["destination_lat"],
        )
        straight = out["straight_line_miles"]
        route_time_ok = (out["route_time_miles"] > 0) & (
            straight.isna() | (out["route_time_miles"] >= 0.95 * straight)
        )
        route_path_ok = (out["route_path_miles"] > 0) & (
            straight.isna() | (out["route_path_miles"] >= 0.95 * straight)
        )
        trip_ok = (
            (out["reported_trip_miles"] > 0)
            & (out["reported_trip_miles"] <= max_fallback_miles)
            & (straight.isna() | (out["reported_trip_miles"] >= 0.5 * straight))
        )
        out["route_time_below_straightline_flag"] = (
            (out["route_time_miles"] > 0)
            & straight.notna()
            & (out["route_time_miles"] < 0.95 * straight)
        )
        out["route_path_below_straightline_flag"] = (
            (out["route_path_miles"] > 0)
            & straight.notna()
            & (out["route_path_miles"] < 0.95 * straight)
        )
        if distance_source == "trip":
            out["selected_distance_source"] = np.select(
                [trip_ok, route_time_ok, route_path_ok],
                ["trip", "route_time", "route_path"],
                default="missing",
            )
        else:
            out["selected_distance_source"] = np.select(
                [route_time_ok, route_path_ok, trip_ok],
                ["route_time", "route_path", "trip"],
                default="missing",
            )
        out["network_miles"] = choose_network_miles(
            out,
            max_fallback_miles=max_fallback_miles,
            distance_source=distance_source,
        )
        out["trip_to_route_time_ratio"] = out["reported_trip_miles"] / out[
            "route_time_miles"
        ].where(out["route_time_miles"] > 0)
        out["trip_to_straight_line_ratio"] = out["reported_trip_miles"] / out[
            "straight_line_miles"
        ].where(out["straight_line_miles"] > 0)
        out["route_to_straight_line_ratio"] = out["route_time_miles"] / out[
            "straight_line_miles"
        ].where(out["straight_line_miles"] > 0)
        return out

    invalid_distance_rows = 0
    for _ in range(3):
        df = assign_origin_and_distance(df)
        invalid_distance = df["network_miles"].isna()
        if not invalid_distance.any():
            break
        invalid_distance_rows += int(invalid_distance.sum())
        df = df.loc[~invalid_distance].copy().reset_index(drop=True)
    else:
        df = df.loc[df["network_miles"].notna()].copy().reset_index(drop=True)

    group = df.groupby(CHAIN_GROUP_COLUMNS, sort=False, dropna=False)
    prev_td = group["_td_num"].shift()
    prev_end = group["end_time_min"].shift()
    prev_dest_x = group["ACTIVITY_X"].shift()
    prev_dest_y = group["ACTIVITY_Y"].shift()
    input_start_x = pd.to_numeric(df["START_X"], errors="coerce")
    input_start_y = pd.to_numeric(df["START_Y"], errors="coerce")

    first_in_group = prev_td.isna()
    td_reset = (~first_in_group) & (df["_td_num"] <= prev_td)
    td_gap = (~first_in_group) & (df["_td_num"] > prev_td + 1)
    impossible_time = (~first_in_group) & (df["start_time_min"] < prev_end)
    coordinate_pair_present = (
        (~first_in_group)
        & prev_dest_x.notna()
        & prev_dest_y.notna()
        & input_start_x.notna()
        & input_start_y.notna()
    )
    x_gap = (input_start_x - prev_dest_x).abs() > coordinate_tolerance
    y_gap = (input_start_y - prev_dest_y).abs() > coordinate_tolerance
    input_origin_mismatch = coordinate_pair_present & (x_gap | y_gap)
    continuity_gap = input_origin_mismatch if origin_mode == "input" else pd.Series(False, index=df.index)
    df["input_origin_mismatch_flag"] = input_origin_mismatch.fillna(False)

    new_chain = first_in_group | td_reset | td_gap | impossible_time | continuity_gap.fillna(False)
    df["trip_chain_sequence"] = new_chain.astype(int).groupby(
        [df[c] for c in CHAIN_GROUP_COLUMNS], sort=False, dropna=False
    ).cumsum()

    chain_cols = CHAIN_GROUP_COLUMNS + ["trip_chain_sequence"]
    unique_chains = df[chain_cols].drop_duplicates().copy()
    unique_chains["vehicle_day_id"] = [
        stable_hash(row, prefix="vd_") for row in unique_chains[chain_cols].itertuples(index=False, name=None)
    ]
    df = df.merge(unique_chains, on=chain_cols, how="left", validate="many_to_one")
    new_chain_rows = int(new_chain.sum())
    td_reset_rows = int(td_reset.sum())
    td_gap_rows = int(td_gap.sum())
    groups_with_td_gap = int(df.loc[td_gap, CHAIN_GROUP_COLUMNS].drop_duplicates().shape[0])
    impossible_time_rows = int(impossible_time.sum())
    continuity_gap_rows = int(continuity_gap.fillna(False).sum())
    input_origin_mismatch_rows = int(input_origin_mismatch.fillna(False).sum())
    df = assign_origin_and_distance(df, within_vehicle_day=True)
    final_origin_invalid_distance = df["network_miles"].isna()
    final_origin_invalid_distance_rows = int(final_origin_invalid_distance.sum())
    if final_origin_invalid_distance.any():
        invalid_distance_rows += final_origin_invalid_distance_rows
        df = df.loc[~final_origin_invalid_distance].copy().reset_index(drop=True)

    vehicle_group = df.groupby("vehicle_day_id", sort=False)
    next_start = vehicle_group["start_time_min"].shift(-1)
    first_start = vehicle_group["start_time_min"].transform("first")
    raw_dwell = pd.to_numeric(df["DWELTIME"], errors="coerce")

    between_trip_dwell = next_start - df["end_time_min"]
    between_trip_dwell = between_trip_dwell.where(between_trip_dwell >= 0.0)
    overnight_dwell = ((first_start % 1440.0) - (df["end_time_min"] % 1440.0)) % 1440.0
    overnight_dwell = overnight_dwell.mask(overnight_dwell <= 1e-9, 1440.0)
    inferred_dwell = between_trip_dwell.where(next_start.notna()).fillna(overnight_dwell)
    df["raw_dwell_min"] = raw_dwell
    df["inferred_dwell_min"] = inferred_dwell
    df["raw_dwell_gap_error_min"] = raw_dwell - inferred_dwell
    nonfinal = next_start.notna()
    dwell_gap_error = (
        raw_dwell.notna()
        & (raw_dwell >= 0.0)
        & nonfinal
        & (df["raw_dwell_gap_error_min"].abs() > 5.0)
    )

    df["dwell_min"] = inferred_dwell
    df["dwell_min"] = df["dwell_min"].fillna(raw_dwell.where(raw_dwell >= 0.0))
    df["dwell_min"] = df["dwell_min"].clip(lower=0.0)
    df["arrival_time_min"] = df["end_time_min"]
    df["departure_time_min"] = df["arrival_time_min"] + df["dwell_min"]
    df["home_record_id"] = df["SYN_RECORD_ID"].astype("string")
    df["home_geoid"] = df["home_geoid"].astype("string")
    df["home_citytown"] = df["CITYTOWN_NAME"].astype("string")

    qa = {
        "invalid_time_rows_dropped": int(invalid_time.sum()),
        "invalid_distance_rows_dropped": int(invalid_distance_rows),
        "invalid_distance_rows_after_final_origin_recompute": final_origin_invalid_distance_rows,
        "vehicle_day_count": int(df["vehicle_day_id"].nunique()),
        "trip_rows_after_qa": int(len(df)),
        "new_chain_rows": new_chain_rows,
        "chain_breaks_td_reset": td_reset_rows,
        "chain_breaks_trip_number_gap": td_gap_rows,
        "groups_with_trip_number_gap_after_drop": groups_with_td_gap,
        "chain_breaks_impossible_time": impossible_time_rows,
        "chain_breaks_coordinate_gap": continuity_gap_rows,
        "input_origin_mismatch_rows": input_origin_mismatch_rows,
        "rows_after_distance_drop": int(len(df)),
        "route_time_below_straightline_count": int(
            df["route_time_below_straightline_flag"].sum()
        ),
        "route_path_below_straightline_count": int(
            df["route_path_below_straightline_flag"].sum()
        ),
        "trip_fallback_used_count": int(df["selected_distance_source"].eq("trip").sum()),
        "all_distance_missing_count": int(invalid_distance_rows),
        "raw_dwell_gap_mismatch_rows": int(dwell_gap_error.sum()),
        "max_abs_dwell_gap_error_min": (
            float(df.loc[dwell_gap_error, "raw_dwell_gap_error_min"].abs().max())
            if dwell_gap_error.any()
            else 0.0
        ),
        "origin_mode": origin_mode,
        "distance_source": distance_source,
        "total_network_miles": float(df["network_miles"].sum()),
        "mean_daily_network_miles": float(
            df.groupby("vehicle_day_id", sort=False)["network_miles"].sum().mean()
        ),
        "coordinate_axis_note": "The model treats *_X as longitude and *_Y as latitude.",
    }
    return df, qa


def classify_stop_opportunities(
    trips: pd.DataFrame,
    *,
    availability_scenario: str,
    availability_choice_model: str,
    charger_assumptions: pd.DataFrame,
    min_dwell_min: float,
    long_public_dwell_min: float,
) -> pd.DataFrame:
    if availability_scenario not in AVAILABILITY_MULTIPLIER:
        raise ValueError(
            f"availability_scenario must be one of {sorted(AVAILABILITY_MULTIPLIER)}"
        )
    if availability_choice_model not in {"gridup", "static"}:
        raise ValueError("availability_choice_model must be 'gridup' or 'static'")

    out = trips.copy()
    activity = out["ACTIVITY_TYPE"].fillna("").astype(str).str.upper().str.strip()
    why = pd.to_numeric(out["WHYTRP1S"], errors="coerce")
    dwell = pd.to_numeric(out["dwell_min"], errors="coerce").fillna(0.0)

    is_home = activity.eq("HOME") | why.eq(1)
    is_work = activity.eq("WORK") | why.eq(10)
    is_civic = (
        activity.str.contains("SCL|CLG|UNIV|CHURCH|MEDICAL|DENTAL", regex=True)
        | why.isin([20, 30])
    )
    feasible = dwell >= min_dwell_min

    stop_type = np.select(
        [
            ~feasible,
            is_home,
            is_work,
            is_civic,
            dwell >= long_public_dwell_min,
        ],
        [
            "no_charging",
            "home",
            "work",
            "civic",
            "long_public",
        ],
        default="quick_public",
    )
    out["charge_location_type"] = stop_type
    out["stop_priority"] = out["charge_location_type"].map(STOP_PRIORITY).fillna(99).astype(int)
    assumption_map = charger_assumptions.set_index("assumption_key")

    home_prop_class = (
        pd.to_numeric(out["HOME_PROP_CLASS"], errors="coerce")
        if "HOME_PROP_CLASS" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    single_family_home = out["charge_location_type"].eq("home") & home_prop_class.isin(
        SINGLE_FAMILY_HOME_PROP_CLASSES
    )
    preferred_key = out["charge_location_type"].astype("string")
    preferred_key = preferred_key.mask(single_family_home, "home_sfh")
    preferred_key = preferred_key.mask(
        out["charge_location_type"].eq("home") & ~single_family_home,
        "home_mfh",
    )
    available_keys = set(assumption_map.index.astype(str))
    fallback_key = out["charge_location_type"].astype("string")
    out["charger_assumption_key"] = preferred_key.where(
        preferred_key.isin(available_keys), fallback_key
    )
    out["charger_assumption_key"] = out["charger_assumption_key"].where(
        out["charger_assumption_key"].isin(available_keys), ""
    )

    out["charger_type"] = out["charger_assumption_key"].map(assumption_map["charger_type"]).fillna("")
    out["rated_power_kw"] = (
        out["charger_assumption_key"].map(assumption_map["rated_power_kw"]).fillna(0.0).astype(float)
    )

    multiplier = AVAILABILITY_MULTIPLIER[availability_scenario]
    out["peak_demand_hour_probability"] = (
        out["charger_assumption_key"]
        .map(assumption_map["peak_demand_hour_probability"])
        .fillna(0.0)
        .astype(float)
        * multiplier
    ).clip(0.0, 1.0)
    if availability_choice_model == "gridup":
        out["charger_availability_probability"] = gridup_time_varying_availability(
            out,
            peak_probability_col="peak_demand_hour_probability",
        )
    else:
        out["charger_availability_probability"] = out["peak_demand_hour_probability"]

    managed_map = assumption_map["managed_eligible_flag"].to_dict()
    out["managed_eligible_flag"] = out["charger_assumption_key"].map(
        lambda x: bool(managed_map.get(x, False))
    )
    out["charging_stop_id"] = [
        stable_hash((vd, row_id), prefix="stop_")
        for vd, row_id in zip(out["vehicle_day_id"], out["source_row_id"])
    ]
    return out


def compress_energy_states(
    states: list[tuple[float, float]],
    *,
    precision_kwh: float = 0.01,
) -> list[tuple[float, float]]:
    buckets: dict[float, float] = defaultdict(float)
    for probability, remaining_need in states:
        if probability <= 1e-12:
            continue
        need = max(0.0, float(remaining_need))
        if need <= 1e-9:
            need = 0.0
        elif precision_kwh > 0:
            need = round(need / precision_kwh) * precision_kwh
        buckets[need] += float(probability)
    return [(probability, need) for need, probability in sorted(buckets.items())]


def build_charging_events(
    stops: pd.DataFrame,
    *,
    scenario_id: str,
    managed: bool,
    efficiency_kwh_per_mile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_energy = (
        stops.groupby("vehicle_day_id", sort=False)["network_miles"].sum()
        * float(efficiency_kwh_per_mile)
    ).rename("daily_energy_kwh")

    feasible = stops.loc[
        stops["charge_location_type"].ne("no_charging")
        & (stops["charger_availability_probability"] > 0.0)
        & (stops["rated_power_kw"] > 0.0)
        & (stops["dwell_min"] > 0.0)
    ].copy()
    if feasible.empty:
        summary = daily_energy.reset_index()
        summary["energy_delivered_kwh"] = 0.0
        summary["unmet_energy_kwh"] = summary["daily_energy_kwh"]
        return pd.DataFrame(), summary

    feasible["daily_energy_kwh"] = feasible["vehicle_day_id"].map(daily_energy)
    feasible["dwell_hr"] = feasible["dwell_min"] / 60.0
    feasible["capacity_if_available_kwh"] = feasible["rated_power_kw"] * feasible["dwell_hr"]
    feasible = feasible.loc[feasible["capacity_if_available_kwh"] > 0.0].copy()

    feasible = feasible.sort_values(
        [
            "vehicle_day_id",
            "stop_priority",
            "dwell_min",
            "arrival_time_min",
            "source_row_id",
        ],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    allocation_rows = []
    for _vehicle_day_id, g in feasible.groupby("vehicle_day_id", sort=False):
        daily_need = float(g["daily_energy_kwh"].iloc[0])
        states = [(1.0, daily_need)]
        for idx, row in g.iterrows():
            active_probability = sum(probability for probability, need in states if need > 1e-9)
            if active_probability <= 1e-12:
                break
            p = float(row["charger_availability_probability"])
            capacity = float(row["capacity_if_available_kwh"])
            next_states: list[tuple[float, float]] = []
            event_probability = 0.0
            expected_energy = 0.0
            for state_probability, remaining_need in states:
                if remaining_need <= 1e-9:
                    next_states.append((state_probability, 0.0))
                    continue
                delivered_if_available = min(remaining_need, capacity)
                available_probability = state_probability * p
                unavailable_probability = state_probability * (1.0 - p)
                if delivered_if_available > 1e-9 and available_probability > 1e-12:
                    event_probability += available_probability
                    expected_energy += available_probability * delivered_if_available
                next_states.append((available_probability, remaining_need - delivered_if_available))
                next_states.append((unavailable_probability, remaining_need))

            if expected_energy > 1e-9:
                allocation_rows.append(
                    {
                        "_idx": idx,
                        "remaining_probability_before": active_probability,
                        "event_probability": event_probability,
                        "conditional_energy_if_available_kwh": expected_energy / event_probability,
                        "energy_delivered_kwh": expected_energy,
                    }
                )
            states = compress_energy_states(next_states)

    if not allocation_rows:
        summary = daily_energy.reset_index()
        summary["energy_delivered_kwh"] = 0.0
        summary["unmet_energy_kwh"] = summary["daily_energy_kwh"]
        summary["scenario_id"] = scenario_id
        summary["managed_flag"] = bool(managed)
        return pd.DataFrame(), summary

    allocation = pd.DataFrame.from_records(allocation_rows).set_index("_idx")
    feasible = feasible.join(allocation, how="inner")
    feasible = feasible.loc[feasible["energy_delivered_kwh"] > 1e-9].copy()

    feasible["actual_energy_if_available_kwh"] = feasible[
        "conditional_energy_if_available_kwh"
    ]
    if "charger_assumption_key" not in feasible.columns:
        feasible["charger_assumption_key"] = feasible["charge_location_type"]
    if "peak_demand_hour_probability" not in feasible.columns:
        feasible["peak_demand_hour_probability"] = feasible["charger_availability_probability"]
    feasible["start_time_min"] = feasible["arrival_time_min"]
    feasible["use_managed_charging_flag"] = bool(managed) & feasible["managed_eligible_flag"]
    managed_mask = feasible["use_managed_charging_flag"]
    feasible["event_duration_min"] = (
        feasible["actual_energy_if_available_kwh"] / feasible["rated_power_kw"] * 60.0
    ).clip(upper=feasible["dwell_min"])
    feasible.loc[managed_mask, "event_duration_min"] = feasible.loc[managed_mask, "dwell_min"]
    feasible["expected_power_kw"] = feasible["rated_power_kw"] * feasible["event_probability"]
    feasible.loc[managed_mask, "expected_power_kw"] = (
        feasible.loc[managed_mask, "energy_delivered_kwh"]
        / feasible.loc[managed_mask, "dwell_hr"]
    )
    feasible["end_time_min"] = feasible["start_time_min"] + feasible["event_duration_min"]
    feasible["scenario_id"] = scenario_id
    feasible["managed_flag"] = bool(managed)

    keep = [
        "scenario_id",
        "managed_flag",
        "vehicle_day_id",
        "charging_stop_id",
        "home_record_id",
        "home_geoid",
        "home_citytown",
        "trip_chain_sequence",
        "TDTRPNUM",
        "ACTIVITY_TYPE",
        "WHYTRP1S",
        "charge_location_type",
        "charger_assumption_key",
        "charger_type",
        "rated_power_kw",
        "peak_demand_hour_probability",
        "charger_availability_probability",
        "remaining_probability_before",
        "event_probability",
        "managed_eligible_flag",
        "use_managed_charging_flag",
        "daily_energy_kwh",
        "energy_delivered_kwh",
        "actual_energy_if_available_kwh",
        "expected_power_kw",
        "start_time_min",
        "end_time_min",
        "event_duration_min",
        "arrival_time_min",
        "departure_time_min",
        "dwell_min",
        "destination_lon",
        "destination_lat",
        "origin_lon",
        "origin_lat",
        "input_origin_lon",
        "input_origin_lat",
        "input_origin_mismatch_flag",
        "home_lon",
        "home_lat",
        "network_miles",
        "HHI",
        "HHS",
        "VEH",
        "URBRUR",
        "HOMEOWN",
    ]
    events = feasible[keep].copy()

    delivered = events.groupby("vehicle_day_id", sort=False)["energy_delivered_kwh"].sum()
    vehicle_summary = daily_energy.reset_index()
    vehicle_summary["energy_delivered_kwh"] = (
        vehicle_summary["vehicle_day_id"].map(delivered).fillna(0.0)
    )
    vehicle_summary["unmet_energy_kwh"] = (
        vehicle_summary["daily_energy_kwh"] - vehicle_summary["energy_delivered_kwh"]
    ).clip(lower=0.0)
    vehicle_summary["scenario_id"] = scenario_id
    vehicle_summary["managed_flag"] = bool(managed)
    return events, vehicle_summary


def _h3_latlng_to_cell(lat: float, lon: float, resolution: int) -> str | None:
    try:
        import h3  # type: ignore
    except Exception:
        return None
    if pd.isna(lat) or pd.isna(lon):
        return None
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(float(lat), float(lon), int(resolution))
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(float(lat), float(lon), int(resolution))
    return None


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 4:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-15) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _point_in_polygon_geometry(lon: float, lat: float, geometry: dict[str, Any]) -> bool:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    polygons = [coords] if geom_type == "Polygon" else coords if geom_type == "MultiPolygon" else []
    for polygon in polygons:
        if not polygon:
            continue
        exterior = polygon[0]
        if not _point_in_ring(lon, lat, exterior):
            continue
        holes = polygon[1:]
        if any(_point_in_ring(lon, lat, hole) for hole in holes):
            continue
        return True
    return False


def _geometry_bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    polygons = [coords] if geom_type == "Polygon" else coords if geom_type == "MultiPolygon" else []
    for polygon in polygons:
        for ring in polygon:
            for lon, lat, *_rest in ring:
                xs.append(float(lon))
                ys.append(float(lat))
    return min(xs), min(ys), max(xs), max(ys)


def first_present(props: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = props.get(name)
        if value not in {None, ""}:
            return value
    return None


def load_zcta_features(
    geojson_path: Path,
    *,
    zip_to_county_path: Path | None,
    county_filter: str | None,
) -> list[dict[str, Any]]:
    with open(geojson_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    allowed_zips: set[str] | None = None
    if zip_to_county_path and county_filter:
        zc = pd.read_csv(zip_to_county_path, dtype={"zip": "string"})
        county = county_filter.strip().upper()
        allowed_zips = set(
            zc.loc[zc["county_name"].astype(str).str.upper().eq(county), "zip"]
            .dropna()
            .astype(str)
            .str.zfill(5)
        )

    out = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        zcta_value = first_present(
            props,
            ["ZCTA5CE20", "GEOID20", "ZCTA5CE10", "GEOID10", "ZCTA5CE", "GEOID"],
        )
        zcta = str(zcta_value or "").zfill(5)
        if allowed_zips is not None and zcta not in allowed_zips:
            continue
        geom = feature.get("geometry") or {}
        if not geom:
            continue
        bbox = _geometry_bbox(geom)
        lon = pd.to_numeric(
            pd.Series([first_present(props, ["INTPTLON20", "INTPTLON10", "INTPTLON"])]),
            errors="coerce",
        ).iloc[0]
        lat = pd.to_numeric(
            pd.Series([first_present(props, ["INTPTLAT20", "INTPTLAT10", "INTPTLAT"])]),
            errors="coerce",
        ).iloc[0]
        out.append({"zcta": zcta, "geometry": geom, "bbox": bbox, "centroid_lon": lon, "centroid_lat": lat})
    if not out:
        raise ValueError(f"No ZCTA features loaded from {geojson_path}")
    return out


def assign_zcta_for_unique_points(
    points: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    zcta_features: list[dict[str, Any]],
    fallback_nearest: bool = False,
    nearest_max_miles: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lon, lat in points[[lon_col, lat_col]].itertuples(index=False, name=None):
        row = {
            "zcta": None,
            "zcta_match_method": "missing",
            "zcta_nearest_distance_miles": np.nan,
        }
        if pd.isna(lon) or pd.isna(lat):
            rows.append(row)
            continue
        lon_f = float(lon)
        lat_f = float(lat)
        match = None
        for feature in zcta_features:
            minx, miny, maxx, maxy = feature["bbox"]
            if lon_f < minx or lon_f > maxx or lat_f < miny or lat_f > maxy:
                continue
            if _point_in_polygon_geometry(lon_f, lat_f, feature["geometry"]):
                match = feature["zcta"]
                break
        if match is not None:
            row["zcta"] = match
            row["zcta_match_method"] = "polygon"
        elif fallback_nearest:
            best_zcta = None
            best_dist = float("inf")
            for feature in zcta_features:
                c_lon = feature.get("centroid_lon")
                c_lat = feature.get("centroid_lat")
                if pd.isna(c_lon) or pd.isna(c_lat):
                    continue
                dist = float(
                    haversine_miles(
                        pd.Series([lon_f]),
                        pd.Series([lat_f]),
                        pd.Series([float(c_lon)]),
                        pd.Series([float(c_lat)]),
                    ).iloc[0]
                )
                if dist < best_dist:
                    best_dist = dist
                    best_zcta = feature["zcta"]
            row["zcta_nearest_distance_miles"] = best_dist
            if best_zcta is not None and (
                nearest_max_miles is None or best_dist <= nearest_max_miles
            ):
                row["zcta"] = best_zcta
                row["zcta_match_method"] = "nearest"
        rows.append(row)
    return pd.DataFrame(rows, index=points.index)


def add_zcta_geographies(
    events: pd.DataFrame,
    *,
    zcta_geojson_path: Path | None,
    zip_to_county_path: Path | None,
    zcta_county_filter: str | None,
    assign_charging_zcta: bool,
    allow_nearest_zcta_fallback: bool,
    nearest_zcta_max_miles: float,
) -> tuple[pd.DataFrame, dict]:
    out = events.copy()
    summary = {
        "zcta_requested": zcta_geojson_path is not None,
        "zcta_geojson_path": str(zcta_geojson_path) if zcta_geojson_path else None,
        "zcta_county_filter": zcta_county_filter,
    }
    if zcta_geojson_path is None:
        out["home_zcta"] = pd.NA
        out["charging_zcta"] = pd.NA
        summary.update(
            {
                "home_zcta_matched_rows": 0,
                "home_zcta_polygon_match_rows": 0,
                "home_zcta_nearest_fallback_rows": 0,
                "home_zcta_missing_rows": len(out),
                "charging_zcta_matched_rows": 0,
            }
        )
        return out, summary

    features = load_zcta_features(
        zcta_geojson_path,
        zip_to_county_path=zip_to_county_path,
        county_filter=zcta_county_filter,
    )
    summary["zcta_feature_count"] = len(features)

    home_points = out[["home_lon", "home_lat"]].drop_duplicates().copy()
    home_assignment = assign_zcta_for_unique_points(
        home_points,
        lon_col="home_lon",
        lat_col="home_lat",
        zcta_features=features,
        fallback_nearest=allow_nearest_zcta_fallback,
        nearest_max_miles=nearest_zcta_max_miles,
    )
    home_points["home_zcta"] = home_assignment["zcta"].astype("string")
    home_points["home_zcta_match_method"] = home_assignment["zcta_match_method"]
    home_points["home_zcta_nearest_distance_miles"] = home_assignment[
        "zcta_nearest_distance_miles"
    ]
    out = out.merge(home_points, on=["home_lon", "home_lat"], how="left")

    if assign_charging_zcta:
        charging_points = out[["destination_lon", "destination_lat"]].drop_duplicates().copy()
        charging_assignment = assign_zcta_for_unique_points(
            charging_points,
            lon_col="destination_lon",
            lat_col="destination_lat",
            zcta_features=features,
            fallback_nearest=allow_nearest_zcta_fallback,
            nearest_max_miles=nearest_zcta_max_miles,
        )
        charging_points["charging_zcta"] = charging_assignment["zcta"].astype("string")
        charging_points["charging_zcta_match_method"] = charging_assignment["zcta_match_method"]
        charging_points["charging_zcta_nearest_distance_miles"] = charging_assignment[
            "zcta_nearest_distance_miles"
        ]
        out = out.merge(charging_points, on=["destination_lon", "destination_lat"], how="left")
    else:
        out["charging_zcta"] = pd.NA
        out["charging_zcta_match_method"] = pd.NA
        out["charging_zcta_nearest_distance_miles"] = np.nan

    summary["home_zcta_unique"] = int(out["home_zcta"].nunique(dropna=True))
    summary["home_zcta_matched_rows"] = int(out["home_zcta"].notna().sum())
    summary["home_zcta_polygon_match_rows"] = int(out["home_zcta_match_method"].eq("polygon").sum())
    summary["home_zcta_nearest_fallback_rows"] = int(out["home_zcta_match_method"].eq("nearest").sum())
    summary["home_zcta_missing_rows"] = int(out["home_zcta"].isna().sum())
    summary["max_home_zcta_nearest_fallback_distance_miles"] = (
        float(
            out.loc[
                out["home_zcta_match_method"].eq("nearest"),
                "home_zcta_nearest_distance_miles",
            ].max()
        )
        if out["home_zcta_match_method"].eq("nearest").any()
        else 0.0
    )
    summary["charging_zcta_unique"] = int(out["charging_zcta"].nunique(dropna=True))
    summary["charging_zcta_matched_rows"] = int(out["charging_zcta"].notna().sum())
    summary["charging_zcta_polygon_match_rows"] = int(
        out["charging_zcta_match_method"].eq("polygon").sum()
    )
    summary["charging_zcta_nearest_fallback_rows"] = int(
        out["charging_zcta_match_method"].eq("nearest").sum()
    )
    summary["charging_zcta_missing_rows"] = int(out["charging_zcta"].isna().sum())
    summary["max_charging_zcta_nearest_fallback_distance_miles"] = (
        float(
            out.loc[
                out["charging_zcta_match_method"].eq("nearest"),
                "charging_zcta_nearest_distance_miles",
            ].max()
        )
        if out["charging_zcta_match_method"].eq("nearest").any()
        else 0.0
    )
    summary["nearest_zcta_fallback_allowed"] = bool(allow_nearest_zcta_fallback)
    summary["nearest_zcta_max_miles"] = float(nearest_zcta_max_miles)
    return out, summary


def add_charging_spatial_ids(
    events: pd.DataFrame,
    *,
    point_precision: int,
    h3_resolution: int | None,
    require_h3: bool,
) -> tuple[pd.DataFrame, dict]:
    out = events.copy()
    out["charging_lon"] = out["destination_lon"].round(point_precision)
    out["charging_lat"] = out["destination_lat"].round(point_precision)
    out["charging_point_id"] = (
        "pt_"
        + out["charging_lat"].map(lambda x: f"{x:.{point_precision}f}" if pd.notna(x) else "na")
        + "_"
        + out["charging_lon"].map(lambda x: f"{x:.{point_precision}f}" if pd.notna(x) else "na")
    )

    summary = {
        "charging_point_precision_decimal_degrees": point_precision,
        "unique_charging_points": int(out["charging_point_id"].nunique()),
        "h3_requested": h3_resolution is not None,
        "h3_available": False,
        "h3_resolution": h3_resolution,
    }

    if h3_resolution is not None:
        unique_points = out[["destination_lat", "destination_lon"]].drop_duplicates().copy()
        unique_points["charging_h3"] = [
            _h3_latlng_to_cell(lat, lon, h3_resolution)
            for lat, lon in unique_points[["destination_lat", "destination_lon"]].itertuples(
                index=False, name=None
            )
        ]
        h3_available = unique_points["charging_h3"].notna().any()
        if require_h3 and not h3_available:
            raise RuntimeError("H3 output requested, but the h3 Python package is not installed")
        out = out.merge(unique_points, on=["destination_lat", "destination_lon"], how="left")
        summary["h3_available"] = bool(h3_available)
        summary["unique_charging_h3"] = int(out["charging_h3"].nunique(dropna=True))
    else:
        out["charging_h3"] = pd.NA
    return out, summary


def format_time_label(minutes: int) -> str:
    minutes = int(minutes) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def iter_event_bins(start: float, end: float, bin_minutes: int):
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return
    current = float(start)
    while current < end - 1e-9:
        bin_start = math.floor(current / bin_minutes) * bin_minutes
        bin_end = bin_start + bin_minutes
        overlap = min(end, bin_end) - current
        if overlap > 1e-9:
            yield int(bin_start % 1440), float(overlap)
        current = bin_end


def build_load_curve(
    events: pd.DataFrame,
    *,
    location_columns: list[str],
    bin_minutes: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    base_columns = [
        "scenario_id",
        "managed_flag",
        "charger_type",
        "charge_location_type",
    ]
    group_columns = base_columns + location_columns + ["time_bin_min"]
    accum: dict[tuple, float] = defaultdict(float)

    row_columns = base_columns + location_columns + [
        "start_time_min",
        "end_time_min",
        "expected_power_kw",
    ]
    for row in events[row_columns].itertuples(index=False, name=None):
        values = dict(zip(row_columns, row))
        power_kw = float(values["expected_power_kw"])
        if power_kw <= 0.0 or not np.isfinite(power_kw):
            continue
        key_prefix = tuple(values[c] for c in base_columns + location_columns)
        for time_bin_min, overlap_min in iter_event_bins(
            float(values["start_time_min"]),
            float(values["end_time_min"]),
            bin_minutes,
        ):
            accum[key_prefix + (time_bin_min,)] += power_kw * overlap_min / 60.0

    records = []
    bin_hours = bin_minutes / 60.0
    for key, kwh in accum.items():
        record = dict(zip(group_columns, key))
        record["time_bin_label"] = format_time_label(record["time_bin_min"])
        record["kwh"] = kwh
        record["kw"] = kwh / bin_hours
        records.append(record)
    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    return out.sort_values(group_columns).reset_index(drop=True)


def aggregate_to_hourly(load_15min: pd.DataFrame, *, location_columns: list[str]) -> pd.DataFrame:
    if load_15min.empty:
        return pd.DataFrame()
    out = load_15min.copy()
    out["hour"] = (out["time_bin_min"] // 60).astype(int)
    group_cols = [
        "scenario_id",
        "managed_flag",
        "charger_type",
        "charge_location_type",
    ] + location_columns + ["hour"]
    hourly = out.groupby(group_cols, dropna=False, as_index=False)["kwh"].sum()
    hourly["time_bin_min"] = hourly["hour"] * 60
    hourly["time_bin_label"] = hourly["time_bin_min"].map(format_time_label)
    hourly["kw"] = hourly["kwh"]
    return hourly.sort_values(group_cols).reset_index(drop=True)


def peak_summary(load: pd.DataFrame, *, location_columns: list[str]) -> pd.DataFrame:
    if load.empty:
        return pd.DataFrame()
    group_cols = ["scenario_id", "managed_flag"] + location_columns
    idx = load.groupby(group_cols, dropna=False)["kw"].idxmax()
    peaks = load.loc[idx, group_cols + ["time_bin_min", "time_bin_label", "kw"]].copy()
    peaks = peaks.rename(columns={"kw": "peak_kw", "time_bin_min": "peak_time_bin_min"})
    daily = load.groupby(group_cols, dropna=False, as_index=False)["kwh"].sum()
    daily = daily.rename(columns={"kwh": "daily_kwh"})
    return peaks.merge(daily, on=group_cols, how="left").sort_values(group_cols).reset_index(drop=True)


def normalize_zcta(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit():
        return s.zfill(5)
    return s


def build_default_adoption_scenarios(home_zctas: Iterable[str], years: list[int]) -> pd.DataFrame:
    params = {
        "low": {"cap": 0.45, "mid": 2037, "rate": 0.23},
        "base": {"cap": 0.70, "mid": 2033, "rate": 0.28},
        "high": {"cap": 0.90, "mid": 2030, "rate": 0.34},
    }
    records = []
    for home_zcta in sorted({str(g) for g in home_zctas if pd.notna(g)}):
        for scenario, p in params.items():
            for year in years:
                fraction = p["cap"] / (1.0 + math.exp(-p["rate"] * (year - p["mid"])))
                records.append(
                    {
                        "adoption_scenario": scenario,
                        "forecast_year": year,
                        "home_zcta": home_zcta,
                        "adoption_fraction": fraction,
                        "vehicle_growth_factor": 1.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def load_adoption_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"home_zcta": "string", "home_zip": "string", "zip": "string"})
    if "home_zcta" not in df.columns:
        if "home_zip" in df.columns:
            df["home_zcta"] = df["home_zip"]
        elif "zip" in df.columns:
            df["home_zcta"] = df["zip"]
    required = {"adoption_scenario", "forecast_year", "home_zcta", "adoption_fraction"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Adoption file {path} missing required columns: {missing}")
    out = df.copy()
    out["home_zcta"] = out["home_zcta"].map(normalize_zcta).astype("string")
    out["forecast_year"] = pd.to_numeric(out["forecast_year"], errors="raise").astype(int)
    out["adoption_fraction"] = pd.to_numeric(out["adoption_fraction"], errors="raise")
    invalid_fraction = out["adoption_fraction"].isna() | ~out["adoption_fraction"].between(0.0, 1.0)
    if invalid_fraction.any():
        examples = out.loc[
            invalid_fraction,
            ["adoption_scenario", "forecast_year", "home_zcta", "adoption_fraction"],
        ].head(5)
        raise ValueError(
            "Adoption fractions must be finite values between 0 and 1. "
            f"Examples:\n{examples.to_string(index=False)}"
        )
    if "vehicle_growth_factor" not in out.columns:
        out["vehicle_growth_factor"] = 1.0
    out["vehicle_growth_factor"] = pd.to_numeric(out["vehicle_growth_factor"], errors="raise")
    out = out.dropna(subset=["home_zcta"])
    key_cols = ["adoption_scenario", "forecast_year", "home_zcta"]
    duplicate_keys = out.duplicated(key_cols, keep=False)
    if duplicate_keys.any():
        examples = out.loc[duplicate_keys, key_cols].drop_duplicates().head(10)
        raise ValueError(
            "Adoption file must have one row per adoption_scenario, forecast_year, "
            f"and home_zcta. Duplicate keys include:\n{examples.to_string(index=False)}"
        )
    return out


def apply_adoption_scaling(
    load: pd.DataFrame,
    adoption: pd.DataFrame,
    *,
    location_columns: list[str],
) -> pd.DataFrame:
    if load.empty or adoption.empty or "home_zcta" not in load.columns:
        return pd.DataFrame()
    modeled_zctas = set(load["home_zcta"].dropna().astype(str))
    coverage = (
        adoption.groupby(["adoption_scenario", "forecast_year"])["home_zcta"]
        .apply(lambda s: set(s.dropna().astype(str)))
    )
    bad_groups = {
        key: sorted(modeled_zctas - zctas)
        for key, zctas in coverage.items()
        if modeled_zctas - zctas
    }
    if bad_groups:
        first_key, missing = next(iter(bad_groups.items()))
        preview = ", ".join(missing[:10])
        raise ValueError(
            "Adoption file does not cover every modeled home_zcta for each "
            f"scenario/year. First incomplete group {first_key}; "
            f"{len(missing)} missing, examples: {preview}"
        )
    merged = load.merge(adoption, on="home_zcta", how="inner", validate="many_to_many")
    merged["kw"] = (
        merged["kw"] * merged["adoption_fraction"] * merged["vehicle_growth_factor"]
    )
    merged["kwh"] = (
        merged["kwh"] * merged["adoption_fraction"] * merged["vehicle_growth_factor"]
    )
    group_cols = [
        "scenario_id",
        "managed_flag",
        "adoption_scenario",
        "forecast_year",
        "charger_type",
        "charge_location_type",
    ] + location_columns + ["time_bin_min", "time_bin_label"]
    return merged.groupby(group_cols, dropna=False, as_index=False)[["kw", "kwh"]].sum()


def write_csv(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def write_adoption_scaled_outputs(
    load: pd.DataFrame,
    adoption: pd.DataFrame,
    *,
    location_columns: list[str],
    output_15_path: Path,
    output_hourly_path: Path,
) -> tuple[int, int]:
    output_15_path.parent.mkdir(parents=True, exist_ok=True)
    output_hourly_path.parent.mkdir(parents=True, exist_ok=True)
    first_15 = True
    first_hourly = True
    total_15 = 0
    total_hourly = 0
    hourly_location_columns = ["adoption_scenario", "forecast_year"] + location_columns

    for _, adoption_part in adoption.groupby(
        ["adoption_scenario", "forecast_year"], sort=True, dropna=False
    ):
        scaled = apply_adoption_scaling(
            load,
            adoption_part,
            location_columns=location_columns,
        )
        if scaled.empty:
            continue
        scaled.to_csv(
            output_15_path,
            mode="w" if first_15 else "a",
            header=first_15,
            index=False,
        )
        total_15 += len(scaled)
        first_15 = False

        hourly = aggregate_to_hourly(scaled, location_columns=hourly_location_columns)
        hourly.to_csv(
            output_hourly_path,
            mode="w" if first_hourly else "a",
            header=first_hourly,
            index=False,
        )
        total_hourly += len(hourly)
        first_hourly = False

    if first_15:
        write_csv(pd.DataFrame(), output_15_path)
    if first_hourly:
        write_csv(pd.DataFrame(), output_hourly_path)
    return total_15, total_hourly


def parse_years(value: str) -> list[int]:
    if not value.strip():
        return []
    years = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        years.append(int(part))
    return sorted(set(years))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Albany behavioral EV charging load curves")
    ap.add_argument(
        "--trip-input",
        default="Albany_Trip_Behaviour/data/raw/Albany_County_NYU_Team.csv",
    )
    ap.add_argument("--output-dir", default="out_behavioral_charging")
    ap.add_argument("--sample-rows", type=int, default=None)
    ap.add_argument("--efficiency-kwh-per-mile", type=float, default=0.42)
    ap.add_argument("--availability-scenario", choices=sorted(AVAILABILITY_MULTIPLIER), default="base")
    ap.add_argument(
        "--availability-choice-model",
        choices=["gridup", "static"],
        default="gridup",
        help="Charger availability model: GridUp inverse-demand adjustment or static peak probabilities.",
    )
    ap.add_argument("--charger-assumptions", default=None)
    ap.add_argument("--modes", default="unmanaged,managed", help="Comma-separated: unmanaged,managed")
    ap.add_argument("--bin-minutes", type=int, default=15)
    ap.add_argument("--min-dwell-min", type=float, default=20.0)
    ap.add_argument("--long-public-dwell-min", type=float, default=90.0)
    ap.add_argument("--max-fallback-trip-miles", type=float, default=500.0)
    ap.add_argument("--distance-source", choices=["trip", "route"], default="route")
    ap.add_argument("--origin-mode", choices=["reconstructed", "input"], default="reconstructed")
    ap.add_argument("--coordinate-tolerance", type=float, default=0.001)
    ap.add_argument("--point-precision", type=int, default=4)
    ap.add_argument("--h3-resolution", type=int, default=8)
    ap.add_argument("--skip-h3", action="store_true")
    ap.add_argument("--require-h3", action="store_true")
    ap.add_argument("--zcta-geojson", default="data/ny_new_york_zip_codes_geo.min.json")
    ap.add_argument("--zip-to-county", default="data/zip_to_county_ny.csv")
    ap.add_argument("--zcta-county-filter", default="ALBANY")
    ap.add_argument("--skip-zcta", action="store_true")
    ap.add_argument("--assign-charging-zcta", action="store_true")
    ap.add_argument("--allow-nearest-zcta-fallback", action="store_true")
    ap.add_argument("--nearest-zcta-max-miles", type=float, default=2.0)
    ap.add_argument("--scale-adoption", action="store_true")
    ap.add_argument("--allow-missing-home-zcta", action="store_true")
    ap.add_argument(
        "--adoption-geography-type",
        choices=["zcta", "usps_zip_mapped_to_zcta"],
        default="usps_zip_mapped_to_zcta",
    )
    ap.add_argument("--use-demo-adoption", action="store_true")
    ap.add_argument("--forecast-years", default="2025,2030,2035,2040")
    ap.add_argument("--adoption-file", default=None)
    ap.add_argument("--write-point-home-load", action="store_true")
    ap.add_argument("--write-trip-table", action="store_true")
    ap.add_argument("--write-stop-table", action="store_true")
    args = ap.parse_args()

    trip_input = resolve_path(args.trip_input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = {m.strip().lower() for m in args.modes.split(",") if m.strip()}
    invalid_modes = modes - {"unmanaged", "managed"}
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {sorted(invalid_modes)}")
    if not modes:
        raise ValueError("--modes must include unmanaged, managed, or both")
    if args.bin_minutes <= 0 or 1440 % args.bin_minutes != 0:
        raise ValueError("--bin-minutes must be a positive divisor of 1440")
    if args.scale_adoption and args.skip_h3:
        raise ValueError(
            "--scale-adoption requires charging H3 outputs; remove --skip-h3 "
            "or run an unscaled QA/debug pass instead"
        )

    raw, read_summary = read_trip_file(trip_input, sample_rows=args.sample_rows)
    trips, trip_qa = reconstruct_vehicle_day_trips(
        raw,
        max_fallback_miles=args.max_fallback_trip_miles,
        distance_source=args.distance_source,
        origin_mode=args.origin_mode,
        coordinate_tolerance=args.coordinate_tolerance,
    )
    charger_assumptions_path = (
        resolve_path(args.charger_assumptions) if args.charger_assumptions else None
    )
    charger_assumptions = load_charger_assumptions(charger_assumptions_path)
    stops = classify_stop_opportunities(
        trips,
        availability_scenario=args.availability_scenario,
        availability_choice_model=args.availability_choice_model,
        charger_assumptions=charger_assumptions,
        min_dwell_min=args.min_dwell_min,
        long_public_dwell_min=args.long_public_dwell_min,
    )

    events_by_mode = []
    vehicle_summaries = []
    for mode in sorted(modes):
        managed = mode == "managed"
        scenario_id = (
            f"{args.availability_scenario}_{mode}_ldv_"
            f"{args.efficiency_kwh_per_mile:.3f}kwh_per_mile"
        )
        events, vehicle_summary = build_charging_events(
            stops,
            scenario_id=scenario_id,
            managed=managed,
            efficiency_kwh_per_mile=args.efficiency_kwh_per_mile,
        )
        events_by_mode.append(events)
        vehicle_summaries.append(vehicle_summary)

    events = pd.concat(events_by_mode, ignore_index=True) if events_by_mode else pd.DataFrame()
    vehicle_summary = (
        pd.concat(vehicle_summaries, ignore_index=True) if vehicle_summaries else pd.DataFrame()
    )
    events, zcta_summary = add_zcta_geographies(
        events,
        zcta_geojson_path=None if args.skip_zcta else resolve_path(args.zcta_geojson),
        zip_to_county_path=None if args.skip_zcta else resolve_path(args.zip_to_county),
        zcta_county_filter=None if args.skip_zcta else args.zcta_county_filter,
        assign_charging_zcta=args.assign_charging_zcta,
        allow_nearest_zcta_fallback=args.allow_nearest_zcta_fallback,
        nearest_zcta_max_miles=args.nearest_zcta_max_miles,
    )
    events, spatial_summary = add_charging_spatial_ids(
        events,
        point_precision=args.point_precision,
        h3_resolution=None if args.skip_h3 else args.h3_resolution,
        require_h3=(not args.skip_h3) or args.require_h3,
    )

    outputs: dict[str, int] = {}
    if args.write_trip_table:
        outputs["vehicle_day_trips.csv"] = write_csv(trips, output_dir / "vehicle_day_trips.csv")
    distance_qa_cols = [
        "source_row_id",
        "home_record_id",
        "PERSONID",
        "vehicle_day_id",
        "trip_chain_sequence",
        "TDTRPNUM",
        "origin_lon",
        "origin_lat",
        "destination_lon",
        "destination_lat",
        "reported_trip_miles",
        "route_time_miles",
        "route_path_miles",
        "straight_line_miles",
        "network_miles",
        "selected_distance_source",
        "route_time_below_straightline_flag",
        "route_path_below_straightline_flag",
        "trip_to_route_time_ratio",
        "trip_to_straight_line_ratio",
        "route_to_straight_line_ratio",
        "raw_dwell_min",
        "inferred_dwell_min",
        "raw_dwell_gap_error_min",
    ]
    outputs["distance_qa.csv"] = write_csv(
        trips[[c for c in distance_qa_cols if c in trips.columns]],
        output_dir / "distance_qa.csv",
    )
    if args.write_stop_table:
        outputs["stop_opportunities.csv"] = write_csv(stops, output_dir / "stop_opportunities.csv")

    outputs["charging_events.csv"] = write_csv(events, output_dir / "charging_events.csv")
    outputs["vehicle_day_energy_summary.csv"] = write_csv(
        vehicle_summary, output_dir / "vehicle_day_energy_summary.csv"
    )

    point_cols = ["charging_point_id", "charging_lon", "charging_lat"]
    load_15_point = build_load_curve(
        events,
        location_columns=point_cols,
        bin_minutes=args.bin_minutes,
    )
    outputs["load_15min_by_point.csv"] = write_csv(
        load_15_point, output_dir / "load_15min_by_point.csv"
    )
    outputs["load_hourly_by_point.csv"] = write_csv(
        aggregate_to_hourly(load_15_point, location_columns=point_cols),
        output_dir / "load_hourly_by_point.csv",
    )

    point_home_cols = point_cols + ["home_zcta"]
    load_15_point_home = pd.DataFrame()
    if args.write_point_home_load or args.scale_adoption:
        load_15_point_home = build_load_curve(
            events,
            location_columns=point_home_cols,
            bin_minutes=args.bin_minutes,
        )
    if args.write_point_home_load:
        outputs["load_15min_by_point_home.csv"] = write_csv(
            load_15_point_home, output_dir / "load_15min_by_point_home.csv"
        )
        outputs["load_hourly_by_point_home.csv"] = write_csv(
            aggregate_to_hourly(load_15_point_home, location_columns=point_home_cols),
            output_dir / "load_hourly_by_point_home.csv",
        )

    home_cols = ["home_zcta"]
    load_15_home = build_load_curve(events, location_columns=home_cols, bin_minutes=args.bin_minutes)
    outputs["load_15min_by_home_zcta.csv"] = write_csv(
        load_15_home, output_dir / "load_15min_by_home_zcta.csv"
    )
    outputs["load_hourly_by_home_zcta.csv"] = write_csv(
        aggregate_to_hourly(load_15_home, location_columns=home_cols),
        output_dir / "load_hourly_by_home_zcta.csv",
    )

    outputs["peak_by_point.csv"] = write_csv(
        peak_summary(load_15_point, location_columns=point_cols),
        output_dir / "peak_by_point.csv",
    )
    outputs["peak_by_home_zcta.csv"] = write_csv(
        peak_summary(load_15_home, location_columns=home_cols),
        output_dir / "peak_by_home_zcta.csv",
    )

    if "charging_h3" in events.columns and events["charging_h3"].notna().any():
        h3_cols = ["charging_h3"]
        load_15_h3 = build_load_curve(
            events.dropna(subset=["charging_h3"]),
            location_columns=h3_cols,
            bin_minutes=args.bin_minutes,
        )
        outputs["load_15min_by_h3.csv"] = write_csv(
            load_15_h3, output_dir / "load_15min_by_h3.csv"
        )
        outputs["load_hourly_by_h3.csv"] = write_csv(
            aggregate_to_hourly(load_15_h3, location_columns=h3_cols),
            output_dir / "load_hourly_by_h3.csv",
        )
        h3_home_cols = ["home_zcta", "charging_h3"]
        full_ev_15_h3 = build_load_curve(
            events.dropna(subset=["home_zcta", "charging_h3"]),
            location_columns=h3_home_cols,
            bin_minutes=args.bin_minutes,
        )
        outputs["full_ev_load_15min_by_home_zcta_charging_h3.csv"] = write_csv(
            full_ev_15_h3,
            output_dir / "full_ev_load_15min_by_home_zcta_charging_h3.csv",
        )
        outputs["full_ev_load_hourly_by_home_zcta_charging_h3.csv"] = write_csv(
            aggregate_to_hourly(full_ev_15_h3, location_columns=h3_home_cols),
            output_dir / "full_ev_load_hourly_by_home_zcta_charging_h3.csv",
        )

    adoption_summary: dict[str, Any] = {
        "adoption_geography_type": args.adoption_geography_type,
    }
    if args.scale_adoption:
        if not args.allow_missing_home_zcta and events["home_zcta"].isna().any():
            missing_rows = int(events["home_zcta"].isna().sum())
            raise ValueError(
                "Adoption scaling requested, but load-bearing events have missing home_zcta. "
                f"Missing event rows: {missing_rows}. Fix ZCTA assignment or set "
                "--allow-missing-home-zcta for an explicit partial-coverage run."
            )
        if args.adoption_file:
            adoption = load_adoption_file(resolve_path(args.adoption_file))
        else:
            if not args.use_demo_adoption:
                raise ValueError(
                    "--scale-adoption requires --adoption-file with home_zcta unless "
                    "--use-demo-adoption is set for software testing"
                )
            years = parse_years(args.forecast_years)
            adoption = build_default_adoption_scenarios(events["home_zcta"].dropna(), years)
            outputs["default_adoption_scenarios.csv"] = write_csv(
                adoption, output_dir / "default_adoption_scenarios.csv"
            )
        modeled_home_zctas = sorted(events["home_zcta"].dropna().astype(str).unique().tolist())
        adoption_home_zctas = sorted(adoption["home_zcta"].dropna().astype(str).unique().tolist())
        missing_from_adoption = sorted(set(modeled_home_zctas) - set(adoption_home_zctas))
        extra_in_adoption = sorted(set(adoption_home_zctas) - set(modeled_home_zctas))
        adoption_summary.update(
            {
                "modeled_home_zcta_count": len(modeled_home_zctas),
                "adoption_home_zcta_count": len(adoption_home_zctas),
                "modeled_home_zctas": modeled_home_zctas,
                "adoption_home_zctas": adoption_home_zctas,
                "missing_modeled_home_zctas_in_adoption": missing_from_adoption,
                "extra_adoption_home_zctas_not_modeled": extra_in_adoption,
                "adoption_scenarios": sorted(
                    adoption["adoption_scenario"].dropna().astype(str).unique().tolist()
                ),
                "forecast_years": sorted(
                    pd.to_numeric(adoption["forecast_year"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                ),
            }
        )
        if "charging_h3" in events.columns and events["charging_h3"].notna().any():
            if "full_ev_15_h3" not in locals():
                full_ev_15_h3 = build_load_curve(
                    events.dropna(subset=["home_zcta", "charging_h3"]),
                    location_columns=["home_zcta", "charging_h3"],
                    bin_minutes=args.bin_minutes,
                )
            scaled_15_rows, scaled_hourly_rows = write_adoption_scaled_outputs(
                full_ev_15_h3,
                adoption,
                location_columns=["charging_h3"],
                output_15_path=output_dir / "scaled_load_15min_by_charging_h3.csv",
                output_hourly_path=output_dir / "scaled_load_hourly_by_charging_h3.csv",
            )
            outputs["scaled_load_15min_by_charging_h3.csv"] = scaled_15_rows
            outputs["scaled_load_hourly_by_charging_h3.csv"] = scaled_hourly_rows
        if args.write_point_home_load:
            scaled_point_15_rows, scaled_point_hourly_rows = write_adoption_scaled_outputs(
                load_15_point_home,
                adoption,
                location_columns=["charging_point_id", "charging_lon", "charging_lat"],
                output_15_path=output_dir / "scaled_load_15min_by_point.csv",
                output_hourly_path=output_dir / "scaled_load_hourly_by_point.csv",
            )
            outputs["scaled_load_15min_by_point.csv"] = scaled_point_15_rows
            outputs["scaled_load_hourly_by_point.csv"] = scaled_point_hourly_rows

    stop_counts = (
        stops["charge_location_type"].value_counts(dropna=False).rename_axis("charge_location_type")
        .reset_index(name="stop_count")
    )
    outputs["stop_type_counts.csv"] = write_csv(stop_counts, output_dir / "stop_type_counts.csv")

    event_summary = (
        events.groupby(["scenario_id", "managed_flag", "charge_location_type", "charger_type"], as_index=False)
        .agg(
            events=("charging_stop_id", "count"),
            expected_kwh=("energy_delivered_kwh", "sum"),
            mean_expected_power_kw=("expected_power_kw", "mean"),
        )
        if not events.empty
        else pd.DataFrame()
    )
    outputs["event_summary_by_type.csv"] = write_csv(
        event_summary, output_dir / "event_summary_by_type.csv"
    )

    total_daily_energy = float(vehicle_summary["daily_energy_kwh"].sum()) if not vehicle_summary.empty else 0.0
    total_delivered = (
        vehicle_summary.groupby(["scenario_id", "managed_flag"], as_index=False)
        .agg(
            vehicle_days=("vehicle_day_id", "count"),
            daily_energy_kwh=("daily_energy_kwh", "sum"),
            energy_delivered_kwh=("energy_delivered_kwh", "sum"),
            unmet_energy_kwh=("unmet_energy_kwh", "sum"),
        )
        if not vehicle_summary.empty
        else pd.DataFrame()
    )
    outputs["scenario_energy_summary.csv"] = write_csv(
        total_delivered, output_dir / "scenario_energy_summary.csv"
    )

    summary = {
        "model": "Albany LDV behavioral charging",
        "read_summary": read_summary,
        "trip_qa": trip_qa,
        "zcta_summary": zcta_summary,
        "spatial_summary": spatial_summary,
        "adoption_summary": adoption_summary,
        "configuration": {
            "efficiency_kwh_per_mile": args.efficiency_kwh_per_mile,
            "availability_scenario": args.availability_scenario,
            "availability_choice_model": args.availability_choice_model,
            "charger_assumptions": str(charger_assumptions_path) if charger_assumptions_path else "built_in_default",
            "modes": sorted(modes),
            "bin_minutes": args.bin_minutes,
            "min_dwell_min": args.min_dwell_min,
            "long_public_dwell_min": args.long_public_dwell_min,
            "distance_source": args.distance_source,
            "origin_mode": args.origin_mode,
            "point_precision": args.point_precision,
            "h3_resolution": args.h3_resolution,
            "skip_h3": args.skip_h3,
            "allow_nearest_zcta_fallback": args.allow_nearest_zcta_fallback,
            "nearest_zcta_max_miles": args.nearest_zcta_max_miles,
            "sample_rows": args.sample_rows,
            "scale_adoption": args.scale_adoption,
            "allow_missing_home_zcta": args.allow_missing_home_zcta,
            "adoption_geography_type": args.adoption_geography_type,
            "use_demo_adoption": args.use_demo_adoption,
            "write_point_home_load": args.write_point_home_load,
        },
        "outputs": outputs,
        "total_vehicle_summary_energy_kwh_across_modes": total_daily_energy,
    }
    with open(output_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
