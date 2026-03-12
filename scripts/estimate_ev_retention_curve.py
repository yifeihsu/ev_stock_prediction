#!/usr/bin/env python3
"""
Estimate an EV retention / survival curve from VIN-level "first-seen" and "last-seen"
snapshots.

Why this exists
---------------
Our adoption flow proxy is the count of VINs that are *first seen* in a region at a
given DMV snapshot (DMV_ID). On-road stock is the count of VINs present in the
snapshot inventory. Stock is not equal to cumulative first-seen inflow because
vehicles can leave the active inventory (expiration, move out of region/state, etc.).

This script estimates a retention curve:
  Pr(still on-road at lag | adopted at time 0)

using VIN-level spans (first snapshot in region, last snapshot in region) and
Kaplan–Meier style right-censoring at the last available snapshot.

Time scale
----------
We estimate survival as a function of *calendar-month lag* between the first and
last snapshot months:
  lag_months = month(last_seen) - month(first_seen)

and use event_time = lag_months + 1 (so survival at lag 0 is always 1).

Inputs
------
- split_part_*.csv (no header; see repo docs)
- Vehicle Descriptions.csv (maps VIN_Key -> Drivetrain_Type)
- ZIP -> county crosswalk (data/zip_to_county_ny.csv)
- Optional ZIP -> region overrides (data/utility_zip_regions.csv)
- NY_DMV_Snapshots.csv (DMV_ID -> snapshot date)

Output
------
- CSV with columns: lag_months, survival_prob, n_at_risk, n_events

Example
-------
python scripts/estimate_ev_retention_curve.py \
  --inputs-glob "split_part_*.csv" \
  --descriptions "Vehicle Descriptions.csv" \
  --zip-to-county data/zip_to_county_ny.csv \
  --zip-to-region data/utility_zip_regions.csv \
  --regions "LIPA:NASSAU,SUFFOLK" \
  --snapshot-map NY_DMV_Snapshots.csv \
  --max-dmv-id 102 \
  --output covariates/retention_LIPA_ev_km.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_ev_vin_keys(descriptions_path: str) -> set[str]:
    ev_keys: set[str] = set()
    with open(descriptions_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row.get("VIN_Key") or "").strip()
            if not k:
                continue
            dt = (row.get("Drivetrain_Type") or "").strip().upper()
            if dt in ("BEV", "PHEV"):
                ev_keys.add(k)
    return ev_keys


def load_zip_to_county(crosswalk_path: str) -> Dict[str, str]:
    with open(crosswalk_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        headers = {h.lower(): h for h in (r.fieldnames or [])}
        zcol = headers.get("zip")
        ccol = headers.get("county_name")
        if not zcol or not ccol:
            raise ValueError("zip_to_county must have columns 'zip' and 'county_name'")
        out: Dict[str, str] = {}
        for row in r:
            z = (row.get(zcol) or "").strip()
            if not z:
                continue
            z5 = z.zfill(5) if z.isdigit() and len(z) <= 5 else z
            cname = (row.get(ccol) or "").strip().upper()
            if cname:
                out[z5] = cname
    return out


def parse_regions(spec: str) -> Dict[str, List[str]]:
    regions: Dict[str, List[str]] = {}
    for seg in (spec or "").split(";"):
        seg = seg.strip()
        if not seg:
            continue
        if ":" not in seg:
            raise ValueError(f"Bad region segment: {seg}")
        name, cs = seg.split(":", 1)
        counties = [c.strip().upper() for c in cs.split(",") if c.strip()]
        if not counties:
            raise ValueError(f"Region {name} has no counties")
        regions[name.strip()] = counties
    if not regions:
        raise ValueError("No regions parsed")
    return regions


def load_zip_to_region_override(path: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        headers = {h.lower(): h for h in (r.fieldnames or [])}
        zcol = headers.get("zip")
        rcol = headers.get("region")
        if not zcol or not rcol:
            raise ValueError("--zip-to-region must have columns 'zip' and 'region'")
        for row in r:
            z = (row.get(zcol) or "").strip()
            if not z:
                continue
            z5 = z.zfill(5) if z.isdigit() and len(z) <= 5 else z
            reg = (row.get(rcol) or "").strip()
            if reg:
                out.setdefault(z5, []).append(reg)
    return out


def load_snapshot_dates(snapshot_map: str) -> Dict[int, pd.Timestamp]:
    snap = pd.read_csv(snapshot_map, encoding="utf-8-sig")
    if "DMV_ID" not in snap.columns or "DMV_Snapshot_Date" not in snap.columns:
        raise ValueError("snapshot map must have DMV_ID and DMV_Snapshot_Date columns")
    snap["DMV_ID"] = pd.to_numeric(snap["DMV_ID"], errors="coerce").astype("Int64")
    snap["DMV_Snapshot_Date"] = pd.to_datetime(snap["DMV_Snapshot_Date"], errors="coerce")
    snap = snap.dropna(subset=["DMV_ID", "DMV_Snapshot_Date"])
    out: Dict[int, pd.Timestamp] = {}
    for did, dt in zip(snap["DMV_ID"].astype(int).tolist(), snap["DMV_Snapshot_Date"].tolist()):
        out[int(did)] = pd.Timestamp(dt)
    return out


def month_index(ts: pd.Timestamp) -> int:
    return int(ts.year) * 12 + int(ts.month)


def kaplan_meier_months(durations_months: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Return survival at each lag_months (0..max) using discrete-time KM."""
    if durations_months.size == 0:
        raise ValueError("No durations provided for survival estimation")
    # event/censor time in months, shifted by +1 so lag 0 survival = 1 always.
    times = durations_months.astype(int) + 1
    max_time = int(times.max())

    counts = np.bincount(times, minlength=max_time + 1)
    event_counts = np.bincount(times[events.astype(bool)], minlength=max_time + 1)

    # At risk just before time t: number with time >= t.
    at_risk = np.cumsum(counts[::-1])[::-1]

    surv = np.ones(max_time + 1, dtype=float)
    # survival at lag 0 = 1
    for t in range(1, max_time + 1):
        n = float(at_risk[t])
        d = float(event_counts[t])
        if n <= 0:
            surv[t] = surv[t - 1]
        else:
            surv[t] = surv[t - 1] * (1.0 - d / n)

    # Map "time" back to "lag months": lag m corresponds to time t=m
    rows = []
    n_total = int(len(times))
    rows.append({"lag_months": 0, "survival_prob": 1.0, "n_at_risk": n_total, "n_events": 0})
    for m in range(1, max_time + 1):
        rows.append(
            {
                "lag_months": int(m),
                "survival_prob": float(surv[m]),
                "n_at_risk": int(at_risk[m]),
                "n_events": int(event_counts[m]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate EV retention curve from last-seen VINs.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", nargs="*")
    g.add_argument("--inputs-glob")
    ap.add_argument("--descriptions", required=True, help="Path to Vehicle Descriptions.csv")
    ap.add_argument("--zip-to-county", required=True)
    ap.add_argument("--zip-to-region", help="Optional ZIP->region override CSV (zip,region)")
    ap.add_argument(
        "--regions",
        required=True,
        help='Region spec like "LIPA:NASSAU,SUFFOLK;CHGE:DUTCHESS,ULSTER"',
    )
    ap.add_argument("--snapshot-map", required=True, help="NY_DMV_Snapshots.csv")
    ap.add_argument(
        "--max-dmv-id",
        type=int,
        default=None,
        help="Ignore rows with DMV_ID greater than this (useful to match precomputed series).",
    )
    ap.add_argument(
        "--cohort-start",
        type=str,
        default=None,
        help="If set (YYYY-MM-DD), include only VINs whose first-seen snapshot date is on/after this date.",
    )
    ap.add_argument("--output", required=True, help="Output CSV path for retention curve")
    args = ap.parse_args()

    inputs = sorted(glob.glob(args.inputs_glob)) if args.inputs_glob else list(args.inputs or [])
    if not inputs:
        raise SystemExit("No input files found")

    snapshot_dates = load_snapshot_dates(args.snapshot_map)
    if not snapshot_dates:
        raise SystemExit("No snapshot dates loaded")

    max_dmv = args.max_dmv_id if args.max_dmv_id is not None else max(snapshot_dates.keys())
    cohort_start_ts = pd.to_datetime(args.cohort_start) if args.cohort_start else None

    print(f"Loading EV VIN_Keys from: {args.descriptions}")
    ev_keys = load_ev_vin_keys(args.descriptions)
    print(f"EV VIN_Keys: {len(ev_keys):,}")

    print(f"Loading ZIP->County: {args.zip_to_county}")
    zip_to_county = load_zip_to_county(args.zip_to_county)
    print(f"ZIPs in crosswalk: {len(zip_to_county):,}")

    regions = parse_regions(args.regions)
    county_to_regions: Dict[str, List[str]] = {}
    for rname, clist in regions.items():
        for c in clist:
            county_to_regions.setdefault(c.upper(), []).append(rname)

    zip_to_region: Dict[str, List[str]] = {}
    if args.zip_to_region:
        print(f"Loading ZIP->Region overrides: {args.zip_to_region}")
        zip_to_region = load_zip_to_region_override(args.zip_to_region)
        print(f"ZIP overrides: {len(zip_to_region):,}")

    # region -> vin -> (first_dmv, last_dmv)
    spans: Dict[str, Dict[str, List[int]]] = {r: {} for r in regions}
    target_regions = set(spans.keys())

    counters = {
        "rows_total": 0,
        "rows_ev": 0,
        "rows_used": 0,
        "rows_skipped_bad_zip": 0,
        "rows_skipped_no_region": 0,
        "rows_skipped_bad_dmv": 0,
        "rows_skipped_over_max_dmv": 0,
    }

    for path in inputs:
        print(f"Scanning {path} ...")
        with open(path, "r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            for row in r:
                counters["rows_total"] += 1
                if len(row) < 9:
                    continue

                vin_key = (row[7] or "").strip()
                if vin_key not in ev_keys:
                    continue
                counters["rows_ev"] += 1

                zip_code = (row[3] or "").strip()
                if not zip_code:
                    counters["rows_skipped_bad_zip"] += 1
                    continue
                z5 = zip_code.zfill(5) if zip_code.isdigit() and len(zip_code) <= 5 else zip_code

                try:
                    dmv_id = int(row[5])
                except Exception:
                    counters["rows_skipped_bad_dmv"] += 1
                    continue
                if dmv_id > max_dmv:
                    counters["rows_skipped_over_max_dmv"] += 1
                    continue

                cregions: List[str] = []
                county = zip_to_county.get(z5)
                if county:
                    cregions = county_to_regions.get(county.upper(), [])
                # ZIP overrides may include regions we are not estimating in this run.
                zregions = [r for r in zip_to_region.get(z5, []) if r in target_regions]
                rlist = list({*cregions, *zregions})
                if not rlist:
                    counters["rows_skipped_no_region"] += 1
                    continue

                vin = row[0]
                for reg in rlist:
                    if reg not in target_regions:
                        continue
                    cur = spans[reg].get(vin)
                    if cur is None:
                        spans[reg][vin] = [dmv_id, dmv_id]
                    else:
                        if dmv_id < cur[0]:
                            cur[0] = dmv_id
                        if dmv_id > cur[1]:
                            cur[1] = dmv_id
                counters["rows_used"] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Estimate per-region survival and write one CSV per region (when multiple provided).
    # If the output path is a directory, write {region}_retention.csv inside it.
    output_is_dir = out_path.suffix.lower() != ".csv"
    out_dir = out_path if output_is_dir else out_path.parent

    for reg, mapping in spans.items():
        if not mapping:
            print(f"Warning: no EV VINs collected for region {reg}; skipping.")
            continue

        first_month = []
        last_month = []
        is_event = []

        for vin, (first_id, last_id) in mapping.items():
            first_dt = snapshot_dates.get(first_id)
            last_dt = snapshot_dates.get(last_id)
            if first_dt is None or last_dt is None:
                continue
            if cohort_start_ts is not None and first_dt < cohort_start_ts:
                continue
            fm = month_index(first_dt)
            lm = month_index(last_dt)
            if lm < fm:
                continue
            first_month.append(fm)
            last_month.append(lm)
            is_event.append(1 if last_id < max_dmv else 0)

        if not first_month:
            print(f"Warning: region {reg} has zero usable VIN spans after filtering.")
            continue

        first_month = np.array(first_month, dtype=int)
        last_month = np.array(last_month, dtype=int)
        durations = last_month - first_month
        events = np.array(is_event, dtype=bool)

        km = kaplan_meier_months(durations_months=durations, events=events)
        km.insert(0, "region", reg)
        km.attrs["n_vins"] = int(len(durations))
        km.attrs["max_dmv_id"] = int(max_dmv)

        out_file = out_dir / (f"retention_{reg}_ev_km.csv" if output_is_dir else out_path.name)
        if not output_is_dir and len(spans) > 1:
            out_file = out_dir / f"retention_{reg}_ev_km.csv"
        km.to_csv(out_file, index=False)
        print(f"Wrote {out_file} (n_vins={len(durations):,}, max_lag={km['lag_months'].max()})")

    print("Done.")
    print("Counters:", counters)


if __name__ == "__main__":
    main()
