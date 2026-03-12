#!/usr/bin/env python3
"""
Build a LIPA light-duty snapshot panel for Bass modeling.

Outputs a CSV with columns:
  DMV_ID,date,stock_ev_t,stock_all_t,flow_ev_t

Definitions
  - Light-duty: Vehicle_GVWR_Category == "Light-Duty (Class 1-2A)"
  - EV: Drivetrain_Type in {BEV, PHEV}, restricted to light-duty rows
  - stock_*_t: approximate unique VINs in each DMV snapshot (HLL)
  - flow_ev_t: first-seen EV VINs in LIPA at each DMV snapshot

Region assignment (LIPA)
  - County-based: NASSAU or SUFFOLK
  - ZIP override: rows in data/utility_zip_regions.csv with region=LIPA
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]


class HyperLogLog:
    def __init__(self, p: int = 16):
        if not (4 <= p <= 20):
            raise ValueError("p must be in [4, 20]")
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        if self.m == 16:
            self.alpha_m = 0.673
        elif self.m == 32:
            self.alpha_m = 0.697
        elif self.m == 64:
            self.alpha_m = 0.709
        else:
            self.alpha_m = 0.7213 / (1 + 1.079 / self.m)

    @staticmethod
    def _hash64(x: str) -> int:
        h = hashlib.blake2b(x.encode("utf-8"), digest_size=8)
        return int.from_bytes(h.digest(), "big", signed=False)

    def add(self, x: str):
        v = self._hash64(x)
        j = v >> (64 - self.p)
        rem = v & ((1 << (64 - self.p)) - 1)
        if rem == 0:
            rho = (64 - self.p) + 1
        else:
            lz = (64 - self.p) - rem.bit_length()
            rho = lz + 1
        if rho > self.registers[j]:
            self.registers[j] = rho

    def count(self) -> int:
        m = self.m
        z_inv = 0.0
        v_zero = 0
        for reg in self.registers:
            z_inv += 2.0 ** (-reg)
            if reg == 0:
                v_zero += 1
        est = self.alpha_m * (m * m) / z_inv
        if est <= 2.5 * m and v_zero > 0:
            import math

            est = m * (math.log(m / v_zero))
        return int(round(est))


def normalize_zip(z: str) -> str:
    z = (z or "").strip()
    if z.isdigit() and len(z) <= 5:
        return z.zfill(5)
    return z


def load_zip_to_county(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        headers = {h.lower(): h for h in (r.fieldnames or [])}
        zc = headers.get("zip")
        cc = headers.get("county_name")
        if not zc or not cc:
            raise ValueError(f"{path} must contain zip and county_name")
        for row in r:
            z = normalize_zip(row.get(zc, ""))
            if not z:
                continue
            c = (row.get(cc) or "").strip().upper()
            if c:
                out.setdefault(z, c)
    return out


def load_zip_to_region(path: Path, *, target_region: str) -> Set[str]:
    if not path.exists():
        return set()
    region = target_region.strip().upper()
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        headers = {h.lower(): h for h in (r.fieldnames or [])}
        zc = headers.get("zip")
        rc = headers.get("region")
        if not zc or not rc:
            raise ValueError(f"{path} must contain zip and region")
        for row in r:
            if (row.get(rc) or "").strip().upper() != region:
                continue
            z = normalize_zip(row.get(zc, ""))
            if z:
                out.add(z)
    return out


def load_light_duty_key_sets(descriptions_path: Path) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    light_keys: Set[str] = set()
    ev_light_keys: Set[str] = set()
    stats = {
        "desc_rows_total": 0,
        "desc_rows_light_duty": 0,
        "desc_rows_light_duty_ev": 0,
    }
    with open(descriptions_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        need = {"VIN_Key", "Vehicle_GVWR_Category", "Drivetrain_Type"}
        missing = [c for c in need if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"{descriptions_path} missing required columns: {missing}")
        for row in r:
            stats["desc_rows_total"] += 1
            key = (row.get("VIN_Key") or "").strip()
            if not key:
                continue
            cat = (row.get("Vehicle_GVWR_Category") or "").strip()
            if cat != "Light-Duty (Class 1-2A)":
                continue
            stats["desc_rows_light_duty"] += 1
            light_keys.add(key)
            dt = (row.get("Drivetrain_Type") or "").strip().upper()
            if dt in ("BEV", "PHEV"):
                ev_light_keys.add(key)
                stats["desc_rows_light_duty_ev"] += 1
    return light_keys, ev_light_keys, stats


def load_snapshot_map(path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                dmv_id = int(row.get("DMV_ID") or 0)
            except Exception:
                continue
            date = (row.get("DMV_Snapshot_Date") or "").strip()
            if dmv_id > 0 and date:
                rows.append((dmv_id, date))
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise ValueError(f"No valid DMV_ID/date rows in {path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build LIPA light-duty panel")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", nargs="*")
    g.add_argument("--inputs-glob", default="split_part_*.csv")
    ap.add_argument("--descriptions", default="Vehicle Descriptions.csv")
    ap.add_argument("--zip-to-county", default="data/zip_to_county_ny.csv")
    ap.add_argument("--zip-to-region", default="data/utility_zip_regions.csv")
    ap.add_argument("--snapshot-map", default="NY_DMV_Snapshots.csv")
    ap.add_argument("--region", default="LIPA")
    ap.add_argument("--counties", default="NASSAU,SUFFOLK")
    ap.add_argument("--hll-p", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=2_000_000)
    ap.add_argument("--output", default="covariates/panel_LIPA_light_duty_snapshot.csv")
    ap.add_argument("--summary", default="covariates/panel_LIPA_light_duty_summary.json")
    args = ap.parse_args()

    inputs = sorted(glob.glob(args.inputs_glob)) if args.inputs_glob else list(args.inputs or [])
    if not inputs:
        raise SystemExit("No input files found")

    descriptions_path = (ROOT / args.descriptions) if not Path(args.descriptions).is_absolute() else Path(args.descriptions)
    zip_to_county_path = (ROOT / args.zip_to_county) if not Path(args.zip_to_county).is_absolute() else Path(args.zip_to_county)
    zip_to_region_path = (ROOT / args.zip_to_region) if not Path(args.zip_to_region).is_absolute() else Path(args.zip_to_region)
    snapshot_map_path = (ROOT / args.snapshot_map) if not Path(args.snapshot_map).is_absolute() else Path(args.snapshot_map)
    output_path = (ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    summary_path = (ROOT / args.summary) if not Path(args.summary).is_absolute() else Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    region = args.region.strip().upper()
    counties = {c.strip().upper() for c in args.counties.split(",") if c.strip()}
    if not counties:
        raise ValueError("--counties cannot be empty")

    zip_to_county = load_zip_to_county(zip_to_county_path)
    zip_overrides = load_zip_to_region(zip_to_region_path, target_region=region)
    light_keys, ev_light_keys, desc_stats = load_light_duty_key_sets(descriptions_path)
    snapshots = load_snapshot_map(snapshot_map_path)

    all_hll: Dict[int, HyperLogLog] = defaultdict(lambda: HyperLogLog(p=args.hll_p))
    ev_hll: Dict[int, HyperLogLog] = defaultdict(lambda: HyperLogLog(p=args.hll_p))
    first_seen_ev: Dict[str, int] = {}

    counters = {
        "rows_total": 0,
        "rows_bad": 0,
        "rows_in_region": 0,
        "rows_light_duty": 0,
        "rows_light_duty_ev": 0,
        "rows_outside_region": 0,
    }

    for p in inputs:
        local = 0
        with open(p, "r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            for row in r:
                counters["rows_total"] += 1
                local += 1
                if args.log_every and local % args.log_every == 0:
                    print(f"[{Path(p).name}] scanned {local:,} rows")
                if len(row) < 8:
                    counters["rows_bad"] += 1
                    continue
                vin = (row[0] or "").strip()
                z5 = normalize_zip(row[3])
                if not vin or not z5:
                    counters["rows_bad"] += 1
                    continue
                try:
                    dmv_id = int(row[5])
                except Exception:
                    counters["rows_bad"] += 1
                    continue

                county = zip_to_county.get(z5)
                in_region = (county in counties) or (z5 in zip_overrides)
                if not in_region:
                    counters["rows_outside_region"] += 1
                    continue
                counters["rows_in_region"] += 1

                vin_key = (row[7] or "").strip()
                if vin_key not in light_keys:
                    continue
                counters["rows_light_duty"] += 1
                all_hll[dmv_id].add(vin)

                if vin_key in ev_light_keys:
                    counters["rows_light_duty_ev"] += 1
                    ev_hll[dmv_id].add(vin)
                    prev = first_seen_ev.get(vin)
                    if prev is None or dmv_id < prev:
                        first_seen_ev[vin] = dmv_id

    flow_ev = Counter(first_seen_ev.values())
    observed_ids = set(all_hll.keys()) | set(ev_hll.keys()) | set(flow_ev.keys())
    max_observed_dmv_id = max(observed_ids) if observed_ids else -1

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["DMV_ID", "date", "stock_ev_t", "stock_all_t", "flow_ev_t"])
        for dmv_id, date in snapshots:
            # Trim trailing snapshots that are entirely unobserved in the source extracts
            # (e.g., DMV_ID 103 when splits only contain data through DMV_ID 102).
            if max_observed_dmv_id > 0 and dmv_id > max_observed_dmv_id:
                continue
            w.writerow(
                [
                    dmv_id,
                    date,
                    ev_hll[dmv_id].count() if dmv_id in ev_hll else 0,
                    all_hll[dmv_id].count() if dmv_id in all_hll else 0,
                    int(flow_ev.get(dmv_id, 0)),
                ]
            )

    summary = {
        "region": region,
        "counties": sorted(counties),
        "zip_override_count": len(zip_overrides),
        "inputs": inputs,
        "output": str(output_path),
        "summary_path": str(summary_path),
        "description_filter": {
            "vehicle_gvwr_category": "Light-Duty (Class 1-2A)",
            "ev_drivetrain_types": ["BEV", "PHEV"],
        },
        "counters": counters,
        "desc_stats": desc_stats,
        "unique_light_duty_ev_vins": len(first_seen_ev),
        "max_observed_dmv_id": max_observed_dmv_id,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Panel written: {output_path}")
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
