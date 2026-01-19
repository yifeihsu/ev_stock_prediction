#!/usr/bin/env python3
"""
Aggregate EV adoption trends by county/region across multiple split CSVs.

What it does
- Streams one or more `full_export` split files and joins `VIN_Key` to drivetrain.
- Maps ZIP -> County (via a provided crosswalk) and filters to configured regions
  (e.g., LIPA, CHGE) defined by county name sets.
- Deduplicates to unique VINs per month using HyperLogLog (HLL) to handle scale.
- Produces monthly unique VIN counts for EV (BEV + PHEV) and All vehicles per region.
- Optionally plots trends if matplotlib is available.

Inputs (assumed schema, no header):
  0 VIN (17 chars standard; non-standard possible)
  1 Registration_Valid_Date (YYYY-MM-DD)
  2 Registration_Expiration_Date (YYYY-MM-DD)
  3 ZIP (5-digit string, may include leading zeros; treat as text)
  4 County_GEOID (empty for NY)
  5 DMV_ID (numeric)
  6 State (e.g., NY)
  7 VIN_Key (first 10 chars of VIN)
  8 Reg Month (YYYY-MM-01)

Requirements
- A ZIP -> County crosswalk CSV with columns: `zip`, `county_name` (case-insensitive).
  Additional columns are ignored.
- `Vehicle Descriptions.csv` for joining `VIN_Key` -> `Drivetrain_Type`.

Usage example
  python scripts/aggregate_ev_trends_by_county.py \
    --inputs-glob "split_part_*.csv" \
    --descriptions "Vehicle Descriptions.csv" \
    --zip-to-county path/to/zip_to_county.csv \
    --regions "LIPA:NASSAU,SUFFOLK;CHGE:DUTCHESS,ULSTER" \
    --output-dir out_ev \
    [--plot]

Notes
- Uses HLL (p=16) per (region, month, metric) where metric in {EV, ALL}.
- County and region matching is case-insensitive.
- Records with unknown ZIP or unmapped county are skipped from region tallies.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
from collections import defaultdict


# ---------------------- HyperLogLog (64-bit) ----------------------

class HyperLogLog:
    """Minimal HyperLogLog implementation for approximate distinct counting.

    - 64-bit hash space (blake2b 8 bytes)
    - p controls buckets m=2^p (default p=16 -> 65,536 registers)
    - supports in-place union via register-wise max
    """

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
        h = hashlib.blake2b(x.encode('utf-8'), digest_size=8)
        return int.from_bytes(h.digest(), 'big', signed=False)

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

    def count(self) -> float:
        m = self.m
        Z_inv = 0.0
        V = 0
        for reg in self.registers:
            Z_inv += 2.0 ** (-reg)
            if reg == 0:
                V += 1
        E = self.alpha_m * (m * m) / Z_inv
        # Small-range correction
        if E <= 2.5 * m and V > 0:
            import math
            E = m * (math.log(m / V))
        return E

    def union_inplace(self, other: "HyperLogLog"):
        if self.p != other.p or self.m != other.m:
            raise ValueError("HLL union requires same p/m")
        regs = self.registers
        oregs = other.registers
        for i in range(self.m):
            if oregs[i] > regs[i]:
                regs[i] = oregs[i]


# ---------------------- IO helpers ----------------------

def load_vin_drivetrain(descriptions_path: str) -> Dict[str, str]:
    """Load VIN_Key -> Drivetrain_Type from Vehicle Descriptions.csv.
    Missing/blank drivetrain mapped to "UNKNOWN".
    """
    need = ("VIN_Key", "Drivetrain_Type")
    out: Dict[str, str] = {}
    with open(descriptions_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in need if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Vehicle Descriptions.csv missing columns: {missing}")
        for row in reader:
            k = (row.get("VIN_Key") or "").strip()
            if not k:
                continue
            d = (row.get("Drivetrain_Type") or "").strip() or "UNKNOWN"
            out[k] = d
    return out


def load_zip_to_county(crosswalk_path: str) -> Dict[str, str]:
    """Load ZIP -> county_name (uppercased) mapping.
    Expected columns: `zip` and `county_name` (case-insensitive). Extra columns ignored.
    """
    with open(crosswalk_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = {h.lower(): h for h in reader.fieldnames or []}
        if "zip" not in headers or "county_name" not in headers:
            raise ValueError(
                f"Crosswalk must have columns 'zip' and 'county_name'. Found: {reader.fieldnames}"
            )
        zip_col = headers["zip"]
        county_col = headers["county_name"]
        mapping: Dict[str, str] = {}
        for row in reader:
            z = (row.get(zip_col) or "").strip()
            if not z:
                continue
            # Normalize ZIP to 5-digit text (retain as given if not 5-digit)
            z5 = z.zfill(5) if z.isdigit() and len(z) <= 5 else z
            cname = (row.get(county_col) or "").strip().upper()
            if not cname:
                continue
            # Prefer first seen, but don't overwrite if dup appears
            mapping.setdefault(z5, cname)
    return mapping


def parse_regions(spec: str) -> Dict[str, List[str]]:
    """Parse region spec like: "LIPA:NASSAU,SUFFOLK;CHGE:DUTCHESS,ULSTER".
    Returns {region_name: [COUNTY_NAME,...]} with uppercase county names.
    """
    regions: Dict[str, List[str]] = {}
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad region spec segment: {part}")
        name, counties = part.split(":", 1)
        cs = [c.strip().upper() for c in counties.split(",") if c.strip()]
        if not cs:
            raise ValueError(f"Region '{name}' has no counties")
        regions[name.strip()] = cs
    if not regions:
        raise ValueError("No regions parsed from --regions")
    return regions


# ---------------------- Core aggregation ----------------------

def aggregate(
    input_paths: Iterable[str],
    vin_map: Dict[str, str],
    zip_to_county: Dict[str, str],
    regions: Dict[str, List[str]],
    *,
    hll_p: int = 16,
) -> Tuple[Dict[Tuple[str, str, str], HyperLogLog], Dict[str, int]]:
    """Aggregate monthly unique VIN counts per (region, month, metric).

    - Groups by key (region, month, metric), where metric in {"EV", "ALL"}.
    - Returns (hll_map, counters) where counters holds simple totals.
    """
    # HLL per group
    hll_map: Dict[Tuple[str, str, str], HyperLogLog] = defaultdict(lambda: HyperLogLog(p=hll_p))
    counters = {
        "rows_total": 0,
        "rows_skipped_no_zip": 0,
        "rows_skipped_no_county": 0,
        "rows_skipped_no_region": 0,
        "rows_skipped_no_month": 0,
    }

    def add(group: Tuple[str, str, str], vin: str):
        hll_map[group].add(vin)

    # Pre-index county->regions
    county_to_regions: Dict[str, List[str]] = {}
    for rname, clist in regions.items():
        for c in clist:
            county_to_regions.setdefault(c, []).append(rname)

    for path in input_paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                counters["rows_total"] += 1
                if len(row) < 9:
                    counters["rows_skipped_no_month"] += 1
                    continue

                vin = row[0]
                zip_code = (row[3] or "").strip()
                month = (row[8] or "").strip()
                if not month:
                    counters["rows_skipped_no_month"] += 1
                    continue
                if not zip_code:
                    counters["rows_skipped_no_zip"] += 1
                    continue

                # Normalize ZIP as 5-digit text when possible
                z5 = zip_code.zfill(5) if zip_code.isdigit() and len(zip_code) <= 5 else zip_code
                county = zip_to_county.get(z5)
                if not county:
                    counters["rows_skipped_no_county"] += 1
                    continue

                rlist = county_to_regions.get(county.upper())
                if not rlist:
                    counters["rows_skipped_no_region"] += 1
                    continue

                vin_key = (row[7] or "").strip()
                drv = vin_map.get(vin_key, "UNKNOWN")
                is_ev = (drv in ("BEV", "PHEV"))

                for rname in rlist:
                    add((rname, month, "ALL"), vin)
                    if is_ev:
                        add((rname, month, "EV"), vin)

    return hll_map, counters


def write_counts_csv(out_path: str, hll_map: Dict[Tuple[str, str, str], HyperLogLog]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "month", "metric", "approx_unique_vins"])
        for (region, month, metric) in sorted(hll_map.keys()):
            w.writerow([region, month, metric, int(round(hll_map[(region, month, metric)].count()))])


def maybe_plot(counts_csv: str, out_png: str, regions: List[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-GUI backend
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        print(f"Plotting skipped (matplotlib/pandas not available): {e}")
        return

    df = pd.read_csv(counts_csv)
    # Pivot to columns EV and ALL per region/month
    pivot = df.pivot_table(index=["region", "month"], columns="metric", values="approx_unique_vins", aggfunc="first").reset_index()
    # Ensure chronological order
    pivot = pivot.sort_values(["region", "month"])  # months are YYYY-MM-01 strings

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in regions:
        sub = pivot[pivot["region"] == r]
        if sub.empty:
            continue
        ax.plot(sub["month"], sub.get("EV", 0), label=f"{r} EV", linewidth=2)
        ax.plot(sub["month"], sub.get("ALL", 0), label=f"{r} All", linewidth=1, alpha=0.7)
    ax.set_title("Monthly Unique Vehicles: EV vs All")
    ax.set_xlabel("Month")
    ax.set_ylabel("Approx Unique VINs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="EV trends by county/region across multiple splits")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", nargs="*", help="Explicit list of input split CSVs")
    g.add_argument("--inputs-glob", help="Glob pattern for input split CSVs (e.g., split_part_*.csv)")
    ap.add_argument("--descriptions", required=True, help="Path to Vehicle Descriptions.csv")
    ap.add_argument("--zip-to-county", required=True, help="CSV mapping ZIP to county_name")
    ap.add_argument(
        "--zip-to-region",
        help="Optional CSV mapping ZIP to region (columns: zip,region). Useful for partial-county coverage.",
    )
    ap.add_argument(
        "--regions",
        default="LIPA:NASSAU,SUFFOLK;CHGE:DUTCHESS,ULSTER",
        help="Region spec like 'LIPA:NASSAU,SUFFOLK;CHGE:DUTCHESS,ULSTER'",
    )
    ap.add_argument("--output-dir", default="out_ev", help="Output directory")
    ap.add_argument("--hll-p", type=int, default=16)
    ap.add_argument("--plot", action="store_true", help="Emit PNG plot if libs are available")
    ap.add_argument(
        "--group-by",
        choices=["month", "snapshot"],
        default="month",
        help="Group by Reg Month (YYYY-MM-01) or DMV snapshot ID (DMV_ID). Default: month",
    )

    args = ap.parse_args()

    if args.inputs_glob:
        input_paths = sorted(glob.glob(args.inputs_glob))
    else:
        input_paths = list(args.inputs or [])
    if not input_paths:
        raise SystemExit("No input files found")

    print(f"Loading VIN drivetrain mapping from: {args.descriptions}")
    vin_map = load_vin_drivetrain(args.descriptions)
    print(f"VIN_Key mappings loaded: {len(vin_map):,}")

    print(f"Loading ZIP->County crosswalk: {args.zip_to_county}")
    zip_to_county = load_zip_to_county(args.zip_to_county)
    print(f"ZIPs in crosswalk: {len(zip_to_county):,}")

    regions = parse_regions(args.regions)
    print(f"Regions: {json.dumps(regions)}")

    # If a ZIP->region override exists, load it
    zip_to_region: Dict[str, List[str]] = {}
    if args.zip_to_region:
        with open(args.zip_to_region, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = {h.lower(): h for h in (reader.fieldnames or [])}
            if "zip" not in headers or "region" not in headers:
                raise ValueError("--zip-to-region must have columns 'zip' and 'region'")
            zc = headers["zip"]
            rc = headers["region"]
            for row in reader:
                z = (row.get(zc) or "").strip()
                if not z:
                    continue
                z5 = z.zfill(5) if z.isdigit() and len(z) <= 5 else z
                r = (row.get(rc) or "").strip()
                if not r:
                    continue
                zip_to_region.setdefault(z5, []).append(r)

    # Wrap aggregate to include ZIP->region overrides: prefer union of county-based and zip-based regions
    def aggregate_with_zip_regions(input_paths):
        # reuse aggregate() but inject region assignment per row
        # key: (region, group_key, metric) where group_key = month (YYYY-MM-01) or snapshot id (string)
        hll_map: Dict[Tuple[str, str, str], HyperLogLog] = defaultdict(lambda: HyperLogLog(p=args.hll_p))
        counters = {
            "rows_total": 0,
            "rows_skipped_no_zip": 0,
            "rows_skipped_no_county": 0,
            "rows_skipped_no_region": 0,
            "rows_skipped_no_month": 0,
        }

        # Pre-index county->regions
        county_to_regions: Dict[str, List[str]] = {}
        for rname, clist in regions.items():
            for c in clist:
                county_to_regions.setdefault(c.upper(), []).append(rname)

        def add(group: Tuple[str, str, str], vin: str):
            hll_map[group].add(vin)

        for path in input_paths:
            print(f"Processing file: {path}")
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                local_rows = 0
                for row in reader:
                    counters["rows_total"] += 1
                    local_rows += 1
                    if local_rows % 5000000 == 0:
                        print(f"  ... {local_rows:,} rows processed in {os.path.basename(path)}; total so far: {counters['rows_total']:,}")
                    if len(row) < 9:
                        counters["rows_skipped_no_month"] += 1
                        continue
                    vin = row[0]
                    zip_code = (row[3] or "").strip()
                    # choose grouping key
                    if args.group_by == "month":
                        group_key = (row[8] or "").strip()
                        if not group_key:
                            counters["rows_skipped_no_month"] += 1
                            continue
                    else:
                        # snapshot grouping by DMV_ID (column 5)
                        try:
                            group_key = str(int(row[5]))
                        except Exception:
                            counters["rows_skipped_no_month"] += 1
                            continue
                    if not zip_code:
                        counters["rows_skipped_no_zip"] += 1
                        continue
                    z5 = zip_code.zfill(5) if zip_code.isdigit() and len(zip_code) <= 5 else zip_code
                    # Regions via county mapping
                    cregions: List[str] = []
                    county = zip_to_county.get(z5)
                    if county:
                        cregions = county_to_regions.get(county.upper(), [])
                    else:
                        counters["rows_skipped_no_county"] += 1
                    # Regions via zip override
                    zregions = zip_to_region.get(z5, [])
                    rlist = list({*cregions, *zregions})
                    if not rlist:
                        counters["rows_skipped_no_region"] += 1
                        continue
                    vin_key = (row[7] or "").strip()
                    drv = vin_map.get(vin_key, "UNKNOWN")
                    is_ev = (drv in ("BEV", "PHEV"))
                    for rname in rlist:
                        add((rname, group_key, "ALL"), vin)
                        if is_ev:
                            add((rname, group_key, "EV"), vin)
            print(f"Finished file: {path} (rows: {local_rows:,})")
        return hll_map, counters

    print(f"Aggregating across {len(input_paths)} files ...")
    hll_map, counters = aggregate_with_zip_regions(input_paths)

    os.makedirs(args.output_dir, exist_ok=True)
    # Output file depends on grouping
    counts_csv = os.path.join(
        args.output_dir,
        f"counts_by_region_{'month' if args.group_by=='month' else 'snapshot'}.csv",
    )
    # Write counts with a generic column header 'group' to cover both month and snapshot keys
    def write_generic_counts(path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region", "group", "metric", "approx_unique_vins"])
            for (region, gkey, metric) in sorted(hll_map.keys()):
                w.writerow([region, gkey, metric, int(round(hll_map[(region, gkey, metric)].count()))])
    write_generic_counts(counts_csv)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "inputs": input_paths,
            "descriptions": args.descriptions,
            "zip_to_county": args.zip_to_county,
            "regions": regions,
            "counters": counters,
        }, f, indent=2)

    if args.plot:
        if args.group_by == "month":
            png = os.path.join(args.output_dir, "ev_trend_lipa_chge.png")
            # maybe_plot expects 'month' column; skip plotting when group-by snapshot
            try:
                # write a temp CSV with month header for plotting
                tmp_csv = os.path.join(args.output_dir, "_tmp_counts_for_plot.csv")
                with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["region", "month", "metric", "approx_unique_vins"])
                    for (region, gkey, metric) in sorted(hll_map.keys()):
                        w.writerow([region, gkey, metric, int(round(hll_map[(region, gkey, metric)].count()))])
                maybe_plot(tmp_csv, png, list(regions.keys()))
                print(f"Plot written: {png}")
            except Exception as e:
                print(f"Plotting failed: {e}")

    print(f"Counts written: {counts_csv}")
    print(f"Summary written: {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
