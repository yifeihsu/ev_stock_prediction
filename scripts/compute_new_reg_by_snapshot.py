#!/usr/bin/env python3
"""
Compute new-registrations (first-seen VINs) per DMV snapshot by region.

Definition
- For each region, a VIN's first-seen snapshot is the minimum DMV_ID among
  rows where the VIN maps to that region (by county/ZIP mapping). A VIN that
  moves into a region later is counted as "new to that region" when it first
  appears there.
- Outputs per (region, DMV_ID): new_all (all first-seen VINs), new_ev (subset
  with Drivetrain_Type in {BEV,PHEV}), and share.

Usage
  python scripts/compute_new_reg_by_snapshot.py \
    --inputs-glob "split_part_*.csv" \
    --descriptions "Vehicle Descriptions.csv" \
    --zip-to-county data/zip_to_county_ny.csv \
    --regions "LIPA:NASSAU,SUFFOLK;CHGE:DUTCHESS,ULSTER" \
    [--zip-to-region data/utility_zip_regions.csv] \
    --snapshot-map NY_DMV_Snapshots.csv \
    --output-dir out_new_reg

Notes
- This is snapshot-based and streams all splits once. Memory footprint is the
  size of the regional VIN dictionaries (order of millions), which is
  acceptable for LIPA/CHGE.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List


def load_vin_drivetrain(descriptions_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(descriptions_path, 'r', encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row.get('VIN_Key') or '').strip()
            if not k:
                continue
            d = (row.get('Drivetrain_Type') or '').strip() or 'UNKNOWN'
            out[k] = d
    return out


def load_zip_to_county(crosswalk_path: str) -> Dict[str, str]:
    with open(crosswalk_path, 'r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        headers = {h.lower(): h for h in (r.fieldnames or [])}
        if 'zip' not in headers or 'county_name' not in headers:
            raise ValueError("zip_to_county must have columns 'zip' and 'county_name'")
        zc = headers['zip']; cc = headers['county_name']
        out: Dict[str, str] = {}
        for row in r:
            z = (row.get(zc) or '').strip()
            if not z:
                continue
            z5 = z.zfill(5) if z.isdigit() and len(z) <= 5 else z
            cname = (row.get(cc) or '').strip().upper()
            if cname:
                out[z5] = cname
    return out


def parse_regions(spec: str) -> Dict[str, List[str]]:
    regions: Dict[str, List[str]] = {}
    for seg in (spec or '').split(';'):
        seg = seg.strip()
        if not seg:
            continue
        if ':' not in seg:
            raise ValueError(f'Bad region segment: {seg}')
        name, cs = seg.split(':', 1)
        counties = [c.strip().upper() for c in cs.split(',') if c.strip()]
        if not counties:
            raise ValueError(f'Region {name} has no counties')
        regions[name.strip()] = counties
    if not regions:
        raise ValueError('No regions parsed')
    return regions


def main():
    ap = argparse.ArgumentParser(description='Compute first-seen (new registrations) per snapshot by region')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--inputs', nargs='*')
    g.add_argument('--inputs-glob')
    ap.add_argument('--descriptions', required=True)
    ap.add_argument('--zip-to-county', required=True)
    ap.add_argument('--regions', required=True)
    ap.add_argument('--zip-to-region')
    ap.add_argument('--snapshot-map', required=True)
    ap.add_argument('--output-dir', default='out_new_reg')
    args = ap.parse_args()

    inputs = sorted(glob.glob(args.inputs_glob)) if args.inputs_glob else list(args.inputs or [])
    if not inputs:
        raise SystemExit('No input files found')

    vin_map = load_vin_drivetrain(args.descriptions)
    zip_to_county = load_zip_to_county(args.zip_to_county)
    regions = parse_regions(args.regions)

    # county->regions
    county_to_regions: Dict[str, List[str]] = {}
    for rname, clist in regions.items():
        for c in clist:
            county_to_regions.setdefault(c.upper(), []).append(rname)

    # Optional ZIP->region override
    zip_to_region: Dict[str, List[str]] = {}
    if args.zip_to_region:
        with open(args.zip_to_region, 'r', encoding='utf-8', newline='') as f:
            r = csv.DictReader(f)
            headers = {h.lower(): h for h in (r.fieldnames or [])}
            if 'zip' not in headers or 'region' not in headers:
                raise ValueError("zip_to_region must have columns 'zip' and 'region'")
            zc = headers['zip']; rc = headers['region']
            for row in r:
                z = (row.get(zc) or '').strip()
                if not z:
                    continue
                z5 = z.zfill(5) if z.isdigit() and len(z) <= 5 else z
                reg = (row.get(rc) or '').strip()
                if reg:
                    zip_to_region.setdefault(z5, []).append(reg)

    # First-seen per region: region -> { VIN -> first DMV_ID }
    first_seen: Dict[str, Dict[str, int]] = {r: {} for r in regions}
    vin_is_ev: Dict[str, bool] = {}

    counters = {
        'rows_total': 0,
        'rows_assigned': 0,
        'rows_skipped_bad_zip': 0,
        'rows_skipped_no_county': 0,
        'rows_skipped_no_region': 0,
        'rows_skipped_bad_dmv_id': 0,
    }

    for path in inputs:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            r = csv.reader(f)
            for row in r:
                counters['rows_total'] += 1
                if len(row) < 9:
                    continue
                vin = row[0]
                zip_code = (row[3] or '').strip()
                if not zip_code:
                    counters['rows_skipped_bad_zip'] += 1
                    continue
                try:
                    dmv_id = int(row[5])
                except Exception:
                    counters['rows_skipped_bad_dmv_id'] += 1
                    continue
                z5 = zip_code.zfill(5) if zip_code.isdigit() and len(zip_code) <= 5 else zip_code
                cregions: List[str] = []
                county = zip_to_county.get(z5)
                if county:
                    cregions = county_to_regions.get(county.upper(), [])
                else:
                    counters['rows_skipped_no_county'] += 1
                zregions = zip_to_region.get(z5, [])
                rlist = list({*cregions, *zregions})
                if not rlist:
                    counters['rows_skipped_no_region'] += 1
                    continue
                # Determine EV flag (stored per VIN once)
                if vin not in vin_is_ev:
                    vk = (row[7] or '').strip()
                    drv = vin_map.get(vk, 'UNKNOWN')
                    vin_is_ev[vin] = (drv in ('BEV', 'PHEV'))
                # Update first-seen per region
                for reg in rlist:
                    cur = first_seen[reg].get(vin)
                    if cur is None or dmv_id < cur:
                        first_seen[reg][vin] = dmv_id
                        counters['rows_assigned'] += 1

    # Aggregate to DMV_ID counts per region
    os.makedirs(args.output_dir, exist_ok=True)
    out_counts = os.path.join(args.output_dir, 'new_reg_by_region_snapshot.csv')
    with open(out_counts, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'snapshot_id', 'new_all', 'new_ev', 'ev_share_pct'])
        for reg, mapping in first_seen.items():
            new_all = defaultdict(int)
            new_ev = defaultdict(int)
            for vin, sid in mapping.items():
                new_all[sid] += 1
                if vin_is_ev.get(vin, False):
                    new_ev[sid] += 1
            for sid in sorted(set(list(new_all.keys()) + list(new_ev.keys()))):
                na = new_all.get(sid, 0)
                ne = new_ev.get(sid, 0)
                share = (100.0 * ne / na) if na else 0.0
                w.writerow([reg, sid, na, ne, f"{share:.4f}"])

    # Write a small summary JSON
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'regions': regions, 'counters': counters, 'output': out_counts}, f, indent=2)

    print('Wrote', out_counts)


if __name__ == '__main__':
    main()

