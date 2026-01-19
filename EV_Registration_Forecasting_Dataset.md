# EV Registration Dataset for Forecasting

This repository packages New York State DMV registration data into a set of modeling‑ready tables for EV adoption and registration forecasting, with a focus on the LIPA and CHGE regions.

The main goal is to give you clean time series of:
- On‑road EV stock (unique registered vehicles)
- On‑road total light‑duty stock (all drivetrains)
- New EV registrations and EV share of new registrations

These can be used directly in diffusion / adoption models (e.g., Bass, logistic, Gompertz) at the regional level.

---

## 1. Data Sources and Geography

- **Raw registration history (`split_part_*.csv`)**
  - VIN‑level registration intervals for NY vehicles (no header).
  - Schema (positional):  
    `0 VIN`, `1 Registration_Valid_Date`, `2 Registration_Expiration_Date`,  
    `3 ZIP`, `4 County_GEOID`, `5 DMV_ID`, `6 State`, `7 VIN_Key`, `8 Reg Month`.

- **Snapshot calendar (`NY_DMV_Snapshots.csv`)**
  - Columns: `DMV_ID`, `DMV_Snapshot_Date`, `DMV_Snapshot_Name`, `State`.
  - Provides the actual dates for each DMV snapshot ID (1–103, from 2011‑03‑03 to 2025‑10‑08).

- **Vehicle attributes (`Vehicle Descriptions.csv`)**
  - Key fields: `VIN_Key`, `Drivetrain_Type`, `Year`, `Vehicle_Type`, `Vehicle_Subtype`, `Vehicle_GVWR_Category`, pricing and range fields, etc.
  - EV definition in this repo: `Drivetrain_Type` in `{BEV, PHEV}`.

- **Geography / region mapping (`data/*`)**
  - `data/zip_to_county_ny.csv` (and `ny_zip_codes.csv`): ZIP → county crosswalk.
  - `data/utility_zip_regions.csv` (optional): ZIP → region overrides for utility territories.
  - Regions are defined as county groups; in the current outputs:
    - `LIPA`: NASSAU, SUFFOLK
    - `CHGE`:
      - **Monthly stock (`out_ev`)**: ALBANY, COLUMBIA, DUTCHESS, GREENE, ORANGE, PUTNAM, ULSTER
      - **Snapshot stock (`out_ev_snapshots`) & new regs (`out_new_reg`)**: DUTCHESS, ULSTER only

> **Important:** CHGE’s county coverage differs between `out_ev` and `out_new_reg`/`out_ev_snapshots`. If you combine series, keep this in mind or rebuild with a consistent region spec.

---

## 2. Core Tables for Forecasting

### 2.1 Monthly on‑road stock by region

**File:** `out_ev/counts_by_region_month.csv`  
**Grain:** one row per (`region`, `month`, `metric`)

Columns:
- `region` – region name (`LIPA`, `CHGE`).
- `month` – first day of month (`YYYY-MM-01`), derived from registration valid dates.
- `metric` – `EV` (BEV+PHEV) or `ALL` (all drivetrains).
- `approx_unique_vins` – approximate **count of distinct VINs** on the road in that region/month.

How it’s computed:
- VINs are mapped to regions via ZIP → county (plus optional ZIP → region overrides).
- For each (`region`, `month`, `metric`), distinct VINs are counted using HyperLogLog (HLL, `p=16`).
- This yields an **approximate** unique count; typical HLL error at this setting is on the order of ±1%.

Coverage notes:
- Data span from **1972‑12‑01** through **2025‑11‑01** (see `out_stats/summary.json`).
- EV counts (`metric = EV`) are available through **2025‑08‑01** for both regions; the last three months currently have `ALL` only (`2025‑09`–`2025‑11`).
- Early pre‑2011 months are very sparse and reflect back‑filled registration intervals; for EV adoption modeling it is usually safest to start the time series around **2011‑03** (first DMV snapshot with statewide coverage).

Typical uses:
- EV stock series: filter to `metric = "EV"` and sort by (`region`, `month`) to get on‑road EV counts over time.
- Market share: `EV / ALL` per (`region`, `month`).
- Bass/logistic modeling: treat `approx_unique_vins` as an approximation of cumulative adopters at each month.

---

### 2.2 Snapshot‑based on‑road stock by region

**File:** `out_ev_snapshots/counts_by_region_snapshot.csv`  
**Grain:** one row per (`region`, `snapshot_id`, `metric`)

Columns:
- `region` – region name (`LIPA`, `CHGE` – here CHGE = DUTCHESS+ULSTER).
- `group` – DMV snapshot ID (`snapshot_id`), integer matching `DMV_ID` in `NY_DMV_Snapshots.csv`.
- `metric` – `EV` or `ALL` as above.
- `approx_unique_vins` – approximate unique VIN count at that snapshot.

Usage:
- Join `group` → `NY_DMV_Snapshots.DMV_ID` to get `DMV_Snapshot_Date`.
- Provides a **monthly (or better) snapshot series** directly tied to DMV files, which some modelers may prefer over the month‑bucketed view in 2.1.

Relationship to 2.1:
- 2.1 buckets registrations by **registration month** from the VIN history.
- 2.2 buckets active registrations by **DMV snapshot**. For most practical forecasting work you can pick whichever time axis fits your modeling assumptions; just don’t mix the two in the same calibration without care.

---

### 2.3 New registrations by snapshot and region

**File:** `out_new_reg/new_reg_by_region_snapshot.csv`  
**Grain:** one row per (`region`, `snapshot_id`)

Columns:
- `region` – region name (`LIPA`, `CHGE` – here CHGE = DUTCHESS+ULSTER).
- `snapshot_id` – DMV snapshot ID (`DMV_ID`), joinable to `NY_DMV_Snapshots`.
- `new_all` – count of VINs **first seen in that region** in this snapshot (all drivetrains).
- `new_ev` – subset of `new_all` where `Drivetrain_Type ∈ {BEV, PHEV}`.
- `ev_share_pct` – `100 * new_ev / new_all` (percentage of new registrations that are EVs).

Definition details:
- A VIN’s **first‑seen snapshot in a region** is the smallest `DMV_ID` among rows where it maps into that region (counties or ZIP overrides).
- If a VIN moves into a region from elsewhere, it is counted as “new to that region” at the first snapshot where it appears in that region.
- VIN‑to‑EV classification is via `Vehicle Descriptions.csv` (`VIN_Key` → `Drivetrain_Type`).

Coverage notes:
- `snapshot_id` values currently run from 1 up through **102** (as of the last computation); new snapshot 103 exists in `NY_DMV_Snapshots.csv` but is **not yet included** in `out_new_reg`.

Typical uses:
- Direct **flow series** for Bass / innovation‑imitation models: treat `new_ev` per snapshot as the adoption flow.
- EV penetration of new sales: `ev_share_pct` by snapshot and region.
- Flow‑stock consistency checks: cumulative `new_ev` (by region) should roughly track the growth in EV stock from section 2.1 / 2.2, modulo scrappage and moves.

---

## 3. How to Build a Forecasting Dataset

This section sketches one way to go from the existing tables to a modeling‑ready panel suitable for Bass or related diffusion models.

### 3.1 Choose geography and time axis

- Pick **region(s)**:
  - `LIPA` (Nassau + Suffolk) is consistent across all outputs.
  - `CHGE` has different county coverage between monthly stock and new‑reg series (see Section 1). For clean modeling, either:
    - Use **LIPA only**, or
    - Re‑run the scripts with a consistent region definition before fitting multi‑region models.

- Pick a **time axis**:
  - Option A – **Snapshot‑based** (recommended with `out_new_reg`):
    - Use `NY_DMV_Snapshots.csv` as the calendar.
    - Join `out_ev_snapshots` and `out_new_reg` on (`region`, `snapshot_id`).
  - Option B – **Calendar month‑based**:
    - Use `month` from `out_ev/counts_by_region_month.csv`.
    - Derive or back‑calculate new registrations per month by differencing EV stock and/or using snapshot‑based flows as an approximation.

### 3.2 Construct key series

For each (`region`, `time`):

- **EV stock**:  
  - From snapshots: `stock_ev_t = approx_unique_vins` where `metric = "EV"` in `out_ev_snapshots`.  
  - From months: `stock_ev_t = approx_unique_vins` where `metric = "EV"` in `out_ev/counts_by_region_month.csv`.

- **Total stock**:  
  - `stock_all_t = approx_unique_vins` where `metric = "ALL"`.

- **New EV registrations (flow)**:
  - Preferred: `flow_ev_t = new_ev` from `out_new_reg` (snapshot‑based).
  - If you need a monthly flow aligned to `month`, you can:
    - Interpolate / allocate snapshot flows to months, or
    - Derive approximate flows as `Δ stock_ev_t` plus assumptions about scrappage and inter‑region moves.

- **EV share of new registrations**:
  - From `out_new_reg`: `ev_share_pct`.
  - Or compute as `new_ev / new_all` if you re‑aggregate.

These give you the standard ingredients for diffusion modeling: a stock series (cumulative adopters) and a flow series (new adopters per period).

### 3.3 Example Bass‑style use

At a high level, for each region:

1. Build a time series `{t, flow_ev_t, stock_ev_t}` on a regular grid (snapshot or month).  
2. Choose a market potential `m` (e.g., long‑run EV stock potential, potentially informed by policy or scenario modeling).  
3. Estimate Bass parameters (`p`, `q`) by fitting:
   - `flow_ev_t ≈ (p + q * stock_ev_t / m) * (m - stock_ev_t)`  
   using nonlinear least squares or maximum likelihood.
4. Forecast future `stock_ev_t` and `flow_ev_t` under different `m`, `p`, `q` scenarios.

The construction of `{flow_ev_t, stock_ev_t}` comes directly from the tables described above; no VIN‑level processing is needed for most modeling work.

---

## 4. Caveats and Data Quality Considerations

- **Approximate counts:**  
  - All `approx_unique_vins` fields come from HyperLogLog‑based approximate distinct counting. Expect small random error; this is usually negligible relative to year‑over‑year growth but is worth remembering if you compare against exact VIN counts from other sources.

- **Region coverage differences:**  
  - As noted, CHGE’s county definition differs between `out_ev` and `out_new_reg`/`out_ev_snapshots`. Check region specs in the corresponding `summary.json` files if you need strict comparability.

- **Missing EV counts in the latest months:**  
  - For `out_ev/counts_by_region_month.csv`, months **2025‑09–2025‑11** currently lack `EV` rows and only have `ALL`. Treat these months as **incomplete** for EV stock modeling until the aggregation is re‑run.

- **Sparse early history:**  
  - Pre‑2011 data exist due to registration interval back‑projection, but the EV market is essentially zero and coverage is thinner. Most analyses should focus on **2011 onward**, and potentially **2014+** where EV volumes are large enough for stable parameter estimation.

- **VIN movement and scrappage:**  
  - New registrations (`new_ev`) count VINs first seen in a region; subsequent moves or de‑registrations are not explicitly modeled here. Stock series still reflect active registrations via DMV snapshots, but if you need detailed churn dynamics you would have to go back to the VIN‑level files.

---

## 5. Scripts and Reproducibility

The key aggregation scripts live under `scripts/`:

- `scripts/aggregate_ev_trends_by_county.py`  
  - Generates `out_ev/counts_by_region_month.csv` and `out_ev_snapshots/counts_by_region_snapshot.csv`.

- `scripts/compute_new_reg_by_snapshot.py`  
  - Generates `out_new_reg/new_reg_by_region_snapshot.csv`.

Both scripts are streaming and can be re‑run with different region definitions, crosswalks, or HLL parameters if you need alternative geographies or accuracy trade‑offs for forecasting work.

If you’d like, we can also add a small example notebook or script that builds a ready‑to‑use Bass model dataset from these tables.

