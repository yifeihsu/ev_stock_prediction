#!/usr/bin/env python3
"""Build a point-level charging-demand map for one Albany H3 region."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_behavioral_charging_model import (  # noqa: E402
    apply_adoption_scaling,
    build_load_curve,
    load_adoption_file,
)


EVENT_COLS = [
    "scenario_id",
    "managed_flag",
    "home_zcta",
    "charger_type",
    "charge_location_type",
    "charging_point_id",
    "charging_lon",
    "charging_lat",
    "charging_h3",
    "start_time_min",
    "end_time_min",
    "expected_power_kw",
]
POINT_COLS = ["charging_point_id", "charging_lon", "charging_lat"]
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "tco": "TCO only",
    "tco_evse": "TCO + EVSE",
}
MODE_LABELS = {
    "managed": "Managed",
    "unmanaged": "Unmanaged",
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def clean_number(value: object, digits: int = 3) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    out = round(out, digits)
    if abs(out - round(out)) < 10 ** (-digits):
        return int(round(out))
    return out


def normalize_mode(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "managed"}:
        return "managed"
    if text in {"false", "0", "no", "unmanaged"}:
        return "unmanaged"
    raise ValueError(f"Unrecognized managed_flag value: {value!r}")


def h3_boundary(cell: str) -> list[list[float]]:
    try:
        import h3  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The h3 Python package is required to build H3 map geometry") from exc

    boundary = h3.cell_to_boundary(cell)
    coords = [[round(lon, 7), round(lat, 7)] for lat, lon in boundary]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def h3_resolution(cell: str) -> int | None:
    try:
        import h3  # type: ignore
    except ImportError:
        return None
    if hasattr(h3, "get_resolution"):
        return int(h3.get_resolution(cell))
    return None


def read_selected_events(path: Path, selected_h3: str, chunksize: int) -> tuple[pd.DataFrame, int]:
    chunks: list[pd.DataFrame] = []
    scanned_rows = 0
    for chunk in pd.read_csv(
        path,
        usecols=EVENT_COLS,
        chunksize=chunksize,
        dtype={
            "home_zcta": "string",
            "charging_h3": "string",
            "charging_point_id": "string",
        },
    ):
        scanned_rows += len(chunk)
        selected = chunk[chunk["charging_h3"].eq(selected_h3)].copy()
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"No charging events found for H3 {selected_h3} in {path}")

    events = pd.concat(chunks, ignore_index=True)
    for col in ["start_time_min", "end_time_min", "expected_power_kw", "charging_lon", "charging_lat"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events = events.dropna(
        subset=[
            "home_zcta",
            "charging_point_id",
            "charging_lon",
            "charging_lat",
            "start_time_min",
            "end_time_min",
            "expected_power_kw",
        ]
    )
    if events.empty:
        raise ValueError(f"Selected H3 {selected_h3} has no usable point-level charging events")
    return events, scanned_rows


def build_scaled_hourly_point_forecast(
    events: pd.DataFrame,
    adoption: pd.DataFrame,
    *,
    bin_minutes: int,
) -> pd.DataFrame:
    load_15 = build_load_curve(
        events,
        location_columns=POINT_COLS + ["home_zcta"],
        bin_minutes=bin_minutes,
    )
    scaled_15 = apply_adoption_scaling(
        load_15,
        adoption,
        location_columns=POINT_COLS,
    )
    if scaled_15.empty:
        raise ValueError("Adoption scaling produced no point-level rows")

    scaled_15 = scaled_15.copy()
    scaled_15["hour"] = (pd.to_numeric(scaled_15["time_bin_min"], errors="coerce") // 60).astype(int)
    scaled_15["mode"] = scaled_15["managed_flag"].map(normalize_mode)
    group_cols = [
        "adoption_scenario",
        "forecast_year",
        "mode",
        "charging_point_id",
        "charging_lon",
        "charging_lat",
        "hour",
    ]
    hourly = scaled_15.groupby(group_cols, dropna=False, as_index=False)[["kw", "kwh"]].sum()
    hourly["kw"] = hourly["kwh"]
    hourly["forecast_year"] = pd.to_numeric(hourly["forecast_year"], errors="raise").astype(int)
    hourly["hour"] = pd.to_numeric(hourly["hour"], errors="raise").astype(int)
    hourly["charging_lon"] = pd.to_numeric(hourly["charging_lon"], errors="coerce")
    hourly["charging_lat"] = pd.to_numeric(hourly["charging_lat"], errors="coerce")
    return hourly.sort_values(group_cols).reset_index(drop=True)


def build_point_features(hourly: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    points = (
        hourly.loc[:, POINT_COLS]
        .drop_duplicates()
        .sort_values("charging_point_id")
        .reset_index(drop=True)
    )
    features: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for idx, row in points.iterrows():
        point_id = str(row["charging_point_id"])
        index[point_id] = int(idx)
        lon = clean_number(row["charging_lon"], 6)
        lat = clean_number(row["charging_lat"], 6)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "i": int(idx),
                    "pointId": point_id,
                    "lon": lon,
                    "lat": lat,
                },
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )
    return features, index


def build_view_payload(hourly: pd.DataFrame, point_index: dict[str, int]) -> dict[str, list[list[Any]]]:
    views: dict[str, list[list[Any]]] = {}
    for (mode, scenario, year, point_id), group in hourly.groupby(
        ["mode", "adoption_scenario", "forecast_year", "charging_point_id"], sort=True
    ):
        hourly_kw = [0.0] * 24
        daily_kwh = 0.0
        for _, row in group.iterrows():
            hour = int(row["hour"])
            if 0 <= hour <= 23:
                hourly_kw[hour] += float(row["kw"])
                daily_kwh += float(row["kwh"])
        peak_kw = max(hourly_kw)
        peak_hour = int(hourly_kw.index(peak_kw))
        view_key = f"{mode}|{scenario}|{int(year)}"
        views.setdefault(view_key, []).append(
            [
                point_index[str(point_id)],
                clean_number(daily_kwh, 3),
                clean_number(peak_kw, 3),
                peak_hour,
                [clean_number(value, 3) or 0 for value in hourly_kw],
            ]
        )

    for records in views.values():
        records.sort(key=lambda row: row[0])
    return views


def write_hourly_csv(hourly: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = hourly.copy()
    out["kw"] = out["kw"].round(6)
    out["kwh"] = out["kwh"].round(6)
    out.to_csv(path, index=False)


def build_metadata(
    *,
    selected_h3: str,
    events: pd.DataFrame,
    hourly: pd.DataFrame,
    source_path: Path,
    adoption_path: Path,
    output_csv: Path,
    scanned_rows: int,
) -> dict[str, Any]:
    scenarios = sorted(hourly["adoption_scenario"].dropna().astype(str).unique().tolist())
    years = sorted(int(value) for value in hourly["forecast_year"].dropna().unique().tolist())
    modes = [mode for mode in ["managed", "unmanaged"] if mode in set(hourly["mode"])]
    default_scenario = "tco_evse" if "tco_evse" in scenarios else scenarios[-1]
    default_mode = "managed" if "managed" in modes else modes[0]
    default_year = max(years)
    latest = hourly[
        (hourly["mode"].eq(default_mode))
        & (hourly["adoption_scenario"].eq(default_scenario))
        & (hourly["forecast_year"].eq(default_year))
    ]
    daily = latest.groupby("charging_point_id", sort=False)["kwh"].sum()
    peak_by_hour = latest.groupby(["charging_point_id", "hour"], sort=False)["kw"].sum().reset_index()
    top_peak = peak_by_hour.sort_values("kw", ascending=False).head(1)
    return {
        "title": "Albany H3 Point Charging Demand Forecast",
        "selectedH3": selected_h3,
        "h3Resolution": h3_resolution(selected_h3),
        "sourceCsv": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
        "adoptionCsv": str(adoption_path.relative_to(ROOT)) if adoption_path.is_relative_to(ROOT) else str(adoption_path),
        "outputCsv": str(output_csv.relative_to(ROOT)) if output_csv.is_relative_to(ROOT) else str(output_csv),
        "scannedEventRows": int(scanned_rows),
        "selectedEventRows": int(len(events)),
        "pointCount": int(hourly["charging_point_id"].nunique()),
        "hourlyRows": int(len(hourly)),
        "homeZctaCount": int(events["home_zcta"].nunique()),
        "scenarios": scenarios,
        "scenarioLabels": {key: SCENARIO_LABELS.get(key, key) for key in scenarios},
        "modes": modes,
        "modeLabels": {key: MODE_LABELS.get(key, key.title()) for key in modes},
        "years": years,
        "defaultScenario": default_scenario,
        "defaultMode": default_mode,
        "defaultYear": default_year,
        "defaultViewTotals": {
            "dailyKwh": clean_number(daily.sum(), 1),
            "activePoints": int((daily > 0).sum()),
            "topPeakPoint": str(top_peak.iloc[0]["charging_point_id"]) if len(top_peak) else None,
            "topPeakHour": int(top_peak.iloc[0]["hour"]) if len(top_peak) else None,
            "topPeakKw": clean_number(top_peak.iloc[0]["kw"], 2) if len(top_peak) else None,
        },
    }


def build_html(app_data: dict[str, Any]) -> str:
    data_json = json.dumps(app_data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Albany H3 Point Charging Demand Forecast</title>
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    :root {
      --ink: #17212b;
      --muted: #5d6977;
      --panel: rgba(255, 255, 255, 0.96);
      --border: rgba(25, 38, 50, 0.18);
      --accent: #0f766e;
      --managed: #3b7d76;
      --unmanaged: #a85f1c;
    }
    html, body, #map {
      width: 100%;
      height: 100%;
      margin: 0;
    }
    body {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #e8edf0;
    }
    #map {
      position: absolute;
      inset: 0;
    }
    .map-panel {
      position: fixed;
      top: 12px;
      left: 50px;
      z-index: 1000;
      width: min(430px, calc(100vw - 68px));
      max-height: calc(100vh - 28px);
      overflow: auto;
      box-sizing: border-box;
      padding: 14px 16px 15px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 16px 36px rgba(17, 24, 39, 0.16);
      backdrop-filter: blur(8px);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 760;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 4px 0 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .field-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }
    .field {
      display: grid;
      gap: 5px;
      min-width: 0;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 680;
      letter-spacing: 0;
    }
    select {
      width: 100%;
      min-height: 34px;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 6px 8px;
      font-size: 13px;
      font-weight: 600;
    }
    .control-band {
      display: grid;
      gap: 8px;
      padding: 10px 0 12px;
      border-top: 1px solid rgba(23, 33, 43, 0.1);
      border-bottom: 1px solid rgba(23, 33, 43, 0.1);
    }
    .label-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .label-line strong {
      font-size: 18px;
      line-height: 1.2;
      font-weight: 760;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e8f4f1;
      color: #0f5e59;
      font-size: 11px;
      font-weight: 760;
      white-space: nowrap;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .button-row {
      display: grid;
      grid-template-columns: 36px 36px 36px 1fr;
      gap: 8px;
      align-items: center;
    }
    button {
      min-width: 36px;
      height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-size: 14px;
      font-weight: 760;
      cursor: pointer;
    }
    button:hover {
      border-color: rgba(15, 118, 110, 0.55);
      color: var(--accent);
    }
    .hour-grid {
      display: grid;
      grid-template-columns: 1fr 72px;
      gap: 10px;
      align-items: center;
    }
    #hourLabel {
      justify-self: end;
      font-size: 15px;
      font-weight: 760;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }
    .stat {
      min-width: 0;
      border-left: 3px solid rgba(15, 118, 110, 0.35);
      padding-left: 8px;
    }
    .stat-label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .stat-value {
      display: block;
      margin-top: 2px;
      font-size: 15px;
      line-height: 1.2;
      font-weight: 780;
      overflow-wrap: anywhere;
    }
    .source-line {
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid rgba(23, 33, 43, 0.1);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .map-legend {
      min-width: 210px;
      padding: 10px 12px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 12px 28px rgba(17, 24, 39, 0.14);
      color: var(--ink);
      font-size: 12px;
      line-height: 1.25;
    }
    .legend-title {
      margin-bottom: 7px;
      font-weight: 780;
    }
    .legend-row {
      display: grid;
      grid-template-columns: 26px 1fr;
      gap: 7px;
      align-items: center;
      margin: 4px 0;
    }
    .legend-swatch {
      width: 24px;
      height: 14px;
      border: 1px solid rgba(23, 33, 43, 0.18);
    }
    .point-tooltip {
      font-weight: 760;
      color: var(--ink);
    }
    .leaflet-popup-content {
      margin: 12px 13px;
      min-width: 315px;
      max-width: 375px;
    }
    .popup-title {
      margin: 0 0 8px;
      font-size: 15px;
      font-weight: 780;
      overflow-wrap: anywhere;
    }
    .popup-grid {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 5px 12px;
      font-size: 12px;
    }
    .popup-grid span:nth-child(odd) {
      color: var(--muted);
    }
    .popup-grid span:nth-child(even) {
      font-weight: 720;
      text-align: right;
    }
    .popup-divider {
      height: 1px;
      background: rgba(23, 33, 43, 0.12);
      margin: 10px 0;
    }
    .chart-title {
      margin: 0 0 6px;
      font-size: 13px;
      font-weight: 780;
    }
    .mini-chart {
      width: 100%;
      height: 112px;
      display: block;
    }
    .legend-inline {
      display: flex;
      gap: 14px;
      margin-top: 5px;
      color: var(--ink);
      font-size: 12px;
      font-weight: 720;
    }
    .legend-dot {
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 5px;
      border-radius: 50%;
    }
    .compare-table {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 5px 12px;
      align-items: end;
      margin-top: 8px;
      font-size: 12px;
    }
    .compare-table .head {
      color: var(--muted);
      font-weight: 760;
    }
    .compare-table .value {
      text-align: right;
      font-weight: 760;
    }
    @media (max-width: 720px) {
      .map-panel {
        top: 10px;
        left: 10px;
        right: 10px;
        width: auto;
        max-height: 54vh;
      }
      .field-grid, .stats-grid {
        grid-template-columns: 1fr;
      }
      .button-row {
        grid-template-columns: 36px 36px 36px;
      }
      .map-legend {
        max-width: calc(100vw - 42px);
      }
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <section class="map-panel" aria-label="Map controls">
    <h1>Point Charging Demand Forecast</h1>
    <div class="subtitle">H3-<span id="h3Resolution"></span> <span id="selectedH3"></span> | <span id="pointCount"></span> modeled points</div>
    <div class="field-grid">
      <label class="field">Scenario
        <select id="scenarioSelect"></select>
      </label>
      <label class="field">Mode
        <select id="modeSelect"></select>
      </label>
      <label class="field">Metric
        <select id="metricSelect">
          <option value="peak">Peak kW</option>
          <option value="daily">Daily kWh</option>
          <option value="hourly">Selected-hour kW</option>
        </select>
      </label>
      <label class="field">Year
        <select id="yearSelect"></select>
      </label>
    </div>
    <div class="control-band">
      <div class="label-line">
        <strong id="yearLabel"></strong>
        <span class="badge" id="viewBadge"></span>
      </div>
      <input id="yearSlider" type="range" min="0" step="1" />
      <div class="button-row">
        <button id="prevYear" type="button" title="Previous year" aria-label="Previous year">&#9664;</button>
        <button id="playYears" type="button" title="Play years" aria-label="Play years">&#9654;</button>
        <button id="nextYear" type="button" title="Next year" aria-label="Next year">&#9654;&#10072;</button>
      </div>
      <div class="hour-grid">
        <input id="hourSlider" type="range" min="0" max="23" step="1" value="18" />
        <span id="hourLabel"></span>
      </div>
    </div>
    <div class="stats-grid">
      <div class="stat">
        <span class="stat-label" id="totalLabel">Total</span>
        <span class="stat-value" id="totalValue"></span>
      </div>
      <div class="stat">
        <span class="stat-label">Highest point</span>
        <span class="stat-value" id="topPoint"></span>
      </div>
      <div class="stat">
        <span class="stat-label">Region peak</span>
        <span class="stat-value" id="regionPeak"></span>
      </div>
      <div class="stat">
        <span class="stat-label">Active points</span>
        <span class="stat-value" id="activePoints"></span>
      </div>
    </div>
    <div class="source-line" id="sourceLine"></div>
  </section>
  <script>
    const APP_DATA = __APP_DATA__;
    const IDX = 0;
    const DAILY = 1;
    const PEAK = 2;
    const PEAK_HOUR = 3;
    const HOURLY = 4;
    const colors = ["#edf8fb", "#b2e2e2", "#66c2a4", "#2ca25f", "#006d2c", "#d95f0e"];
    const map = L.map("map", { zoomControl: true, preferCanvas: true });

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors"
    }).addTo(map);

    const scenarioSelect = document.getElementById("scenarioSelect");
    const modeSelect = document.getElementById("modeSelect");
    const metricSelect = document.getElementById("metricSelect");
    const yearSelect = document.getElementById("yearSelect");
    const yearSlider = document.getElementById("yearSlider");
    const hourSlider = document.getElementById("hourSlider");
    let yearIndex = APP_DATA.metadata.years.indexOf(APP_DATA.metadata.defaultYear);
    let playTimer = null;
    let breaks = [];
    let pointLayer = null;
    const recordCache = new Map();

    function addOptions(select, values, labels) {
      for (const value of values) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = labels[value] || value;
        select.appendChild(option);
      }
    }

    addOptions(scenarioSelect, APP_DATA.metadata.scenarios, APP_DATA.metadata.scenarioLabels);
    addOptions(modeSelect, APP_DATA.metadata.modes, APP_DATA.metadata.modeLabels);
    addOptions(yearSelect, APP_DATA.metadata.years.map(String), {});
    scenarioSelect.value = APP_DATA.metadata.defaultScenario;
    modeSelect.value = APP_DATA.metadata.defaultMode;
    yearSelect.value = String(APP_DATA.metadata.defaultYear);
    yearSlider.max = String(APP_DATA.metadata.years.length - 1);
    yearSlider.value = String(yearIndex);
    document.getElementById("h3Resolution").textContent = APP_DATA.metadata.h3Resolution;
    document.getElementById("selectedH3").textContent = APP_DATA.metadata.selectedH3;
    document.getElementById("pointCount").textContent = APP_DATA.metadata.pointCount.toLocaleString();
    document.getElementById("sourceLine").textContent = `${APP_DATA.metadata.outputCsv} | ${APP_DATA.metadata.selectedEventRows.toLocaleString()} selected events`;

    function currentYear() {
      return APP_DATA.metadata.years[yearIndex];
    }

    function currentViewKey(year = currentYear(), mode = modeSelect.value) {
      return `${mode}|${scenarioSelect.value}|${year}`;
    }

    function viewRecords(key = currentViewKey()) {
      if (!recordCache.has(key)) {
        const records = new Map();
        for (const record of APP_DATA.views[key] || []) {
          records.set(record[IDX], record);
        }
        recordCache.set(key, records);
      }
      return recordCache.get(key);
    }

    function recordFor(feature, mode = modeSelect.value) {
      return viewRecords(currentViewKey(currentYear(), mode)).get(feature.properties.i) || null;
    }

    function hourValue(record) {
      if (!record) return 0;
      return record[HOURLY][Number(hourSlider.value)] || 0;
    }

    function valueFor(record) {
      if (!record) return 0;
      if (metricSelect.value === "daily") return record[DAILY] || 0;
      if (metricSelect.value === "hourly") return hourValue(record);
      return record[PEAK] || 0;
    }

    function metricMeta() {
      if (metricSelect.value === "daily") {
        return { label: "Daily kWh", unit: "kWh", decimals: 1 };
      }
      if (metricSelect.value === "hourly") {
        return { label: `${hourLabelText()} kW`, unit: "kW", decimals: 2 };
      }
      return { label: "Peak kW", unit: "kW", decimals: 2 };
    }

    function hourLabelText() {
      return `${String(hourSlider.value).padStart(2, "0")}:00`;
    }

    function formatNumber(value, digits = 0) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: digits
      });
    }

    function formatMetric(value) {
      const meta = metricMeta();
      return `${formatNumber(value, meta.decimals)} ${meta.unit}`;
    }

    function valuesForScale() {
      const values = [];
      for (const year of APP_DATA.metadata.years) {
        const records = viewRecords(currentViewKey(year));
        for (const record of records.values()) {
          const value = valueFor(record);
          if (Number.isFinite(value)) values.push(value);
        }
      }
      return values.sort((a, b) => a - b);
    }

    function quantile(values, q) {
      if (!values.length) return 0;
      const position = (values.length - 1) * q;
      const lower = Math.floor(position);
      const upper = Math.ceil(position);
      if (lower === upper) return values[lower];
      return values[lower] + (values[upper] - values[lower]) * (position - lower);
    }

    function computeBreaks() {
      const values = valuesForScale();
      const raw = [0, 0.2, 0.4, 0.6, 0.8, 0.95, 1].map(q => quantile(values, q));
      for (let i = 1; i < raw.length; i += 1) {
        if (raw[i] <= raw[i - 1]) raw[i] = raw[i - 1] + 0.001;
      }
      breaks = raw;
    }

    function colorFor(value) {
      if (!Number.isFinite(value) || value <= 0) return "#d8dee4";
      for (let i = 0; i < breaks.length - 1; i += 1) {
        if (value <= breaks[i + 1]) return colors[i];
      }
      return colors[colors.length - 1];
    }

    function radiusFor(value) {
      if (!Number.isFinite(value) || value <= 0) return 4;
      const maxValue = breaks[breaks.length - 1] || 1;
      return Math.max(4, Math.min(18, 4 + 14 * Math.sqrt(value / maxValue)));
    }

    function markerStyle(feature) {
      const value = valueFor(recordFor(feature));
      return {
        radius: radiusFor(value),
        color: "#21313c",
        weight: 0.8,
        opacity: 0.85,
        fillColor: colorFor(value),
        fillOpacity: value > 0 ? 0.78 : 0.28
      };
    }

    function chartPath(values, x0, y0, width, height, maxValue) {
      if (!values || !values.length || maxValue <= 0) return "";
      return values.map((value, hour) => {
        const x = x0 + (width * hour / 23);
        const y = y0 + height - (height * value / maxValue);
        return `${hour === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
    }

    function hourlyChartHtml(managed, unmanaged) {
      const managedHourly = managed ? managed[HOURLY] : Array(24).fill(0);
      const unmanagedHourly = unmanaged ? unmanaged[HOURLY] : Array(24).fill(0);
      const selectedHour = Number(hourSlider.value);
      const maxValue = Math.max(0.001, ...managedHourly, ...unmanagedHourly);
      const x0 = 42;
      const y0 = 8;
      const width = 240;
      const height = 74;
      const xSelected = x0 + (width * selectedHour / 23);
      return `
        <div class="popup-divider"></div>
        <div class="chart-title">Managed vs unmanaged hourly demand</div>
        <svg class="mini-chart" viewBox="0 0 312 112" role="img" aria-label="Hourly demand comparison">
          <line x1="${x0}" y1="${y0 + height}" x2="${x0 + width}" y2="${y0 + height}" stroke="#c8ced4" />
          <line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y0 + height}" stroke="#c8ced4" />
          <line x1="${xSelected.toFixed(1)}" y1="${y0}" x2="${xSelected.toFixed(1)}" y2="${y0 + height}" stroke="#9aa4af" stroke-dasharray="4 4" />
          <text x="0" y="${y0 + 5}" font-size="11" fill="#5d6977">${formatNumber(maxValue, 1)}</text>
          <text x="16" y="${y0 + height + 4}" font-size="11" fill="#5d6977">0</text>
          <text x="${x0}" y="105" font-size="11" fill="#5d6977">00:00</text>
          <text x="${x0 + width - 31}" y="105" font-size="11" fill="#5d6977">23:00</text>
          <path d="${chartPath(managedHourly, x0, y0, width, height, maxValue)}" fill="none" stroke="var(--managed)" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
          <path d="${chartPath(unmanagedHourly, x0, y0, width, height, maxValue)}" fill="none" stroke="var(--unmanaged)" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
        </svg>
        <div class="legend-inline">
          <span><span class="legend-dot" style="background:var(--managed)"></span>Managed</span>
          <span><span class="legend-dot" style="background:var(--unmanaged)"></span>Unmanaged</span>
        </div>`;
    }

    function comparisonTableHtml(managed, unmanaged) {
      return `
        <div class="compare-table">
          <span></span><span class="head">Daily</span><span class="head">Peak</span>
          <span>Managed</span><span class="value">${formatNumber(managed ? managed[DAILY] : 0, 1)} kWh</span><span class="value">${formatNumber(managed ? managed[PEAK] : 0, 2)} kW</span>
          <span>Unmanaged</span><span class="value">${formatNumber(unmanaged ? unmanaged[DAILY] : 0, 1)} kWh</span><span class="value">${formatNumber(unmanaged ? unmanaged[PEAK] : 0, 2)} kW</span>
        </div>`;
    }

    function popupHtml(feature) {
      const selected = recordFor(feature);
      const managed = recordFor(feature, "managed");
      const unmanaged = recordFor(feature, "unmanaged");
      const peakHour = selected ? selected[PEAK_HOUR] : null;
      const coords = feature.geometry.coordinates;
      return `
        <div class="popup-title">${feature.properties.pointId}</div>
        <div class="popup-grid">
          <span>Year</span><span>${currentYear()}</span>
          <span>Scenario</span><span>${APP_DATA.metadata.scenarioLabels[scenarioSelect.value]}</span>
          <span>Mode</span><span>${APP_DATA.metadata.modeLabels[modeSelect.value]}</span>
          <span>Coordinates</span><span>${formatNumber(coords[1], 4)}, ${formatNumber(coords[0], 4)}</span>
          <span>Daily energy</span><span>${formatNumber(selected ? selected[DAILY] : 0, 1)} kWh</span>
          <span>Peak demand</span><span>${formatNumber(selected ? selected[PEAK] : 0, 2)} kW</span>
          <span>Peak hour</span><span>${peakHour === null ? "n/a" : String(peakHour).padStart(2, "0") + ":00"}</span>
          <span>${hourLabelText()}</span><span>${formatNumber(selected ? hourValue(selected) : 0, 2)} kW</span>
        </div>
        ${hourlyChartHtml(managed, unmanaged)}
        ${comparisonTableHtml(managed, unmanaged)}`;
    }

    function tooltipHtml(feature) {
      return `<span class="point-tooltip">${feature.properties.pointId}</span><br>${formatMetric(valueFor(recordFor(feature)))}`;
    }

    function updateLayerStyles() {
      pointLayer.eachLayer(layer => {
        layer.setStyle(markerStyle(layer.feature));
        layer.bindTooltip(tooltipHtml(layer.feature), { sticky: true });
        if (layer.isPopupOpen()) layer.setPopupContent(popupHtml(layer.feature));
      });
    }

    function currentRows() {
      return APP_DATA.points.features.map(feature => ({
        feature,
        record: recordFor(feature)
      }));
    }

    function updateStats() {
      const rows = currentRows();
      const values = rows.map(row => ({
        feature: row.feature,
        record: row.record,
        value: valueFor(row.record)
      }));
      const top = values.slice().sort((a, b) => b.value - a.value)[0];
      const dailyTotal = rows.reduce((sum, row) => sum + (row.record ? row.record[DAILY] || 0 : 0), 0);
      const hourlyTotals = Array.from({ length: 24 }, (_, hour) =>
        rows.reduce((sum, row) => sum + (row.record ? row.record[HOURLY][hour] || 0 : 0), 0)
      );
      const regionPeak = Math.max(...hourlyTotals);
      const regionPeakHour = hourlyTotals.indexOf(regionPeak);
      const active = rows.filter(row => row.record && row.record[DAILY] > 0).length;
      const selectedTotal = values.reduce((sum, row) => sum + row.value, 0);
      const meta = metricMeta();

      document.getElementById("yearLabel").textContent = currentYear();
      document.getElementById("viewBadge").textContent = `${APP_DATA.metadata.modeLabels[modeSelect.value]} | ${APP_DATA.metadata.scenarioLabels[scenarioSelect.value]}`;
      document.getElementById("hourLabel").textContent = hourLabelText();
      document.getElementById("totalLabel").textContent =
        metricSelect.value === "daily" ? "Total energy" :
        metricSelect.value === "hourly" ? "Total at hour" :
        "Sum of point peaks";
      document.getElementById("totalValue").textContent = `${formatNumber(metricSelect.value === "daily" ? dailyTotal : selectedTotal, meta.decimals)} ${meta.unit}`;
      document.getElementById("topPoint").textContent = top ? `${top.feature.properties.pointId} (${formatMetric(top.value)})` : "n/a";
      document.getElementById("regionPeak").textContent = `${formatNumber(regionPeak, 2)} kW at ${String(regionPeakHour).padStart(2, "0")}:00`;
      document.getElementById("activePoints").textContent = active.toLocaleString();
    }

    function updateLegend() {
      const meta = metricMeta();
      const legend = document.getElementById("legendContent");
      const rows = colors.map((color, index) => {
        const low = breaks[index];
        const high = breaks[index + 1];
        const label = index === 0
          ? `up to ${formatNumber(high, meta.decimals)}`
          : `${formatNumber(low, meta.decimals)} to ${formatNumber(high, meta.decimals)}`;
        return `<div class="legend-row"><span class="legend-swatch" style="background:${color}"></span><span>${label}</span></div>`;
      }).join("");
      legend.innerHTML = `<div class="legend-title">${meta.label}</div>${rows}`;
    }

    function refresh({ recomputeBreaks = false } = {}) {
      if (recomputeBreaks) computeBreaks();
      yearSlider.value = String(yearIndex);
      yearSelect.value = String(currentYear());
      updateLayerStyles();
      updateStats();
      updateLegend();
    }

    function setYearIndex(nextIndex) {
      yearIndex = Math.max(0, Math.min(APP_DATA.metadata.years.length - 1, nextIndex));
      refresh();
    }

    function stopPlayback() {
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
        document.getElementById("playYears").innerHTML = "&#9654;";
        document.getElementById("playYears").setAttribute("aria-label", "Play years");
      }
    }

    function togglePlayback() {
      if (playTimer) {
        stopPlayback();
        return;
      }
      document.getElementById("playYears").innerHTML = "&#10074;&#10074;";
      document.getElementById("playYears").setAttribute("aria-label", "Pause years");
      playTimer = setInterval(() => {
        yearIndex = yearIndex >= APP_DATA.metadata.years.length - 1 ? 0 : yearIndex + 1;
        refresh();
      }, 900);
    }

    const legendControl = L.control({ position: "bottomright" });
    legendControl.onAdd = function () {
      const div = L.DomUtil.create("div", "map-legend");
      div.id = "legendContent";
      L.DomEvent.disableClickPropagation(div);
      return div;
    };
    legendControl.addTo(map);

    L.geoJSON(APP_DATA.h3Boundary, {
      style: {
        color: "#111827",
        weight: 2.2,
        opacity: 0.9,
        fillColor: "#ffffff",
        fillOpacity: 0.04
      },
      interactive: false
    }).addTo(map);

    computeBreaks();
    pointLayer = L.geoJSON(APP_DATA.points, {
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, markerStyle(feature)),
      onEachFeature: (feature, layer) => {
        layer.bindPopup(popupHtml(feature));
        layer.bindTooltip(tooltipHtml(feature), { sticky: true });
        layer.on({
          mouseover: event => event.target.setStyle({ weight: 2.2, color: "#111827", fillOpacity: 0.9 }),
          mouseout: event => event.target.setStyle(markerStyle(feature)),
          click: event => event.target.setPopupContent(popupHtml(feature))
        });
      }
    }).addTo(map);
    map.fitBounds(pointLayer.getBounds(), { padding: [28, 28] });
    updateStats();
    updateLegend();

    scenarioSelect.addEventListener("change", () => refresh({ recomputeBreaks: true }));
    modeSelect.addEventListener("change", () => refresh({ recomputeBreaks: true }));
    metricSelect.addEventListener("change", () => refresh({ recomputeBreaks: true }));
    hourSlider.addEventListener("input", () => refresh({ recomputeBreaks: metricSelect.value === "hourly" }));
    yearSlider.addEventListener("input", event => {
      stopPlayback();
      setYearIndex(Number(event.target.value));
    });
    yearSelect.addEventListener("change", event => {
      stopPlayback();
      yearIndex = APP_DATA.metadata.years.indexOf(Number(event.target.value));
      refresh();
    });
    document.getElementById("prevYear").addEventListener("click", () => {
      stopPlayback();
      setYearIndex(yearIndex - 1);
    });
    document.getElementById("nextYear").addEventListener("click", () => {
      stopPlayback();
      setYearIndex(yearIndex + 1);
    });
    document.getElementById("playYears").addEventListener("click", togglePlayback);
  </script>
</body>
</html>
"""
    return template.replace("__APP_DATA__", data_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h3",
        default="882b890311fffff",
        help="Selected H3 cell to map.",
    )
    parser.add_argument(
        "--events",
        default="out_behavioral_charging_full_20260615_232335_nearest8_streamed/charging_events.csv",
        help="Full behavioral charging events CSV.",
    )
    parser.add_argument(
        "--adoption",
        default="models/adoption_forecast_albany_zip_for_charging.csv",
        help="Charging-model adoption scenario file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output standalone HTML map. Defaults to visualization/albany_h3_<h3>_point_charging_demand_map.html.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output point-level hourly forecast CSV. Defaults to visualization/albany_h3_<h3>_point_charging_demand_forecast.csv.",
    )
    parser.add_argument("--bin-minutes", type=int, default=15)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_h3 = str(args.h3)
    events_path = resolve(args.events)
    adoption_path = resolve(args.adoption)
    output_path = resolve(
        args.output or f"visualization/albany_h3_{selected_h3}_point_charging_demand_map.html"
    )
    output_csv = resolve(
        args.output_csv
        or f"visualization/albany_h3_{selected_h3}_point_charging_demand_forecast.csv"
    )

    events, scanned_rows = read_selected_events(events_path, selected_h3, args.chunksize)
    adoption = load_adoption_file(adoption_path)
    hourly = build_scaled_hourly_point_forecast(
        events,
        adoption,
        bin_minutes=args.bin_minutes,
    )
    write_hourly_csv(hourly, output_csv)

    point_features, point_index = build_point_features(hourly)
    app_data = {
        "metadata": build_metadata(
            selected_h3=selected_h3,
            events=events,
            hourly=hourly,
            source_path=events_path,
            adoption_path=adoption_path,
            output_csv=output_csv,
            scanned_rows=scanned_rows,
        ),
        "h3Boundary": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"h3": selected_h3},
                    "geometry": {"type": "Polygon", "coordinates": [h3_boundary(selected_h3)]},
                }
            ],
        },
        "points": {"type": "FeatureCollection", "features": point_features},
        "views": build_view_payload(hourly, point_index),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(app_data), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(f"Wrote {output_csv.relative_to(ROOT)}")
    print(
        "Embedded "
        f"{app_data['metadata']['pointCount']} points from {app_data['metadata']['selectedEventRows']} events "
        f"inside H3 {selected_h3}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
