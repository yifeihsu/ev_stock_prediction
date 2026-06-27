#!/usr/bin/env python3
"""Build a standalone Leaflet map for Albany H3-8 charging demand forecasts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
USECOLS = [
    "managed_flag",
    "adoption_scenario",
    "forecast_year",
    "charging_h3",
    "hour",
    "kw",
    "kwh",
]
GROUP_COLS = ["mode", "adoption_scenario", "forecast_year", "charging_h3", "hour"]
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


def aggregate_hourly_load(path: Path, chunksize: int) -> tuple[pd.DataFrame, int]:
    chunks: list[pd.DataFrame] = []
    row_count = 0
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize):
        row_count += len(chunk)
        chunk = chunk.dropna(subset=["charging_h3", "hour"]).copy()
        chunk["mode"] = chunk["managed_flag"].map(normalize_mode)
        chunk["adoption_scenario"] = chunk["adoption_scenario"].astype(str)
        chunk["forecast_year"] = pd.to_numeric(chunk["forecast_year"], errors="coerce").astype("Int64")
        chunk["hour"] = pd.to_numeric(chunk["hour"], errors="coerce").astype("Int64")
        chunk["kw"] = pd.to_numeric(chunk["kw"], errors="coerce").fillna(0.0)
        chunk["kwh"] = pd.to_numeric(chunk["kwh"], errors="coerce").fillna(0.0)
        chunk = chunk.dropna(subset=["forecast_year", "hour"])
        chunk["forecast_year"] = chunk["forecast_year"].astype(int)
        chunk["hour"] = chunk["hour"].astype(int)
        grouped = (
            chunk.groupby(GROUP_COLS, sort=False, observed=True)[["kw", "kwh"]]
            .sum()
            .reset_index()
        )
        chunks.append(grouped)

    if not chunks:
        raise ValueError(f"No H3 charging-demand rows found in {path}")

    agg = (
        pd.concat(chunks, ignore_index=True)
        .groupby(GROUP_COLS, sort=True, observed=True)[["kw", "kwh"]]
        .sum()
        .reset_index()
    )
    agg["charging_h3"] = agg["charging_h3"].astype(str)
    agg["forecast_year"] = agg["forecast_year"].astype(int)
    agg["hour"] = agg["hour"].astype(int)
    return agg, row_count


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


def build_geojson(cells: list[str]) -> dict[str, Any]:
    features = []
    for idx, cell in enumerate(cells):
        features.append(
            {
                "type": "Feature",
                "properties": {"i": idx, "h3": cell},
                "geometry": {"type": "Polygon", "coordinates": [h3_boundary(cell)]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_view_payload(agg: pd.DataFrame, cells: list[str]) -> dict[str, list[list[Any]]]:
    cell_index = {cell: idx for idx, cell in enumerate(cells)}
    views: dict[str, list[list[Any]]] = {}
    for (mode, scenario, year, cell), group in agg.groupby(
        ["mode", "adoption_scenario", "forecast_year", "charging_h3"], sort=True
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
                cell_index[str(cell)],
                clean_number(daily_kwh, 3),
                clean_number(peak_kw, 3),
                peak_hour,
                [clean_number(value, 3) or 0 for value in hourly_kw],
            ]
        )

    for records in views.values():
        records.sort(key=lambda row: row[0])
    return views


def read_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_metadata(
    agg: pd.DataFrame,
    *,
    source_path: Path,
    source_rows: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    cells = sorted(agg["charging_h3"].unique().tolist())
    resolutions = sorted({value for cell in cells if (value := h3_resolution(cell)) is not None})
    scenarios = sorted(agg["adoption_scenario"].unique().tolist())
    years = sorted(int(value) for value in agg["forecast_year"].unique().tolist())
    modes = [mode for mode in ["managed", "unmanaged"] if mode in set(agg["mode"])]
    default_scenario = "tco_evse" if "tco_evse" in scenarios else scenarios[-1]
    default_mode = "unmanaged" if "unmanaged" in modes else modes[0]
    default_year = max(years)
    latest = agg[
        (agg["mode"].eq(default_mode))
        & (agg["adoption_scenario"].eq(default_scenario))
        & (agg["forecast_year"].eq(default_year))
    ]
    daily = latest.groupby("charging_h3", sort=False)["kwh"].sum()
    peak_by_hour = latest.groupby(["charging_h3", "hour"], sort=False)["kw"].sum().reset_index()
    top_peak = peak_by_hour.sort_values("kw", ascending=False).head(1)
    configuration = summary.get("configuration", {})
    return {
        "title": "Albany H3 Charging Demand Forecast",
        "sourceCsv": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
        "sourceRows": int(source_rows),
        "aggregatedRows": int(len(agg)),
        "h3Count": int(len(cells)),
        "h3Resolution": int(resolutions[0]) if len(resolutions) == 1 else configuration.get("h3_resolution"),
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
            "activeH3": int((daily > 0).sum()),
            "topPeakH3": str(top_peak.iloc[0]["charging_h3"]) if len(top_peak) else None,
            "topPeakHour": int(top_peak.iloc[0]["hour"]) if len(top_peak) else None,
            "topPeakKw": clean_number(top_peak.iloc[0]["kw"], 1) if len(top_peak) else None,
        },
        "configuration": {
            "scaleAdoption": configuration.get("scale_adoption"),
            "efficiencyKwhPerMile": configuration.get("efficiency_kwh_per_mile"),
            "binMinutes": configuration.get("bin_minutes"),
            "allChargerTypesAggregated": True,
        },
    }


def build_html(app_data: dict[str, Any]) -> str:
    data_json = json.dumps(app_data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Albany H3 Charging Demand Forecast</title>
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    :root {
      --ink: #17212b;
      --muted: #5d6977;
      --panel: rgba(255, 255, 255, 0.95);
      --border: rgba(25, 38, 50, 0.18);
      --accent: #0f766e;
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
      width: min(420px, calc(100vw - 68px));
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
    }
    .map-legend {
      min-width: 200px;
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
    .h3-tooltip {
      font-weight: 760;
      color: var(--ink);
    }
    .leaflet-popup-content {
      margin: 12px 13px;
      min-width: 235px;
    }
    .popup-title {
      margin: 0 0 8px;
      font-size: 15px;
      font-weight: 780;
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
    .profile-card {
      margin-top: 10px;
      padding-top: 9px;
      border-top: 1px solid rgba(23, 33, 43, 0.1);
    }
    .profile-title {
      margin-bottom: 6px;
      font-size: 12px;
      font-weight: 780;
    }
    .profile-svg {
      display: block;
      width: 100%;
      max-width: 280px;
      height: auto;
      overflow: visible;
    }
    .profile-axis {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      margin-top: 2px;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.2;
    }
    .profile-axis span:nth-child(2) {
      text-align: center;
    }
    .profile-axis span:nth-child(3) {
      text-align: right;
    }
    .profile-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 12px;
      margin-top: 7px;
      font-size: 11px;
      font-weight: 700;
    }
    .profile-legend span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      white-space: nowrap;
    }
    .legend-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      display: inline-block;
    }
    .profile-stats {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 4px 10px;
      margin-top: 7px;
      font-size: 11px;
      line-height: 1.25;
    }
    .profile-stats span:nth-child(1),
    .profile-stats span:nth-child(2),
    .profile-stats span:nth-child(3) {
      color: var(--muted);
      font-weight: 700;
    }
    .profile-stats span:nth-child(3n + 2),
    .profile-stats span:nth-child(3n + 3) {
      text-align: right;
      font-weight: 720;
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
    <h1>Albany H3 Charging Demand Forecast</h1>
    <div class="subtitle"><span id="h3Count"></span> H3-<span id="h3Resolution"></span> cells, all charger types</div>
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
        <span class="stat-label">Highest cell</span>
        <span class="stat-value" id="topCell"></span>
      </div>
      <div class="stat">
        <span class="stat-label">System peak</span>
        <span class="stat-value" id="systemPeak"></span>
      </div>
      <div class="stat">
        <span class="stat-label">Active cells</span>
        <span class="stat-value" id="activeCells"></span>
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
    const colors = ["#f7fcf0", "#c7e9b4", "#7fcdbb", "#41b6c4", "#225ea8", "#d73027"];
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
    let h3Layer = null;
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
    document.getElementById("h3Count").textContent = APP_DATA.metadata.h3Count.toLocaleString();
    document.getElementById("h3Resolution").textContent = APP_DATA.metadata.h3Resolution;
    document.getElementById("sourceLine").textContent = `${APP_DATA.metadata.sourceCsv} (${APP_DATA.metadata.sourceRows.toLocaleString()} rows)`;

    function currentYear() {
      return APP_DATA.metadata.years[yearIndex];
    }

    function currentViewKey(year = currentYear()) {
      return viewKeyFor(modeSelect.value, scenarioSelect.value, year);
    }

    function viewKeyFor(mode, scenario = scenarioSelect.value, year = currentYear()) {
      return `${mode}|${scenario}|${year}`;
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

    function recordFor(feature) {
      return recordForMode(feature, modeSelect.value);
    }

    function recordForMode(feature, mode, scenario = scenarioSelect.value, year = currentYear()) {
      return viewRecords(viewKeyFor(mode, scenario, year)).get(feature.properties.i) || null;
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
        if (raw[i] <= raw[i - 1]) raw[i] = raw[i - 1] + 1;
      }
      breaks = raw;
    }

    function colorFor(value) {
      if (!Number.isFinite(value) || value <= 0) return "#e1e7ec";
      for (let i = 0; i < breaks.length - 1; i += 1) {
        if (value <= breaks[i + 1]) return colors[i];
      }
      return colors[colors.length - 1];
    }

    function profilePoints(values, maxValue, width, height, pad) {
      const innerWidth = width - pad.left - pad.right;
      const innerHeight = height - pad.top - pad.bottom;
      return values.map((value, hour) => {
        const x = pad.left + innerWidth * (hour / 23);
        const y = pad.top + innerHeight - innerHeight * (value / maxValue);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ");
    }

    function profilePlotHtml(feature) {
      const series = [
        { mode: "managed", color: "#0f766e", record: recordForMode(feature, "managed") },
        { mode: "unmanaged", color: "#b45309", record: recordForMode(feature, "unmanaged") }
      ].filter(item => item.record && APP_DATA.metadata.modes.includes(item.mode));
      if (series.length < 2) return "";

      for (const item of series) {
        item.label = APP_DATA.metadata.modeLabels[item.mode] || item.mode;
        item.values = Array.from({ length: 24 }, (_, hour) => Number(item.record[HOURLY][hour] || 0));
      }
      const maxValue = Math.max(1, ...series.flatMap(item => item.values));
      const width = 268;
      const height = 96;
      const pad = { top: 8, right: 8, bottom: 17, left: 30 };
      const selectedHour = Number(hourSlider.value);
      const selectedX = pad.left + (width - pad.left - pad.right) * (selectedHour / 23);
      const gridY = [0, 0.5, 1].map(ratio => pad.top + (height - pad.top - pad.bottom) * ratio);
      const lines = series.map(item =>
        `<polyline points="${profilePoints(item.values, maxValue, width, height, pad)}" fill="none" stroke="${item.color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" />`
      ).join("");
      const markers = series.map(item => {
        const value = item.values[selectedHour] || 0;
        const y = pad.top + (height - pad.top - pad.bottom) * (1 - value / maxValue);
        return `<circle cx="${selectedX.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="${item.color}" stroke="#fff" stroke-width="1.3" />`;
      }).join("");
      const legend = series.map(item =>
        `<span><i class="legend-dot" style="background:${item.color}"></i>${item.label}</span>`
      ).join("");
      const stats = series.map(item =>
        `<span>${item.label}</span><span>${formatNumber(item.record[DAILY], 1)} kWh</span><span>${formatNumber(item.record[PEAK], 2)} kW</span>`
      ).join("");

      return `
        <div class="profile-card">
          <div class="profile-title">Managed vs unmanaged hourly demand</div>
          <svg class="profile-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Hourly charging demand profile">
            <rect x="${pad.left}" y="${pad.top}" width="${width - pad.left - pad.right}" height="${height - pad.top - pad.bottom}" fill="#f8faf9" stroke="rgba(23,33,43,0.14)" />
            ${gridY.map(y => `<line x1="${pad.left}" x2="${width - pad.right}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="rgba(23,33,43,0.11)" />`).join("")}
            <text x="0" y="${pad.top + 4}" fill="#5d6977" font-size="10">${formatNumber(maxValue, 1)}</text>
            <text x="10" y="${height - pad.bottom + 3}" fill="#5d6977" font-size="10">0</text>
            <line x1="${selectedX.toFixed(1)}" x2="${selectedX.toFixed(1)}" y1="${pad.top}" y2="${height - pad.bottom}" stroke="rgba(17,24,39,0.28)" stroke-dasharray="3 3" />
            ${lines}
            ${markers}
          </svg>
          <div class="profile-axis"><span>00:00</span><span>12:00</span><span>23:00</span></div>
          <div class="profile-legend">${legend}</div>
          <div class="profile-stats">
            <span></span><span>Daily</span><span>Peak</span>
            ${stats}
          </div>
        </div>`;
    }

    function featureStyle(feature) {
      const value = valueFor(recordFor(feature));
      return {
        color: "#253340",
        weight: 0.35,
        opacity: 0.7,
        fillColor: colorFor(value),
        fillOpacity: value > 0 ? 0.72 : 0.22
      };
    }

    function popupHtml(feature) {
      const record = recordFor(feature);
      const h3 = feature.properties.h3;
      const daily = record ? record[DAILY] : 0;
      const peak = record ? record[PEAK] : 0;
      const peakHour = record ? record[PEAK_HOUR] : null;
      const hourly = record ? hourValue(record) : 0;
      return `
        <div class="popup-title">H3 ${h3}</div>
        <div class="popup-grid">
          <span>Year</span><span>${currentYear()}</span>
          <span>Scenario</span><span>${APP_DATA.metadata.scenarioLabels[scenarioSelect.value]}</span>
          <span>Mode</span><span>${APP_DATA.metadata.modeLabels[modeSelect.value]}</span>
          <span>Daily energy</span><span>${formatNumber(daily, 1)} kWh</span>
          <span>Peak demand</span><span>${formatNumber(peak, 2)} kW</span>
          <span>Peak hour</span><span>${peakHour === null ? "n/a" : String(peakHour).padStart(2, "0") + ":00"}</span>
          <span>Demand at ${hourLabelText()}</span><span>${formatNumber(hourly, 2)} kW</span>
        </div>
        ${profilePlotHtml(feature)}`;
    }

    function tooltipHtml(feature) {
      const record = recordFor(feature);
      return `<span class="h3-tooltip">${feature.properties.h3}</span><br>${formatMetric(valueFor(record))}`;
    }

    function updateLayerStyles() {
      h3Layer.eachLayer(layer => {
        layer.setStyle(featureStyle(layer.feature));
        layer.bindTooltip(tooltipHtml(layer.feature), { sticky: true });
        if (layer.isPopupOpen()) layer.setPopupContent(popupHtml(layer.feature));
      });
    }

    function currentRows() {
      return APP_DATA.geojson.features.map(feature => ({
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
      const systemPeak = Math.max(...hourlyTotals);
      const systemPeakHour = hourlyTotals.indexOf(systemPeak);
      const active = rows.filter(row => row.record && row.record[DAILY] > 0).length;
      const selectedTotal = values.reduce((sum, row) => sum + row.value, 0);
      const meta = metricMeta();

      document.getElementById("yearLabel").textContent = currentYear();
      document.getElementById("viewBadge").textContent = `${APP_DATA.metadata.modeLabels[modeSelect.value]} | ${APP_DATA.metadata.scenarioLabels[scenarioSelect.value]}`;
      document.getElementById("hourLabel").textContent = hourLabelText();
      document.getElementById("totalLabel").textContent =
        metricSelect.value === "daily" ? "Total energy" :
        metricSelect.value === "hourly" ? "Total at hour" :
        "Sum of cell peaks";
      document.getElementById("totalValue").textContent = `${formatNumber(metricSelect.value === "daily" ? dailyTotal : selectedTotal, meta.decimals)} ${meta.unit}`;
      document.getElementById("topCell").textContent = top ? `${top.feature.properties.h3} (${formatMetric(top.value)})` : "n/a";
      document.getElementById("systemPeak").textContent = `${formatNumber(systemPeak, 2)} kW at ${String(systemPeakHour).padStart(2, "0")}:00`;
      document.getElementById("activeCells").textContent = active.toLocaleString();
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

    computeBreaks();
    h3Layer = L.geoJSON(APP_DATA.geojson, {
      style: featureStyle,
      onEachFeature: (feature, layer) => {
        layer.bindPopup(popupHtml(feature));
        layer.bindTooltip(tooltipHtml(feature), { sticky: true });
        layer.on({
          mouseover: event => event.target.setStyle({ weight: 1.8, color: "#111827", fillOpacity: 0.88 }),
          mouseout: event => h3Layer.resetStyle(event.target),
          click: event => event.target.setPopupContent(popupHtml(feature))
        });
      }
    }).addTo(map);
    map.fitBounds(h3Layer.getBounds(), { padding: [24, 24] });
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
        "--load-csv",
        default="out_behavioral_charging_full_20260615_232335_nearest8_streamed/scaled_load_hourly_by_charging_h3.csv",
        help="Adoption-scaled hourly charging load by H3.",
    )
    parser.add_argument(
        "--summary-json",
        default="out_behavioral_charging_full_20260615_232335_nearest8_streamed/model_summary.json",
        help="Behavioral charging model summary JSON.",
    )
    parser.add_argument(
        "--output",
        default="visualization/albany_h3_charging_demand_forecast_map.html",
        help="Output standalone HTML map.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=750_000,
        help="CSV rows per aggregation chunk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_path = resolve(args.load_csv)
    summary_path = resolve(args.summary_json) if args.summary_json else None
    output_path = resolve(args.output)

    agg, source_rows = aggregate_hourly_load(load_path, args.chunksize)
    cells = sorted(agg["charging_h3"].unique().tolist())
    summary = read_summary(summary_path)
    app_data = {
        "metadata": build_metadata(
            agg,
            source_path=load_path,
            source_rows=source_rows,
            summary=summary,
        ),
        "geojson": build_geojson(cells),
        "views": build_view_payload(agg, cells),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(app_data), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(
        "Embedded "
        f"{len(cells)} H3 cells, {len(app_data['views'])} scenario/year/mode views, "
        f"{len(agg)} aggregated hourly rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
