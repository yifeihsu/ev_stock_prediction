#!/usr/bin/env python3
"""Build a standalone Leaflet map for the Albany ZIP adoption forecast."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


FORECAST_COLUMNS = {
    "baseline": {
        "stock": "stock_ev_t_hat_baseline",
        "flow": "flow_ev_t_hat_baseline",
    },
    "tco": {
        "stock": "stock_ev_t_hat_tco",
        "flow": "flow_ev_t_hat_tco",
    },
    "tco_evse": {
        "stock": "stock_ev_t_hat_tco_evse",
        "flow": "flow_ev_t_hat_tco_evse",
    },
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def normalize_zip(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5) if text.isdigit() and len(text) <= 5 else text


def clean_number(value: object, digits: int | None = None) -> float | None:
    if value is None or pd.isna(value):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    if digits is not None:
        return round(out, digits)
    return out


def read_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"zip": str})
    required = {
        "zip",
        "date",
        "model_period",
        "zip_status",
        "stock_ev_t_obs",
        "flow_ev_t_obs",
        "market_size",
        "total_vehicle_market_proxy",
    }
    for scenario_cols in FORECAST_COLUMNS.values():
        required.update(scenario_cols.values())
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Forecast file is missing required columns: {missing}")

    df = df.copy()
    df["zip"] = df["zip"].map(normalize_zip)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values(["date", "zip"]).reset_index(drop=True)


def read_filtered_geojson(path: Path, zips: set[str]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    features: list[dict[str, Any]] = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        zip_code = normalize_zip(props.get("ZCTA5CE10") or props.get("zip"))
        if zip_code not in zips:
            continue
        intpt_lat = clean_number(props.get("INTPTLAT10"), 6)
        intpt_lon = clean_number(props.get("INTPTLON10"), 6)
        area_sq_km = clean_number(clean_number(props.get("ALAND10")) / 1_000_000, 2) if props.get("ALAND10") else None
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "zip": zip_code,
                    "area_sq_km": area_sq_km,
                    "intpt_lat": intpt_lat,
                    "intpt_lon": intpt_lon,
                },
                "geometry": feature.get("geometry"),
            }
        )

    found = {feature["properties"]["zip"] for feature in features}
    missing = sorted(zips - found)
    if missing:
        raise ValueError(f"Missing ZIP polygons in {path}: {missing}")

    return {"type": "FeatureCollection", "features": sorted(features, key=lambda f: f["properties"]["zip"])}


def build_forecast_payload(df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    payload: dict[str, dict[str, dict[str, Any]]] = {}
    for date, group in df.groupby("date", sort=True):
        date_rows: dict[str, dict[str, Any]] = {}
        for _, row in group.iterrows():
            stock = {
                scenario: clean_number(row[columns["stock"]], 3)
                for scenario, columns in FORECAST_COLUMNS.items()
            }
            flow = {
                scenario: clean_number(row[columns["flow"]], 3)
                for scenario, columns in FORECAST_COLUMNS.items()
            }
            date_rows[str(row["zip"])] = {
                "period": str(row["model_period"]),
                "status": str(row["zip_status"]),
                "observedStock": clean_number(row["stock_ev_t_obs"], 3),
                "observedFlow": clean_number(row["flow_ev_t_obs"], 3),
                "stock": stock,
                "flow": flow,
                "marketSize": clean_number(row["market_size"], 3),
                "vehicleMarket": clean_number(row["total_vehicle_market_proxy"], 3),
            }
        payload[date] = date_rows
    return payload


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def summarize(df: pd.DataFrame, *, title: str, forecast_path: Path, geojson_path: Path) -> dict[str, Any]:
    dates = sorted(df["date"].unique().tolist())
    forecast_end = dates[-1]
    latest_observed = df.loc[df["model_period"].eq("observed_fit"), "date"].max()
    end = df[df["date"].eq(forecast_end)].copy()
    end["zip"] = end["zip"].map(normalize_zip)
    top = (
        end.sort_values("stock_ev_t_hat_tco_evse", ascending=False)
        .loc[:, ["zip", "stock_ev_t_hat_tco_evse", "flow_ev_t_hat_tco_evse"]]
        .head(5)
    )
    return {
        "title": title,
        "forecastCsv": _relative_or_absolute(forecast_path),
        "geojson": _relative_or_absolute(geojson_path),
        "zipCount": int(df["zip"].nunique()),
        "rowCount": int(len(df)),
        "dateMin": dates[0],
        "dateMax": forecast_end,
        "latestObservedDate": latest_observed,
        "defaultScenario": "tco_evse",
        "forecastEndTotals": {
            "stockBaseline": clean_number(end["stock_ev_t_hat_baseline"].sum(), 1),
            "stockTco": clean_number(end["stock_ev_t_hat_tco"].sum(), 1),
            "stockTcoEvse": clean_number(end["stock_ev_t_hat_tco_evse"].sum(), 1),
            "flowTcoEvse": clean_number(end["flow_ev_t_hat_tco_evse"].sum(), 1),
        },
        "topForecastEndZips": [
            {
                "zip": str(row["zip"]),
                "stockTcoEvse": clean_number(row["stock_ev_t_hat_tco_evse"], 1),
                "flowTcoEvse": clean_number(row["flow_ev_t_hat_tco_evse"], 1),
            }
            for _, row in top.iterrows()
        ],
    }


def build_html(app_data: dict[str, Any]) -> str:
    data_json = json.dumps(app_data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    map_title = html.escape(str(app_data["metadata"]["title"]))
    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__MAP_TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    :root {
      --ink: #17212b;
      --muted: #5c6875;
      --panel: rgba(255, 255, 255, 0.95);
      --border: rgba(26, 39, 52, 0.18);
      --accent: #0f766e;
      --focus: #b45309;
    }
    html, body, #map {
      width: 100%;
      height: 100%;
      margin: 0;
    }
    body {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #e9eef1;
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
      width: min(390px, calc(100vw - 68px));
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
    .timeline {
      display: grid;
      gap: 8px;
      padding: 10px 0 12px;
      border-top: 1px solid rgba(23, 33, 43, 0.1);
      border-bottom: 1px solid rgba(23, 33, 43, 0.1);
    }
    .date-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    #dateLabel {
      font-size: 18px;
      font-weight: 760;
    }
    #periodBadge {
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
    .timeline-buttons {
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
    .check-row {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 7px;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 620;
    }
    .check-row input {
      width: 15px;
      height: 15px;
      accent-color: var(--accent);
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
    .top-list {
      margin: 12px 0 0;
      padding: 10px 0 0;
      border-top: 1px solid rgba(23, 33, 43, 0.1);
      font-size: 12px;
      color: var(--muted);
    }
    .top-list strong {
      color: var(--ink);
    }
    .map-legend {
      min-width: 190px;
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
    .zip-tooltip {
      font-weight: 760;
      color: var(--ink);
    }
    .leaflet-popup-content {
      margin: 12px 13px;
      min-width: 210px;
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
    @media (max-width: 680px) {
      .map-panel {
        top: 10px;
        left: 10px;
        right: 10px;
        width: auto;
        max-height: 48vh;
      }
      .field-grid, .stats-grid {
        grid-template-columns: 1fr;
      }
      .timeline-buttons {
        grid-template-columns: 36px 36px 36px;
      }
      .check-row {
        justify-content: flex-start;
        grid-column: 1 / -1;
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
    <h1>__MAP_TITLE__</h1>
    <div class="subtitle"><span id="zipCount"></span> ZIPs, <span id="dateRange"></span></div>
    <div class="field-grid">
      <label class="field">Scenario
        <select id="scenarioSelect">
          <option value="tco_evse">TCO + EVSE</option>
          <option value="tco">TCO only</option>
          <option value="baseline">Baseline</option>
        </select>
      </label>
      <label class="field">Measure
        <select id="measureSelect">
          <option value="stock">EV stock</option>
          <option value="flow">Monthly EV flow</option>
          <option value="share">EV stock share</option>
        </select>
      </label>
    </div>
    <div class="timeline">
      <div class="date-line">
        <span id="dateLabel"></span>
        <span id="periodBadge"></span>
      </div>
      <input id="dateSlider" type="range" min="0" step="1" />
      <div class="timeline-buttons">
        <button id="prevDate" type="button" title="Previous date" aria-label="Previous date">&#9664;</button>
        <button id="playDates" type="button" title="Play dates" aria-label="Play dates">&#9654;</button>
        <button id="nextDate" type="button" title="Next date" aria-label="Next date">&#9654;&#10072;</button>
        <label class="check-row"><input id="observedToggle" type="checkbox" checked />Observed history</label>
      </div>
    </div>
    <div class="stats-grid">
      <div class="stat">
        <span class="stat-label" id="totalLabel">County total</span>
        <span class="stat-value" id="totalValue"></span>
      </div>
      <div class="stat">
        <span class="stat-label">Highest ZIP</span>
        <span class="stat-value" id="topZip"></span>
      </div>
      <div class="stat">
        <span class="stat-label">Forecast end</span>
        <span class="stat-value" id="endTotal"></span>
      </div>
      <div class="stat">
        <span class="stat-label">Latest observed</span>
        <span class="stat-value" id="latestObserved"></span>
      </div>
    </div>
    <div class="top-list" id="topList"></div>
  </section>
  <script>
    const APP_DATA = __APP_DATA__;
    const dates = APP_DATA.dates;
    const colors = ["#f7fbff", "#c7e9e1", "#7fcdbb", "#2c7fb8", "#fdae61"];
    const map = L.map("map", { zoomControl: true, preferCanvas: true });

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors"
    }).addTo(map);

    const scenarioSelect = document.getElementById("scenarioSelect");
    const measureSelect = document.getElementById("measureSelect");
    const observedToggle = document.getElementById("observedToggle");
    const dateSlider = document.getElementById("dateSlider");
    const dateLabel = document.getElementById("dateLabel");
    const periodBadge = document.getElementById("periodBadge");
    let dateIndex = dates.length - 1;
    let breaks = [];
    let zipLayer = null;
    let playTimer = null;

    scenarioSelect.value = APP_DATA.metadata.defaultScenario;
    dateSlider.max = String(dates.length - 1);
    dateSlider.value = String(dateIndex);
    document.getElementById("zipCount").textContent = APP_DATA.metadata.zipCount.toLocaleString();
    document.getElementById("dateRange").textContent = `${APP_DATA.metadata.dateMin} to ${APP_DATA.metadata.dateMax}`;
    document.getElementById("endTotal").textContent = `${formatNumber(APP_DATA.metadata.forecastEndTotals.stockTcoEvse, 0)} EVs`;
    document.getElementById("latestObserved").textContent = APP_DATA.metadata.latestObservedDate;
    document.getElementById("topList").innerHTML = `Top forecast ZIPs: ${APP_DATA.metadata.topForecastEndZips.map(row => `<strong>${row.zip}</strong> ${formatNumber(row.stockTcoEvse, 0)}`).join(" | ")}`;

    function scenarioLabel() {
      return scenarioSelect.options[scenarioSelect.selectedIndex].text;
    }

    function measureMeta() {
      const measure = measureSelect.value;
      if (measure === "flow") {
        return { label: "Monthly EV flow", unit: "EVs/month", decimals: 1 };
      }
      if (measure === "share") {
        return { label: "EV stock share", unit: "%", decimals: 1 };
      }
      return { label: "EV stock", unit: "EVs", decimals: 0 };
    }

    function currentDate() {
      return dates[dateIndex];
    }

    function recordFor(zip, date = currentDate()) {
      return APP_DATA.forecastByDate[date]?.[zip] || null;
    }

    function isObservedDate(record) {
      return record && record.period === "observed_fit";
    }

    function stockFor(record) {
      if (!record) return null;
      if (observedToggle.checked && isObservedDate(record) && record.observedStock !== null) {
        return record.observedStock;
      }
      return record.stock[scenarioSelect.value];
    }

    function flowFor(record) {
      if (!record) return null;
      if (observedToggle.checked && isObservedDate(record) && record.observedFlow !== null) {
        return record.observedFlow;
      }
      return record.flow[scenarioSelect.value];
    }

    function valueFor(record) {
      if (!record) return null;
      if (measureSelect.value === "flow") return flowFor(record);
      if (measureSelect.value === "share") {
        const stock = stockFor(record);
        return stock !== null && record.vehicleMarket > 0 ? (100 * stock / record.vehicleMarket) : null;
      }
      return stockFor(record);
    }

    function formatNumber(value, digits = 0) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: digits
      });
    }

    function formatMetric(value) {
      const meta = measureMeta();
      const formatted = formatNumber(value, meta.decimals);
      return meta.unit === "%" ? `${formatted}%` : `${formatted} ${meta.unit}`;
    }

    function allMetricValues() {
      const values = [];
      for (const date of dates) {
        for (const zip of APP_DATA.zips) {
          const value = valueFor(recordFor(zip, date));
          if (value !== null && Number.isFinite(value)) values.push(value);
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
      const values = allMetricValues();
      const raw = [0, 0.2, 0.4, 0.6, 0.8, 1].map(q => quantile(values, q));
      for (let i = 1; i < raw.length; i += 1) {
        if (raw[i] <= raw[i - 1]) raw[i] = raw[i - 1] + 1;
      }
      breaks = raw;
    }

    function colorFor(value) {
      if (value === null || !Number.isFinite(value)) return "#d8dee4";
      for (let i = 0; i < breaks.length - 1; i += 1) {
        if (value <= breaks[i + 1]) return colors[i];
      }
      return colors[colors.length - 1];
    }

    function featureStyle(feature) {
      const record = recordFor(feature.properties.zip);
      const value = valueFor(record);
      return {
        color: "#253340",
        weight: 0.8,
        opacity: 0.8,
        fillColor: colorFor(value),
        fillOpacity: value === null ? 0.28 : 0.78
      };
    }

    function popupHtml(zip) {
      const record = recordFor(zip);
      if (!record) return `<div class="popup-title">ZIP ${zip}</div>`;
      const stock = stockFor(record);
      const flow = flowFor(record);
      const share = record.vehicleMarket > 0 && stock !== null ? (100 * stock / record.vehicleMarket) : null;
      return `
        <div class="popup-title">ZIP ${zip}</div>
        <div class="popup-grid">
          <span>Date</span><span>${currentDate()}</span>
          <span>Period</span><span>${record.period === "future_forecast" ? "Forecast" : "Observed fit"}</span>
          <span>Scenario</span><span>${scenarioLabel()}</span>
          <span>Displayed stock</span><span>${formatNumber(stock, 0)} EVs</span>
          <span>Displayed flow</span><span>${formatNumber(flow, 1)} EVs/month</span>
          <span>Stock share</span><span>${formatNumber(share, 1)}%</span>
          <span>Observed stock</span><span>${formatNumber(record.observedStock, 0)} EVs</span>
          <span>Market potential</span><span>${formatNumber(record.marketSize, 0)}</span>
        </div>`;
    }

    function updateLayerStyles() {
      zipLayer.eachLayer(layer => {
        layer.setStyle(featureStyle(layer.feature));
        const zip = layer.feature.properties.zip;
        layer.bindTooltip(`<span class="zip-tooltip">ZIP ${zip}</span><br>${formatMetric(valueFor(recordFor(zip)))}`, { sticky: true });
        if (layer.isPopupOpen()) layer.setPopupContent(popupHtml(zip));
      });
    }

    function updateStats() {
      const rows = APP_DATA.zips.map(zip => ({ zip, record: recordFor(zip) }));
      const values = rows
        .map(row => ({ zip: row.zip, value: valueFor(row.record), record: row.record }))
        .filter(row => row.value !== null && Number.isFinite(row.value));
      const top = values.slice().sort((a, b) => b.value - a.value)[0];
      const meta = measureMeta();
      dateLabel.textContent = currentDate();
      const dateRecord = rows.find(row => row.record)?.record;
      periodBadge.textContent = dateRecord?.period === "future_forecast" ? "Future forecast" : "Observed fit";

      if (measureSelect.value === "share") {
        const stockTotal = rows.reduce((sum, row) => sum + (stockFor(row.record) || 0), 0);
        const marketTotal = rows.reduce((sum, row) => sum + (row.record?.vehicleMarket || 0), 0);
        document.getElementById("totalLabel").textContent = "County share";
        document.getElementById("totalValue").textContent = `${formatNumber(100 * stockTotal / marketTotal, 1)}%`;
      } else {
        const total = values.reduce((sum, row) => sum + row.value, 0);
        document.getElementById("totalLabel").textContent = "County total";
        document.getElementById("totalValue").textContent = `${formatNumber(total, meta.decimals)} ${meta.unit}`;
      }
      document.getElementById("topZip").textContent = top ? `${top.zip} (${formatMetric(top.value)})` : "n/a";
    }

    function updateLegend() {
      const meta = measureMeta();
      const legend = document.getElementById("legendContent");
      const rows = colors.map((color, index) => {
        const low = breaks[index];
        const high = breaks[index + 1];
        const label = index === 0
          ? `up to ${formatNumber(high, meta.decimals)}`
          : `${formatNumber(low, meta.decimals)} to ${formatNumber(high, meta.decimals)}`;
        return `<div class="legend-row"><span class="legend-swatch" style="background:${color}"></span><span>${label}</span></div>`;
      }).join("");
      legend.innerHTML = `<div class="legend-title">${meta.label} ${meta.unit === "%" ? "(%)" : ""}</div>${rows}`;
    }

    function refresh({ recomputeBreaks = false } = {}) {
      if (recomputeBreaks) computeBreaks();
      dateSlider.value = String(dateIndex);
      updateLayerStyles();
      updateStats();
      updateLegend();
    }

    function setDateIndex(nextIndex) {
      dateIndex = Math.max(0, Math.min(dates.length - 1, nextIndex));
      refresh();
    }

    function stopPlayback() {
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
        document.getElementById("playDates").innerHTML = "&#9654;";
        document.getElementById("playDates").setAttribute("aria-label", "Play dates");
      }
    }

    function togglePlayback() {
      if (playTimer) {
        stopPlayback();
        return;
      }
      document.getElementById("playDates").innerHTML = "&#10074;&#10074;";
      document.getElementById("playDates").setAttribute("aria-label", "Pause dates");
      playTimer = setInterval(() => {
        if (dateIndex >= dates.length - 1) {
          dateIndex = 0;
        } else {
          dateIndex += 1;
        }
        refresh();
      }, 650);
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
    zipLayer = L.geoJSON(APP_DATA.geojson, {
      style: featureStyle,
      onEachFeature: (feature, layer) => {
        const zip = feature.properties.zip;
        layer.bindPopup(popupHtml(zip));
        layer.bindTooltip(`<span class="zip-tooltip">ZIP ${zip}</span><br>${formatMetric(valueFor(recordFor(zip)))}`, { sticky: true });
        layer.on({
          mouseover: event => event.target.setStyle({ weight: 2.2, color: "#111827", fillOpacity: 0.9 }),
          mouseout: event => zipLayer.resetStyle(event.target),
          click: event => event.target.setPopupContent(popupHtml(zip))
        });
      }
    }).addTo(map);
    map.fitBounds(zipLayer.getBounds(), { padding: [24, 24] });
    updateStats();
    updateLegend();

    dateSlider.addEventListener("input", event => {
      stopPlayback();
      setDateIndex(Number(event.target.value));
    });
    scenarioSelect.addEventListener("change", () => refresh({ recomputeBreaks: true }));
    measureSelect.addEventListener("change", () => refresh({ recomputeBreaks: true }));
    observedToggle.addEventListener("change", () => refresh({ recomputeBreaks: true }));
    document.getElementById("prevDate").addEventListener("click", () => {
      stopPlayback();
      setDateIndex(dateIndex - 1);
    });
    document.getElementById("nextDate").addEventListener("click", () => {
      stopPlayback();
      setDateIndex(dateIndex + 1);
    });
    document.getElementById("playDates").addEventListener("click", togglePlayback);
  </script>
</body>
</html>
"""
    return template.replace("__APP_DATA__", data_json).replace("__MAP_TITLE__", map_title)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast",
        default="models/adoption_forecast_albany_zip_bass_central_hudson_covariates_snapshot.csv",
        help="ZIP-level forecast CSV.",
    )
    parser.add_argument(
        "--zcta-geojson",
        default="data/ny_new_york_zip_codes_geo.min.json",
        help="New York ZIP/ZCTA GeoJSON.",
    )
    parser.add_argument(
        "--output",
        default="visualization/albany_zip_adoption_forecast_map.html",
        help="Output standalone HTML map.",
    )
    parser.add_argument(
        "--title",
        default="Albany ZIP EV Adoption Forecast",
        help="Title shown in the standalone map.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    forecast_path = resolve(args.forecast)
    geojson_path = resolve(args.zcta_geojson)
    output_path = resolve(args.output)

    df = read_forecast(forecast_path)
    zips = set(df["zip"].unique())
    app_data = {
        "metadata": summarize(df, title=str(args.title), forecast_path=forecast_path, geojson_path=geojson_path),
        "zips": sorted(zips),
        "dates": sorted(df["date"].unique().tolist()),
        "geojson": read_filtered_geojson(geojson_path, zips),
        "forecastByDate": build_forecast_payload(df),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(app_data), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(f"Embedded {len(zips)} ZIPs across {len(app_data['dates'])} dates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
