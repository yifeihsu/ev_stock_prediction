#!/usr/bin/env python3
"""Plot observed vs pooled-vs-independent forecasts for one representative ZIP."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=str, default="11746", help="Representative ZIP to plot.")
    ap.add_argument(
        "--pooled-forecast",
        type=str,
        default="models/zip_mixed_effects_forecast_lipa_zip_mixed_eval8.csv",
        help="Forecast CSV produced by the pooled mixed-effects model.",
    )
    ap.add_argument(
        "--independent-forecast",
        type=str,
        default=None,
        help="Optional independent ZIP Bass forecast CSV. Defaults to models/bass_forecast_zip<ZIP>_mixcmp.csv",
    )
    ap.add_argument(
        "--holdout-start",
        type=str,
        default="2025-01-01",
        help="Holdout boundary shown in the figure.",
    )
    ap.add_argument(
        "--plot-start",
        type=str,
        default="2018-01-01",
        help="Left bound for displayed dates. Keeps the figure focused on the monthly era.",
    )
    ap.add_argument(
        "--smooth-train-window",
        type=int,
        default=1,
        help="Optional rolling window (months) used to smooth simulation-based training fits in the flow panel.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Defaults to models/zip_mixed_effects_rep_zip<ZIP>.png",
    )
    return ap


def resolve(path_str: str | None, *, zip_code: str) -> Path:
    if path_str is None:
        return ROOT / "models" / f"bass_forecast_zip{zip_code}_mixcmp.csv"
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = build_parser().parse_args()
    zip_code = str(args.zip).strip()
    pooled_path = resolve(args.pooled_forecast, zip_code=zip_code)
    indep_path = resolve(args.independent_forecast, zip_code=zip_code)
    output_path = (
        resolve(args.output, zip_code=zip_code)
        if args.output
        else ROOT / "models" / f"zip_mixed_effects_rep_zip{zip_code}.png"
    )

    pooled = pd.read_csv(pooled_path)
    pooled["zip"] = pooled["zip"].astype(str).str.zfill(5)
    pooled = pooled[pooled["zip"] == zip_code].copy()
    if pooled.empty:
        raise SystemExit(f"No pooled rows found for ZIP {zip_code} in {pooled_path}")
    pooled["date"] = pd.to_datetime(pooled["date"])

    indep = pd.read_csv(indep_path)
    indep["date"] = pd.to_datetime(indep["date"])

    holdout_ts = pd.to_datetime(args.holdout_start)
    plot_start = pd.to_datetime(args.plot_start)
    pooled_train = pooled[pooled["date"] < holdout_ts].copy()
    indep_train = indep[indep["date"] < holdout_ts].copy()
    pooled_plot = pooled[pooled["date"] >= plot_start].copy()
    indep_plot = indep[indep["date"] >= plot_start].copy()
    pooled_train_plot = pooled_train[pooled_train["date"] >= plot_start].copy()
    indep_train_plot = indep_train[indep_train["date"] >= plot_start].copy()

    smooth_window = max(int(args.smooth_train_window), 1)
    pooled_train_plot["flow_ev_t_hat_fit_smooth"] = pooled_train_plot["flow_ev_t_hat_fit"]
    indep_train_plot["flow_ev_t_hat_fit_smooth"] = indep_train_plot["flow_ev_t_hat_fit"]
    if smooth_window > 1:
        pooled_train_plot["flow_ev_t_hat_fit_smooth"] = (
            pooled_train_plot["flow_ev_t_hat_fit"].rolling(smooth_window, min_periods=1).mean()
        )
        indep_train_plot["flow_ev_t_hat_fit_smooth"] = (
            indep_train_plot["flow_ev_t_hat_fit"].rolling(smooth_window, min_periods=1).mean()
        )
    pooled_fit_label = (
        "Pooled mixed model: train fit (simulate)"
        if smooth_window == 1
        else f"Pooled mixed model: train fit ({smooth_window}m mean)"
    )
    indep_fit_label = (
        "Independent ZIP Bass: train fit (simulate)"
        if smooth_window == 1
        else f"Independent ZIP Bass: train fit ({smooth_window}m mean)"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    fig.patch.set_facecolor("white")
    obs_color = "#0b84f3"
    pooled_color = "#d95f02"
    indep_color = "#6a737d"
    holdout_mask = pooled_plot["date"] >= holdout_ts
    observed_stock = pooled_plot[pooled_plot["stock_ev_t"].notna()].copy()

    # Flow panel
    ax = axes[0]
    ax.plot(
        pooled_plot["date"],
        pooled_plot["flow_ev_t"],
        color=obs_color,
        linewidth=2.0,
        label="Observed flow",
    )
    ax.plot(
        pooled_train_plot["date"],
        pooled_train_plot["flow_ev_t_hat_fit_smooth"],
        color=pooled_color,
        linewidth=1.8,
        alpha=0.75,
        label=pooled_fit_label,
    )
    ax.plot(
        pooled_plot[holdout_mask]["date"],
        pooled_plot[holdout_mask]["flow_ev_t_hat_anchor"],
        color=pooled_color,
        linewidth=2.2,
        linestyle="--",
        label="Pooled mixed model: holdout forecast",
    )
    ax.plot(
        indep_train_plot["date"],
        indep_train_plot["flow_ev_t_hat_fit_smooth"],
        color=indep_color,
        linewidth=1.6,
        alpha=0.75,
        label=indep_fit_label,
    )
    ax.plot(
        indep_plot[indep_plot["date"] >= holdout_ts]["date"],
        indep_plot[indep_plot["date"] >= holdout_ts]["flow_ev_t_hat_anchor"],
        color=indep_color,
        linewidth=2.0,
        linestyle="--",
        label="Independent ZIP Bass: holdout forecast",
    )
    ax.axvline(holdout_ts, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title(f"ZIP {zip_code}: First-Seen EV Flow", loc="left", fontsize=13)
    ax.set_ylabel("EVs per snapshot")
    ax.grid(True, alpha=0.25)

    # Stock panel
    ax = axes[1]
    ax.plot(
        observed_stock["date"],
        observed_stock["stock_ev_t"],
        color=obs_color,
        linewidth=1.8,
        alpha=0.95,
        marker="o",
        markersize=4.2,
        label="Observed stock (snapshot months)",
        zorder=3,
    )
    ax.plot(
        pooled_plot[holdout_mask]["date"],
        pooled_plot[holdout_mask]["stock_ev_t_hat_anchor"],
        color=pooled_color,
        linewidth=2.6,
        linestyle="--",
        label="Pooled mixed model: holdout stock forecast",
    )
    ax.plot(
        indep_plot[indep_plot["date"] >= holdout_ts]["date"],
        indep_plot[indep_plot["date"] >= holdout_ts]["stock_ev_t_hat_anchor"],
        color=indep_color,
        linewidth=2.2,
        linestyle="--",
        label="Independent ZIP Bass: holdout stock forecast",
    )
    ax.axvline(holdout_ts, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title(f"ZIP {zip_code}: On-Road EV Stock", loc="left", fontsize=13)
    ax.set_ylabel("Unique VINs")
    ax.set_xlabel("Snapshot date")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.01,
        0.96,
        "Stock is reconstructed from cumulative flow + retention.\nTraining stock fits are omitted to avoid conflating fit flow with holdout stock forecasts.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        color="#465261",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d9dee5", "alpha": 0.92},
    )
    inset = ax.inset_axes([0.60, 0.10, 0.36, 0.34])
    zoom_start = holdout_ts - pd.DateOffset(months=6)
    zoom_mask = observed_stock["date"] >= zoom_start
    inset.plot(
        observed_stock.loc[zoom_mask, "date"],
        observed_stock.loc[zoom_mask, "stock_ev_t"],
        color=obs_color,
        linewidth=1.5,
        marker="o",
        markersize=3.3,
    )
    inset.plot(
        pooled_plot[holdout_mask]["date"],
        pooled_plot[holdout_mask]["stock_ev_t_hat_anchor"],
        color=pooled_color,
        linewidth=2.0,
        linestyle="--",
    )
    inset.plot(
        indep_plot[indep_plot["date"] >= holdout_ts]["date"],
        indep_plot[indep_plot["date"] >= holdout_ts]["stock_ev_t_hat_anchor"],
        color=indep_color,
        linewidth=1.8,
        linestyle="--",
    )
    inset.axvline(holdout_ts, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    inset.set_title("Holdout zoom", fontsize=8.8)
    inset.grid(True, alpha=0.18)
    inset.tick_params(axis="both", labelsize=7)
    inset.set_xlim(zoom_start, pooled_plot["date"].max() + pd.DateOffset(months=1))

    axes[0].legend(
        loc="upper left",
        ncol=2,
        fontsize=9.0,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d9dee5",
    )
    axes[1].legend(
        loc="upper right",
        fontsize=9.5,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d9dee5",
    )
    fig.suptitle(
        f"Representative ZIP Forecast Comparison: {zip_code}",
        fontsize=15.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.955,
        "Displayed window starts at 2018-01-01. Training flow lines are simulation-based fits; dashed lines are recursive holdout forecasts.",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#465261",
    )

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
