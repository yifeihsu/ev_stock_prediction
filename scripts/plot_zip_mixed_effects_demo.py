#!/usr/bin/env python3
"""Create a demonstration figure for pooled vs independent ZIP forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--comparison-csv",
        type=str,
        default="models/zip_mixed_effects_comparison_lipa_zip_mixed_eval8.csv",
        help="CSV produced from pooled-vs-independent ZIP comparison.",
    )
    ap.add_argument(
        "--summary-json",
        type=str,
        default="models/zip_mixed_effects_comparison_lipa_zip_mixed_eval8.json",
        help="JSON summary produced from pooled-vs-independent ZIP comparison.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="models/zip_mixed_effects_demo_lipa_zip_mixed_eval8.png",
        help="Output PNG path.",
    )
    return ap


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def plot_panel(ax, df: pd.DataFrame, pooled_col: str, indep_col: str, title: str, xlabel: str) -> None:
    order = df.sort_values(pooled_col, ascending=True).reset_index(drop=True)
    y = np.arange(len(order))
    ax.hlines(y, order[indep_col], order[pooled_col], color="#c9ced6", linewidth=2.0, zorder=1)
    ax.scatter(order[indep_col], y, color="#8a94a6", label="Independent ZIP Bass", s=48, zorder=3)
    ax.scatter(order[pooled_col], y, color="#0b84f3", label="Pooled ZIP mixed model", s=56, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(order["zip"])
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)


def main() -> None:
    args = build_parser().parse_args()
    comp_path = _resolve(args.comparison_csv)
    summary_path = _resolve(args.summary_json)
    out_path = _resolve(args.output)

    df = pd.read_csv(comp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    pooled = summary["pooled_overall"]
    indep = summary["independent_weighted_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharey=False)
    fig.patch.set_facecolor("white")

    plot_panel(
        axes[0],
        df,
        pooled_col="pooled_rmse_flow",
        indep_col="indep_rmse_flow",
        title="Holdout Flow RMSE by ZIP",
        xlabel="RMSE on first-seen EV flow",
    )
    plot_panel(
        axes[1],
        df,
        pooled_col="pooled_rmse_stock",
        indep_col="indep_rmse_stock",
        title="Holdout Stock RMSE by ZIP",
        xlabel="RMSE on on-road EV stock",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("Pooled ZIP Mixed-Effects Model vs Independent ZIP Bass", fontsize=17, y=0.995)
    fig.text(
        0.5,
        0.92,
        (
            f"8 LIPA ZIPs, holdout starting 2025-01-01 | "
            f"Flow RMSE improved in {summary['flow_rmse_improved_zip_count']}/8 ZIPs | "
            f"Stock RMSE improved in {summary['stock_rmse_improved_zip_count']}/8 ZIPs"
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="#384454",
    )

    fig.text(
        0.5,
        0.04,
        (
            f"Overall pooled metrics: flow RMSE {pooled['rmse_flow']:.2f}, stock RMSE {pooled['rmse_stock']:.2f}, "
            f"flow Poisson NLL {pooled['poisson_nll_flow']:.2f}. "
            f"Independent weighted means: flow RMSE {indep['rmse_flow']:.2f}, "
            f"stock RMSE {indep['rmse_stock']:.2f}, flow Poisson NLL {indep['poisson_nll_flow']:.2f}."
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#4a5565",
    )

    plt.tight_layout(rect=[0.03, 0.08, 0.97, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
