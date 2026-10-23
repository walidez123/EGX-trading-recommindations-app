"""
Institutional Candlestick & Volume-Profile Chart Generator (`mplfinance` + `matplotlib`).
Supports:
- Multi-Timeframe Candles: Daily ('1D'), Weekly ('1W'), and Monthly ('1M')
- Both EGX Stocks and 3-Year EGX Market Indices (EGX 30, EGX 70 EWI, EGX 33 Shariah)
- Dynamic Target Price (+X.X%), Suggested Target Timeframe/Date, and Dynamic Stop-Loss (-Y.Y%)
- 50-MA, 200-MA, Bollinger Bands, Volume Point of Control (VPOC), Fibonacci 61.8%, Volume & RSI(14)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from backend.config import CHARTS_DIR, TIMEFRAME_LABELS, is_egx_index, normalize_egx_ticker, normalize_timeframe


def generate_vision_chart(
    tech_data: Dict[str, Any],
    bars_to_plot: Optional[int] = None,
) -> Tuple[Path, str]:
    """
    Renders a multi-panel candlestick chart PNG for any EGX Stock or Index (`EGX30`, `EGX70`, `EGX33`)
    on the selected candle timeframe (`1D`, `1W`, or `1M`).
    Returns (absolute_png_path, relative_web_url).
    """
    ticker = normalize_egx_ticker(tech_data["ticker"])
    tf = normalize_timeframe(tech_data.get("candle_timeframe", "1D"))
    index_flag = is_egx_index(ticker) or tech_data.get("asset_type") == "INDEX"

    safe_name = ticker.replace(".", "_").replace(":", "_")
    suffix = f"_{tf.lower()}" if tf != "1D" else ""
    out_path = CHARTS_DIR / f"{safe_name}{suffix}_swing_chart.png"
    web_url = f"/static/charts/{safe_name}{suffix}_swing_chart.png"

    if bars_to_plot is None:
        if index_flag:
            bars_to_plot = 300 if tf == "1D" else (160 if tf == "1W" else 60)
        else:
            bars_to_plot = 130 if tf == "1D" else (110 if tf == "1W" else 60)

    df: pd.DataFrame = tech_data["_raw_df"].tail(bars_to_plot).copy()
    ind = tech_data["indicators"]
    geom = tech_data["strategy_geometry"]
    fib = tech_data.get("fibonacci_6m") or tech_data.get("fibonacci") or {}

    entry_price = float(geom["entry_price"])
    target_price = float(geom.get("target_price") or geom.get("target_price_30pct") or entry_price * 1.15)
    target_upside_pct = float(geom.get("target_upside_pct") or geom.get("take_profit_pct") or 15.0)
    target_tf_label = geom.get("target_timeframe_label") or f"{geom.get('target_days', 60)}d"
    stop_price = float(geom.get("stop_loss_price") or geom.get("initial_stop_loss_2_5x_atr") or entry_price * 0.92)
    stop_pct = float(geom.get("stop_distance_pct") or 8.0)
    vpoc_price = float(ind.get("vpoc_6m") or entry_price)
    fib_618 = float(fib.get("fib_61_8_golden") or entry_price)
    unit_lbl = "pts" if index_flag else "EGP"

    # Prepare additional plots
    apds = []
    if "SMA50" in df.columns and not df["SMA50"].dropna().empty:
        apds.append(mpf.make_addplot(df["SMA50"], color="#38bdf8", width=1.4, panel=0, secondary_y=False, label="MA 50"))
    if "SMA200" in df.columns and not df["SMA200"].dropna().empty:
        apds.append(mpf.make_addplot(df["SMA200"], color="#f59e0b", width=1.6, panel=0, secondary_y=False, label="MA 200"))
    if "BB_UPPER" in df.columns and not df["BB_UPPER"].dropna().empty:
        apds.append(mpf.make_addplot(df["BB_UPPER"], color="#a855f7", width=0.8, linestyle="--", panel=0, secondary_y=False))
        apds.append(mpf.make_addplot(df["BB_LOWER"], color="#a855f7", width=0.8, linestyle="--", panel=0, secondary_y=False))

    # RSI(14) in panel 2
    if "RSI14" in df.columns:
        apds.append(
            mpf.make_addplot(
                df["RSI14"],
                panel=2,
                color="#c084fc",
                width=1.3,
                ylabel=f"RSI(14) [{tf}]",
                ylim=(15, 85),
                secondary_y=False,
            )
        )
        rsi_70 = pd.Series(70.0, index=df.index)
        rsi_30 = pd.Series(30.0, index=df.index)
        apds.append(mpf.make_addplot(rsi_70, panel=2, color="#ef4444", width=0.7, linestyle=":", secondary_y=False))
        apds.append(mpf.make_addplot(rsi_30, panel=2, color="#10b981", width=0.7, linestyle=":", secondary_y=False))

    # Custom institutional dark style
    mc = mpf.make_marketcolors(
        up="#10b981",
        down="#ef4444",
        edge="inherit",
        wick="inherit",
        volume={"up": "#10b98188", "down": "#ef444488"},
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#0b0f19",
        figcolor="#0b0f19",
        gridcolor="#1e293b",
        gridstyle="--",
        rc={
            "font.size": 9,
            "axes.labelcolor": "#94a3b8",
            "axes.edgecolor": "#334155",
            "xtick.color": "#94a3b8",
            "ytick.color": "#94a3b8",
        },
    )

    hlines_list = [target_price, entry_price, vpoc_price, fib_618, stop_price]
    hlines_colors = ["#10b981", "#38bdf8", "#eab308", "#a855f7", "#f43f5e"]
    hlines_styles = ["--", "-", ":", "-.", "--"]

    has_volume = float(df["Volume"].sum()) > 0
    panel_ratios = (5, 1.4, 1.6) if has_volume else (5, 1.6)

    fig, axes = mpf.plot(
        df,
        type="candle",
        volume=has_volume,
        addplot=apds,
        style=style,
        figsize=(13, 8),
        panel_ratios=panel_ratios,
        hlines=dict(
            hlines=hlines_list,
            colors=hlines_colors,
            linestyle=hlines_styles,
            linewidths=[1.5, 1.2, 1.1, 1.0, 1.5],
        ),
        returnfig=True,
        tight_layout=True,
        ylim=(
            min(float(df["Low"].min()) * 0.96, stop_price * 0.96),
            max(float(df["High"].max()) * 1.04, target_price * 1.03),
        ),
    )

    ax_price = axes[0]
    x_len = len(df)
    x_start = int(x_len * 0.72)
    x_range = np.arange(x_start, x_len)
    ax_price.fill_between(
        x_range,
        entry_price,
        target_price,
        color="#10b981",
        alpha=0.10,
    )
    ax_price.fill_between(
        x_range,
        stop_price,
        entry_price,
        color="#f43f5e",
        alpha=0.12,
    )

    annotations = [
        (target_price, f" Target (+{target_upside_pct:.1f}% in {target_tf_label}): {target_price:,.2f} {unit_lbl}", "#10b981"),
        (entry_price, f" Current/Entry: {entry_price:,.2f} {unit_lbl}", "#38bdf8"),
        (vpoc_price, f" VPOC: {vpoc_price:,.2f} {unit_lbl}", "#eab308"),
        (fib_618, f" Fib 61.8%: {fib_618:,.2f} {unit_lbl}", "#c084fc"),
        (stop_price, f" Stop-Loss (-{stop_pct:.1f}%): {stop_price:,.2f} {unit_lbl}", "#f43f5e"),
    ]
    for y_val, label_txt, col in annotations:
        ax_price.text(
            x_len - 1,
            y_val,
            label_txt,
            color=col,
            fontsize=8.5,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor=col, alpha=0.85, lw=0.8),
        )

    display_name = tech_data.get("company_name") or tech_data.get("full_name") or tech_data.get("name") or ticker
    title_str = (
        f"{ticker} ({display_name}) [{TIMEFRAME_LABELS[tf]}] | "
        f"Level: {entry_price:,.2f} {unit_lbl} | Target (+{target_upside_pct:.1f}% / {target_tf_label}): {target_price:,.2f} | "
        f"Stop: {stop_price:,.2f} (R:R {geom['reward_to_risk']}:1)"
    )
    ax_price.set_title(title_str, color="#f8fafc", fontsize=10.0, fontweight="bold", pad=12)

    fig.savefig(str(out_path), dpi=140, bbox_inches="tight", facecolor="#0b0f19")
    plt.close(fig)
    return out_path, web_url
