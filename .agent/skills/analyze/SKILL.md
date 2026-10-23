---
name: analyze
description: >-
  Use this skill when the user invokes "analyze" (/analyze) with a stock ticker (e.g. COMI, SWDY)
  or an EGX index (EGX30, EGX70, EGX33) and an optional candle timeframe (1D Daily, 1W Weekly, 1M Monthly).
  Automatically runs the Python technical/index & chart generation model to produce the candlestick graph,
  inspects the chart with Vision, saves the recommendation by date/stock/source into SQLite, and provides a clear
  recommendation with exact Entry Price, Dynamic Target Price, Suggested Target Timeframe/Date, and Stop-Loss.
---

# `analyze` — 1-Command Stock & 3-Year EGX Index Analysis (`1D` / `1W` / `1M` Candles)

When the user asks to `analyze <TICKER_OR_INDEX>` (e.g., `/analyze COMI`, `/analyze EGX30`, `/analyze EGX70 1W`, `/analyze EGX33 1M`):

## Step 1: Run the Python Analysis & Graph Generator Model
Execute [run_analyze.py](./scripts/run_analyze.py) via `run_command` passing `<TICKER_OR_INDEX>` and optional candle timeframe (`1D`, `1W`, or `1M`, default `1D`). The Python engine fetches the latest EGX market data (including 3+ years of history for `EGX30`, `EGX70`, `EGX33`), computes the dynamic Target Price, Suggested Target Timeframe/Date, and Stop-Loss, generates the chart image in `static/charts/`, and saves the recommendation into `data/egx_agent.db` (`recommendations` & `theses` tables):

```powershell
python ".agent/skills/analyze/scripts/run_analyze.py" "<TICKER_OR_INDEX>" "<TIMEFRAME_1D_1W_OR_1M>"
```

## Step 2: Inspect the Generated Graph Image
Call `read_file` on the generated chart PNG path printed by the script (e.g., `static/charts/<TICKER>_swing_chart.png`) so you visually verify the trend, candle structure, and price levels.

## Step 3: Present a Clear, Actionable Recommendation
Explain the result in plain, easy-to-understand language:
1. **Verdict & Candle Timeframe:** Clear recommendation (`🟢 BUY`, `🟡 WAIT FOR DIP / ACCUMULATE`, or `🔴 AVOID / DEFENSIVE`) and candle timeframe (`Daily 1D`, `Weekly 1W`, or `Monthly 1M`).
2. **The Key Action Numbers:**
   - **Current / Entry Level:** `X.XX EGP` (or `pts` for indices)
   - **Dynamic Profit Target:** `Y.YY EGP (+A.A%)` *(Data-driven target based on structural resistance & volatility)*
   - **Suggested Target Timeframe & Date:** `<TARGET_TIMEFRAME_LABEL>` *(Expected completion by `<TARGET_DATE>`)*
   - **Dynamic Safety Stop-Loss:** `Z.ZZ EGP (-B.B%)` *(Exit if price closes below this level to protect capital)*
   - **Reward-to-Risk Ratio:** `<R:R>x`
   - **Suggested Position Size (for stocks):** `<SHARES>` shares (`<AMOUNT> EGP`, safely within the 20% max rule)
   - **3-Year Index Snapshot (if `EGX30`, `EGX70`, or `EGX33`):** Include `1M`, `6M`, `1Y`, and `3Y` Return %, `3Y CAGR %`, and `3Y Max Drawdown %`.
3. **Why (in 3 Simple Bullet Points):** Explain the trend structure, momentum/volume, and why the dynamic target and stop-loss make sense. Confirm that the recommendation has been saved to the **Recommendations Tracker** for automatic future evaluation.
