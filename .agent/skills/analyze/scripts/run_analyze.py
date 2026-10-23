"""
Helper script for the `analyze` skill.
Supports both EGX Stocks (e.g. COMI.CA, SWDY.CA) and 3-Year EGX Indices (EGX30, EGX70, EGX33)
across Daily ('1D'), Weekly ('1W'), and Monthly ('1M') candles.
Computes Dynamic Target Price, Suggested Target Timeframe/Date, and Dynamic Stop-Loss,
generates the mplfinance chart PNG, and saves the recommendation by date/stock/source in `data/egx_agent.db`.
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))

from backend.ai_agent import run_hybrid_stock_analysis
from backend.config import normalize_timeframe
from backend.database import init_db


async def _run(ticker: str, timeframe: str = "1D") -> None:
    init_db()
    tf = normalize_timeframe(timeframe)
    res = await run_hybrid_stock_analysis(
        ticker=ticker,
        user_query=f"Run {tf} candle analysis on {ticker} with dynamic target price, target time, and stop-loss",
        auto_save_thesis=True,
        timeframe=tf,
        source_override=f"Antigravity /analyze Skill ({tf})",
    )
    tech = res["technical"]
    geom = tech["strategy_geometry"]
    ind = tech["indicators"]
    syn = res["synthesis"]
    rec = res.get("saved_recommendation") or {}
    idx_3y = res.get("index_3y")

    trend_simple = (
        f"UPTREND on {geom['candle_timeframe_label']} (Price above 50-MA & 200-MA)"
        if tech["price"]["current"] >= ind["sma50"] and ind["sma50"] >= ind["sma200"]
        else (
            f"PULLBACK IN BULL TREND on {geom['candle_timeframe_label']} (Above 200-MA)"
            if tech["price"]["current"] >= ind["sma200"]
            else f"BELOW 200-MA on {geom['candle_timeframe_label']} (Defensive / Wait)"
        )
    )

    rsi = ind["rsi_14_daily"]
    buyer_strength_simple = (
        f"OVERSOLD / VALUE ZONE ({tf} RSI {rsi})"
        if rsi < 38
        else (
            f"OVERBOUGHT / STRETCHED ({tf} RSI {rsi})"
            if rsi > 72
            else f"CONSTRUCTIVE MOMENTUM ({tf} RSI {rsi})"
        )
    )

    summary = {
        "ticker": res["ticker"],
        "company_name": res["company_name"],
        "asset_type": res["asset_type"],
        "sector": res["sector"],
        "candle_timeframe": tf,
        "candle_timeframe_label": geom["candle_timeframe_label"],
        "chart_image_path": res["chart_image_path"],
        "chart_image_url": res["chart_image_url"],
        "thesis_id": res["saved_thesis"]["thesis_id"] if res.get("saved_thesis") else "",
        "recommendation_id": rec.get("recommendation_id", ""),
        "saved_by_source": rec.get("source", f"Antigravity /analyze Skill ({tf})"),
        "saved_on_date": rec.get("created_date", ""),
        "simple_recommendation": syn["action_recommendation"],
        "confidence_score_out_of_100": syn["conviction_score"],
        "dynamic_action_plan": {
            "entry_price": geom["entry_price"],
            "dynamic_target_price": geom["target_price"],
            "target_upside_pct": f"+{geom['target_upside_pct']}%",
            "suggested_target_timeframe": geom["target_timeframe_label"],
            "suggested_target_date": geom["target_date"],
            "dynamic_stop_loss": geom["stop_loss_price"],
            "risk_if_stopped_out_pct": f"-{geom['stop_distance_pct']}%",
            "reward_to_risk_ratio": f"{geom['reward_to_risk']}:1",
            "suggested_shares_to_buy": geom["recommended_shares"],
            "suggested_investment_egp": geom["recommended_position_egp"],
            "portfolio_share_pct": f"{geom['recommended_allocation_pct']}% (Max allowed: 20%)",
        },
        "plain_english_signals": {
            "trend_direction": trend_simple,
            "buyer_strength": buyer_strength_simple,
            "volume_activity": f"{tech['price']['rvol']}x normal {tf} trading volume",
        },
    }

    if idx_3y:
        summary["index_3_year_performance"] = idx_3y["performance_3y"]
        summary["index_regime"] = idx_3y["regime"]

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_analyze.py <TICKER_OR_INDEX> [1D|1W|1M]")
        sys.exit(1)
    tf_arg = sys.argv[2] if len(sys.argv) >= 3 else "1D"
    asyncio.run(_run(sys.argv[1], tf_arg))
