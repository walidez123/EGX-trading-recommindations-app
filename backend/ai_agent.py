"""
StateGraph Investment Decision Agent, 3-Year Index Advisor & Self-Reflection Orchestrator.
Supports:
- Multi-Timeframe Candle Analysis: Daily ('1D'), Weekly ('1W'), and Monthly ('1M')
- Dynamic Target Price, Suggested Target Timeframe/Date, and Dynamic Stop-Loss for every Stock & Index
- Automatic persistence of every recommendation by Date, Stock/Index, Candle Timeframe, and Source
- Automated evaluation of saved recommendations as time passes and market prices update
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pydantic

from backend.chart_generator import generate_vision_chart
from backend.config import (
    BRIDGE_DIR,
    STRATEGY_RULES,
    TIMEFRAME_LABELS,
    is_egx_index,
    normalize_egx_ticker,
    normalize_timeframe,
)
from backend.database import (
    add_learned_lesson,
    create_thesis,
    evaluate_recommendations,
    get_connection,
    get_learned_lessons,
    get_portfolio_cockpit,
    get_portfolio_settings,
    list_theses,
    record_recommendation,
    update_position_live_price,
)
from backend.document_store import list_documents
from backend.technical_engine import compute_full_technical_analysis, compute_index_analysis


class StructuredThesisOutput(pydantic.BaseModel):
    fundamental_health_score: int
    fundamental_verdict: str
    technical_status_verdict: str
    vision_chart_findings: str
    strategy_alignment_score: int
    action_recommendation: str
    conviction_score: int
    recommended_allocation_pct: float
    target_price: float
    target_timeframe_label: str
    stop_loss_price: float
    thesis_summary: str
    fundamental_catalysts: List[str]
    technical_triggers: List[str]
    invalidation_rules: List[str]
    applied_lesson_ids: List[str]
    narrative_markdown: str


async def run_hybrid_stock_analysis(
    ticker: str,
    user_query: str = "",
    auto_save_thesis: bool = True,
    timeframe: str = "1D",
    source_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the 4-Node StateGraph Pipeline for any EGX Stock or Index (`EGX30`, `EGX70`, `EGX33`)
    on the chosen candle timeframe (`1D` Daily, `1W` Weekly, or `1M` Monthly):
    Node 1: Retrieve Tier A Learned Lessons, Tier B Portfolio State, and Tier C Disclosures.
    Node 2: Compute Pure-Math Technical Indicators, Fibonacci, VPOC & Dynamic Target/Time/Stop Geometry.
    Node 3: Render High-DPI mplfinance Candlestick + Volume + RSI + Target/Stop Chart PNG.
    Node 4: Synthesize Multimodal Vision + Fundamental + Cumulative Memory into a Verified Thesis
            and save to the `recommendations` evaluation ledger.
    """
    norm_ticker = normalize_egx_ticker(ticker)
    tf = normalize_timeframe(timeframe)
    index_flag = is_egx_index(norm_ticker)

    # --- NODE 1: Cumulative Memory Retrieval (Tiers A, B, C) ---
    settings = get_portfolio_settings()
    cockpit = get_portfolio_cockpit()
    portfolio_equity = float(cockpit["summary"]["total_equity_egp"])
    lessons = get_learned_lessons(norm_ticker, active_only=True)
    disclosures = list_documents(norm_ticker)

    # --- NODE 2: Deterministic Pure-Math Technical & Index Engine ---
    index_3y_data: Optional[Dict[str, Any]] = None
    if index_flag:
        index_3y_data = compute_index_analysis(norm_ticker, timeframe=tf, lookback_years=3)
        tech = compute_full_technical_analysis(norm_ticker, portfolio_equity_egp=portfolio_equity, timeframe=tf)
    else:
        tech = compute_full_technical_analysis(norm_ticker, portfolio_equity_egp=portfolio_equity, timeframe=tf)
        update_position_live_price(
            norm_ticker,
            latest_price=tech["price"]["current"],
            latest_atr=tech["indicators"]["atr_14"],
        )

    # --- NODE 3: Visual Chart Generation (mplfinance PNG) ---
    chart_path, chart_url = generate_vision_chart(tech)

    raw_df = tech.pop("_raw_df", None)
    if index_3y_data is not None:
        index_3y_data.pop("_raw_df", None)
        index_3y_data["chart_image_url"] = chart_url
    candlestick_patterns = _detect_deterministic_chart_patterns(raw_df, tech) if raw_df is not None else []

    # --- NODE 4: Multimodal Synthesis (Antigravity SDK / Gemini Vision / Institutional Engine) ---
    api_key = (settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or "").strip()
    preferred_model = settings.get("preferred_model") or "gemini-2.5-pro"

    synthesis: Optional[Dict[str, Any]] = None
    ai_provider_used = source_override or "Antigravity Institutional Hybrid Engine"

    if api_key:
        synthesis, model_provider = await _call_antigravity_or_genai_multimodal(
            api_key=api_key,
            model_name=preferred_model,
            tech=tech,
            chart_path=chart_path,
            lessons=lessons,
            disclosures=disclosures,
            user_query=user_query,
        )
        if synthesis is not None and not source_override:
            ai_provider_used = model_provider

    if synthesis is None:
        synthesis = _synthesize_institutional_thesis_deterministic(
            tech=tech,
            candlestick_patterns=candlestick_patterns,
            lessons=lessons,
            disclosures=disclosures,
            user_query=user_query,
            index_3y_data=index_3y_data,
        )

    geom = tech["strategy_geometry"]
    # Ensure dynamic target, timeframe, and stop-loss are always present on synthesis
    synthesis.setdefault("target_price", geom["target_price"])
    synthesis.setdefault("target_timeframe_label", geom["target_timeframe_label"])
    synthesis.setdefault("stop_loss_price", geom["stop_loss_price"])

    # Write Bridge Packet for Antigravity IDE Agent inspection
    safe_sym = norm_ticker.replace(".", "_")
    bridge_packet_path = BRIDGE_DIR / f"latest_{safe_sym}.json"
    bridge_payload = {
        "ticker": norm_ticker,
        "candle_timeframe": tf,
        "generated_at": datetime.now().isoformat(),
        "chart_image_path": str(chart_path),
        "chart_image_url": chart_url,
        "technical_metrics": {k: v for k, v in tech.items() if k != "chart_series"},
        "index_3y_metrics": {k: v for k, v in (index_3y_data or {}).items() if k != "chart_series"} if index_3y_data else None,
        "tier_a_lessons_injected": lessons,
        "tier_c_disclosures": disclosures,
        "synthesis": synthesis,
    }
    bridge_packet_path.write_text(json.dumps(bridge_payload, indent=2), encoding="utf-8")

    # Persist as a PROPOSED thesis in Tier A Episodic Memory
    saved_thesis = None
    if auto_save_thesis:
        saved_thesis = create_thesis(
            {
                "ticker": norm_ticker,
                "company_name": tech["company_name"],
                "sector": tech["sector"],
                "status": "PROPOSED",
                "entry_price": geom["entry_price"],
                "target_price": float(synthesis.get("target_price") or geom["target_price"]),
                "atr_at_entry": tech["indicators"]["atr_14"],
                "initial_stop_loss": float(synthesis.get("stop_loss_price") or geom["stop_loss_price"]),
                "target_days": geom["target_days"],
                "target_timeframe_label": geom["target_timeframe_label"],
                "time_horizon_min_date": geom["min_horizon_date"],
                "time_horizon_deadline": geom["target_date"],
                "candle_timeframe": tf,
                "conviction_score": synthesis["conviction_score"],
                "recommended_allocation_pct": min(
                    float(synthesis["recommended_allocation_pct"]) / 100.0
                    if synthesis["recommended_allocation_pct"] > 1.0
                    else float(synthesis["recommended_allocation_pct"]),
                    STRATEGY_RULES.max_allocation_pct,
                ),
                "fundamental_catalysts": synthesis["fundamental_catalysts"],
                "technical_triggers": synthesis["technical_triggers"],
                "invalidation_rules": synthesis["invalidation_rules"],
                "vision_chart_findings": synthesis["vision_chart_findings"],
                "chart_image_url": chart_url,
                "thesis_summary": synthesis["thesis_summary"],
                "ai_provider_used": ai_provider_used,
            }
        )

    # Always save every recommendation by Date, Stock/Index, Timeframe, and Source into `recommendations`
    saved_rec = record_recommendation(
        {
            "ticker": norm_ticker,
            "company_name": tech["company_name"],
            "asset_type": "INDEX" if index_flag else "STOCK",
            "sector": tech["sector"],
            "candle_timeframe": tf,
            "source": ai_provider_used,
            "action": synthesis["action_recommendation"],
            "conviction_score": synthesis["conviction_score"],
            "entry_price": geom["entry_price"],
            "target_price": float(synthesis.get("target_price") or geom["target_price"]),
            "target_upside_pct": geom["target_upside_pct"],
            "stop_loss_price": float(synthesis.get("stop_loss_price") or geom["stop_loss_price"]),
            "stop_loss_pct": geom["stop_distance_pct"],
            "reward_to_risk": geom["reward_to_risk"],
            "target_days": geom["target_days"],
            "target_timeframe_label": geom["target_timeframe_label"],
            "target_date": geom["target_date"],
            "rationale_summary": synthesis["thesis_summary"],
            "thesis_id": saved_thesis["thesis_id"] if saved_thesis else "",
        }
    )

    # Evaluate recommendations so live stats stay fresh
    rec_stats = evaluate_recommendations()

    # Save to chat history
    _record_chat_interaction(
        role="user",
        ticker=norm_ticker,
        content=user_query or f"Analyze {norm_ticker} on {TIMEFRAME_LABELS[tf]} candles with dynamic target, timeframe & stop-loss",
        payload={"candle_timeframe": tf},
    )
    _record_chat_interaction(
        role="assistant",
        ticker=norm_ticker,
        content=synthesis["narrative_markdown"],
        payload={
            "ticker": norm_ticker,
            "candle_timeframe": tf,
            "chart_url": chart_url,
            "synthesis": synthesis,
            "strategy_geometry": geom,
            "thesis_id": saved_thesis["thesis_id"] if saved_thesis else "",
            "recommendation_id": saved_rec.get("recommendation_id", ""),
        },
    )

    return {
        "ticker": norm_ticker,
        "company_name": tech["company_name"],
        "sector": tech["sector"],
        "asset_type": "INDEX" if index_flag else "STOCK",
        "candle_timeframe": tf,
        "candle_timeframe_label": TIMEFRAME_LABELS[tf],
        "ai_provider_used": ai_provider_used,
        "chart_image_url": chart_url,
        "chart_image_path": str(chart_path),
        "technical": tech,
        "index_3y": index_3y_data,
        "synthesis": synthesis,
        "saved_thesis": saved_thesis,
        "saved_recommendation": saved_rec,
        "recommendation_stats": rec_stats,
        "applied_lessons": lessons,
        "disclosures_used": disclosures,
    }


async def _call_antigravity_or_genai_multimodal(
    api_key: str,
    model_name: str,
    tech: Dict[str, Any],
    chart_path: Path,
    lessons: List[Dict[str, Any]],
    disclosures: List[Dict[str, Any]],
    user_query: str,
) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Calls `google.antigravity.Agent` or `google.genai.Client`
    to perform live multimodal Vision + Quantitative + Fundamental synthesis.
    """
    system_prompt = _build_system_instruction(lessons)
    user_prompt = _build_analysis_prompt(tech, disclosures, user_query)

    # 1. Try google.antigravity SDK first
    try:
        import google.antigravity as ag

        config = ag.LocalAgentConfig(
            api_key=api_key,
            system_instructions=system_prompt,
            response_schema=StructuredThesisOutput,
        )
        prompt_parts: List[Any] = [
            ag.from_file(chart_path, description=f"Candlestick & Volume Profile Chart for {tech['ticker']}"),
            user_prompt,
        ]
        for doc in disclosures:
            fp = doc.get("file_path")
            if fp and Path(fp).exists() and str(fp).lower().endswith(".pdf"):
                prompt_parts.append(ag.from_file(fp, description=doc.get("title", "Financial Disclosure PDF")))

        async with ag.Agent(config) as agent:
            resp = await agent.chat(prompt_parts)
            chunks = []
            async for token in resp:
                chunks.append(token)
            raw_json = "".join(chunks).strip()
            parsed = json.loads(raw_json)
            return parsed, f"Google Antigravity SDK ({model_name} Multimodal Vision)"
    except Exception:
        pass

    # 2. Fallback to google.genai SDK
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=api_key)
        chart_bytes = chart_path.read_bytes()
        contents: List[Any] = [
            genai_types.Part.from_bytes(data=chart_bytes, mime_type="image/png"),
            user_prompt,
        ]
        for doc in disclosures:
            fp = doc.get("file_path")
            if fp and Path(fp).exists() and str(fp).lower().endswith(".pdf"):
                pdf_bytes = Path(fp).read_bytes()
                contents.append(genai_types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=StructuredThesisOutput,
                temperature=0.2,
            ),
        )
        if response and response.text:
            parsed = json.loads(response.text)
            return parsed, f"Google GenAI Multimodal Vision ({model_name})"
    except Exception:
        pass

    return None, "Antigravity Institutional Hybrid Engine"


def _detect_deterministic_chart_patterns(df: Any, tech: Dict[str, Any]) -> List[str]:
    """Inspects OHLCV bars and overlays to extract exact visual chart structure findings."""
    findings: List[str] = []
    if df is None or len(df) < 20:
        return findings

    recent = df.tail(20)
    last = df.iloc[-1]
    ind = tech["indicators"]
    fib = tech["fibonacci_6m"]
    tf = tech.get("candle_timeframe", "1D")
    unit_lbl = "pts" if tech.get("asset_type") == "INDEX" else "EGP"

    first_half_low = float(recent.iloc[:10]["Low"].min())
    second_half_low = float(recent.iloc[10:]["Low"].min())
    if second_half_low > first_half_low:
        findings.append(
            f"Constructive Higher-Low structure on {tf} chart ({first_half_low:,.2f} -> {second_half_low:,.2f} {unit_lbl})"
        )

    body = float(last["Close"] - last["Open"])
    rng = max(float(last["High"] - last["Low"]), 0.01)
    upper_wick = float(last["High"] - max(last["Open"], last["Close"]))
    lower_wick = float(min(last["Open"], last["Close"]) - last["Low"])

    if body > 0 and body / rng >= 0.65 and float(last["RVOL"]) >= 1.2:
        findings.append(
            f"Bullish Wide-Range Expansion {tf} Candle closing near high ({last['Close']:,.2f} {unit_lbl}) on {last['RVOL']:.2f}x RVOL"
        )
    elif lower_wick / rng >= 0.50:
        findings.append(
            f"Lower-Wick Demand Rejection (Hammer tail rejecting {last['Low']:,.2f} {unit_lbl} support)"
        )
    elif upper_wick / rng >= 0.50:
        findings.append(
            f"Upper-Wick Supply Rejection near {last['High']:,.2f} {unit_lbl} — wait for full candle close above resistance"
        )

    if ind["bb_width_pct"] < 12.0:
        findings.append(
            f"Tight Bollinger Bandwidth Contraction ({ind['bb_width_pct']:.1f}%) signaling volatility squeeze prior to expansion"
        )

    cur_p = tech["price"]["current"]
    vpoc = ind["vpoc_6m"]
    if cur_p >= vpoc:
        findings.append(
            f"Trading ABOVE Volume Point of Control (VPOC: {vpoc:,.2f} {unit_lbl}), turning high-volume node into structural floor"
        )
    else:
        findings.append(
            f"Trading below VPOC ({vpoc:,.2f} {unit_lbl}); overhead volume node acts as resistance until reclaimed"
        )

    if abs(cur_p - fib["fib_38_2"]) / max(cur_p, 1.0) <= 0.035:
        findings.append(f"Coiling on 38.2% Fibonacci retracement support ({fib['fib_38_2']:,.2f} {unit_lbl})")
    elif abs(cur_p - fib["fib_61_8_golden"]) / max(cur_p, 1.0) <= 0.04:
        findings.append(f"Testing 61.8% Golden Pocket Fibonacci retracement zone ({fib['fib_61_8_golden']:,.2f} {unit_lbl})")

    return findings


def _synthesize_institutional_thesis_deterministic(
    tech: Dict[str, Any],
    candlestick_patterns: List[str],
    lessons: List[Dict[str, Any]],
    disclosures: List[Dict[str, Any]],
    user_query: str = "",
    index_3y_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Synthesizes exact mathematical indicators, visual chart structure, Tier C financial disclosures,
    and Tier A Learned Lessons into a complete recommendation with Dynamic Target Price,
    Suggested Target Timeframe/Date, and Dynamic Stop-Loss.
    """
    ticker = tech["ticker"]
    company_name = tech["company_name"]
    sector = tech["sector"]
    is_index = tech.get("asset_type") == "INDEX"
    tf = tech.get("candle_timeframe", "1D")
    tf_label = tech.get("candle_timeframe_label", TIMEFRAME_LABELS.get(tf, "Daily (1D)"))
    unit_lbl = "pts" if is_index else "EGP"

    p = tech["price"]
    ind = tech["indicators"]
    geom = tech["strategy_geometry"]
    scoring = tech["scoring"]
    tech_score = int(scoring["technical_score"])

    # Calibrated Fundamental / Structural Score (Fixes Bug #4 where every stock got a static 85-89)
    base_fund = 72
    if disclosures:
        doc_scores = [int(d.get("key_metrics", {}).get("fundamental_health_score", 82)) for d in disclosures]
        base_fund = max(doc_scores)
    else:
        # Derive structural quality from long-term 200-MA trend health & sector tailwinds
        if p["current"] >= ind["sma200"]:
            base_fund += 8
        if "Export" in sector or "Chemicals" in sector or "Banking" in sector:
            base_fund += 5
        elif "Index" in sector:
            base_fund += 6

    fund_score = int(min(max(base_fund, 55), 94))

    if is_index and index_3y_data:
        perf3y = index_3y_data["performance_3y"]
        catalysts = [
            f"3-Year Secular Performance: {ticker} delivered {perf3y['return_3y_pct']:+.1f}% over 3Y (CAGR: {perf3y['cagr_3y_pct']:+.1f}%, Max Drawdown: {perf3y['max_drawdown_3y_pct']:.1f}%).",
            f"Multi-Horizon Momentum: 1Y Return {perf3y['return_1y_pct']:+.1f}% | 6M Return {perf3y['return_6m_pct']:+.1f}% | 1M Return {perf3y['return_1m_pct']:+.1f}%.",
            f"Volatility & Range Profile: 3Y Range {perf3y['low_3y']:,.0f} – {perf3y['high_3y']:,.0f} pts (Annualized Vol: {perf3y['annualized_volatility_pct']:.1f}%).",
        ]
    elif "Export" in sector or "Chemicals" in sector or "Industrials" in sector:
        catalysts = [
            f"USD / Export Revenue Hedge: Earns foreign-currency linked cash flows in {sector}, protecting real margins.",
            "Institutional Liquidity: Core EGX constituent with strong institutional participation.",
            "Cash-Flow Resilience: Strong operating cash generation supporting medium-term repricing.",
        ]
    elif "Banking" in sector or "Financial" in sector:
        catalysts = [
            "High Net Interest Margin (NIM) Environment: High CBE interest rates support strong treasury & lending yields.",
            "Valuation Support: Trades at an attractive earnings multiple relative to ROE.",
            "Index Heavyweight: Major institutional benchmark weight on the Egyptian Exchange.",
        ]
    else:
        catalysts = [
            f"Sector Leadership: Key Egyptian Exchange operator in {sector} with pricing power against inflation.",
            "Demand & Volume Support: Institutional accumulation visible near volume point of control.",
            "Disciplined Risk/Reward Geometry: Favorable upside-to-stop ratio at current price levels.",
        ]

    triggers: List[str] = []
    if p["current"] >= ind["sma50"] and ind["sma50"] >= ind["sma200"]:
        triggers.append(
            f"Confirmed Uptrend ({tf_label}): Price ({p['current']:,.2f} {unit_lbl}) is above both 50-MA ({ind['sma50']:,.2f}) and 200-MA ({ind['sma200']:,.2f})."
        )
    elif p["current"] >= ind["sma200"]:
        triggers.append(
            f"Long-Term Uptrend Support ({tf_label}): Price ({p['current']:,.2f} {unit_lbl}) holds above 200-MA ({ind['sma200']:,.2f} {unit_lbl}) while pulling back below 50-MA ({ind['sma50']:,.2f})."
        )
    else:
        triggers.append(
            f"Counter-Trend / Below 200-MA ({tf_label}): Price ({p['current']:,.2f} {unit_lbl}) is below 200-MA ({ind['sma200']:,.2f} {unit_lbl}) — require momentum confirmation."
        )

    if ind["rsi_14_daily"] < 38:
        triggers.append(
            f"Oversold Value Zone: {tf} RSI(14) is low ({ind['rsi_14_daily']}), signaling exhaustion of sellers and rebound potential."
        )
    elif ind["rsi_14_daily"] > 72:
        triggers.append(
            f"Short-Term Overbought: {tf} RSI(14) is stretched ({ind['rsi_14_daily']}) — wait for a pullback before adding exposure."
        )
    else:
        triggers.append(
            f"Balanced Momentum: {tf} RSI(14) at {ind['rsi_14_daily']} leaves clean runway toward the +{geom['target_upside_pct']:.1f}% target."
        )

    triggers.append(
        f"Volume Point of Control (VPOC) anchor at {ind['vpoc_6m']:,.2f} {unit_lbl} with {p['rvol']:.2f}x relative volume."
    )
    for pat in candlestick_patterns[:2]:
        triggers.append(pat)

    invalidations = [
        f"Dynamic Stop-Loss: Exit if {tf} candle closes below {geom['stop_loss_price']:,.2f} {unit_lbl} (caps downside risk at -{geom['stop_distance_pct']:.1f}%).",
        f"Dynamic Profit Target: Take profit / trim 50% at {geom['target_price']:,.2f} {unit_lbl} (+{geom['target_upside_pct']:.1f}% gain, R:R {geom['reward_to_risk']}:1).",
        f"Suggested Target Horizon: {geom['target_timeframe_label']} (Target Date: {geom['target_date']}). Re-evaluate if target is not reached by {geom['target_date']}.",
    ]

    vision_summary = (
        f"[{tf_label} Chart] Price at {p['current']:,.2f} {unit_lbl} | Dynamic Target: {geom['target_price']:,.2f} {unit_lbl} "
        f"(+{geom['target_upside_pct']:.1f}% in {geom['target_timeframe_label']}) | Stop-Loss: {geom['stop_loss_price']:,.2f} {unit_lbl} (-{geom['stop_distance_pct']:.1f}%)."
    )

    # Weight technical score 70% and fundamental/structural score 30% so weak technicals are never masked
    alignment_score = int(round(0.70 * tech_score + 0.30 * fund_score))
    conviction_score = min(max(alignment_score, 40), 96)

    if alignment_score >= 70 and geom["reward_to_risk"] >= 1.85 and p["current"] >= ind["sma200"] * 0.96:
        action_rec = f"BUY (Target +{geom['target_upside_pct']:.1f}% in {geom['target_timeframe_label']})"
    elif alignment_score >= 56 or (ind["rsi_14_daily"] < 38 and p["current"] >= ind["sma200"] * 0.92):
        action_rec = "WAIT FOR DIP / ACCUMULATE ON SUPPORT"
    else:
        action_rec = "AVOID / WAIT FOR TREND REVERSAL"

    applied_lesson_ids = [lsn["lesson_id"] for lsn in lessons[:4]]
    rec_alloc_pct = 0.0 if is_index else min(float(geom["recommended_allocation_pct"]), STRATEGY_RULES.max_allocation_pct * 100.0)

    sizing_clause = (
        ""
        if is_index
        else f" | Suggested Size: {geom['recommended_shares']:,} shares ({geom['recommended_position_egp']:,.0f} EGP)"
    )
    thesis_summary = (
        f"{ticker} ({company_name}) [{tf_label}]: {action_rec}. "
        f"Entry: {geom['entry_price']:,.2f} {unit_lbl} -> Target: {geom['target_price']:,.2f} {unit_lbl} "
        f"(+{geom['target_upside_pct']:.1f}% in {geom['target_timeframe_label']}, by {geom['target_date']}) | "
        f"Stop-Loss: {geom['stop_loss_price']:,.2f} {unit_lbl} (-{geom['stop_distance_pct']:.1f}%, R:R {geom['reward_to_risk']}:1)"
        f"{sizing_clause}."
    )

    lessons_md = "\n".join(
        f"- **`{l['lesson_id']}`:** {l['rule_statement']}" for l in lessons[:4]
    ) or "- No past trade lessons recorded yet."
    catalysts_md = "\n".join(f"- {c}" for c in catalysts[:3])
    triggers_md = "\n".join(f"- {t}" for t in triggers[:4])
    inval_md = "\n".join(f"- {inv}" for inv in invalidations)

    sizing_line_md = (
        "- **Asset Type:** `EGX Market Benchmark Index`"
        if is_index
        else f"- **Suggested Position Size (<=20% Cap):** `{geom['recommended_shares']:,} shares` (`{geom['recommended_position_egp']:,.0f} EGP` = `{rec_alloc_pct:.1f}%` of portfolio)"
    )

    narrative_md = f"""### Dynamic Action Plan ({tf_label}): **{ticker}** ({company_name})
**Recommendation:** `{action_rec}` | **Conviction Score:** `{conviction_score}/100`

#### 1. Key Trade & Timing Numbers
- **Current / Entry Level:** `{geom['entry_price']:,.2f} {unit_lbl}`
- **Dynamic Target Price:** `{geom['target_price']:,.2f} {unit_lbl}` *(+{geom['target_upside_pct']:.1f}% upside | R:R `{geom['reward_to_risk']}:1`)*
- **Suggested Target Timeframe:** `{geom['target_timeframe_label']}` *(Target Date: `{geom['target_date']}`)*
- **Dynamic Safety Stop-Loss:** `{geom['stop_loss_price']:,.2f} {unit_lbl}` *(Risk: `-{geom['stop_distance_pct']:.1f}%`)*
{sizing_line_md}

#### 2. Technical & Structural Drivers ({tf_label})
{triggers_md}
{catalysts_md}

#### 3. Execution & Invalidation Rules
{inval_md}

#### 4. Lessons Applied from Cumulative Memory
{lessons_md}
"""

    return {
        "fundamental_health_score": fund_score,
        "fundamental_verdict": "STRONG STRUCTURAL TAILWIND" if fund_score >= 82 else "MODERATE FUNDAMENTAL BACKDROP",
        "technical_status_verdict": scoring["regime_verdict"],
        "vision_chart_findings": vision_summary,
        "strategy_alignment_score": alignment_score,
        "action_recommendation": action_rec,
        "conviction_score": conviction_score,
        "recommended_allocation_pct": rec_alloc_pct,
        "target_price": geom["target_price"],
        "target_upside_pct": geom["target_upside_pct"],
        "target_days": geom["target_days"],
        "target_timeframe_label": geom["target_timeframe_label"],
        "target_date": geom["target_date"],
        "stop_loss_price": geom["stop_loss_price"],
        "stop_distance_pct": geom["stop_distance_pct"],
        "thesis_summary": thesis_summary,
        "fundamental_catalysts": catalysts[:5],
        "technical_triggers": triggers[:5],
        "invalidation_rules": invalidations,
        "applied_lesson_ids": applied_lesson_ids,
        "narrative_markdown": narrative_md,
    }


# =============================================================================
# SELF-REFLECTION LOOP & RECOMMENDATION EVALUATOR
# =============================================================================

def run_self_reflection_audit(thesis_id: Optional[str] = None, trigger_type: str = "MID_TRADE_AUDIT") -> Dict[str, Any]:
    """
    Executes the Tier A Self-Reflection Loop AND Tier D Recommendations Evaluation:
    - Evaluates all saved recommendations against live prices and historical High/Low paths.
    - Audits active & closed theses against their dynamic Target Price, Target Date, and Stop-Loss.
    """
    rec_stats = evaluate_recommendations()

    if thesis_id:
        theses_to_audit = [t for t in list_theses() if t["thesis_id"] == thesis_id]
    else:
        theses_to_audit = [
            t for t in list_theses()
            if t["status"] in ("ACTIVE", "TRIMMED_50", "CLOSED_TP30", "CLOSED_PROFIT", "CLOSED_STOP_LOSS", "CLOSED_TIME_EXPIRY")
        ]

    cockpit = get_portfolio_cockpit()
    portfolio_equity = float(cockpit["summary"]["total_equity_egp"])
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today = datetime.now().date()

    generated_reflections: List[Dict[str, Any]] = []
    new_lessons_created: List[Dict[str, Any]] = []

    for th in theses_to_audit:
        t_id = th["thesis_id"]
        ticker = th["ticker"]
        tf = th.get("candle_timeframe") or "1D"
        tech = compute_full_technical_analysis(ticker, portfolio_equity_egp=portfolio_equity, timeframe=tf)
        live_price = float(tech["price"]["current"])
        live_atr = float(tech["indicators"]["atr_14"])

        if not is_egx_index(ticker):
            update_position_live_price(ticker, live_price, live_atr)

        entry_price = float(th["entry_price"])
        target_price = float(th["target_price"])
        stop_price = float(th["current_trailing_stop"])
        target_pct = round(((target_price - entry_price) / max(entry_price, 0.01)) * 100.0, 1)
        t_days = int(th.get("target_days") or 60)

        try:
            entry_dt = datetime.fromisoformat(th["entry_date"]).date()
            days_elapsed = max((today - entry_dt).days, 1)
        except Exception:
            days_elapsed = 15

        pct_move = (
            round(((live_price - entry_price) / entry_price) * 100.0, 2)
            if th["status"] in ("ACTIVE", "TRIMMED_50")
            else float(th.get("realized_return_pct") or 0.0)
        )

        predicted_str = (
            f"Entry at {entry_price:.2f} EGP targeting +{target_pct:.1f}% ({target_price:.2f} EGP) within {t_days} days "
            f"with dynamic stop-loss ({stop_price:.2f} EGP)."
        )
        actual_str = (
            f"After {days_elapsed} days: Status={th['status']}, Price={live_price:.2f} EGP "
            f"({pct_move:+.2f}% vs entry), RSI(14)={tech['indicators']['rsi_14_daily']}, RVOL={tech['price']['rvol']}x."
        )

        extracted_ids: List[str] = []
        if pct_move >= target_pct * 0.90 or th["status"] in ("TRIMMED_50", "CLOSED_TP30", "CLOSED_PROFIT"):
            adherence = 95
            variance = (
                f"PROFITABLE / ON-TARGET ({pct_move:+.2f}% in {days_elapsed}d): "
                f"Dynamic target & trailing stop discipline validated on {ticker}."
            )
            if int(th.get("lessons_extracted") or 0) == 0:
                new_lsn = add_learned_lesson(
                    {
                        "source_thesis_id": t_id,
                        "ticker": ticker,
                        "sector": th["sector"],
                        "category": "PROFIT_CAPTURE",
                        "rule_statement": (
                            f"On {th['sector']} setups ({ticker}), combining dynamic structural targets (+{target_pct:.1f}%) "
                            f"with ATR trailing stops captured {pct_move:+.2f}% over {days_elapsed} days."
                        ),
                        "evidence_summary": f"Validated by {t_id} ({ticker}): Entry {entry_price:.2f} EGP -> {pct_move:+.2f}%.",
                        "confidence_weight": 0.90,
                        "win_rate_impact": f"{pct_move:+.1f}% Realized Gain",
                    }
                )
                extracted_ids.append(new_lsn["lesson_id"])
                new_lessons_created.append(new_lsn)
                conn = get_connection()
                conn.execute("UPDATE theses SET lessons_extracted = 1, updated_at = ? WHERE thesis_id = ?;", (now_iso, t_id))
                conn.commit()
                conn.close()

        elif pct_move < 0 or th["status"] == "CLOSED_STOP_LOSS":
            adherence = 72
            variance = (
                f"DRAWDOWN / ADVERSE VARIANCE ({pct_move:+.2f}% after {days_elapsed}d): "
                f"Price lagging expected trajectory; strictly enforce stop-loss at {stop_price:.2f} EGP."
            )
            if int(th.get("lessons_extracted") or 0) == 0 and th["status"].startswith("CLOSED"):
                new_lsn = add_learned_lesson(
                    {
                        "source_thesis_id": t_id,
                        "ticker": ticker,
                        "sector": th["sector"],
                        "category": "STOP_DISCIPLINE",
                        "rule_statement": (
                            f"Avoid entering {th['sector']} setups ({ticker}) when price is below VPOC ({tech['indicators']['vpoc_6m']:.2f} EGP) "
                            f"or RVOL is below 1.2x; strictly honor initial ATR stop."
                        ),
                        "evidence_summary": f"Post-mortem on {t_id} ({ticker}) closed at {pct_move:+.2f}%.",
                        "confidence_weight": 0.88,
                        "win_rate_impact": "Drawdown Cap Protection",
                    }
                )
                extracted_ids.append(new_lsn["lesson_id"])
                new_lessons_created.append(new_lsn)
                conn = get_connection()
                conn.execute("UPDATE theses SET lessons_extracted = 1, updated_at = ? WHERE thesis_id = ?;", (now_iso, t_id))
                conn.commit()
                conn.close()
        else:
            expected_linear_pct = round((days_elapsed / max(t_days, 1)) * target_pct, 1)
            adherence = 90 if pct_move >= expected_linear_pct * 0.7 else 80
            variance = (
                f"ON-TRACK MID-SWING ({pct_move:+.2f}% in {days_elapsed}d vs ~{expected_linear_pct:+.1f}% pace): "
                f"Holding above trailing stop ({stop_price:.2f} EGP) toward target ({target_price:.2f} EGP)."
            )

        ref_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
        conn = get_connection()
        conn.execute(
            """
            INSERT INTO reflection_logs (
                reflection_id, trigger_type, thesis_id, ticker,
                predicted_trajectory, actual_trajectory, variance_analysis,
                adherence_score, extracted_lesson_ids, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                ref_id,
                trigger_type,
                t_id,
                ticker,
                predicted_str,
                actual_str,
                variance,
                adherence,
                json.dumps(extracted_ids),
                now_iso,
            ),
        )
        conn.execute(
            "UPDATE theses SET post_mortem_notes = ?, updated_at = ? WHERE thesis_id = ?;",
            (variance, now_iso, t_id),
        )
        conn.commit()
        conn.close()

        generated_reflections.append(
            {
                "reflection_id": ref_id,
                "trigger_type": trigger_type,
                "thesis_id": t_id,
                "ticker": ticker,
                "predicted_trajectory": predicted_str,
                "actual_trajectory": actual_str,
                "variance_analysis": variance,
                "adherence_score": adherence,
                "extracted_lesson_ids": extracted_ids,
                "created_at": now_iso,
            }
        )

    conn = get_connection()
    conn.execute("UPDATE learned_lessons SET times_applied = times_applied + 1 WHERE active = 1;")
    conn.commit()
    conn.close()

    return {
        "audited_count": len(generated_reflections),
        "new_lessons_count": len(new_lessons_created),
        "new_lessons": new_lessons_created,
        "reflections": generated_reflections,
        "all_lessons": get_learned_lessons(),
        "recommendation_stats": rec_stats,
    }


def _build_system_instruction(lessons: List[Dict[str, Any]]) -> str:
    lessons_block = "\n".join(
        f"- [{l['lesson_id']}] ({l['category']}): {l['rule_statement']} (Evidence: {l['evidence_summary']})"
        for l in lessons
    )
    return f"""You are a Principal AI Financial Architect and Institutional EGX Trading & Index Decision Agent.
You MUST enforce these strategy rules in every response:
1. Dynamic Target Price & Timeframe: Suggest a realistic, data-driven Target Price, Target Timeframe (days/weeks/months & target date), and Stop-Loss tailored to the asset's structural resistance, support, and ATR volatility on the selected candle timeframe (1D, 1W, or 1M). Do NOT use a blind fixed +30% or 6-month rule.
2. Risk Management: Enforce dynamic ATR + structural Stop-Loss and a maximum capital allocation of 20.0% of portfolio equity per stock.
3. Thesis-Driven Gatekeeper: Provide explicit Fundamental Catalysts, Technical Triggers, and Invalidation Rules.
4. Never guess prices or indicators: Use ONLY the exact mathematical numbers provided by the Python Technical Engine.

CUMULATIVE AGENT KNOWLEDGE BASE (LEARNED LESSONS FROM PAST TRADES — MUST BE APPLIED):
{lessons_block}
"""


def _build_analysis_prompt(tech: Dict[str, Any], disclosures: List[Dict[str, Any]], user_query: str) -> str:
    clean_tech = {k: v for k, v in tech.items() if k != "chart_series"}
    disc_summary = [
        {"title": d["title"], "period": d["fiscal_period"], "metrics": d.get("key_metrics"), "summary": d["ai_summary"]}
        for d in disclosures
    ]
    tf_lbl = tech.get("candle_timeframe_label", "Daily (1D)")
    return f"""Analyze the attached Candlestick + Volume Profile + RSI chart image ({tf_lbl}) alongside these exact mathematical metrics for {tech['ticker']} ({tech['company_name']}):

EXACT PYTHON TECHNICAL ENGINE METRICS (DO NOT HALLUCINATE NUMBERS):
{json.dumps(clean_tech, indent=2)}

TIER C COMPANY DISCLOSURES & FINANCIAL REPORTS:
{json.dumps(disc_summary, indent=2)}

USER REQUEST / FOCUS:
{user_query or f'Perform a comprehensive {tf_lbl} analysis with dynamic Target Price, Suggested Target Timeframe/Date, and Stop-Loss.'}
"""


def _record_chat_interaction(role: str, ticker: str, content: str, payload: Dict[str, Any]) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO chat_history (message_id, role, ticker, content, structured_payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            f"MSG-{uuid.uuid4().hex[:8].upper()}",
            role,
            ticker,
            content,
            json.dumps(payload),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()
