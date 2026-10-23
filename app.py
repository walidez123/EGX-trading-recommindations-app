"""
FastAPI Backend Server for the EGX Investment & Index Decision Agent.
Exposes REST endpoints for:
- Tier A: Episodic Thesis Ledger & Self-Reflection Loop (Learned Lessons Knowledge Base)
- Tier B: Portfolio Cockpit (EGP Cash, 20% Cap, Dynamic Target / Partial Trim, Dynamic ATR Stop & Countdown)
- Tier C: Native Multimodal PDF Financial Disclosure Store
- Tier D: Recommendations Ledger & Live Time/Price Evaluation Engine
- 3+ Year EGX Index Analysis: EGX 30 (EGX:EGX30), EGX 70 EWI (EGX:EGX70EWI), and EGX 33 Shariah (EGX:SHARIAH)
- Multi-Timeframe Candle Selection: Daily ('1D'), Weekly ('1W'), and Monthly ('1M')
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.ai_agent import run_hybrid_stock_analysis, run_self_reflection_audit
from backend.chart_generator import generate_vision_chart
from backend.config import (
    BASE_DIR,
    CSV_UPLOADS_DIR,
    EGX_INDICES_DIRECTORY,
    EGX_TICKER_DIRECTORY,
    is_egx_index,
    normalize_egx_ticker,
    normalize_timeframe,
)
from backend.database import (
    add_learned_lesson,
    create_thesis,
    evaluate_recommendations,
    execute_position_action,
    get_connection,
    get_learned_lessons,
    get_portfolio_cockpit,
    get_portfolio_settings,
    get_recommendation_stats,
    init_db,
    list_recommendations,
    list_reflection_logs,
    list_theses,
    open_position_from_thesis,
    record_direct_trade,
    record_recommendation,
    reset_all_data,
    update_portfolio_settings,
    update_position_live_price,
)
from backend.document_store import ingest_disclosure_document, list_documents
from backend.technical_engine import (
    compute_all_indices_overview,
    compute_full_technical_analysis,
    compute_index_analysis,
)

app = FastAPI(
    title="EGX Investment & 3Y Index Decision Agent API",
    description="Local-First Cumulative Memory, 3-Year EGX Index Analyzer (EGX30/70/33), Dynamic Target/Time/Stop Engine & Recommendation Evaluator",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def on_startup() -> None:
    init_db()


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class AnalyzeRequest(BaseModel):
    ticker: str
    query: str = ""
    auto_save_thesis: bool = True
    timeframe: str = "1D"  # '1D' (Daily), '1W' (Weekly), or '1M' (Monthly)
    source: Optional[str] = None


class CreateThesisRequest(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    entry_price: float
    target_price: Optional[float] = None
    target_days: Optional[int] = 60
    atr_at_entry: Optional[float] = None
    initial_stop_loss: Optional[float] = None
    candle_timeframe: str = "1D"
    conviction_score: int = 82
    recommended_allocation_pct: float = 0.15
    fundamental_catalysts: List[str]
    technical_triggers: List[str]
    invalidation_rules: List[str]
    thesis_summary: str = ""


class CreateRecommendationRequest(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    source: str = "Manual / Analyst Entry"
    action: str = "BUY"
    candle_timeframe: str = "1D"
    conviction_score: int = 80
    entry_price: float
    target_price: float
    stop_loss_price: float
    target_days: int = 45
    target_date: Optional[str] = None
    created_date: Optional[str] = None
    rationale_summary: str = ""


class OpenPositionRequest(BaseModel):
    thesis_id: str
    allocation_egp: Optional[float] = None
    custom_entry_price: Optional[float] = None


class PositionActionRequest(BaseModel):
    action: str  # 'PARTIAL_TRIM_50' or 'FULL_EXIT'
    execution_price: Optional[float] = None
    notes: str = ""


class ReflectionRequest(BaseModel):
    thesis_id: Optional[str] = None
    trigger_type: str = "MID_TRADE_AUDIT"


class LessonRequest(BaseModel):
    ticker: str = "ALL"
    sector: str = "ALL"
    category: str = "ENTRY_TIMING"
    rule_statement: str
    evidence_summary: str = ""
    confidence_weight: float = 0.90
    win_rate_impact: str = "Manual Institutional Rule"


class SettingsUpdateRequest(BaseModel):
    cash_balance_egp: Optional[float] = None
    initial_capital_egp: Optional[float] = None
    max_allocation_pct: Optional[float] = None
    atr_multiplier: Optional[float] = None
    ai_mode: Optional[str] = None
    gemini_api_key: Optional[str] = None
    preferred_model: Optional[str] = None


class DirectTradeRequest(BaseModel):
    ticker: str
    side: str = "BUY"
    shares: int
    price: float
    fees_egp: float = 0.0
    company_name: Optional[str] = None
    trade_date: Optional[str] = None
    notes: str = "Logged via Trade Entry"


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
async def serve_dashboard() -> FileResponse:
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))


@app.get("/api/state")
async def get_full_system_state() -> Dict[str, Any]:
    """Returns unified Tier A, Tier B, Tier C, and Tier D state for the frontend cockpit."""
    conn = get_connection()
    chat_rows = conn.execute("SELECT * FROM chat_history ORDER BY created_at DESC LIMIT 20;").fetchall()
    conn.close()

    rec_stats = evaluate_recommendations()

    return {
        "portfolio": get_portfolio_cockpit(),
        "theses": list_theses(),
        "lessons": get_learned_lessons(active_only=False),
        "reflections": list_reflection_logs(),
        "documents": list_documents(),
        "recommendations": rec_stats.get("recommendations", []),
        "recommendation_stats": rec_stats,
        "egx_directory": EGX_TICKER_DIRECTORY,
        "egx_indices_directory": EGX_INDICES_DIRECTORY,
        "chat_history": [dict(r) for r in reversed(chat_rows)],
    }


@app.get("/api/technical/{ticker}")
async def get_ticker_technicals(
    ticker: str,
    timeframe: str = Query("1D", description="Candle timeframe: 1D, 1W, or 1M"),
) -> Dict[str, Any]:
    """Computes pure-math technicals, Fibs, VPOC, dynamic target/time/stop, and renders the mplfinance chart."""
    try:
        tf = normalize_timeframe(timeframe)
        cockpit = get_portfolio_cockpit()
        equity = float(cockpit["summary"]["total_equity_egp"])
        tech = compute_full_technical_analysis(ticker, portfolio_equity_egp=equity, timeframe=tf)
        if not is_egx_index(tech["ticker"]):
            update_position_live_price(
                tech["ticker"],
                latest_price=tech["price"]["current"],
                latest_atr=tech["indicators"]["atr_14"],
            )
        _, chart_url = generate_vision_chart(tech)
        tech.pop("_raw_df", None)
        tech["chart_image_url"] = chart_url
        return tech
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/indices")
async def get_all_indices_endpoint(
    timeframe: str = Query("1D", description="Candle timeframe: 1D, 1W, or 1M"),
    lookback_years: int = Query(3, ge=3, le=5, description="Minimum historical years (default 3)"),
) -> Dict[str, Any]:
    """
    Computes 3+ Year Index Analysis & Relative Strength Comparison for EGX 30, EGX 70 EWI, and EGX 33 Shariah
    on the selected candle timeframe ('1D', '1W', or '1M'), and records their recommendations.
    """
    try:
        tf = normalize_timeframe(timeframe)
        overview = compute_all_indices_overview(timeframe=tf, lookback_years=lookback_years)
        for sym, idx_data in overview["indices"].items():
            geom = idx_data["strategy_geometry"]
            reg = idx_data["regime"]
            record_recommendation(
                {
                    "ticker": sym,
                    "company_name": idx_data["full_name"],
                    "asset_type": "INDEX",
                    "sector": "EGX Market Index",
                    "candle_timeframe": tf,
                    "source": f"EGX 3Y Index Analyzer ({tf})",
                    "action": reg["action_recommendation"],
                    "conviction_score": reg["conviction_score"],
                    "entry_price": geom["entry_price"],
                    "target_price": geom["target_price"],
                    "target_upside_pct": geom["target_upside_pct"],
                    "stop_loss_price": geom["stop_loss_price"],
                    "stop_loss_pct": geom["stop_distance_pct"],
                    "reward_to_risk": geom["reward_to_risk"],
                    "target_days": geom["target_days"],
                    "target_timeframe_label": geom["target_timeframe_label"],
                    "target_date": geom["target_date"],
                    "rationale_summary": reg["summary"],
                }
            )
        overview["recommendation_stats"] = evaluate_recommendations()
        return overview
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/indices/{symbol}")
async def get_single_index_endpoint(
    symbol: str,
    timeframe: str = Query("1D", description="Candle timeframe: 1D, 1W, or 1M"),
    lookback_years: int = Query(3, ge=3, le=5),
) -> Dict[str, Any]:
    """
    Runs full 3+ Year Index Analysis for a specific EGX index ('EGX30', 'EGX70', or 'EGX33')
    on the selected candle timeframe ('1D', '1W', '1M'), renders its chart PNG, and records its recommendation.
    """
    try:
        tf = normalize_timeframe(timeframe)
        idx_data = compute_index_analysis(symbol, timeframe=tf, lookback_years=lookback_years)
        chart_path, chart_url = generate_vision_chart(idx_data)
        idx_data.pop("_raw_df", None)
        idx_data["chart_image_url"] = chart_url
        idx_data["chart_image_path"] = str(chart_path)

        geom = idx_data["strategy_geometry"]
        reg = idx_data["regime"]
        saved_rec = record_recommendation(
            {
                "ticker": idx_data["ticker"],
                "company_name": idx_data["full_name"],
                "asset_type": "INDEX",
                "sector": "EGX Market Index",
                "candle_timeframe": tf,
                "source": f"EGX 3Y Index Analyzer ({tf})",
                "action": reg["action_recommendation"],
                "conviction_score": reg["conviction_score"],
                "entry_price": geom["entry_price"],
                "target_price": geom["target_price"],
                "target_upside_pct": geom["target_upside_pct"],
                "stop_loss_price": geom["stop_loss_price"],
                "stop_loss_pct": geom["stop_distance_pct"],
                "reward_to_risk": geom["reward_to_risk"],
                "target_days": geom["target_days"],
                "target_timeframe_label": geom["target_timeframe_label"],
                "target_date": geom["target_date"],
                "rationale_summary": reg["summary"],
            }
        )
        idx_data["saved_recommendation"] = saved_rec
        idx_data["recommendation_stats"] = evaluate_recommendations()
        return idx_data
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze")
async def analyze_ticker_endpoint(req: AnalyzeRequest) -> Dict[str, Any]:
    """Executes the full 4-Node StateGraph Hybrid Analysis on the chosen candle timeframe ('1D', '1W', '1M')."""
    try:
        result = await run_hybrid_stock_analysis(
            ticker=req.ticker,
            user_query=req.query,
            auto_save_thesis=req.auto_save_thesis,
            timeframe=req.timeframe,
            source_override=req.source,
        )
        result["portfolio"] = get_portfolio_cockpit()
        result["theses"] = list_theses()
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/recommendations")
async def get_recommendations_endpoint(
    ticker: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates all recommendations against latest prices and returns the filtered list + accuracy stats."""
    stats = evaluate_recommendations()
    filtered = list_recommendations(ticker=ticker, source=source, status=status)
    return {
        "stats": stats,
        "recommendations": filtered,
    }


@app.post("/api/recommendations")
async def create_manual_recommendation_endpoint(req: CreateRecommendationRequest) -> Dict[str, Any]:
    """Saves a custom/manual recommendation by date, stock/index, and source, then evaluates all recommendations."""
    try:
        saved = record_recommendation(req.model_dump(exclude_none=True))
        stats = evaluate_recommendations()
        return {"recommendation": saved, "stats": stats, "recommendations": stats["recommendations"]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/recommendations/evaluate")
async def evaluate_recommendations_endpoint() -> Dict[str, Any]:
    """Triggers a live evaluation pass across all saved recommendations."""
    stats = evaluate_recommendations()
    return {"stats": stats, "recommendations": stats["recommendations"]}


@app.post("/api/theses")
async def create_manual_thesis_endpoint(req: CreateThesisRequest) -> Dict[str, Any]:
    try:
        created = create_thesis(req.model_dump())
        return {"thesis": created, "theses": list_theses()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/positions/open")
async def open_position_endpoint(req: OpenPositionRequest) -> Dict[str, Any]:
    """Opens a position in Tier B strictly backed by a verified Tier A Thesis and capped at <=20% equity."""
    try:
        cockpit = open_position_from_thesis(
            thesis_id=req.thesis_id,
            allocation_egp=req.allocation_egp,
            custom_entry_price=req.custom_entry_price,
        )
        return {"portfolio": cockpit, "theses": list_theses()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/positions/{position_id}/action")
async def position_action_endpoint(position_id: str, req: PositionActionRequest) -> Dict[str, Any]:
    """
    Executes PARTIAL_TRIM_50 or FULL_EXIT.
    When FULL_EXIT occurs, automatically runs Post-Mortem Self-Reflection and Recommendation Evaluation!
    """
    try:
        outcome = execute_position_action(
            position_id=position_id,
            action=req.action,
            execution_price=req.execution_price,
            notes=req.notes,
        )
        reflection_result = None
        if req.action == "FULL_EXIT":
            reflection_result = run_self_reflection_audit(
                thesis_id=outcome["thesis_id"],
                trigger_type="THESIS_CLOSE_POSTMORTEM",
            )
        rec_stats = evaluate_recommendations()
        return {
            "outcome": outcome,
            "reflection": reflection_result,
            "portfolio": get_portfolio_cockpit(),
            "theses": list_theses(),
            "lessons": get_learned_lessons(active_only=False),
            "reflections": list_reflection_logs(),
            "recommendations": rec_stats["recommendations"],
            "recommendation_stats": rec_stats,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/reflection/run")
async def trigger_reflection_endpoint(req: ReflectionRequest) -> Dict[str, Any]:
    """Runs the Self-Reflection Loop & Recommendation Evaluation across active & closed theses/recommendations."""
    try:
        res = run_self_reflection_audit(thesis_id=req.thesis_id, trigger_type=req.trigger_type)
        res["portfolio"] = get_portfolio_cockpit()
        res["theses"] = list_theses()
        res["reflections"] = list_reflection_logs()
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/lessons")
async def add_custom_lesson_endpoint(req: LessonRequest) -> Dict[str, Any]:
    lesson = add_learned_lesson(req.model_dump())
    return {"lesson": lesson, "lessons": get_learned_lessons(active_only=False)}


@app.post("/api/documents/upload")
async def upload_disclosure_endpoint(
    ticker: str = Form(...),
    title: str = Form(...),
    doc_type: str = Form("EARNINGS_RELEASE"),
    fiscal_period: str = Form("FY2025/2026"),
    raw_text: str = Form(""),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Ingests a company PDF earnings report or disclosure into Tier C Document Store."""
    file_bytes = await file.read() if file else None
    filename = file.filename if file else None
    doc = ingest_disclosure_document(
        ticker=ticker,
        title=title,
        doc_type=doc_type,
        fiscal_period=fiscal_period,
        file_bytes=file_bytes,
        filename=filename,
        raw_text_input=raw_text,
    )
    return {"document": doc, "documents": list_documents()}


@app.post("/api/csv/upload")
async def upload_ohlcv_csv_endpoint(
    ticker: str = Form(...),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Saves an uploaded TradingView / Mubasher OHLCV CSV file for deterministic technical calculation."""
    norm_t = normalize_egx_ticker(ticker)
    target_path = CSV_UPLOADS_DIR / f"{norm_t}.csv"
    content = await file.read()
    target_path.write_bytes(content)
    return {"status": "ok", "ticker": norm_t, "csv_path": str(target_path)}


@app.post("/api/settings")
async def update_settings_endpoint(req: SettingsUpdateRequest) -> Dict[str, Any]:
    updated = update_portfolio_settings(req.model_dump(exclude_none=True))
    return {"settings": updated, "portfolio": get_portfolio_cockpit()}


@app.post("/api/trades/direct")
async def direct_trade_endpoint(req: DirectTradeRequest) -> Dict[str, Any]:
    """Logs a direct BUY or SELL trade straight into SQLite."""
    try:
        trade_res = record_direct_trade(
            ticker=req.ticker,
            side=req.side,
            shares=req.shares,
            price=req.price,
            fees_egp=req.fees_egp,
            company_name=req.company_name,
            trade_date=req.trade_date,
            notes=req.notes,
        )
        rec_stats = evaluate_recommendations()
        return {
            "trade": trade_res,
            "portfolio": get_portfolio_cockpit(),
            "theses": list_theses(),
            "lessons": get_learned_lessons(active_only=False),
            "reflections": list_reflection_logs(),
            "recommendations": rec_stats["recommendations"],
            "recommendation_stats": rec_stats,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/reset")
async def reset_system_endpoint() -> Dict[str, Any]:
    """Wipes all theses, positions, lessons, reflections, recommendations, and documents for a fresh start."""
    portfolio = reset_all_data(initial_capital_egp=1_000_000.0)
    return {
        "portfolio": portfolio,
        "theses": [],
        "lessons": [],
        "reflections": [],
        "documents": [],
        "recommendations": [],
        "recommendation_stats": get_recommendation_stats(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
