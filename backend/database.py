"""
3-Tier Cumulative Memory, Portfolio Cockpit & Recommendations Evaluation Engine (SQLite + JSON).
Implements:
- Tier A: Episodic Memory, Thesis Ledger & Learned Lessons (Self-Reflection Knowledge Base)
- Tier B: Structured State & Portfolio Tracker (Positions, Cash EGP, Dynamic Target / Partial Trim, Dynamic Countdown, ATR Trailing Stop)
- Tier C: Native Document Store for Company Disclosures, Earnings Releases & Board Reports
- Tier D: Recommendations Ledger & Automated Time/Price Path Evaluation Engine
  (saves every recommendation by date, stock/index, candle timeframe, and source, and evaluates outcomes & accuracy over time)
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.config import (
    DB_PATH,
    EGX_INDICES_DIRECTORY,
    EGX_TICKER_DIRECTORY,
    STRATEGY_RULES,
    get_index_meta,
    is_egx_index,
    normalize_egx_ticker,
    normalize_timeframe,
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Initializes the SQLite database schema, migrates columns if needed, and repairs any legacy state."""
    conn = get_connection()
    cur = conn.cursor()

    # =========================================================================
    # TIER A: EPISODIC MEMORY, THESIS LEDGER & SELF-REFLECTION KNOWLEDGE BASE
    # =========================================================================
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS theses (
            thesis_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT DEFAULT 'EGX Equity',
            status TEXT NOT NULL DEFAULT 'PROPOSED',
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            target_price REAL NOT NULL,
            time_horizon_min_date TEXT NOT NULL,
            time_horizon_deadline TEXT NOT NULL,
            holding_months_target INTEGER DEFAULT 3,
            initial_stop_loss REAL NOT NULL,
            current_trailing_stop REAL NOT NULL,
            highest_price_since_entry REAL NOT NULL,
            atr_at_entry REAL NOT NULL,
            conviction_score INTEGER DEFAULT 75,
            reward_to_risk REAL DEFAULT 2.5,
            recommended_allocation_pct REAL DEFAULT 0.15,
            fundamental_catalysts TEXT NOT NULL,
            technical_triggers TEXT NOT NULL,
            invalidation_rules TEXT NOT NULL,
            vision_chart_findings TEXT DEFAULT '',
            chart_image_url TEXT DEFAULT '',
            thesis_summary TEXT DEFAULT '',
            post_mortem_notes TEXT DEFAULT '',
            actual_outcome TEXT DEFAULT '',
            realized_return_pct REAL DEFAULT 0.0,
            lessons_extracted INTEGER DEFAULT 0,
            ai_provider_used TEXT DEFAULT 'Antigravity Hybrid Engine',
            candle_timeframe TEXT DEFAULT '1D',
            target_days INTEGER DEFAULT 60,
            target_timeframe_label TEXT DEFAULT '~2 Months (60 Days)',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )

    # Ensure new columns exist on existing `theses` table
    existing_thesis_cols = {r["name"] for r in cur.execute("PRAGMA table_info(theses);").fetchall()}
    for col_name, col_def in [
        ("candle_timeframe", "TEXT DEFAULT '1D'"),
        ("target_days", "INTEGER DEFAULT 60"),
        ("target_timeframe_label", "TEXT DEFAULT '~2 Months (60 Days)'"),
    ]:
        if col_name not in existing_thesis_cols:
            cur.execute(f"ALTER TABLE theses ADD COLUMN {col_name} {col_def};")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS learned_lessons (
            lesson_id TEXT PRIMARY KEY,
            source_thesis_id TEXT,
            ticker TEXT DEFAULT 'ALL',
            sector TEXT DEFAULT 'ALL',
            category TEXT NOT NULL,
            rule_statement TEXT NOT NULL,
            evidence_summary TEXT NOT NULL,
            confidence_weight REAL DEFAULT 0.85,
            win_rate_impact TEXT DEFAULT '',
            times_applied INTEGER DEFAULT 1,
            active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reflection_logs (
            reflection_id TEXT PRIMARY KEY,
            trigger_type TEXT NOT NULL,
            thesis_id TEXT,
            ticker TEXT NOT NULL,
            predicted_trajectory TEXT NOT NULL,
            actual_trajectory TEXT NOT NULL,
            variance_analysis TEXT NOT NULL,
            adherence_score INTEGER DEFAULT 85,
            extracted_lesson_ids TEXT DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        """
    )

    # =========================================================================
    # TIER B: STRUCTURED STATE & PORTFOLIO TRACKER
    # =========================================================================
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            base_currency TEXT DEFAULT 'EGP',
            initial_capital_egp REAL DEFAULT 1000000.0,
            cash_balance_egp REAL DEFAULT 1000000.0,
            max_allocation_pct REAL DEFAULT 0.20,
            take_profit_pct REAL DEFAULT 0.20,
            partial_trim_ratio REAL DEFAULT 0.50,
            atr_multiplier REAL DEFAULT 2.2,
            max_holding_days INTEGER DEFAULT 180,
            ai_mode TEXT DEFAULT 'antigravity_auto',
            gemini_api_key TEXT DEFAULT '',
            preferred_model TEXT DEFAULT 'gemini-2.5-pro',
            updated_at TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            position_id TEXT PRIMARY KEY,
            thesis_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT DEFAULT 'EGX Equity',
            status TEXT NOT NULL DEFAULT 'OPEN',
            entry_date TEXT NOT NULL,
            deadline_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL NOT NULL,
            highest_price REAL NOT NULL,
            target_price REAL NOT NULL,
            initial_stop_loss REAL NOT NULL,
            trailing_stop_loss REAL NOT NULL,
            atr_value REAL NOT NULL,
            initial_shares INTEGER NOT NULL,
            current_shares INTEGER NOT NULL,
            trimmed_shares INTEGER DEFAULT 0,
            trim_price REAL DEFAULT 0.0,
            realized_pnl_egp REAL DEFAULT 0.0,
            exit_date TEXT DEFAULT '',
            exit_price REAL DEFAULT 0.0,
            exit_reason TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            updated_at TEXT NOT NULL,
            FOREIGN KEY (thesis_id) REFERENCES theses(thesis_id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_events (
            event_id TEXT PRIMARY KEY,
            position_id TEXT NOT NULL,
            thesis_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            shares INTEGER NOT NULL,
            price REAL NOT NULL,
            total_value_egp REAL NOT NULL,
            pnl_egp REAL DEFAULT 0.0,
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );
        """
    )

    # =========================================================================
    # TIER C: SEMANTIC & NATIVE MULTIMODAL DOCUMENT STORE (DISCLOSURES / PDFs)
    # =========================================================================
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            title TEXT NOT NULL,
            fiscal_period TEXT DEFAULT '',
            file_path TEXT DEFAULT '',
            extracted_text TEXT DEFAULT '',
            key_metrics_json TEXT DEFAULT '{}',
            ai_summary TEXT DEFAULT '',
            uploaded_at TEXT NOT NULL
        );
        """
    )

    # =========================================================================
    # TIER D: RECOMMENDATIONS LEDGER & LIVE TIME/PRICE EVALUATION ENGINE
    # =========================================================================
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            recommendation_id TEXT PRIMARY KEY,
            created_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            asset_type TEXT NOT NULL DEFAULT 'STOCK',
            sector TEXT DEFAULT 'EGX Equity',
            candle_timeframe TEXT NOT NULL DEFAULT '1D',
            source TEXT NOT NULL,
            action TEXT NOT NULL,
            conviction_score INTEGER DEFAULT 75,
            entry_price REAL NOT NULL,
            target_price REAL NOT NULL,
            target_upside_pct REAL NOT NULL,
            stop_loss_price REAL NOT NULL,
            stop_loss_pct REAL NOT NULL,
            reward_to_risk REAL NOT NULL,
            target_days INTEGER NOT NULL,
            target_timeframe_label TEXT NOT NULL,
            target_date TEXT NOT NULL,
            rationale_summary TEXT DEFAULT '',
            thesis_id TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            current_price REAL NOT NULL,
            current_return_pct REAL DEFAULT 0.0,
            peak_price_since_rec REAL NOT NULL,
            trough_price_since_rec REAL NOT NULL,
            max_favorable_excursion_pct REAL DEFAULT 0.0,
            max_adverse_excursion_pct REAL DEFAULT 0.0,
            progress_to_target_pct REAL DEFAULT 0.0,
            days_elapsed INTEGER DEFAULT 0,
            days_remaining INTEGER DEFAULT 0,
            resolved_date TEXT DEFAULT '',
            resolved_price REAL DEFAULT 0.0,
            resolved_return_pct REAL DEFAULT 0.0,
            accuracy_score REAL DEFAULT 0.0,
            evaluation_notes TEXT DEFAULT '',
            last_evaluated_at TEXT NOT NULL
        );
        """
    )

    # Chat & Analysis History Table for Terminal Continuity
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            ticker TEXT DEFAULT '',
            content TEXT NOT NULL,
            structured_payload TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        """
    )

    conn.commit()
    _seed_initial_data_if_empty(conn)
    _repair_and_backfill_legacy_records(conn)
    conn.close()


def _seed_initial_data_if_empty(conn: sqlite3.Connection) -> None:
    """Initializes clean default portfolio settings if missing."""
    cur = conn.cursor()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute("SELECT COUNT(*) as cnt FROM portfolio_settings;")
    if cur.fetchone()["cnt"] == 0:
        cur.execute(
            """
            INSERT INTO portfolio_settings (
                id, base_currency, initial_capital_egp, cash_balance_egp,
                max_allocation_pct, take_profit_pct, partial_trim_ratio,
                atr_multiplier, max_holding_days, ai_mode, gemini_api_key,
                preferred_model, updated_at
            ) VALUES (1, 'EGP', 1000000.0, 1000000.0, 0.20, 0.20, 0.50, 2.2, 180, 'antigravity_auto', '', 'gemini-2.5-pro', ?);
            """,
            (now_iso,),
        )
    conn.commit()


def _repair_and_backfill_legacy_records(conn: sqlite3.Connection) -> None:
    """
    1. Repairs any historical thesis where a profitable exit (realized_return_pct > 0) was
       incorrectly marked 'CLOSED_STOP_LOSS' due to the legacy PARTIAL_TRIM_50 stop ratchet bug.
    2. Backfills existing theses into the `recommendations` table if not already present.
    """
    cur = conn.cursor()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Repair misclassified profitable theses
    cur.execute(
        """
        UPDATE theses
        SET status = 'CLOSED_PROFIT',
            actual_outcome = 'PROFITABLE EXIT (+' || ROUND(realized_return_pct, 2) || '% Realized)',
            post_mortem_notes = 'PROFITABLE SWING EXIT (+' || ROUND(realized_return_pct, 2) || '% Realized): Closed in profit above entry price.'
        WHERE status = 'CLOSED_STOP_LOSS' AND realized_return_pct > 0;
        """
    )

    # 2. Backfill theses into recommendations if recommendations table is empty
    rec_cnt = cur.execute("SELECT COUNT(*) as cnt FROM recommendations;").fetchone()["cnt"]
    if rec_cnt == 0:
        theses_rows = cur.execute("SELECT * FROM theses ORDER BY created_at ASC;").fetchall()
        for r in theses_rows:
            th = dict(r)
            entry_p = float(th["entry_price"])
            target_p = float(th["target_price"])
            stop_p = float(th["initial_stop_loss"])
            upside_pct = round(((target_p - entry_p) / entry_p) * 100.0, 2) if entry_p > 0 else 15.0
            stop_pct = round(((entry_p - stop_p) / entry_p) * 100.0, 2) if entry_p > 0 else 8.0
            rr = round(upside_pct / max(stop_pct, 0.1), 2)
            created_dt = (th.get("entry_date") or th.get("created_at", now_iso))[:10]
            deadline_dt = (th.get("time_horizon_deadline") or created_dt)[:10]
            try:
                t_days = max((datetime.fromisoformat(deadline_dt).date() - datetime.fromisoformat(created_dt).date()).days, 14)
            except Exception:
                t_days = 60
            tf_lbl = th.get("target_timeframe_label") or f"~{max(round(t_days / 30, 1), 1)} Months ({t_days} Days)"
            rec_id = f"REC-{th['thesis_id'].replace('TH-', '')}"
            cur.execute(
                """
                INSERT OR IGNORE INTO recommendations (
                    recommendation_id, created_date, created_at, ticker, company_name,
                    asset_type, sector, candle_timeframe, source, action, conviction_score,
                    entry_price, target_price, target_upside_pct, stop_loss_price, stop_loss_pct,
                    reward_to_risk, target_days, target_timeframe_label, target_date,
                    rationale_summary, thesis_id, status, current_price, current_return_pct,
                    peak_price_since_rec, trough_price_since_rec, max_favorable_excursion_pct,
                    max_adverse_excursion_pct, progress_to_target_pct, days_elapsed, days_remaining,
                    resolved_date, resolved_price, resolved_return_pct, accuracy_score,
                    evaluation_notes, last_evaluated_at
                ) VALUES (?, ?, ?, ?, ?, 'STOCK', ?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, 0.0, ?, ?, 0.0, 0.0, 0.0, 0, ?, '', 0.0, 0.0, 0.0, 'Pending live price evaluation', ?);
                """,
                (
                    rec_id,
                    created_dt,
                    th.get("created_at") or now_iso,
                    th["ticker"],
                    th["company_name"],
                    th.get("sector") or "EGX Equity",
                    th.get("candle_timeframe") or "1D",
                    th.get("ai_provider_used") or "Antigravity Hybrid Engine",
                    int(th.get("conviction_score") or 75),
                    entry_p,
                    target_p,
                    upside_pct,
                    stop_p,
                    stop_pct,
                    rr,
                    t_days,
                    tf_lbl,
                    deadline_dt,
                    th.get("thesis_summary") or "",
                    th["thesis_id"],
                    entry_p,
                    float(th.get("highest_price_since_entry") or entry_p),
                    entry_p,
                    t_days,
                    now_iso,
                ),
            )
    conn.commit()


def reset_all_data(initial_capital_egp: float = 1_000_000.0) -> Dict[str, Any]:
    """
    Clears all theses, open/closed positions, trade events, learned lessons,
    reflection logs, recommendations, uploaded disclosures, and chat history for a fresh start.
    """
    conn = get_connection()
    cur = conn.cursor()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute("DELETE FROM trade_events;")
    cur.execute("DELETE FROM positions;")
    cur.execute("DELETE FROM reflection_logs;")
    cur.execute("DELETE FROM learned_lessons;")
    cur.execute("DELETE FROM recommendations;")
    cur.execute("DELETE FROM theses;")
    cur.execute("DELETE FROM documents;")
    cur.execute("DELETE FROM chat_history;")

    cur.execute(
        """
        UPDATE portfolio_settings
        SET initial_capital_egp = ?,
            cash_balance_egp = ?,
            updated_at = ?
        WHERE id = 1;
        """,
        (float(initial_capital_egp), float(initial_capital_egp), now_iso),
    )
    conn.commit()
    conn.close()
    return get_portfolio_cockpit()


# =============================================================================
# DATA ACCESS & PORTFOLIO / THESIS STATE OPERATIONS
# =============================================================================

def get_portfolio_settings() -> Dict[str, Any]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM portfolio_settings WHERE id = 1;").fetchone()
    conn.close()
    return dict(row) if row else {}


def update_portfolio_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_connection()
    allowed_fields = {
        "base_currency",
        "initial_capital_egp",
        "cash_balance_egp",
        "max_allocation_pct",
        "take_profit_pct",
        "partial_trim_ratio",
        "atr_multiplier",
        "max_holding_days",
        "ai_mode",
        "gemini_api_key",
        "preferred_model",
    }
    filtered = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
    if filtered:
        filtered["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in filtered.keys())
        values = list(filtered.values())
        conn.execute(f"UPDATE portfolio_settings SET {set_clause} WHERE id = 1;", values)
        conn.commit()
    conn.close()
    return get_portfolio_settings()


def list_theses(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    if status_filter:
        rows = conn.execute(
            "SELECT * FROM theses WHERE status = ? ORDER BY created_at DESC;", (status_filter,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM theses ORDER BY created_at DESC;").fetchall()
    conn.close()

    results = []
    for r in rows:
        item = dict(r)
        for json_col in ("fundamental_catalysts", "technical_triggers", "invalidation_rules"):
            try:
                item[json_col] = json.loads(item.get(json_col) or "[]")
            except Exception:
                item[json_col] = []
        # Provide aliases so any frontend table expecting `stop_loss` or `actual_return_pct` works seamlessly
        item["stop_loss"] = item.get("initial_stop_loss")
        item["actual_return_pct"] = item.get("realized_return_pct")
        results.append(item)
    return results


def get_thesis_by_id(thesis_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM theses WHERE thesis_id = ?;", (thesis_id,)).fetchone()
    conn.close()
    if not row:
        return None
    item = dict(row)
    for json_col in ("fundamental_catalysts", "technical_triggers", "invalidation_rules"):
        try:
            item[json_col] = json.loads(item.get(json_col) or "[]")
        except Exception:
            item[json_col] = []
    item["stop_loss"] = item.get("initial_stop_loss")
    item["actual_return_pct"] = item.get("realized_return_pct")
    return item


def create_thesis(thesis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new Investment Thesis in Tier A Episodic Memory using Dynamic Target Price,
    Dynamic Target Timeframe/Date, and Dynamic Stop-Loss.
    """
    catalysts = thesis_data.get("fundamental_catalysts") or []
    triggers = thesis_data.get("technical_triggers") or []
    invalidations = thesis_data.get("invalidation_rules") or []

    if not catalysts or not triggers or not invalidations:
        raise ValueError(
            "Strategy Rule Violation: Every Investment Thesis MUST include explicit Fundamental Catalysts, "
            "Technical Triggers, and Invalidation Rules."
        )

    entry_price = float(thesis_data["entry_price"])
    if entry_price <= 0:
        raise ValueError("Entry price must be greater than 0.")

    atr_val = float(thesis_data.get("atr_at_entry") or round(entry_price * 0.032, 2))
    initial_stop = float(
        thesis_data.get("initial_stop_loss")
        or thesis_data.get("stop_loss_price")
        or round(entry_price - STRATEGY_RULES.atr_trailing_multiplier * atr_val, 2)
    )
    risk_per_share = max(entry_price - initial_stop, 0.01)

    # Dynamic target price (no longer hardcoded to +30%)
    if thesis_data.get("target_price") and float(thesis_data["target_price"]) > entry_price:
        target_price = round(float(thesis_data["target_price"]), 2)
    else:
        target_price = round(entry_price + max(2.2 * risk_per_share, entry_price * 0.14), 2)

    reward_per_share = max(target_price - entry_price, 0.01)
    rr_ratio = round(reward_per_share / risk_per_share, 2)

    today = datetime.now().date()
    entry_date_str = thesis_data.get("entry_date") or today.isoformat()
    try:
        entry_dt = datetime.fromisoformat(entry_date_str).date()
    except Exception:
        entry_dt = today

    target_days = int(thesis_data.get("target_days") or 60)
    target_timeframe_label = thesis_data.get("target_timeframe_label") or f"~{max(round(target_days / 30, 1), 1)} Months ({target_days} Days)"
    min_horizon_date = thesis_data.get("time_horizon_min_date") or (entry_dt + timedelta(days=max(target_days // 2, 10))).isoformat()
    deadline_date = thesis_data.get("time_horizon_deadline") or thesis_data.get("target_date") or (entry_dt + timedelta(days=target_days)).isoformat()
    candle_tf = normalize_timeframe(thesis_data.get("candle_timeframe", "1D"))

    alloc_pct = min(
        float(thesis_data.get("recommended_allocation_pct") or 0.15),
        STRATEGY_RULES.max_allocation_pct,
    )

    thesis_id = thesis_data.get("thesis_id") or f"TH-{today.year}-{uuid.uuid4().hex[:6].upper()}"
    ticker = normalize_egx_ticker(thesis_data["ticker"])
    company_name = thesis_data.get("company_name") or _lookup_company_name(ticker)
    sector = thesis_data.get("sector") or _lookup_sector(ticker)
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO theses (
            thesis_id, ticker, company_name, sector, status,
            entry_date, entry_price, target_price, time_horizon_min_date,
            time_horizon_deadline, holding_months_target, initial_stop_loss,
            current_trailing_stop, highest_price_since_entry, atr_at_entry,
            conviction_score, reward_to_risk, recommended_allocation_pct,
            fundamental_catalysts, technical_triggers, invalidation_rules,
            vision_chart_findings, chart_image_url, thesis_summary,
            post_mortem_notes, actual_outcome, realized_return_pct,
            lessons_extracted, ai_provider_used, candle_timeframe,
            target_days, target_timeframe_label, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            thesis_id,
            ticker,
            company_name,
            sector,
            thesis_data.get("status", "PROPOSED"),
            entry_dt.isoformat(),
            entry_price,
            target_price,
            min_horizon_date,
            deadline_date,
            max(1, int(round(target_days / 30))),
            initial_stop,
            float(thesis_data.get("current_trailing_stop") or initial_stop),
            float(thesis_data.get("highest_price_since_entry") or entry_price),
            atr_val,
            int(thesis_data.get("conviction_score", 80)),
            rr_ratio,
            alloc_pct,
            json.dumps(catalysts),
            json.dumps(triggers),
            json.dumps(invalidations),
            thesis_data.get("vision_chart_findings", ""),
            thesis_data.get("chart_image_url", ""),
            thesis_data.get("thesis_summary", ""),
            thesis_data.get("post_mortem_notes", ""),
            thesis_data.get("actual_outcome", "PROPOSED — Awaiting Position Execution"),
            float(thesis_data.get("realized_return_pct", 0.0)),
            int(thesis_data.get("lessons_extracted", 0)),
            thesis_data.get("ai_provider_used", "Antigravity Hybrid Engine"),
            candle_tf,
            target_days,
            target_timeframe_label,
            now_iso,
            now_iso,
        ),
    )
    conn.commit()
    conn.close()
    return get_thesis_by_id(thesis_id) or {}


def get_learned_lessons(ticker: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
    conn = get_connection()
    query = "SELECT * FROM learned_lessons"
    conditions = []
    params: List[Any] = []
    if active_only:
        conditions.append("active = 1")
    if ticker:
        norm_t = normalize_egx_ticker(ticker)
        sector = _lookup_sector(norm_t)
        conditions.append("(ticker = 'ALL' OR ticker = ? OR sector = 'ALL' OR sector = ?)")
        params.extend([norm_t, sector])
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY confidence_weight DESC, created_at DESC;"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    out = []
    for r in rows:
        d = dict(r)
        # Provide aliases for frontend compatibility
        d["rule_title"] = d.get("rule_statement", "")
        d["lesson_text"] = d.get("evidence_summary", "")
        out.append(d)
    return out


def add_learned_lesson(lesson_data: Dict[str, Any]) -> Dict[str, Any]:
    lesson_id = lesson_data.get("lesson_id") or f"LSN-EGX-{uuid.uuid4().hex[:6].upper()}"
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO learned_lessons (
            lesson_id, source_thesis_id, ticker, sector, category,
            rule_statement, evidence_summary, confidence_weight,
            win_rate_impact, times_applied, active, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            lesson_id,
            lesson_data.get("source_thesis_id", ""),
            lesson_data.get("ticker", "ALL"),
            lesson_data.get("sector", "ALL"),
            lesson_data.get("category", "ENTRY_TIMING"),
            lesson_data["rule_statement"],
            lesson_data.get("evidence_summary", ""),
            float(lesson_data.get("confidence_weight", 0.85)),
            lesson_data.get("win_rate_impact", "Learned from Reflection"),
            int(lesson_data.get("times_applied", 1)),
            1,
            now_iso,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM learned_lessons WHERE lesson_id = ?;", (lesson_id,)).fetchone()
    conn.close()
    if not row:
        return {}
    d = dict(row)
    d["rule_title"] = d.get("rule_statement", "")
    d["lesson_text"] = d.get("evidence_summary", "")
    return d


def list_reflection_logs() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM reflection_logs ORDER BY created_at DESC;").fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        try:
            d["extracted_lesson_ids"] = json.loads(d.get("extracted_lesson_ids") or "[]")
        except Exception:
            d["extracted_lesson_ids"] = []
        results.append(d)
    return results


# =============================================================================
# TIER D: RECOMMENDATIONS LEDGER & LIVE TIME/PRICE EVALUATION ENGINE
# =============================================================================

def record_recommendation(rec_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Saves a stock or index recommendation by Date, Stock/Index, Candle Timeframe, and Source.
    If the exact same (created_date, ticker, candle_timeframe, source) was already recorded today,
    updates that record in-place so repeated clicks on the same day don't spam duplicate rows,
    while preserving every distinct date, stock, timeframe, and source.
    """
    now = datetime.now()
    now_iso = now.strftime("%Y-%m-%d %H:%M:%S")
    created_date = (rec_data.get("created_date") or now.date().isoformat())[:10]
    ticker = normalize_egx_ticker(rec_data["ticker"])
    asset_type = rec_data.get("asset_type") or ("INDEX" if is_egx_index(ticker) else "STOCK")
    company_name = rec_data.get("company_name") or _lookup_company_name(ticker)
    sector = rec_data.get("sector") or _lookup_sector(ticker)
    candle_tf = normalize_timeframe(rec_data.get("candle_timeframe", "1D"))
    source = (rec_data.get("source") or "Antigravity Hybrid Engine").strip()
    action = (rec_data.get("action") or "BUY").strip().upper()
    conviction = int(rec_data.get("conviction_score") or 75)

    entry_price = round(float(rec_data["entry_price"]), 2)
    target_price = round(float(rec_data["target_price"]), 2)
    stop_loss_price = round(float(rec_data["stop_loss_price"]), 2)

    target_upside_pct = round(
        float(rec_data.get("target_upside_pct") or (((target_price - entry_price) / max(entry_price, 0.01)) * 100.0)),
        2,
    )
    stop_loss_pct = round(
        float(rec_data.get("stop_loss_pct") or (((entry_price - stop_loss_price) / max(entry_price, 0.01)) * 100.0)),
        2,
    )
    rr = round(
        float(rec_data.get("reward_to_risk") or (target_upside_pct / max(stop_loss_pct, 0.1))),
        2,
    )

    target_days = max(int(rec_data.get("target_days") or 45), 1)
    target_timeframe_label = rec_data.get("target_timeframe_label") or f"~{target_days} Days"
    try:
        base_dt = datetime.fromisoformat(created_date).date()
    except Exception:
        base_dt = now.date()
    target_date = (rec_data.get("target_date") or (base_dt + timedelta(days=target_days)).isoformat())[:10]

    rationale = rec_data.get("rationale_summary") or ""
    thesis_id = rec_data.get("thesis_id") or ""

    conn = get_connection()
    existing = conn.execute(
        """
        SELECT recommendation_id FROM recommendations
        WHERE created_date = ? AND ticker = ? AND candle_timeframe = ? AND source = ?;
        """,
        (created_date, ticker, candle_tf, source),
    ).fetchone()

    rec_id = (
        existing["recommendation_id"]
        if existing
        else (rec_data.get("recommendation_id") or f"REC-{created_date.replace('-', '')}-{uuid.uuid4().hex[:6].upper()}")
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO recommendations (
            recommendation_id, created_date, created_at, ticker, company_name,
            asset_type, sector, candle_timeframe, source, action, conviction_score,
            entry_price, target_price, target_upside_pct, stop_loss_price, stop_loss_pct,
            reward_to_risk, target_days, target_timeframe_label, target_date,
            rationale_summary, thesis_id, status, current_price, current_return_pct,
            peak_price_since_rec, trough_price_since_rec, max_favorable_excursion_pct,
            max_adverse_excursion_pct, progress_to_target_pct, days_elapsed, days_remaining,
            resolved_date, resolved_price, resolved_return_pct, accuracy_score,
            evaluation_notes, last_evaluated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, 0.0, ?, ?, 0.0, 0.0, 0.0, 0, ?, '', 0.0, 0.0, 0.0, 'Saved — awaiting time & price movement', ?);
        """,
        (
            rec_id,
            created_date,
            now_iso,
            ticker,
            company_name,
            asset_type,
            sector,
            candle_tf,
            source,
            action,
            conviction,
            entry_price,
            target_price,
            target_upside_pct,
            stop_loss_price,
            stop_loss_pct,
            rr,
            target_days,
            target_timeframe_label,
            target_date,
            rationale,
            thesis_id,
            entry_price,
            entry_price,
            entry_price,
            target_days,
            now_iso,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM recommendations WHERE recommendation_id = ?;", (rec_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}


def evaluate_recommendations() -> Dict[str, Any]:
    """
    Evaluates all saved recommendations against actual historical daily High/Low/Close bars
    and current live market prices as time passes:
    - Computes `days_elapsed`, `days_remaining`, `current_price`, `current_return_pct`
    - Computes `peak_price_since_rec` (MFE %) and `trough_price_since_rec` (MAE %)
    - Checks path-dependent resolution along daily bars after `created_date`:
      * `TARGET_HIT` (Accuracy = 100%)
      * `STOP_LOSS_HIT` (Accuracy = 0%)
      * `EXPIRED_PROFITABLE` / `EXPIRED_MISSED` (when `today > target_date`)
      * `WAIT_VALIDATED` / `WAIT_MISSED` (when `action` is WAIT / AVOID / DEFENSIVE)
      * `ACTIVE` (tracks live progress toward target and on-track accuracy score)
    """
    from backend.technical_engine import _fetch_tradingview_live_egx_bar, fetch_ohlcv_dataframe

    conn = get_connection()
    rows = conn.execute("SELECT * FROM recommendations ORDER BY created_at DESC;").fetchall()
    if not rows:
        conn.close()
        return get_recommendation_stats()

    today = datetime.now().date()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Group by ticker so we only fetch each ticker's OHLCV once per evaluation pass
    tickers = sorted({r["ticker"] for r in rows})
    ticker_bars: Dict[str, Any] = {}
    ticker_live: Dict[str, float] = {}

    for tkr in tickers:
        try:
            df_t, _ = fetch_ohlcv_dataframe(tkr, lookback_days=450, timeframe="1D")
            ticker_bars[tkr] = df_t
            if not df_t.empty:
                ticker_live[tkr] = round(float(df_t["Close"].iloc[-1]), 2)
        except Exception:
            live_b = _fetch_tradingview_live_egx_bar(tkr)
            if live_b:
                ticker_live[tkr] = live_b["close"]

    for r in rows:
        rec = dict(r)
        rec_id = rec["recommendation_id"]
        tkr = rec["ticker"]
        entry_p = max(float(rec["entry_price"]), 0.01)
        target_p = float(rec["target_price"])
        stop_p = float(rec["stop_loss_price"])
        action_upper = (rec["action"] or "BUY").upper()

        try:
            created_dt = datetime.fromisoformat(rec["created_date"][:10]).date()
        except Exception:
            created_dt = today
        try:
            target_dt = datetime.fromisoformat(rec["target_date"][:10]).date()
        except Exception:
            target_dt = created_dt + timedelta(days=int(rec.get("target_days") or 45))

        days_elapsed = max((today - created_dt).days, 0)
        days_remaining = (target_dt - today).days

        df_t = ticker_bars.get(tkr)
        cur_p = ticker_live.get(tkr, float(rec["current_price"] or entry_p))
        peak_p = max(float(rec.get("peak_price_since_rec") or entry_p), cur_p)
        trough_p = min(float(rec.get("trough_price_since_rec") or entry_p), cur_p)

        resolved_status = rec["status"]
        resolved_date = rec.get("resolved_date") or ""
        resolved_price = float(rec.get("resolved_price") or 0.0)
        resolved_ret = float(rec.get("resolved_return_pct") or 0.0)

        # Inspect daily bars strictly on or after `created_date` up to `target_date`
        if df_t is not None and not df_t.empty:
            sub_df = df_t[df_t.index.date >= created_dt]
            if not sub_df.empty:
                cur_p = round(float(sub_df["Close"].iloc[-1]), 2)
                peak_p = round(max(float(sub_df["High"].max()), entry_p, cur_p), 2)
                trough_p = round(min(float(sub_df["Low"].min()), entry_p, cur_p), 2)

                # Only scan path if not already permanently resolved
                if resolved_status not in ("TARGET_HIT", "STOP_LOSS_HIT"):
                    is_buy_call = ("BUY" in action_upper or "ACCUMULATE" in action_upper) and "WAIT" not in action_upper
                    if is_buy_call:
                        # Check subsequent bars (after entry bar, or including bars > created_dt)
                        post_entry_df = sub_df[sub_df.index.date > created_dt]
                        for bar_ts, bar_row in post_entry_df.iterrows():
                            b_date = bar_ts.date()
                            if b_date > target_dt:
                                break
                            b_high = float(bar_row["High"])
                            b_low = float(bar_row["Low"])
                            if b_high >= target_p:
                                resolved_status = "TARGET_HIT"
                                resolved_date = b_date.isoformat()
                                resolved_price = target_p
                                resolved_ret = round(((target_p - entry_p) / entry_p) * 100.0, 2)
                                break
                            if b_low <= stop_p:
                                resolved_status = "STOP_LOSS_HIT"
                                resolved_date = b_date.isoformat()
                                resolved_price = stop_p
                                resolved_ret = round(((stop_p - entry_p) / entry_p) * 100.0, 2)
                                break

                        # Also check latest live price if same day or today
                        if resolved_status not in ("TARGET_HIT", "STOP_LOSS_HIT"):
                            if cur_p >= target_p:
                                resolved_status = "TARGET_HIT"
                                resolved_date = today.isoformat()
                                resolved_price = cur_p
                                resolved_ret = round(((cur_p - entry_p) / entry_p) * 100.0, 2)
                            elif cur_p <= stop_p:
                                resolved_status = "STOP_LOSS_HIT"
                                resolved_date = today.isoformat()
                                resolved_price = cur_p
                                resolved_ret = round(((cur_p - entry_p) / entry_p) * 100.0, 2)

        cur_ret_pct = round(((cur_p - entry_p) / entry_p) * 100.0, 2)
        mfe_pct = round(((peak_p - entry_p) / entry_p) * 100.0, 2)
        mae_pct = round(((trough_p - entry_p) / entry_p) * 100.0, 2)
        target_upside = max(float(rec["target_upside_pct"]), 0.1)
        progress_pct = round(max(-100.0, min(100.0, (cur_ret_pct / target_upside) * 100.0)), 1)

        is_buy_call = ("BUY" in action_upper or "ACCUMULATE" in action_upper) and "WAIT" not in action_upper

        # Determine final evaluation status, accuracy_score, and evaluation_notes
        if resolved_status == "TARGET_HIT":
            accuracy_score = 100.0
            progress_pct = 100.0
            eval_note = f"✅ TARGET HIT: Reached {resolved_price:,.2f} ({resolved_ret:+.2f}%) on {resolved_date}."
        elif resolved_status == "STOP_LOSS_HIT":
            accuracy_score = 0.0
            eval_note = f"🛑 STOP-LOSS HIT: Breached {stop_p:,.2f} ({resolved_ret:+.2f}%) on {resolved_date}."
        elif not is_buy_call:
            # WAIT / AVOID / DEFENSIVE recommendation evaluation
            if mae_pct <= -2.5 or cur_ret_pct <= 1.0:
                resolved_status = "WAIT_VALIDATED" if days_elapsed >= 1 else "ACTIVE"
                accuracy_score = 90.0 if mae_pct <= -2.5 else 75.0
                eval_note = (
                    f"🛡️ DEFENSIVE CALL VALIDATED: Price dipped to {trough_p:,.2f} ({mae_pct:+.2f}%) / now {cur_ret_pct:+.2f}% after '{rec['action']}' warning."
                )
            elif cur_ret_pct >= target_upside:
                resolved_status = "WAIT_MISSED"
                accuracy_score = 25.0
                eval_note = f"⚠️ MISSED RALLY: Price rose {cur_ret_pct:+.2f}% despite '{rec['action']}' posture."
            else:
                resolved_status = "ACTIVE"
                accuracy_score = 70.0
                eval_note = f"⏳ WATCHING ({days_elapsed}d elapsed): Price at {cur_p:,.2f} ({cur_ret_pct:+.2f}% vs entry)."
        else:
            # BUY recommendation not yet hitting Target or Stop
            if days_remaining < 0:
                resolved_date = rec["target_date"]
                resolved_price = cur_p
                resolved_ret = cur_ret_pct
                if cur_ret_pct > 0:
                    resolved_status = "EXPIRED_PROFITABLE"
                    accuracy_score = round(min(92.0, 55.0 + max(progress_pct, 0.0) * 0.40), 1)
                    eval_note = (
                        f"⌛ TIME HORIZON ENDED IN PROFIT ({cur_ret_pct:+.2f}%): Peak reached +{mfe_pct:.2f}% (Target was +{target_upside:.1f}%)."
                    )
                else:
                    resolved_status = "EXPIRED_MISSED"
                    accuracy_score = round(max(10.0, 35.0 + cur_ret_pct * 2.0), 1)
                    eval_note = (
                        f"⌛ TIME HORIZON EXPIRED ({cur_ret_pct:+.2f}%): Target {target_p:,.2f} not reached by {rec['target_date']}."
                    )
            else:
                resolved_status = "ACTIVE"
                if cur_ret_pct > 0:
                    accuracy_score = round(min(95.0, 62.0 + max(progress_pct, 0.0) * 0.35), 1)
                    eval_note = (
                        f"📈 ON TRACK ({cur_ret_pct:+.2f}% in {days_elapsed}d | {progress_pct:.0f}% to Target {target_p:,.2f} | {days_remaining}d left)."
                    )
                elif days_elapsed == 0 and abs(cur_ret_pct) < 0.05:
                    accuracy_score = 75.0
                    eval_note = (
                        f"🆕 NEWLY SAVED: Tracking toward Target {target_p:,.2f} (+{target_upside:.1f}%) by {rec['target_date']} ({days_remaining}d left)."
                    )
                else:
                    accuracy_score = round(max(25.0, 55.0 + cur_ret_pct * 3.0), 1)
                    eval_note = (
                        f"📉 PULLBACK ({cur_ret_pct:+.2f}% in {days_elapsed}d): Holding above Stop-Loss {stop_p:,.2f} ({days_remaining}d left to {rec['target_date']})."
                    )

        conn.execute(
            """
            UPDATE recommendations
            SET status = ?,
                current_price = ?,
                current_return_pct = ?,
                peak_price_since_rec = ?,
                trough_price_since_rec = ?,
                max_favorable_excursion_pct = ?,
                max_adverse_excursion_pct = ?,
                progress_to_target_pct = ?,
                days_elapsed = ?,
                days_remaining = ?,
                resolved_date = ?,
                resolved_price = ?,
                resolved_return_pct = ?,
                accuracy_score = ?,
                evaluation_notes = ?,
                last_evaluated_at = ?
            WHERE recommendation_id = ?;
            """,
            (
                resolved_status,
                cur_p,
                cur_ret_pct,
                peak_p,
                trough_p,
                mfe_pct,
                mae_pct,
                progress_pct,
                days_elapsed,
                days_remaining,
                resolved_date,
                resolved_price,
                resolved_ret,
                accuracy_score,
                eval_note,
                now_iso,
                rec_id,
            ),
        )

    conn.commit()
    conn.close()
    return get_recommendation_stats()


def list_recommendations(
    ticker: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = get_connection()
    query = "SELECT * FROM recommendations"
    conds = []
    params: List[Any] = []
    if ticker:
        conds.append("ticker = ?")
        params.append(normalize_egx_ticker(ticker))
    if source:
        conds.append("source = ?")
        params.append(source)
    if status:
        conds.append("status = ?")
        params.append(status)
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY created_date DESC, created_at DESC;"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recommendation_stats() -> Dict[str, Any]:
    """Computes aggregate accuracy, win-rate, and source performance metrics across all saved recommendations."""
    recs = list_recommendations()
    total = len(recs)
    if total == 0:
        return {
            "total_recommendations": 0,
            "active_count": 0,
            "resolved_count": 0,
            "target_hit_count": 0,
            "stop_loss_hit_count": 0,
            "expired_profitable_count": 0,
            "wait_validated_count": 0,
            "overall_accuracy_score_pct": 0.0,
            "positive_or_validated_rate_pct": 0.0,
            "average_return_pct": 0.0,
            "average_mfe_pct": 0.0,
            "by_source": [],
            "recommendations": [],
        }

    target_hits = sum(1 for r in recs if r["status"] == "TARGET_HIT")
    stop_hits = sum(1 for r in recs if r["status"] == "STOP_LOSS_HIT")
    exp_prof = sum(1 for r in recs if r["status"] == "EXPIRED_PROFITABLE")
    wait_val = sum(1 for r in recs if r["status"] == "WAIT_VALIDATED")
    active_cnt = sum(1 for r in recs if r["status"] == "ACTIVE")
    resolved_cnt = total - active_cnt

    favorable_cnt = sum(
        1
        for r in recs
        if r["status"] in ("TARGET_HIT", "EXPIRED_PROFITABLE", "WAIT_VALIDATED")
        or (r["status"] == "ACTIVE" and float(r["current_return_pct"]) >= 0.0)
    )
    avg_acc = round(sum(float(r["accuracy_score"] or 0.0) for r in recs) / total, 1)
    avg_ret = round(sum(float(r["current_return_pct"] or 0.0) for r in recs) / total, 2)
    avg_mfe = round(sum(float(r["max_favorable_excursion_pct"] or 0.0) for r in recs) / total, 2)

    # Group by Source
    by_src_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in recs:
        by_src_map.setdefault(r["source"], []).append(r)

    by_source = []
    for src_name, items in by_src_map.items():
        cnt = len(items)
        s_acc = round(sum(float(x["accuracy_score"] or 0.0) for x in items) / cnt, 1)
        s_ret = round(sum(float(x["current_return_pct"] or 0.0) for x in items) / cnt, 2)
        s_wins = sum(1 for x in items if x["status"] in ("TARGET_HIT", "EXPIRED_PROFITABLE", "WAIT_VALIDATED") or float(x["current_return_pct"]) > 0)
        by_source.append(
            {
                "source": src_name,
                "count": cnt,
                "avg_accuracy_pct": s_acc,
                "avg_return_pct": s_ret,
                "favorable_count": s_wins,
            }
        )
    by_source.sort(key=lambda x: x["avg_accuracy_pct"], reverse=True)

    return {
        "total_recommendations": total,
        "active_count": active_cnt,
        "resolved_count": resolved_cnt,
        "target_hit_count": target_hits,
        "stop_loss_hit_count": stop_hits,
        "expired_profitable_count": exp_prof,
        "wait_validated_count": wait_val,
        "overall_accuracy_score_pct": avg_acc,
        "positive_or_validated_rate_pct": round((favorable_cnt / total) * 100.0, 1),
        "average_return_pct": avg_ret,
        "average_mfe_pct": avg_mfe,
        "by_source": by_source,
        "recommendations": recs,
    }


def _sync_open_positions_live_prices() -> None:
    """
    Automatically syncs all open positions with live TradingView Egypt prices
    so Current Price and Unrealized Profit/Loss are always up-to-date in real time.
    """
    import urllib.request

    conn = get_connection()
    open_rows = conn.execute(
        "SELECT position_id, thesis_id, ticker, current_price, highest_price, atr_value, trailing_stop_loss FROM positions WHERE status != 'CLOSED';"
    ).fetchall()
    if not open_rows:
        conn.close()
        return

    symbol_map: Dict[str, str] = {}
    for r in open_rows:
        norm_t = normalize_egx_ticker(r["ticker"])
        sym_only = norm_t.replace(".CA", "").upper()
        symbol_map[f"EGX:{sym_only}"] = norm_t

    payload = {
        "symbols": {"tickers": list(symbol_map.keys())},
        "columns": ["close", "high"],
    }
    live_quotes: Dict[str, Tuple[float, float]] = {}
    try:
        req = urllib.request.Request(
            "https://scanner.tradingview.com/egypt/scan",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            },
        )
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in data.get("data") or []:
            tv_sym = item.get("s")
            d = item.get("d") or []
            if tv_sym in symbol_map and len(d) >= 1 and d[0] is not None:
                c_price = round(float(d[0]), 2)
                h_price = round(float(d[1]), 2) if len(d) > 1 and d[1] is not None else c_price
                live_quotes[symbol_map[tv_sym]] = (c_price, h_price)
    except Exception:
        pass

    if live_quotes:
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in open_rows:
            norm_t = normalize_egx_ticker(r["ticker"])
            if norm_t in live_quotes:
                live_close, live_high = live_quotes[norm_t]
                atr_val = float(r["atr_value"] or round(live_close * 0.032, 2))
                highest = max(float(r["highest_price"]), live_close, live_high)
                candidate_stop = round(highest - STRATEGY_RULES.atr_trailing_multiplier * atr_val, 2)
                new_trailing_stop = max(float(r["trailing_stop_loss"]), candidate_stop)
                conn.execute(
                    """
                    UPDATE positions
                    SET current_price = ?, highest_price = ?, trailing_stop_loss = ?, updated_at = ?
                    WHERE position_id = ?;
                    """,
                    (live_close, highest, new_trailing_stop, now_iso, r["position_id"]),
                )
                conn.execute(
                    """
                    UPDATE theses
                    SET current_trailing_stop = ?, highest_price_since_entry = ?, updated_at = ?
                    WHERE thesis_id = ?;
                    """,
                    (new_trailing_stop, highest, now_iso, r["thesis_id"]),
                )
        conn.commit()
    conn.close()


def get_portfolio_cockpit() -> Dict[str, Any]:
    """
    Computes live Tier B Portfolio State with dynamic Target Price, dynamic Target Date countdown,
    and dynamic ATR trailing stop.
    """
    _sync_open_positions_live_prices()
    settings = get_portfolio_settings()
    conn = get_connection()
    rows = conn.execute("SELECT * FROM positions ORDER BY status ASC, entry_date DESC;").fetchall()
    conn.close()

    today = datetime.now().date()
    open_positions = []
    closed_positions = []
    total_open_market_value = 0.0
    total_open_cost_basis = 0.0
    total_unrealized_pnl = 0.0
    total_realized_pnl = 0.0

    for r in rows:
        pos = dict(r)
        entry_price = float(pos["entry_price"])
        current_price = float(pos["current_price"])
        target_price = float(pos["target_price"])
        trailing_stop = float(pos["trailing_stop_loss"])
        current_shares = int(pos["current_shares"])
        initial_shares = int(pos["initial_shares"])

        try:
            entry_dt = datetime.fromisoformat(pos["entry_date"]).date()
        except Exception:
            entry_dt = today - timedelta(days=15)

        try:
            deadline_dt = datetime.fromisoformat(pos["deadline_date"]).date()
        except Exception:
            deadline_dt = entry_dt + timedelta(days=60)

        total_planned_days = max((deadline_dt - entry_dt).days, 1)
        days_held = max((today - entry_dt).days, 0)
        days_remaining = (deadline_dt - today).days
        time_progress_pct = round(min(max((days_held / total_planned_days) * 100.0, 0.0), 100.0), 1)

        market_value_egp = round(current_shares * current_price, 2)
        active_cost_basis_egp = round(current_shares * entry_price, 2)
        unrealized_pnl_egp = round(market_value_egp - active_cost_basis_egp, 2)
        unrealized_pnl_pct = round(((current_price - entry_price) / entry_price) * 100.0, 2) if entry_price > 0 else 0.0

        target_gain_pct = round(((target_price - entry_price) / entry_price) * 100.0, 2) if entry_price > 0 else 15.0
        progress_to_tp_pct = round((unrealized_pnl_pct / max(target_gain_pct, 0.1)) * 100.0, 1)

        exit_signal = "HOLD"
        exit_signal_detail = f"Thesis intact — tracking toward Target {target_price:.2f} EGP (+{target_gain_pct:.1f}%) above trailing stop."
        if pos["status"] != "CLOSED":
            if current_price <= trailing_stop:
                exit_signal = "STOP_LOSS_HIT"
                exit_signal_detail = f"ALERT: Price ({current_price:.2f} EGP) breached Trailing Stop ({trailing_stop:.2f} EGP). Execute exit & post-mortem!"
            elif days_remaining <= 0:
                exit_signal = "TIME_HORIZON_EXPIRED"
                exit_signal_detail = f"ALERT: Target holding window expired ({days_held}d held). Evaluate exit or roll target!"
            elif current_price >= target_price and pos["status"] == "OPEN":
                exit_signal = "PARTIAL_TRIM_TARGET"
                exit_signal_detail = f"TARGET HIT (+{unrealized_pnl_pct:.1f}% >= +{target_gain_pct:.1f}%): Trim 50% ({current_shares // 2} shares) now & trail runner!"
            elif pos["status"] == "TRIMMED_50_TRAILING":
                exit_signal = "TRAILING_RUNNER_50PCT"
                exit_signal_detail = f"50% Trimmed. Trailing remaining {current_shares} shares (Stop: {trailing_stop:.2f} EGP | {days_remaining}d left)."

        pos.update(
            {
                "days_held": days_held,
                "days_remaining": days_remaining,
                "total_planned_days": total_planned_days,
                "time_progress_pct": time_progress_pct,
                "market_value_egp": market_value_egp,
                "cost_basis_egp": active_cost_basis_egp,
                "initial_cost_basis_egp": round(initial_shares * entry_price, 2),
                "unrealized_pnl_egp": unrealized_pnl_egp,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "target_gain_pct": target_gain_pct,
                "progress_to_tp_pct": progress_to_tp_pct,
                "progress_to_30pct_target": progress_to_tp_pct,  # Legacy alias
                "partial_tp_taken": pos["status"] == "TRIMMED_50_TRAILING",
                "distance_to_tp_pct": round(((target_price - current_price) / current_price) * 100.0, 2) if current_price > 0 else 0.0,
                "distance_to_stop_pct": round(((current_price - trailing_stop) / current_price) * 100.0, 2) if current_price > 0 else 0.0,
                "exit_signal": exit_signal,
                "exit_signal_detail": exit_signal_detail,
            }
        )

        total_realized_pnl += float(pos.get("realized_pnl_egp") or 0.0)
        if pos["status"] != "CLOSED":
            total_open_market_value += market_value_egp
            total_open_cost_basis += active_cost_basis_egp
            total_unrealized_pnl += unrealized_pnl_egp
            open_positions.append(pos)
        else:
            closed_positions.append(pos)

    cash_balance = float(settings.get("cash_balance_egp", 1000000.0))
    total_equity_egp = round(cash_balance + total_open_market_value, 2)
    initial_capital_egp = float(settings.get("initial_capital_egp", 1000000.0))
    total_return_egp = round(total_equity_egp - initial_capital_egp, 2)
    total_return_pct = round((total_return_egp / initial_capital_egp) * 100.0, 2) if initial_capital_egp > 0 else 0.0
    total_unrealized_pnl_pct = round((total_unrealized_pnl / total_open_cost_basis) * 100.0, 2) if total_open_cost_basis > 0 else 0.0

    sector_exposure: Dict[str, float] = {}
    for pos in open_positions:
        alloc_pct = round((pos["market_value_egp"] / total_equity_egp) * 100.0, 2) if total_equity_egp > 0 else 0.0
        pos["portfolio_weight_pct"] = alloc_pct
        pos["cap_compliant"] = alloc_pct <= (STRATEGY_RULES.max_allocation_pct * 100.0 + 0.5)
        sec = pos.get("sector") or "EGX Equity"
        sector_exposure[sec] = round(sector_exposure.get(sec, 0.0) + pos["market_value_egp"], 2)

    return {
        "settings": settings,
        "summary": {
            "base_currency": "EGP",
            "initial_capital_egp": initial_capital_egp,
            "cash_balance_egp": round(cash_balance, 2),
            "cash_weight_pct": round((cash_balance / total_equity_egp) * 100.0, 2) if total_equity_egp > 0 else 100.0,
            "open_market_value_egp": round(total_open_market_value, 2),
            "invested_value_egp": round(total_open_market_value, 2),
            "invested_weight_pct": round((total_open_market_value / total_equity_egp) * 100.0, 2) if total_equity_egp > 0 else 0.0,
            "total_equity_egp": total_equity_egp,
            "total_unrealized_pnl_egp": round(total_unrealized_pnl, 2),
            "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
            "total_realized_pnl_egp": round(total_realized_pnl, 2),
            "total_return_egp": total_return_egp,
            "total_return_pct": total_return_pct,
            "open_positions_count": len(open_positions),
            "max_single_stock_cap_pct": STRATEGY_RULES.max_allocation_pct * 100.0,
            "max_single_stock_egp": round(total_equity_egp * STRATEGY_RULES.max_allocation_pct, 2),
        },
        "sector_exposure": sector_exposure,
        "open_positions": open_positions,
        "closed_positions": closed_positions,
    }


def open_position_from_thesis(
    thesis_id: str,
    allocation_egp: Optional[float] = None,
    custom_entry_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Opens a live position in Tier B backed by a verified Tier A Investment Thesis,
    inheriting the thesis's dynamic target price, dynamic stop-loss, and dynamic target deadline.
    """
    thesis = get_thesis_by_id(thesis_id)
    if not thesis:
        raise ValueError(f"Cannot open position: Thesis '{thesis_id}' does not exist. All trades must be thesis-driven.")

    cockpit = get_portfolio_cockpit()
    total_equity = cockpit["summary"]["total_equity_egp"]
    cash_balance = cockpit["summary"]["cash_balance_egp"]
    max_allowed_egp = total_equity * STRATEGY_RULES.max_allocation_pct

    rec_alloc_pct = min(float(thesis.get("recommended_allocation_pct") or 0.15), STRATEGY_RULES.max_allocation_pct)
    desired_egp = allocation_egp if allocation_egp is not None else (total_equity * rec_alloc_pct)

    if desired_egp > max_allowed_egp * 1.001:
        raise ValueError(
            f"Risk Rule Violation: Requested allocation ({desired_egp:,.0f} EGP) exceeds strict 20% max single-stock cap ({max_allowed_egp:,.0f} EGP)."
        )
    if desired_egp > cash_balance:
        raise ValueError(
            f"Insufficient Cash: Requested {desired_egp:,.0f} EGP but available cash is {cash_balance:,.0f} EGP."
        )

    entry_price = float(custom_entry_price or thesis["entry_price"])
    shares = int(desired_egp // entry_price)
    if shares <= 0:
        raise ValueError("Allocation amount is too small to purchase at least 1 share.")

    actual_cost_egp = round(shares * entry_price, 2)
    thesis_entry = max(float(thesis["entry_price"]), 0.01)
    thesis_target_ratio = float(thesis["target_price"]) / thesis_entry
    target_price = round(entry_price * thesis_target_ratio, 2)
    atr_val = float(thesis.get("atr_at_entry") or round(entry_price * 0.032, 2))
    stop_loss = float(thesis.get("initial_stop_loss") or round(entry_price - STRATEGY_RULES.atr_trailing_multiplier * atr_val, 2))

    today = datetime.now().date()
    target_days = int(thesis.get("target_days") or 60)
    deadline_date = thesis.get("time_horizon_deadline") or (today + timedelta(days=target_days)).isoformat()
    position_id = f"POS-{today.year}-{uuid.uuid4().hex[:6].upper()}"
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO positions (
            position_id, thesis_id, ticker, company_name, sector, status,
            entry_date, deadline_date, entry_price, current_price,
            highest_price, target_price, initial_stop_loss, trailing_stop_loss,
            atr_value, initial_shares, current_shares, trimmed_shares,
            trim_price, realized_pnl_egp, exit_date, exit_price, exit_reason,
            notes, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, 0.0, '', 0.0, '', ?, ?);
        """,
        (
            position_id,
            thesis_id,
            thesis["ticker"],
            thesis["company_name"],
            thesis["sector"],
            today.isoformat(),
            deadline_date,
            entry_price,
            entry_price,
            entry_price,
            target_price,
            stop_loss,
            stop_loss,
            atr_val,
            shares,
            shares,
            "Opened via Thesis-Driven Execution Gatekeeper.",
            now_iso,
        ),
    )

    conn.execute(
        "UPDATE theses SET status = 'ACTIVE', entry_price = ?, target_price = ?, updated_at = ? WHERE thesis_id = ?;",
        (entry_price, target_price, now_iso, thesis_id),
    )
    conn.execute(
        "UPDATE portfolio_settings SET cash_balance_egp = cash_balance_egp - ?, updated_at = ? WHERE id = 1;",
        (actual_cost_egp, now_iso),
    )
    conn.execute(
        """
        INSERT INTO trade_events (event_id, position_id, thesis_id, ticker, event_type, shares, price, total_value_egp, pnl_egp, notes, created_at)
        VALUES (?, ?, ?, ?, 'ENTRY_BUY', ?, ?, ?, 0.0, 'Initial thesis-backed entry', ?);
        """,
        (f"EVT-{uuid.uuid4().hex[:8].upper()}", position_id, thesis_id, thesis["ticker"], shares, entry_price, actual_cost_egp, now_iso),
    )
    conn.commit()
    conn.close()
    return get_portfolio_cockpit()


def execute_position_action(
    position_id: str,
    action: str,
    execution_price: Optional[float] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Executes lifecycle actions on an active position:
    - 'PARTIAL_TRIM_50': Trims 50% of shares, adds proceeds to cash, and ratchets trailing stop safely below current price
    - 'FULL_EXIT': Closes 100% of remaining shares, accurately classifies outcome (never mislabeling a profitable exit as STOP_LOSS),
      and triggers Post-Mortem Self-Reflection.
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM positions WHERE position_id = ?;", (position_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Position '{position_id}' not found.")

    pos = dict(row)
    if pos["status"] == "CLOSED":
        conn.close()
        raise ValueError(f"Position '{position_id}' is already closed.")

    price = float(execution_price or pos["current_price"])
    entry_price = float(pos["entry_price"])
    current_shares = int(pos["current_shares"])
    atr_val = max(float(pos["atr_value"]), price * 0.015)
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = datetime.now().date().isoformat()

    if action == "PARTIAL_TRIM_50":
        trim_shares = max(current_shares // 2, 1)
        remaining_shares = current_shares - trim_shares
        proceeds_egp = round(trim_shares * price, 2)
        realized_gain = round(trim_shares * (price - entry_price), 2)
        new_realized_total = round(float(pos["realized_pnl_egp"]) + realized_gain, 2)

        # Safe trailing stop ratchet: never set above (price - 1.0x ATR)
        highest = max(float(pos["highest_price"]), price)
        atr_ratchet = highest - STRATEGY_RULES.atr_trailing_multiplier * atr_val
        breakeven_floor = entry_price if price > entry_price * 1.03 else float(pos["trailing_stop_loss"])
        new_trailing_stop = round(min(max(float(pos["trailing_stop_loss"]), breakeven_floor, atr_ratchet), price - 0.8 * atr_val), 2)

        conn.execute(
            """
            UPDATE positions
            SET status = 'TRIMMED_50_TRAILING',
                current_shares = ?,
                trimmed_shares = trimmed_shares + ?,
                trim_price = ?,
                current_price = ?,
                highest_price = ?,
                trailing_stop_loss = ?,
                realized_pnl_egp = ?,
                exit_reason = '50% Partial Trim executed; trailing remaining 50%',
                notes = ?,
                updated_at = ?
            WHERE position_id = ?;
            """,
            (
                remaining_shares,
                trim_shares,
                price,
                price,
                highest,
                new_trailing_stop,
                new_realized_total,
                notes or f"50% trimmed at {price:.2f} EGP (+{((price-entry_price)/entry_price)*100:.1f}%). Stop ratcheted to {new_trailing_stop:.2f} EGP.",
                now_iso,
                position_id,
            ),
        )
        conn.execute(
            "UPDATE theses SET status = 'TRIMMED_50', current_trailing_stop = ?, highest_price_since_entry = ?, updated_at = ? WHERE thesis_id = ?;",
            (new_trailing_stop, highest, now_iso, pos["thesis_id"]),
        )
        conn.execute(
            "UPDATE portfolio_settings SET cash_balance_egp = cash_balance_egp + ?, updated_at = ? WHERE id = 1;",
            (proceeds_egp, now_iso),
        )
        conn.execute(
            """
            INSERT INTO trade_events (event_id, position_id, thesis_id, ticker, event_type, shares, price, total_value_egp, pnl_egp, notes, created_at)
            VALUES (?, ?, ?, ?, 'PARTIAL_TRIM_50', ?, ?, ?, ?, '50% Partial Trim', ?);
            """,
            (f"EVT-{uuid.uuid4().hex[:8].upper()}", position_id, pos["thesis_id"], pos["ticker"], trim_shares, price, proceeds_egp, realized_gain, now_iso),
        )
        conn.commit()
        conn.close()
        return {"action": "PARTIAL_TRIM_50", "position_id": position_id, "thesis_id": pos["thesis_id"], "realized_gain_egp": realized_gain}

    else:
        # FULL_EXIT
        proceeds_egp = round(current_shares * price, 2)
        realized_gain = round(current_shares * (price - entry_price), 2)
        total_realized_pnl = round(float(pos["realized_pnl_egp"]) + realized_gain, 2)
        initial_cost = float(pos["initial_shares"]) * entry_price
        blended_return_pct = round((total_realized_pnl / initial_cost) * 100.0, 2) if initial_cost > 0 else 0.0

        target_p = float(pos["target_price"])
        if price >= target_p * 0.98 or blended_return_pct >= ((target_p - entry_price) / entry_price) * 85.0:
            thesis_status = "CLOSED_TP30"
            outcome_label = f"TARGET WIN ({blended_return_pct:+.2f}% Realized)"
        elif blended_return_pct > 0:
            thesis_status = "CLOSED_PROFIT"
            outcome_label = f"PROFITABLE EXIT ({blended_return_pct:+.2f}% Realized)"
        elif price <= float(pos["trailing_stop_loss"]) * 1.01 or blended_return_pct < 0:
            thesis_status = "CLOSED_STOP_LOSS"
            outcome_label = f"STOP-LOSS / DRAWDOWN EXIT ({blended_return_pct:+.2f}% Realized)"
        else:
            thesis_status = "CLOSED_TIME_EXPIRY"
            outcome_label = f"TIME/MANUAL EXIT ({blended_return_pct:+.2f}% Realized)"

        conn.execute(
            """
            UPDATE positions
            SET status = 'CLOSED',
                current_shares = 0,
                current_price = ?,
                realized_pnl_egp = ?,
                exit_date = ?,
                exit_price = ?,
                exit_reason = ?,
                notes = ?,
                updated_at = ?
            WHERE position_id = ?;
            """,
            (
                price,
                total_realized_pnl,
                today_str,
                price,
                notes or thesis_status,
                f"Closed at {price:.2f} EGP | Total Realized PnL: {total_realized_pnl:,.2f} EGP ({blended_return_pct:+.2f}%)",
                now_iso,
                position_id,
            ),
        )
        conn.execute(
            """
            UPDATE theses
            SET status = ?,
                actual_outcome = ?,
                realized_return_pct = ?,
                updated_at = ?
            WHERE thesis_id = ?;
            """,
            (thesis_status, outcome_label, blended_return_pct, now_iso, pos["thesis_id"]),
        )
        conn.execute(
            "UPDATE portfolio_settings SET cash_balance_egp = cash_balance_egp + ?, updated_at = ? WHERE id = 1;",
            (proceeds_egp, now_iso),
        )
        conn.execute(
            """
            INSERT INTO trade_events (event_id, position_id, thesis_id, ticker, event_type, shares, price, total_value_egp, pnl_egp, notes, created_at)
            VALUES (?, ?, ?, ?, 'FULL_EXIT', ?, ?, ?, ?, ?, ?);
            """,
            (f"EVT-{uuid.uuid4().hex[:8].upper()}", position_id, pos["thesis_id"], pos["ticker"], current_shares, price, proceeds_egp, realized_gain, outcome_label, now_iso),
        )
        conn.commit()
        conn.close()
        return {
            "action": "FULL_EXIT",
            "position_id": position_id,
            "thesis_id": pos["thesis_id"],
            "ticker": pos["ticker"],
            "blended_return_pct": blended_return_pct,
            "total_realized_pnl_egp": total_realized_pnl,
            "thesis_status": thesis_status,
        }


def update_position_live_price(ticker: str, latest_price: float, latest_atr: Optional[float] = None) -> None:
    """
    Updates open positions for `ticker` with the latest market price and ratchets the
    ATR(14) trailing stop upward whenever a new high is achieved (never downward).
    """
    norm_t = normalize_egx_ticker(ticker)
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM positions WHERE ticker = ? AND status != 'CLOSED';", (norm_t,)
    ).fetchall()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for r in rows:
        pos = dict(r)
        atr_val = float(latest_atr or pos["atr_value"])
        highest = max(float(pos["highest_price"]), float(latest_price))
        candidate_stop = round(highest - STRATEGY_RULES.atr_trailing_multiplier * atr_val, 2)
        new_trailing_stop = max(float(pos["trailing_stop_loss"]), candidate_stop)

        conn.execute(
            """
            UPDATE positions
            SET current_price = ?, highest_price = ?, atr_value = ?, trailing_stop_loss = ?, updated_at = ?
            WHERE position_id = ?;
            """,
            (round(latest_price, 2), round(highest, 2), round(atr_val, 2), new_trailing_stop, now_iso, pos["position_id"]),
        )
        conn.execute(
            """
            UPDATE theses
            SET current_trailing_stop = ?, highest_price_since_entry = ?, updated_at = ?
            WHERE thesis_id = ?;
            """,
            (new_trailing_stop, round(highest, 2), now_iso, pos["thesis_id"]),
        )

    conn.commit()
    conn.close()


def _lookup_company_name(ticker: str) -> str:
    norm_t = normalize_egx_ticker(ticker)
    idx_meta = get_index_meta(norm_t)
    if idx_meta:
        return idx_meta["full_name"]
    for item in EGX_TICKER_DIRECTORY:
        if item["ticker"] == norm_t:
            return item["name"]
    return f"{norm_t} (Egyptian Exchange)"


def _lookup_sector(ticker: str) -> str:
    norm_t = normalize_egx_ticker(ticker)
    idx_meta = get_index_meta(norm_t)
    if idx_meta:
        return idx_meta["sector"]
    for item in EGX_TICKER_DIRECTORY:
        if item["ticker"] == norm_t:
            return item["sector"]
    return "EGX Equity"


def record_direct_trade(
    ticker: str,
    side: str,
    shares: int,
    price: float,
    fees_egp: float = 0.0,
    company_name: Optional[str] = None,
    trade_date: Optional[str] = None,
    notes: str = "Logged via Screenshot (a-trade skill)",
) -> Dict[str, Any]:
    """
    Directly records a BUY or SELL trade into SQLite (`theses`, `positions`, `trade_events`, `recommendations`, and `portfolio_settings`),
    computing dynamic Target Price, dynamic Target Timeframe/Date, and dynamic Stop-Loss.
    """
    from backend.technical_engine import compute_full_technical_analysis

    norm_t = normalize_egx_ticker(ticker)
    clean_side = side.strip().upper()
    shares = int(shares)
    price = round(float(price), 2)
    fees_egp = round(float(fees_egp or 0.0), 2)

    if shares <= 0 or price <= 0:
        raise ValueError("Shares and price must both be greater than 0.")

    comp_name = company_name or _lookup_company_name(norm_t)
    sector = _lookup_sector(norm_t)
    today = datetime.now().date()
    entry_dt_str = trade_date or today.isoformat()
    try:
        entry_dt = datetime.fromisoformat(entry_dt_str).date()
    except Exception:
        entry_dt = today

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if clean_side in ("BUY", "BOT", "LONG", "PURCHASE"):
        # Compute dynamic target, stop-loss, and timeframe from live technical engine
        try:
            tech = compute_full_technical_analysis(norm_t, timeframe="1D")
            geom = tech["strategy_geometry"]
            atr_est = float(tech["indicators"]["atr_14"])
            ratio_target = float(geom["target_price"]) / max(float(geom["entry_price"]), 0.01)
            ratio_stop = float(geom["stop_loss_price"]) / max(float(geom["entry_price"]), 0.01)
            target_price = round(price * ratio_target, 2)
            stop_loss = round(price * ratio_stop, 2)
            target_days = int(geom["target_days"])
            target_tf_label = geom["target_timeframe_label"]
        except Exception:
            atr_est = round(price * 0.032, 2)
            stop_loss = round(price - 2.2 * atr_est, 2)
            target_price = round(price + 4.8 * atr_est, 2)
            target_days = 55
            target_tf_label = "~8 Weeks (55 Days)"

        target_upside_pct = round(((target_price - price) / price) * 100.0, 2)
        stop_loss_pct = round(((price - stop_loss) / price) * 100.0, 2)
        rr = round(target_upside_pct / max(stop_loss_pct, 0.1), 2)
        deadline_str = (entry_dt + timedelta(days=target_days)).isoformat()
        min_date_str = (entry_dt + timedelta(days=max(target_days // 2, 10))).isoformat()

        cockpit = get_portfolio_cockpit()
        total_equity = max(float(cockpit["summary"]["total_equity_egp"]), 1.0)
        gross_cost = round(shares * price, 2)
        total_debit = round(gross_cost + fees_egp, 2)
        alloc_pct = round(min(gross_cost / total_equity, STRATEGY_RULES.max_allocation_pct), 4)

        thesis_id = f"TH-{today.year}-{uuid.uuid4().hex[:6].upper()}"
        position_id = f"POS-{today.year}-{uuid.uuid4().hex[:6].upper()}"

        conn = get_connection()
        conn.execute(
            """
            INSERT INTO theses (
                thesis_id, ticker, company_name, sector, status,
                entry_date, entry_price, target_price, time_horizon_min_date,
                time_horizon_deadline, holding_months_target, initial_stop_loss,
                current_trailing_stop, highest_price_since_entry, atr_at_entry,
                conviction_score, reward_to_risk, recommended_allocation_pct,
                fundamental_catalysts, technical_triggers, invalidation_rules,
                vision_chart_findings, chart_image_url, thesis_summary,
                post_mortem_notes, actual_outcome, realized_return_pct,
                lessons_extracted, ai_provider_used, candle_timeframe,
                target_days, target_timeframe_label, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 85, ?, ?, ?, ?, ?, ?, '', ?, '', 'ACTIVE — Logged from Trade Entry', 0.0, 0, 'a-trade / Direct Trade Logger', '1D', ?, ?, ?, ?);
            """,
            (
                thesis_id,
                norm_t,
                comp_name,
                sector,
                entry_dt.isoformat(),
                price,
                target_price,
                min_date_str,
                deadline_str,
                max(1, int(round(target_days / 30))),
                stop_loss,
                stop_loss,
                price,
                atr_est,
                rr,
                alloc_pct,
                json.dumps([f"Direct broker execution on {comp_name} ({norm_t})", f"Dynamic swing target +{target_upside_pct:.1f}% in {target_tf_label}"]),
                json.dumps([f"Bought {shares:,} shares at {price:.2f} EGP (Total: {total_debit:,.2f} EGP incl. {fees_egp:.2f} EGP fees)"]),
                json.dumps([
                    f"Safety Stop-Loss at {stop_loss:.2f} EGP (-{stop_loss_pct:.1f}%)",
                    f"Profit Target at {target_price:.2f} EGP (+{target_upside_pct:.1f}%)",
                    f"Suggested Target Horizon: {target_tf_label} (by {deadline_str})",
                ]),
                notes,
                f"Bought {shares:,} shares of {norm_t} at {price:.2f} EGP. Target: {target_price:.2f} EGP (+{target_upside_pct:.1f}% in {target_tf_label}) | Stop-Loss: {stop_loss:.2f} EGP.",
                target_days,
                target_tf_label,
                now_iso,
                now_iso,
            ),
        )

        conn.execute(
            """
            INSERT INTO positions (
                position_id, thesis_id, ticker, company_name, sector, status,
                entry_date, deadline_date, entry_price, current_price,
                highest_price, target_price, initial_stop_loss, trailing_stop_loss,
                atr_value, initial_shares, current_shares, trimmed_shares,
                trim_price, realized_pnl_egp, exit_date, exit_price, exit_reason,
                notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, 0.0, '', 0.0, '', ?, ?);
            """,
            (
                position_id,
                thesis_id,
                norm_t,
                comp_name,
                sector,
                entry_dt.isoformat(),
                deadline_str,
                price,
                price,
                price,
                target_price,
                stop_loss,
                stop_loss,
                atr_est,
                shares,
                shares,
                f"{notes} | Fees: {fees_egp:.2f} EGP",
                now_iso,
            ),
        )

        conn.execute(
            "UPDATE portfolio_settings SET cash_balance_egp = cash_balance_egp - ?, updated_at = ? WHERE id = 1;",
            (total_debit, now_iso),
        )
        conn.execute(
            """
            INSERT INTO trade_events (event_id, position_id, thesis_id, ticker, event_type, shares, price, total_value_egp, pnl_egp, notes, created_at)
            VALUES (?, ?, ?, ?, 'SCREENSHOT_BUY', ?, ?, ?, 0.0, ?, ?);
            """,
            (f"EVT-{uuid.uuid4().hex[:8].upper()}", position_id, thesis_id, norm_t, shares, price, total_debit, f"{notes} (Fees: {fees_egp:.2f} EGP)", now_iso),
        )
        conn.commit()
        conn.close()

        # Also record in Recommendations Ledger
        record_recommendation(
            {
                "created_date": entry_dt.isoformat(),
                "ticker": norm_t,
                "company_name": comp_name,
                "asset_type": "STOCK",
                "sector": sector,
                "candle_timeframe": "1D",
                "source": "Direct Trade / a-trade Skill",
                "action": "BUY",
                "conviction_score": 85,
                "entry_price": price,
                "target_price": target_price,
                "target_upside_pct": target_upside_pct,
                "stop_loss_price": stop_loss,
                "stop_loss_pct": stop_loss_pct,
                "reward_to_risk": rr,
                "target_days": target_days,
                "target_timeframe_label": target_tf_label,
                "target_date": deadline_str,
                "rationale_summary": f"Direct BUY execution ({shares} shares @ {price:.2f} EGP). {notes}",
                "thesis_id": thesis_id,
            }
        )

        return {
            "side": "BUY",
            "ticker": norm_t,
            "company_name": comp_name,
            "shares": shares,
            "price_egp": price,
            "fees_egp": fees_egp,
            "total_amount_egp": total_debit,
            "target_price": target_price,
            "target_price_30pct": target_price,  # Backward compatibility alias
            "target_upside_pct": target_upside_pct,
            "target_timeframe_label": target_tf_label,
            "stop_loss_price": stop_loss,
            "deadline_date": deadline_str,
            "thesis_id": thesis_id,
            "position_id": position_id,
        }

    else:
        # SELL order
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM positions WHERE ticker = ? AND status != 'CLOSED' ORDER BY entry_date DESC LIMIT 1;",
            (norm_t,),
        ).fetchone()

        gross_proceeds = round(shares * price, 2)
        net_proceeds = round(max(gross_proceeds - fees_egp, 0.0), 2)

        if row:
            pos = dict(row)
            pos_id = pos["position_id"]
            th_id = pos["thesis_id"]
            cur_sh = int(pos["current_shares"])
            entry_p = float(pos["entry_price"])
            realized_gain = round(shares * (price - entry_p) - fees_egp, 2)
            new_realized = round(float(pos["realized_pnl_egp"]) + realized_gain, 2)

            if shares < cur_sh:
                rem_sh = cur_sh - shares
                conn.execute(
                    """
                    UPDATE positions
                    SET status = 'TRIMMED_50_TRAILING',
                        current_shares = ?,
                        trimmed_shares = trimmed_shares + ?,
                        trim_price = ?,
                        current_price = ?,
                        realized_pnl_egp = ?,
                        notes = ?,
                        updated_at = ?
                    WHERE position_id = ?;
                    """,
                    (rem_sh, shares, price, price, new_realized, f"Sold {shares} sh @ {price:.2f} EGP ({notes})", now_iso, pos_id),
                )
                conn.execute(
                    "UPDATE theses SET status = 'TRIMMED_50', updated_at = ? WHERE thesis_id = ?;",
                    (now_iso, th_id),
                )
                action_taken = "PARTIAL_SELL"
            else:
                init_cost = float(pos["initial_shares"]) * entry_p
                ret_pct = round((new_realized / init_cost) * 100.0, 2) if init_cost > 0 else 0.0
                th_status = "CLOSED_TP30" if price >= float(pos["target_price"]) * 0.98 else ("CLOSED_PROFIT" if ret_pct > 0 else "CLOSED_STOP_LOSS")
                conn.execute(
                    """
                    UPDATE positions
                    SET status = 'CLOSED',
                        current_shares = 0,
                        current_price = ?,
                        realized_pnl_egp = ?,
                        exit_date = ?,
                        exit_price = ?,
                        exit_reason = ?,
                        notes = ?,
                        updated_at = ?
                    WHERE position_id = ?;
                    """,
                    (price, new_realized, entry_dt.isoformat(), price, notes, f"Full Sell @ {price:.2f} EGP | Net PnL: {new_realized:+,.2f} EGP", now_iso, pos_id),
                )
                conn.execute(
                    "UPDATE theses SET status = ?, actual_outcome = ?, realized_return_pct = ?, updated_at = ? WHERE thesis_id = ?;",
                    (th_status, f"CLOSED ({ret_pct:+.2f}% Realized)", ret_pct, now_iso, th_id),
                )
                action_taken = "FULL_SELL_CLOSED"

            conn.execute(
                "UPDATE portfolio_settings SET cash_balance_egp = cash_balance_egp + ?, updated_at = ? WHERE id = 1;",
                (net_proceeds, now_iso),
            )
            conn.execute(
                """
                INSERT INTO trade_events (event_id, position_id, thesis_id, ticker, event_type, shares, price, total_value_egp, pnl_egp, notes, created_at)
                VALUES (?, ?, ?, ?, 'SCREENSHOT_SELL', ?, ?, ?, ?, ?, ?);
                """,
                (f"EVT-{uuid.uuid4().hex[:8].upper()}", pos_id, th_id, norm_t, shares, price, net_proceeds, realized_gain, f"{notes} (Fees: {fees_egp:.2f} EGP)", now_iso),
            )
            conn.commit()
            conn.close()
            return {
                "side": "SELL",
                "action_taken": action_taken,
                "ticker": norm_t,
                "company_name": comp_name,
                "shares_sold": shares,
                "price_egp": price,
                "fees_egp": fees_egp,
                "net_proceeds_egp": net_proceeds,
                "realized_pnl_egp": realized_gain,
                "position_id": pos_id,
                "thesis_id": th_id,
            }
        else:
            conn.execute(
                "UPDATE portfolio_settings SET cash_balance_egp = cash_balance_egp + ?, updated_at = ? WHERE id = 1;",
                (net_proceeds, now_iso),
            )
            conn.execute(
                """
                INSERT INTO trade_events (event_id, position_id, thesis_id, ticker, event_type, shares, price, total_value_egp, pnl_egp, notes, created_at)
                VALUES (?, 'POS-DIRECT', 'TH-DIRECT', ?, 'SCREENSHOT_SELL', ?, ?, ?, 0.0, ?, ?);
                """,
                (f"EVT-{uuid.uuid4().hex[:8].upper()}", norm_t, shares, price, net_proceeds, f"{notes} (Fees: {fees_egp:.2f} EGP)", now_iso),
            )
            conn.commit()
            conn.close()
            return {
                "side": "SELL",
                "action_taken": "STANDALONE_SELL_CREDITED",
                "ticker": norm_t,
                "company_name": comp_name,
                "shares_sold": shares,
                "price_egp": price,
                "fees_egp": fees_egp,
                "net_proceeds_egp": net_proceeds,
            }
