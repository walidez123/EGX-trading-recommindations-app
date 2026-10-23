"""
Tier C: Native Multimodal PDF & Financial Disclosure Store.
Ingests company earnings releases, board reports, and EGX disclosures.
Preserves full PDF files on disk for direct Gemini / Antigravity multimodal context ingestion
while indexing extracted text and key financial metrics in SQLite.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

from backend.config import DISCLOSURES_DIR, normalize_egx_ticker
from backend.database import get_connection


def ingest_disclosure_document(
    ticker: str,
    title: str,
    doc_type: str = "EARNINGS_RELEASE",
    fiscal_period: str = "FY2025/2026",
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    raw_text_input: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stores an uploaded PDF or text disclosure in `data/disclosures/<TICKER>/`
    and records extracted text + key financial ratios in SQLite `documents`.
    """
    norm_ticker = normalize_egx_ticker(ticker)
    ticker_dir = DISCLOSURES_DIR / norm_ticker.replace(".", "_")
    ticker_dir.mkdir(parents=True, exist_ok=True)

    saved_path_str = ""
    extracted_text = raw_text_input or ""

    if file_bytes and filename:
        safe_fname = re.sub(r"[^a-zA-Z0-9_.\-]", "_", filename)
        target_path = ticker_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_fname}"
        target_path.write_bytes(file_bytes)
        saved_path_str = str(target_path)

        if safe_fname.lower().endswith(".pdf"):
            try:
                reader = PdfReader(str(target_path))
                pages_txt = []
                for idx, page in enumerate(reader.pages[:40]):
                    t = page.extract_text() or ""
                    if t.strip():
                        pages_txt.append(f"--- [Page {idx + 1}] ---\n{t.strip()}")
                extracted_text = "\n\n".join(pages_txt)
            except Exception as exc:
                extracted_text = f"[PDF saved for native multimodal pass: {target_path.name}. Text extraction note: {exc}]"
        else:
            try:
                extracted_text = file_bytes.decode("utf-8", errors="replace")
            except Exception:
                extracted_text = ""

    metrics = _extract_heuristic_financial_metrics(extracted_text)
    summary = _build_fundamental_summary(norm_ticker, title, fiscal_period, extracted_text, metrics)

    doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO documents (
            doc_id, ticker, doc_type, title, fiscal_period,
            file_path, extracted_text, key_metrics_json, ai_summary, uploaded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            doc_id,
            norm_ticker,
            doc_type,
            title,
            fiscal_period,
            saved_path_str,
            extracted_text[:65000],
            json.dumps(metrics),
            summary,
            now_iso,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM documents WHERE doc_id = ?;", (doc_id,)).fetchone()
    conn.close()
    res = dict(row) if row else {}
    try:
        res["key_metrics"] = json.loads(res.get("key_metrics_json") or "{}")
    except Exception:
        res["key_metrics"] = {}
    return res


def list_documents(ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    if ticker:
        norm_t = normalize_egx_ticker(ticker)
        rows = conn.execute(
            "SELECT * FROM documents WHERE ticker = ? ORDER BY uploaded_at DESC;", (norm_t,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC;").fetchall()
    conn.close()

    out = []
    for r in rows:
        item = dict(r)
        try:
            item["key_metrics"] = json.loads(item.get("key_metrics_json") or "{}")
        except Exception:
            item["key_metrics"] = {}
        out.append(item)
    return out


def _extract_heuristic_financial_metrics(text: str) -> Dict[str, Any]:
    """Extracts common fundamental indicators (P/E, ROE, growth, export share) from disclosure text."""
    metrics: Dict[str, Any] = {"fundamental_health_score": 82}
    if not text:
        return metrics

    pe_match = re.search(r"(?:P/E|PE|price[- ]to[- ]earnings)\s*(?:ratio|of|:)?\s*(\d+(?:\.\d+)?)\s*x?", text, re.IGNORECASE)
    if pe_match:
        metrics["pe_ratio"] = float(pe_match.group(1))

    roe_match = re.search(r"(?:ROE|ROAE|return on (?:average )?equity)\s*(?:of|:|reached)?\s*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if roe_match:
        metrics["roe"] = f"{roe_match.group(1)}%"

    rev_match = re.search(r"(?:revenue|sales|top[- ]line|income)\s+(?:growth|grew|increased|surged)\s+(?:by|of)?\s*\+?(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if rev_match:
        metrics["revenue_growth_yoy"] = f"+{rev_match.group(1)}%"
        metrics["fundamental_health_score"] = 89

    export_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:export|USD|hard[- ]currency)", text, re.IGNORECASE)
    if export_match:
        metrics["fx_export_revenue_share"] = f"{export_match.group(1)}%"
        metrics["fundamental_health_score"] = 92

    return metrics


def _build_fundamental_summary(
    ticker: str, title: str, fiscal_period: str, text: str, metrics: Dict[str, Any]
) -> str:
    if not text:
        return f"{title} ({fiscal_period}) uploaded for {ticker} multimodal context."
    snippet = " ".join(text.split())[:360]
    metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items() if k != "fundamental_health_score")
    if metrics_str:
        return f"[{fiscal_period}] Key Extracted Metrics ({metrics_str}). Excerpt: {snippet}..."
    return f"[{fiscal_period}] Excerpt: {snippet}..."
