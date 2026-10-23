# EGX AlphaCore v3.0 — 3-Year Index Analyzer, Dynamic Target/Time/Stop Advisor & Recommendation Evaluator

A local-first, production-grade **Investment & Index Decision Agent** tailored for the **Egyptian Exchange (EGX)** featuring:
- **3+ Year EGX Market Index Analysis**: Full multi-year historical analysis (`1W`, `1M`, `3M`, `6M`, `YTD`, `1Y`, `3Y` return %, `3Y CAGR`, `3Y Max Drawdown`, `Annualized Volatility`, `3Y Range`, and relative strength) for **EGX 30** (`EGX:EGX30`), **EGX 70 EWI** (`EGX:EGX70EWI`), and **EGX 33 Shariah** (`EGX:SHARIAH`).
- **Multi-Timeframe Candle Selection (`1D` / `1W` / `1M`)**: Analyze any stock or index on **Daily (`1D`)**, **Weekly (`1W`)**, or **Monthly (`1M`)** candles.
- **Dynamic Target Price, Target Timeframe/Date & Stop-Loss**: Replaces rigid fixed rules with data-driven targets and holding horizons calculated from structural resistance/support, Fibonacci extensions, Volume Point of Control (VPOC), and `ATR(14)` velocity.
- **Recommendations Ledger & Automated Live Evaluation**: Every recommendation is saved by **Date**, **Stock/Index**, **Candle Timeframe**, and **Source**, and automatically evaluated against subsequent daily High/Low bars and live prices (`TARGET_HIT`, `STOP_LOSS_HIT`, `EXPIRED_PROFITABLE`, `WAIT_VALIDATED`, `ACTIVE`, MFE/MAE, and Source Accuracy Leaderboard).

---

## 🏗️ Project Architecture & Directory Structure

```text
EGX project A/
├── app.py                      # FastAPI backend server & REST API endpoints (v3.0)
├── index.html                  # Interactive 5-Tab Dashboard (Tailwind + TradingView Lightweight Charts)
├── requirements.txt            # Python 3.11+ package dependencies
├── run.bat                     # 1-Click Windows launcher
├── .agent/
│   └── skills/
│       ├── analyze/
│       │   ├── SKILL.md        # 1-Command Stock & Index Analysis Skill (supports 1D / 1W / 1M)
│       │   └── scripts/run_analyze.py
│       └── a-trade/
│           ├── SKILL.md        # Screenshot Trade Extractor & Direct DB Logger
│           └── scripts/save_trade.py
├── backend/
│   ├── __init__.py
│   ├── config.py               # Strategy rules, EGX30/70/33 index directory & 1D/1W/1M timeframe helpers
│   ├── database.py             # 4-Tier SQLite Cumulative Memory & Recommendations Evaluation Engine
│   ├── technical_engine.py     # TradingView 3.5Y feed, pure-math indicators & dynamic target/time/stop engine
│   ├── chart_generator.py      # High-DPI mplfinance Candlestick + Volume Profile + RSI PNG generator
│   ├── document_store.py       # Tier C Native PDF & Financial Disclosure ingestion engine
│   └── ai_agent.py             # 4-Node StateGraph Orchestrator & Automated Self-Reflection Loop
├── static/
│   └── charts/                 # Generated mplfinance chart PNGs for Vision synthesis & UI preview
└── data/
    ├── egx_agent.db            # Persistent SQLite database (WAL mode)
    ├── disclosures/            # Uploaded company earnings & board report PDFs per ticker
    ├── csv_uploads/            # Custom TradingView/Mubasher OHLCV CSV overrides
    └── bridge_queue/           # Live JSON+PNG packets for Antigravity IDE Zero-Key Agent inspection
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Start Server & Open Dashboard
Double-click `run.bat` or run:
```powershell
python app.py
```
Then open **`http://127.0.0.1:8000`** in your browser.

### 3. CLI / Skill Commands
```powershell
# Analyze any EGX stock or index on Daily (1D), Weekly (1W), or Monthly (1M) candles:
python ".agent/skills/analyze/scripts/run_analyze.py" EGX30 1D
python ".agent/skills/analyze/scripts/run_analyze.py" EGX70 1W
python ".agent/skills/analyze/scripts/run_analyze.py" EGX33 1M
python ".agent/skills/analyze/scripts/run_analyze.py" COMI.CA 1D
```
