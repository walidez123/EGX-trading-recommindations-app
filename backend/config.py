"""
Core Strategy & System Configuration for the EGX Investment Decision Agent.
Supports:
- Dynamic Target Price, Dynamic Target Timeframe/Date, and Dynamic ATR/Structure Stop-Loss
  (replacing any rigid fixed +30% / 6-month constraint)
- Multi-Timeframe Candle Selection: Daily (1D), Weekly (1W), and Monthly (1M)
- 3+ Year EGX Index Analysis: EGX 30 (EGX:EGX30), EGX 70 EWI (EGX:EGX70EWI), and EGX 33 Shariah (EGX:SHARIAH)
- Risk Management: Dynamic ATR(14) trailing stop-loss, strict 20% max capital allocation per stock
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DISCLOSURES_DIR = DATA_DIR / "disclosures"
CSV_UPLOADS_DIR = DATA_DIR / "csv_uploads"
BRIDGE_DIR = DATA_DIR / "bridge_queue"
CHARTS_DIR = BASE_DIR / "static" / "charts"
DB_PATH = DATA_DIR / "egx_agent.db"

for directory in [DATA_DIR, DISCLOSURES_DIR, CSV_UPLOADS_DIR, BRIDGE_DIR, CHARTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StrategyRules:
    base_currency: str = "EGP"
    default_initial_capital: float = 1_000_000.0  # 1,000,000 EGP default starting cash
    min_target_days: int = 10                     # Minimum dynamic target window (days)
    max_target_days: int = 365                    # Maximum dynamic target window (days)
    min_holding_days: int = 21                    # Legacy compatibility fallback
    max_holding_days: int = 180                   # Legacy compatibility fallback
    take_profit_pct: float = 0.20                 # Legacy fallback baseline when dynamic target not supplied
    partial_trim_ratio: float = 0.50              # Optional 50% partial trim at primary target
    max_allocation_pct: float = 0.20              # Strict max 20% of total portfolio equity per single stock
    max_portfolio_risk_per_trade_pct: float = 0.025  # Max 2.5% total equity risked per trade (entry to initial stop)
    atr_period: int = 14                          # 14-bar ATR for volatility measurement
    atr_trailing_multiplier: float = 2.2          # Dynamic ATR(14) stop-loss baseline multiplier
    min_reward_to_risk: float = 1.8               # Minimum R:R target threshold


STRATEGY_RULES = StrategyRules()

TIMEFRAME_LABELS: Dict[str, str] = {
    "1D": "Daily (1D)",
    "1W": "Weekly (1W)",
    "1M": "Monthly (1M)",
}


def normalize_timeframe(raw_tf: Optional[str]) -> str:
    """
    Normalizes user timeframe input into '1D' (Daily), '1W' (Weekly), or '1M' (Monthly).
    """
    if not raw_tf:
        return "1D"
    cleaned = str(raw_tf).strip().upper()
    if cleaned in ("1W", "W", "WK", "WEEK", "WEEKLY"):
        return "1W"
    if cleaned in ("1M", "M", "MO", "MON", "MONTH", "MONTHLY"):
        return "1M"
    return "1D"


# Major Egyptian Exchange (EGX) Market Indices
EGX_INDICES_DIRECTORY: List[Dict[str, Any]] = [
    {
        "ticker": "EGX30",
        "symbol": "EGX30",
        "tv_symbol": "EGX:EGX30",
        "yf_symbol": "^CASE30",
        "name": "EGX 30 Index",
        "full_name": "EGX 30 Benchmark Index (Top 30 Large-Cap & Liquid Equities)",
        "sector": "EGX Market Index (Large-Cap)",
        "category": "Large-Cap Institutional Benchmark",
        "anchor_price": 51052.30,
    },
    {
        "ticker": "EGX70",
        "symbol": "EGX70",
        "tv_symbol": "EGX:EGX70EWI",
        "yf_symbol": "EGX70.CA",
        "name": "EGX 70 EWI Index",
        "full_name": "EGX 70 Equal Weight Index (Small & Mid-Cap Market Breadth)",
        "sector": "EGX Market Index (Small/Mid-Cap EWI)",
        "category": "Equal-Weight Retail & Mid-Cap Breadth",
        "anchor_price": 17184.30,
    },
    {
        "ticker": "EGX33",
        "symbol": "EGX33",
        "tv_symbol": "EGX:SHARIAH",
        "yf_symbol": "EGX33.CA",
        "name": "EGX 33 Shariah Index",
        "full_name": "EGX 33 Shariah Compliant Index (Top 33 Shariah-Compliant Leaders)",
        "sector": "EGX Market Index (Shariah)",
        "category": "Shariah-Compliant Benchmark",
        "anchor_price": 6037.81,
    },
]

INDEX_ALIAS_MAP: Dict[str, str] = {
    "EGX30": "EGX30",
    "EXG30": "EGX30",
    "EGX 30": "EGX30",
    "EXG 30": "EGX30",
    "EGX30.CA": "EGX30",
    "EGX:EGX30": "EGX30",
    "^CASE30": "EGX30",
    "CASE30": "EGX30",
    "EGX70": "EGX70",
    "EXG70": "EGX70",
    "EGX 70": "EGX70",
    "EXG 70": "EGX70",
    "EGX70EWI": "EGX70",
    "EGX70.CA": "EGX70",
    "EGX70EWI.CA": "EGX70",
    "EGX:EGX70EWI": "EGX70",
    "EGX:EGX70": "EGX70",
    "EGX33": "EGX33",
    "EXG33": "EGX33",
    "EGX 33": "EGX33",
    "EXG 33": "EGX33",
    "EGX33.CA": "EGX33",
    "SHARIAH": "EGX33",
    "SHARIAH.CA": "EGX33",
    "EGX:SHARIAH": "EGX33",
    "EGX:EGX33": "EGX33",
}


def is_egx_index(raw_ticker: str) -> bool:
    """Returns True if `raw_ticker` refers to EGX30, EGX70, or EGX33."""
    if not raw_ticker:
        return False
    cleaned = raw_ticker.strip().upper()
    return cleaned in INDEX_ALIAS_MAP


def get_index_meta(raw_ticker: str) -> Optional[Dict[str, Any]]:
    """Returns metadata dict from EGX_INDICES_DIRECTORY if `raw_ticker` is an EGX index."""
    if not raw_ticker:
        return None
    canonical = INDEX_ALIAS_MAP.get(raw_ticker.strip().upper())
    if not canonical:
        return None
    for item in EGX_INDICES_DIRECTORY:
        if item["ticker"] == canonical:
            return item
    return None


# Curated EGX30 & Popular Egyptian Exchange Equities Directory
EGX_TICKER_DIRECTORY: List[Dict[str, str]] = [
    {"ticker": "COMI.CA", "symbol": "COMI", "name": "Commercial International Bank (CIB)", "sector": "Banking"},
    {"ticker": "FWRY.CA", "symbol": "FWRY", "name": "Fawry for Banking Technology & Electronic Payments", "sector": "FinTech"},
    {"ticker": "TMGH.CA", "symbol": "TMGH", "name": "Talaat Moustafa Group Holding", "sector": "Real Estate"},
    {"ticker": "SWDY.CA", "symbol": "SWDY", "name": "Elsewedy Electric", "sector": "Industrials & Cables"},
    {"ticker": "ABUK.CA", "symbol": "ABUK", "name": "Abu Qir Fertilizers & Chemical Industries", "sector": "Chemicals & Export"},
    {"ticker": "MFPC.CA", "symbol": "MFPC", "name": "Misr Fertilizers Production Co (MOPCO)", "sector": "Chemicals & Export"},
    {"ticker": "HRHO.CA", "symbol": "HRHO", "name": "EFG Holding (EFG Hermes)", "sector": "Financial Services"},
    {"ticker": "ETEL.CA", "symbol": "ETEL", "name": "Telecom Egypt (WE)", "sector": "Telecommunications"},
    {"ticker": "EFID.CA", "symbol": "EFID", "name": "Edita Food Industries", "sector": "Food & Beverage"},
    {"ticker": "ORWE.CA", "symbol": "ORWE", "name": "Oriental Weavers Carpets", "sector": "Consumer & Export"},
    {"ticker": "EAST.CA", "symbol": "EAST", "name": "Eastern Company", "sector": "Consumer Staples"},
    {"ticker": "ESRS.CA", "symbol": "ESRS", "name": "Ezz Steel", "sector": "Basic Resources"},
    {"ticker": "ALCN.CA", "symbol": "ALCN", "name": "Alexandria Container & Cargo Handling", "sector": "Shipping & Export"},
    {"ticker": "AMOC.CA", "symbol": "AMOC", "name": "Alexandria Mineral Oils Company", "sector": "Energy & Petrochemicals"},
    {"ticker": "SKPC.CA", "symbol": "SKPC", "name": "Sidi Kerir Petrochemicals (SIDPEC)", "sector": "Petrochemicals"},
    {"ticker": "PHDC.CA", "symbol": "PHDC", "name": "Palm Hills Developments", "sector": "Real Estate"},
    {"ticker": "HELI.CA", "symbol": "HELI", "name": "Heliopolis Housing & Development", "sector": "Real Estate"},
    {"ticker": "GBCO.CA", "symbol": "GBCO", "name": "GB Corp (Ghabbour Auto)", "sector": "Automotive & NBFI"},
    {"ticker": "CIEB.CA", "symbol": "CIEB", "name": "Credit Agricole Egypt", "sector": "Banking"},
    {"ticker": "ADIB.CA", "symbol": "ADIB", "name": "Abu Dhabi Islamic Bank Egypt", "sector": "Banking"},
    {"ticker": "CLHO.CA", "symbol": "CLHO", "name": "Cleopatra Hospital Group", "sector": "Healthcare"},
    {"ticker": "JUFO.CA", "symbol": "JUFO", "name": "Juhayna Food Industries", "sector": "Food & Beverage"},
]


def normalize_egx_ticker(raw_ticker: str) -> str:
    """
    Normalizes user ticker input for EGX stocks and indices.
    Examples:
      'comi' -> 'COMI.CA'
      'COMI.CA' -> 'COMI.CA'
      'EGX:FWRY' -> 'FWRY.CA'
      'egx30' / 'exg30' -> 'EGX30'
      'egx70' / 'egx70ewi' -> 'EGX70'
      'egx33' / 'shariah' -> 'EGX33'
    """
    cleaned = raw_ticker.strip().upper()
    if cleaned in INDEX_ALIAS_MAP:
        return INDEX_ALIAS_MAP[cleaned]

    if ":" in cleaned:
        prefix, symbol = cleaned.split(":", 1)
        if symbol in INDEX_ALIAS_MAP:
            return INDEX_ALIAS_MAP[symbol]
        if prefix in ("EGX", "CAIRO"):
            return f"{symbol}.CA"
        cleaned = symbol

    if cleaned.endswith(".CA"):
        base_sym = cleaned[:-3]
        if base_sym in INDEX_ALIAS_MAP:
            return INDEX_ALIAS_MAP[base_sym]

    if "." not in cleaned:
        return f"{cleaned}.CA"
    return cleaned
