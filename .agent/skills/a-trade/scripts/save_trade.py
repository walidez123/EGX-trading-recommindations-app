"""
Helper script for the `a-trade` skill.
Saves an extracted BUY or SELL trade from a screenshot directly into `data/egx_agent.db`.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))

from backend.database import get_portfolio_cockpit, init_db, record_direct_trade


def main() -> None:
    parser = argparse.ArgumentParser(description="Save trade directly into EGX SQLite DB.")
    parser.add_argument("--ticker", required=True, help="EGX Ticker symbol (e.g. COMI.CA or COMI)")
    parser.add_argument("--side", required=True, choices=["BUY", "SELL", "buy", "sell"], help="BUY or SELL")
    parser.add_argument("--shares", required=True, type=int, help="Number of shares")
    parser.add_argument("--price", required=True, type=float, help="Execution price per share in EGP")
    parser.add_argument("--fees", default=0.0, type=float, help="Extra broker fees/commission in EGP")
    parser.add_argument("--company", default="", help="Optional company name")
    parser.add_argument("--date", default="", help="Optional trade date YYYY-MM-DD")
    parser.add_argument("--notes", default="Logged via a-trade screenshot skill", help="Trade notes")
    args = parser.parse_args()

    init_db()
    trade_result = record_direct_trade(
        ticker=args.ticker,
        side=args.side.upper(),
        shares=args.shares,
        price=args.price,
        fees_egp=args.fees,
        company_name=args.company or None,
        trade_date=args.date or None,
        notes=args.notes,
    )
    cockpit = get_portfolio_cockpit()

    output = {
        "status": "SAVED_TO_DB",
        "trade": trade_result,
        "updated_cash_balance_egp": cockpit["summary"]["cash_balance_egp"],
        "total_portfolio_equity_egp": cockpit["summary"]["total_equity_egp"],
        "open_positions_count": cockpit["summary"]["open_positions_count"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
