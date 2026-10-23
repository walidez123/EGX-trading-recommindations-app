---
name: a-trade
description: >-
  Use this skill when the user invokes "a trade" (/a-trade) or sends a screenshot of a stock trade/order receipt.
  Extracts whether the trade is a BUY or SELL, which EGX company/ticker, how many shares, execution price per share,
  and any extra fees/commission, then saves the trade directly into the SQLite database (data/egx_agent.db) with
  a dynamic Target Price, Suggested Target Timeframe/Date, and Dynamic Stop-Loss.
---

# `a-trade` — Screenshot Trade Extractor & Direct Database Logger

When the user sends a screenshot of a trade (from Thndr, EFG Hermes, Mubasher, or any brokerage app) or invokes `a-trade`:

## Step 1: Extract Trade Data from the Screenshot
Inspect the image carefully and extract:
1. **Side (`BUY` or `SELL`)**: Is it a buy/purchase order or a sell order?
2. **Company / Ticker**: Map the company name or symbol to its EGX ticker (e.g., CIB -> `COMI.CA`, Elsewedy -> `SWDY.CA`, Fawry -> `FWRY.CA`, Talaat Moustafa -> `TMGH.CA`, Abu Qir -> `ABUK.CA`, MOPCO -> `MFPC.CA`, EFG -> `HRHO.CA`, Telecom Egypt -> `ETEL.CA`, Ezz Steel -> `ESRS.CA`).
3. **Quantity (`shares`)**: Number of shares bought or sold. (If only Total Amount and Price are shown, compute `int(total_amount // price)`).
4. **Price per Share (`price`)**: Execution price per share in EGP.
5. **Extra Fees / Commission (`fees`)**: Any broker commission, regulatory fees, or difference between `Total Paid` and `(Shares * Price)` in EGP (default `0.0` if none shown).

## Step 2: Save Directly to the Database (`data/egx_agent.db`)
Run the helper script [save_trade.py](./scripts/save_trade.py) using `run_command`:

```powershell
python ".agent/skills/a-trade/scripts/save_trade.py" --ticker "<TICKER>" --side "<BUY_OR_SELL>" --shares <SHARES> --price <PRICE> --fees <FEES_EGP> --company "<COMPANY_NAME>"
```

## Step 3: Confirm in Simple, Beginner-Friendly Language
Report the saved result clearly to the user:
- **Action Recorded:** `BUY` or `SELL`
- **Company & Shares:** `<SHARES>` shares of `<COMPANY>` (`<TICKER>`) at `<PRICE> EGP`
- **Extra Fees / Total Amount:** `<FEES> EGP` extra fees (`<TOTAL> EGP` total)
- **If BUY:** Show their automatic **Dynamic Target Price (`+X.X%`)**, **Suggested Target Timeframe & Target Date**, and **Dynamic Safety Stop-Loss Price**.
- **If SELL:** Show their **Realized Profit/Loss (EGP)** and updated **Available Cash Balance (EGP)**.
