@echo off
title EGX AlphaCore - Swing Investment Decision Agent (3-6M / +30% TP)
echo ============================================================================
echo   EGX AlphaCore: Cumulative Memory ^& Hybrid Vision Swing Decision Agent
echo   Strategy: 3-6M Horizon ^| +30%% TP (50%% Trim) ^| 2.5x ATR Stop ^| Max 20%% Cap
echo ============================================================================
echo.
echo Starting FastAPI + SQLite + mplfinance Vision Backend on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
python app.py
pause
