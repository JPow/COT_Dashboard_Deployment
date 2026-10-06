"""
Modular Backtest Engine
=======================
Shared components for strategy backtesting across all models.

Modules:
    data        – Load COT daily/weekly data and IBKR intraday cache (``data/``)
    indicators  – ATR, RSI, moving averages, N-day bands
    setups      – Setup detectors (narrowing range, inside days, COT+RSI, N-day)
    entries     – Entry filters (ORB, daily breakout, close, next_open, N-day)
    stops       – Exit / stop-management strategies
    filters     – Optional COT / RSI / MA entry filters
    backtester  – Unified backtest engine (setup → entry → stop, optional costs)
    portfolio   – Aggregate per-market results
    metrics     – Performance analytics
    charts      – Plotly visualisation helpers
    cot_rsi     – Legacy COT+RSI helpers for notebooks
"""
