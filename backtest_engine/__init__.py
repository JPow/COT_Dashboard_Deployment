"""
Modular Backtest Engine
=======================
Shared components for strategy backtesting across all models.

Modules:
    data        – Load COT daily/weekly data and IBKR intraday cache
    indicators  – ATR, RSI, moving averages
    setups      – Setup detectors (narrowing range, inside days, COT+RSI extremes)
    entries     – Entry filters (ORB breakout, daily breakout, market-on-close)
    stops       – Exit / stop-management strategies
    backtester  – Unified backtest engine that wires setup → entry → stop
    metrics     – Performance analytics (win rate, Sharpe, drawdown, …)
    charts      – Plotly visualisation helpers
"""
