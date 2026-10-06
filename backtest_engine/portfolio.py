"""Aggregate per-market backtest results into portfolio-level metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def combine_equity_curves(all_results: dict, initial_capital: float) -> list[float]:
    """Sum per-market equity changes bar-by-bar (aligned by trade index)."""
    if not all_results:
        return [initial_capital]
    curves = []
    for bundle in all_results.values():
        eq = bundle.get('results', {}).get('equity_curve', [])
        if eq:
            curves.append(eq)
    if not curves:
        return [initial_capital]
    max_len = max(len(c) for c in curves)
    combined = []
    for i in range(max_len):
        total = 0.0
        for eq in curves:
            idx = min(i, len(eq) - 1)
            total += eq[idx]
        combined.append(total)
    return combined


def aggregate_trades(all_results: dict) -> pd.DataFrame:
    frames = []
    for market, bundle in all_results.items():
        trades = bundle.get('results', {}).get('trades')
        if trades is None or trades.empty:
            continue
        t = trades.copy()
        if 'market' not in t.columns:
            t['market'] = market
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
