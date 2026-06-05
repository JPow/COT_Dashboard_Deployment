"""
COT Hypothesis Test — Trend-Following N-Day Breakout Strategy

Tests H0: No COT filter configuration improves a diversified trend-following
breakout portfolio (lookbacks 5–100 in steps of 5) across all markets.

Strategy:
  Long  : High > highest High of prior N days  → enter at N-day high
  Short : Low  < lowest Low  of prior N days   → enter at N-day low
  Stop  : 2×ATR from entry → breakeven at 1:1 R/R → trail at 2×ATR

Usage:
    from trend_following_test import run_full_tf_test
    results = run_full_tf_test(cot_df, markets)
"""

import sys
import os
import itertools
import warnings
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest_engine.data import load_cot_data, prepare_base_data
from backtest_engine.indicators import add_standard_indicators
from backtest_engine.backtester import run_backtest

# Reuse shared pieces from the ORB hypothesis test
from hypothesis_test import (
    build_cot_grid, grid_label, grid_to_dataframe,
    max_consecutive_losers, compute_aggregate_metrics,
    build_aligned_pnl_matrix, run_spa_test,
    WALK_FORWARD_WINDOWS, INITIAL_CAPITAL, RISK_PCT,
    _extract_daily_pnl, _apply_cot_filter,
)

# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------

LOOKBACKS = list(range(5, 101, 5))  # 5, 10, 15, … , 100
ATR_STOP_MULT = 2.0
ATR_TRAIL_MULT = 2.0
STOP_BUFFER = 0.01


# ===================================================================
# ATR Two-Phase Stop (for trend-following breakout)
# ===================================================================

class ATRTwoPhaseStop:
    """Phase 1: fixed stop at entry ± N×ATR.
    Phase 2: after 1:1 R/R, move to breakeven then trail with ATR.
    """

    def __init__(self, atr_stop_mult=ATR_STOP_MULT,
                 atr_trail_mult=ATR_TRAIL_MULT):
        self.atr_stop_mult = atr_stop_mult
        self.atr_trail_mult = atr_trail_mult

    def initial_stop(self, direction, entry_price, row, **_kw):
        atr = row.get("ATR", np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = abs(entry_price) * 0.02
        if direction == 1:
            return entry_price - self.atr_stop_mult * atr - STOP_BUFFER
        return entry_price + self.atr_stop_mult * atr + STOP_BUFFER

    def stop_distance(self, direction, entry_price, row, **_kw):
        atr = row.get("ATR", np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = abs(entry_price) * 0.02
        return self.atr_stop_mult * atr + STOP_BUFFER

    def update(self, direction, entry_price, stop_loss, row, bar_idx,
               entry_idx, phase=1, stop_distance=0, **_kw):
        close = row["Close"]
        high = row.get("High", close)
        low = row.get("Low", close)
        atr = row.get("ATR", np.nan)

        exit_reason = None
        exit_price = None

        if direction == 1 and low <= stop_loss:
            exit_reason = f"Stop (Phase {phase})"
            exit_price = stop_loss
        elif direction == -1 and high >= stop_loss:
            exit_reason = f"Stop (Phase {phase})"
            exit_price = stop_loss

        if exit_reason:
            return stop_loss, exit_reason, exit_price, {"phase": 1}

        if phase == 1:
            unrealised = (
                (close - entry_price) if direction == 1
                else (entry_price - close)
            )
            if stop_distance > 0 and unrealised >= stop_distance:
                phase = 2
                stop_loss = entry_price

        if phase == 2 and not pd.isna(atr) and atr > 0:
            if direction == 1:
                trail = close - self.atr_trail_mult * atr
                stop_loss = max(stop_loss, trail)
            else:
                trail = close + self.atr_trail_mult * atr
                stop_loss = min(stop_loss, trail)

        return stop_loss, None, None, {
            "phase": phase, "stop_distance": stop_distance
        }


# ===================================================================
# Data preparation — daily N-day breakout
# ===================================================================

def _prepare_market_indicators(cot_df, market, start_date, end_date):
    """Prepare base daily data with indicators and COT for one market.

    Shared across all lookbacks — computed once.
    Returns a DataFrame or empty DataFrame.
    """
    df = prepare_base_data(cot_df, market)
    if df.empty:
        return pd.DataFrame()
    df = add_standard_indicators(df, atr_period=10)
    return df


def _add_breakout_signals(df, lookback):
    """Add N-day breakout signals to a prepared DataFrame.

    Long  : High > max(High) of prior N days → enter at that N-day high
    Short : Low  < min(Low)  of prior N days → enter at that N-day low
    """
    out = df.copy()
    out["n_day_high"] = out["High"].shift(1).rolling(lookback, min_periods=lookback).max()
    out["n_day_low"] = out["Low"].shift(1).rolling(lookback, min_periods=lookback).min()

    out["signal"] = 0
    out["entry_price"] = np.nan

    long_bo = out["High"] > out["n_day_high"]
    short_bo = out["Low"] < out["n_day_low"]

    long_only = long_bo & ~short_bo
    short_only = short_bo & ~long_bo
    both = long_bo & short_bo

    out.loc[long_only, "signal"] = 1
    out.loc[long_only, "entry_price"] = out.loc[long_only, "n_day_high"]

    out.loc[short_only, "signal"] = -1
    out.loc[short_only, "entry_price"] = out.loc[short_only, "n_day_low"]

    if both.any():
        mid = (out["n_day_high"] + out["n_day_low"]) / 2
        long_both = both & (out["Open"] >= mid)
        short_both = both & ~long_both
        out.loc[long_both, "signal"] = 1
        out.loc[long_both, "entry_price"] = out.loc[long_both, "n_day_high"]
        out.loc[short_both, "signal"] = -1
        out.loc[short_both, "entry_price"] = out.loc[short_both, "n_day_low"]

    return out


# ===================================================================
# Backtest runner — one lookback, one COT config, all markets
# ===================================================================

def _run_lookback_config(indicator_cache, markets, lookback, cot_params,
                         start_date, end_date, stop_strategy):
    """Run a single (lookback, COT-config) across all markets.

    Returns (portfolio_daily_pnl, pooled_trades).
    """
    is_base = not (
        cot_params.get("cot_filter")
        or cot_params.get("cot_direction_filter")
        or cot_params.get("cot_roc_filter")
    )

    pnl_parts = []
    trade_frames = []

    for market in markets:
        if market not in indicator_cache:
            continue

        base_df = indicator_cache[market]
        data = _add_breakout_signals(base_df, lookback)

        if start_date:
            data = data[data["Date"] >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data["Date"] <= pd.Timestamp(end_date)]
        data = data.reset_index(drop=True)

        if data.empty:
            continue

        if not is_base:
            data = _apply_cot_filter(data, cot_params)

        result = run_backtest(
            data, market, stop_strategy,
            initial_capital=INITIAL_CAPITAL, risk_pct=RISK_PCT,
        )

        pnl_s = _extract_daily_pnl(data, result)
        if not pnl_s.empty:
            pnl_parts.append(pnl_s)
        if not result["trades"].empty:
            trade_frames.append(result["trades"])

    if pnl_parts:
        portfolio_pnl = (
            pd.concat(pnl_parts, axis=1).fillna(0.0).sum(axis=1).sort_index()
        )
    else:
        portfolio_pnl = pd.Series(dtype=float)

    trades_df = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames else pd.DataFrame()
    )
    return portfolio_pnl, trades_df


# ===================================================================
# Window runner — all lookbacks × all COT configs
# ===================================================================

def run_grid_for_window(cot_df, markets, cot_grid, lookbacks,
                        start_date, end_date, verbose=True):
    """Run every (lookback × COT-config) pair for one OOS window.

    Aggregates PnL across all lookbacks into a diversified TF portfolio.
    Returns dict {cot_idx: {daily_pnl, trades, label, params}}
    """
    stop_strategy = ATRTwoPhaseStop()

    if verbose:
        print("  Pre-computing indicators for all markets …")
    indicator_cache = {}
    for market in markets:
        df = _prepare_market_indicators(cot_df, market, start_date, end_date)
        if not df.empty:
            indicator_cache[market] = df
    if verbose:
        print(f"  {len(indicator_cache)} markets with data")

    n_cot = len(cot_grid)
    n_lb = len(lookbacks)

    # For each COT config, accumulate PnL across all lookbacks
    config_pnl: dict[int, list[pd.Series]] = {i: [] for i in range(n_cot)}
    config_trades: dict[int, list[pd.DataFrame]] = {i: [] for i in range(n_cot)}

    for lb_idx, lookback in enumerate(lookbacks):
        if verbose:
            print(f"  Lookback {lookback:>3}d  ({lb_idx + 1}/{n_lb})")

        # Pre-compute base breakout signals for this lookback
        base_breakout_cache = {}
        for market, base_df in indicator_cache.items():
            data = _add_breakout_signals(base_df, lookback)
            if start_date:
                data = data[data["Date"] >= pd.Timestamp(start_date)]
            if end_date:
                data = data[data["Date"] <= pd.Timestamp(end_date)]
            data = data.reset_index(drop=True)
            if not data.empty:
                base_breakout_cache[market] = data

        for cot_idx, cot_params in enumerate(cot_grid):
            is_base = not (
                cot_params.get("cot_filter")
                or cot_params.get("cot_direction_filter")
                or cot_params.get("cot_roc_filter")
            )

            pnl_parts = []
            trade_parts = []

            for market in markets:
                if market not in base_breakout_cache:
                    continue

                if is_base:
                    data = base_breakout_cache[market]
                else:
                    data = _apply_cot_filter(
                        base_breakout_cache[market], cot_params
                    )

                result = run_backtest(
                    data, market, stop_strategy,
                    initial_capital=INITIAL_CAPITAL, risk_pct=RISK_PCT,
                )

                pnl_s = _extract_daily_pnl(data, result)
                if not pnl_s.empty:
                    pnl_parts.append(pnl_s)
                if not result["trades"].empty:
                    trade_parts.append(result["trades"])

            if pnl_parts:
                lb_pnl = (
                    pd.concat(pnl_parts, axis=1).fillna(0.0)
                    .sum(axis=1).sort_index()
                )
                config_pnl[cot_idx].append(lb_pnl)
            if trade_parts:
                config_trades[cot_idx].append(
                    pd.concat(trade_parts, ignore_index=True)
                )

    # Aggregate across lookbacks: sum daily PnL
    results = {}
    for cot_idx in range(n_cot):
        pnl_list = config_pnl[cot_idx]
        if pnl_list:
            agg_pnl = (
                pd.concat(pnl_list, axis=1).fillna(0.0)
                .sum(axis=1).sort_index()
            )
        else:
            agg_pnl = pd.Series(dtype=float)

        tl = config_trades[cot_idx]
        agg_trades = (
            pd.concat(tl, ignore_index=True) if tl else pd.DataFrame()
        )

        results[cot_idx] = {
            "daily_pnl": agg_pnl,
            "trades": agg_trades,
            "label": grid_label(cot_grid[cot_idx]),
            "params": cot_grid[cot_idx],
        }

    return results


# ===================================================================
# Main entry point
# ===================================================================

def run_full_tf_test(cot_df, markets,
                     windows=None, cot_grid=None, lookbacks=None,
                     n_bootstrap=1000, verbose=True):
    """Orchestrate the full walk-forward SPA test for trend-following.

    Returns a results dict with:
      spa, metrics_df, base_pnl, variant_pnl, dates,
      cot_grid, variant_indices, per_window_best, windows, lookbacks.
    """
    windows = windows or WALK_FORWARD_WINDOWS
    cot_grid = cot_grid or build_cot_grid()
    lookbacks = lookbacks or LOOKBACKS

    if verbose:
        n_combos = len(cot_grid) * len(lookbacks)
        print(f"Trend-following test: {len(lookbacks)} lookbacks × "
              f"{len(cot_grid)} COT configs = {n_combos} combos per window")

    # ---- Run each OOS window ----
    all_window_results = []
    for w in windows:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"{w['name']}  |  OOS: {w['test_start']} → {w['test_end']}")
            print(f"{'=' * 60}")
        wr = run_grid_for_window(
            cot_df, markets, cot_grid, lookbacks,
            start_date=w["test_start"], end_date=w["test_end"],
            verbose=verbose,
        )
        all_window_results.append(wr)

    # ---- Align PnL matrix across windows ----
    if verbose:
        print(f"\n{'=' * 60}")
        print("Building aligned PnL matrix …")
    base_pnl, variant_pnl, dates, variant_indices = build_aligned_pnl_matrix(
        all_window_results
    )
    T, K = len(base_pnl), variant_pnl.shape[1]
    if verbose:
        print(f"  OOS trading days: {T},  COT variants: {K}")

    # ---- SPA test ----
    if verbose:
        print(f"\nRunning SPA test ({n_bootstrap} bootstrap replications) …")
    spa_result = run_spa_test(-base_pnl, -variant_pnl, n_bootstrap)
    if verbose:
        verdict = "REJECT H₀" if spa_result["reject_h0"] else "FAIL TO REJECT H₀"
        print(f"  p-value = {spa_result['pvalue']:.4f}  →  {verdict}")

    # ---- Per-variant aggregate metrics ----
    if verbose:
        print("\nComputing per-variant metrics …")

    # Capital base for the diversified portfolio = INITIAL_CAPITAL × n_lookbacks
    portfolio_capital = INITIAL_CAPITAL * len(lookbacks)

    def _pool(config_idx):
        tl, pl = [], []
        for wr in all_window_results:
            r = wr[config_idx]
            if not r["trades"].empty:
                tl.append(r["trades"])
            if not r["daily_pnl"].empty:
                pl.append(r["daily_pnl"])
        trades_all = pd.concat(tl, ignore_index=True) if tl else pd.DataFrame()
        pnl_all = pd.concat(pl).sort_index() if pl else pd.Series(dtype=float)
        return trades_all, pnl_all

    rows = []
    base_trades, base_pnl_s = _pool(0)
    base_m = compute_aggregate_metrics(
        base_trades, base_pnl_s, initial_capital=portfolio_capital
    )
    base_m.update({"label": "Base (No COT)", "config_idx": 0, "sharpe_diff": 0.0})
    rows.append(base_m)

    for vi in variant_indices:
        t, p = _pool(vi)
        m = compute_aggregate_metrics(t, p, initial_capital=portfolio_capital)
        m.update({
            "label": all_window_results[0][vi]["label"],
            "config_idx": vi,
            "sharpe_diff": m["sharpe"] - base_m["sharpe"],
        })
        rows.append(m)

    metrics_df = pd.DataFrame(rows).sort_values("sharpe_diff", ascending=False)

    # ---- Per-window consistency ----
    per_window_best = []
    for w_idx, wr in enumerate(all_window_results):
        best_idx, best_sharpe = 0, -np.inf
        for idx, res in wr.items():
            if res["daily_pnl"].empty:
                continue
            dr = res["daily_pnl"] / portfolio_capital
            std = dr.std()
            s = (dr.mean() / std) * np.sqrt(252) if std > 0 else 0
            if s > best_sharpe:
                best_sharpe, best_idx = s, idx
        per_window_best.append({
            "window": windows[w_idx]["name"],
            "best_config_idx": best_idx,
            "best_label": wr[best_idx]["label"],
            "best_sharpe": round(best_sharpe, 3),
        })

    if verbose:
        print("\nDone.\n")

    return {
        "spa": spa_result,
        "metrics_df": metrics_df,
        "base_pnl": base_pnl,
        "variant_pnl": variant_pnl,
        "dates": dates,
        "cot_grid": cot_grid,
        "variant_indices": variant_indices,
        "per_window_best": per_window_best,
        "all_window_results": all_window_results,
        "windows": windows,
        "lookbacks": lookbacks,
        "portfolio_capital": portfolio_capital,
    }
