"""
Lookback Robustness Analysis — Finding Safe Ranges

Maps the full performance surface (lookbacks 5–100, step 1) with block-
bootstrap confidence intervals, identifies contiguous robust ranges, and
validates them via walk-forward to avoid overfitting.

Usage:
    from lookback_robustness import (
        run_performance_surface, find_robust_ranges,
        run_walk_forward_validation,
    )
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest_engine.data import prepare_base_data
from backtest_engine.indicators import add_standard_indicators
from backtest_engine.backtester import run_backtest

from trend_following_test import (
    ATRTwoPhaseStop,
    _add_breakout_signals,
    INITIAL_CAPITAL, RISK_PCT,
)
from hypothesis_test import _extract_daily_pnl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LOOKBACKS = list(range(5, 101))          # 5, 6, 7, … , 100
MIN_RANGE_WIDTH = 5                          # consecutive viable lookbacks
MIN_CAGR_TARGET = 20.0                       # annual return target (%)
BOOTSTRAP_REPS = 1000
BLOCK_LENGTH = 20                            # block bootstrap block size (days)

WF_WINDOWS = [
    {"train_start": "2022-01-01", "train_end": "2023-06-30",
     "test_start": "2023-07-01",  "test_end": "2024-03-31",
     "name": "WF-1"},
    {"train_start": "2022-01-01", "train_end": "2024-03-31",
     "test_start": "2024-04-01",  "test_end": "2024-12-31",
     "name": "WF-2"},
    {"train_start": "2022-01-01", "train_end": "2024-12-31",
     "test_start": "2025-01-01",  "test_end": "2025-09-30",
     "name": "WF-3"},
    {"train_start": "2022-01-01", "train_end": "2025-09-30",
     "test_start": "2025-10-01",  "test_end": "2026-04-02",
     "name": "WF-4"},
]


# ===================================================================
# Indicator pre-computation (shared across lookbacks)
# ===================================================================

def _build_indicator_cache(cot_df, markets):
    """Compute base indicators once per market (no date filter)."""
    cache = {}
    for market in markets:
        df = prepare_base_data(cot_df, market)
        if df.empty:
            continue
        df = add_standard_indicators(df, atr_period=10)
        cache[market] = df
    return cache


# ===================================================================
# Single-lookback runner
# ===================================================================

def _run_single_lookback(indicator_cache, markets, lookback,
                         start_date, end_date):
    """Run one lookback across all markets.

    Returns (aggregate_daily_pnl: pd.Series, pooled_trades: pd.DataFrame).
    """
    stop = ATRTwoPhaseStop()
    pnl_parts, trade_parts = [], []

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

        result = run_backtest(
            data, market, stop,
            initial_capital=INITIAL_CAPITAL, risk_pct=RISK_PCT,
        )
        pnl_s = _extract_daily_pnl(data, result)
        if not pnl_s.empty:
            pnl_parts.append(pnl_s)
        if not result["trades"].empty:
            trade_parts.append(result["trades"])

    if pnl_parts:
        agg_pnl = (
            pd.concat(pnl_parts, axis=1).fillna(0.0)
            .sum(axis=1).sort_index()
        )
    else:
        agg_pnl = pd.Series(dtype=float)

    trades = (
        pd.concat(trade_parts, ignore_index=True)
        if trade_parts else pd.DataFrame()
    )
    return agg_pnl, trades


# ===================================================================
# Metrics from daily PnL
# ===================================================================

def _compute_metrics(daily_pnl, trades_df, capital=INITIAL_CAPITAL):
    """Sharpe, CAGR, max DD, win rate, trade count, profit factor."""
    out = {
        "sharpe": 0.0, "cagr": 0.0, "max_dd_pct": 0.0,
        "win_rate": 0.0, "trades": 0, "profit_factor": 0.0,
        "net_profit": 0.0,
    }
    if daily_pnl.empty:
        return out

    dr = daily_pnl / capital
    std = dr.std()
    if std > 0:
        out["sharpe"] = (dr.mean() / std) * np.sqrt(252)

    total_days = (daily_pnl.index[-1] - daily_pnl.index[0]).days
    years = max(total_days / 365.25, 0.01)
    final = capital + daily_pnl.sum()
    out["cagr"] = ((final / capital) ** (1 / years) - 1) * 100 if final > 0 else -100

    eq = capital + daily_pnl.cumsum()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    out["max_dd_pct"] = abs(dd.min()) * 100

    out["net_profit"] = daily_pnl.sum()

    if not trades_df.empty:
        out["trades"] = len(trades_df)
        w = trades_df[trades_df["pnl"] > 0]
        l = trades_df[trades_df["pnl"] < 0]
        out["win_rate"] = len(w) / len(trades_df) * 100
        gp = w["pnl"].sum() if not w.empty else 0
        gl = abs(l["pnl"].sum()) if not l.empty else 0
        out["profit_factor"] = gp / gl if gl > 0 else float("inf")

    return out


# ===================================================================
# Block bootstrap confidence intervals
# ===================================================================

def _block_bootstrap_ci(daily_pnl, capital=INITIAL_CAPITAL,
                        n_boot=BOOTSTRAP_REPS,
                        block_len=BLOCK_LENGTH, alpha=0.05):
    """Block-bootstrap 95% CIs for Sharpe and CAGR.

    Returns dict with sharpe_lo, sharpe_hi, cagr_lo, cagr_hi.
    """
    arr = daily_pnl.values
    n = len(arr)
    if n < block_len * 2:
        return {"sharpe_lo": np.nan, "sharpe_hi": np.nan,
                "cagr_lo": np.nan, "cagr_hi": np.nan}

    total_days = (daily_pnl.index[-1] - daily_pnl.index[0]).days
    years = max(total_days / 365.25, 0.01)

    rng = np.random.default_rng(42)
    sharpes, cagrs = [], []
    n_blocks = int(np.ceil(n / block_len))

    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        blocks = [arr[s: s + block_len] for s in starts]
        sample = np.concatenate(blocks)[:n]

        dr = sample / capital
        std = dr.std()
        s = (dr.mean() / std) * np.sqrt(252) if std > 0 else 0.0
        sharpes.append(s)

        final = capital + sample.sum()
        c = ((final / capital) ** (1 / years) - 1) * 100 if final > 0 else -100
        cagrs.append(c)

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    return {
        "sharpe_lo": float(np.percentile(sharpes, lo)),
        "sharpe_hi": float(np.percentile(sharpes, hi)),
        "cagr_lo": float(np.percentile(cagrs, lo)),
        "cagr_hi": float(np.percentile(cagrs, hi)),
    }


# ===================================================================
# Performance surface
# ===================================================================

def run_performance_surface(cot_df, markets,
                            lookbacks=None,
                            start_date="2022-01-01",
                            end_date="2026-04-02",
                            do_bootstrap=True,
                            verbose=True):
    """Run every lookback and compute metrics + bootstrap CIs.

    Returns a DataFrame indexed by lookback with columns:
        sharpe, cagr, max_dd_pct, win_rate, trades, profit_factor,
        net_profit, sharpe_lo, sharpe_hi, cagr_lo, cagr_hi, viable.
    Also returns {lookback: daily_pnl Series} for later use.
    """
    lookbacks = lookbacks or ALL_LOOKBACKS

    if verbose:
        print("Pre-computing indicators …")
    cache = _build_indicator_cache(cot_df, markets)
    if verbose:
        print(f"  {len(cache)} markets ready")

    rows = []
    pnl_store = {}
    total = len(lookbacks)

    for i, lb in enumerate(lookbacks):
        if verbose and (i % 10 == 0 or i == total - 1):
            print(f"  Lookback {lb:>3}d  ({i + 1}/{total})")

        pnl, trades = _run_single_lookback(
            cache, markets, lb, start_date, end_date
        )
        pnl_store[lb] = pnl
        m = _compute_metrics(pnl, trades)
        m["lookback"] = lb

        if do_bootstrap and not pnl.empty:
            ci = _block_bootstrap_ci(pnl)
            m.update(ci)
        else:
            m.update({"sharpe_lo": np.nan, "sharpe_hi": np.nan,
                       "cagr_lo": np.nan, "cagr_hi": np.nan})

        rows.append(m)

    df = pd.DataFrame(rows).set_index("lookback")
    df["viable"] = df["cagr_lo"] >= MIN_CAGR_TARGET

    if verbose:
        n_viable = df["viable"].sum()
        print(f"\n  Viable lookbacks (lower CI of CAGR ≥ {MIN_CAGR_TARGET}%): "
              f"{n_viable} / {total}")

    return df, pnl_store


# ===================================================================
# Robust range finder
# ===================================================================

def find_robust_ranges(surface_df, min_cagr=MIN_CAGR_TARGET,
                       min_width=MIN_RANGE_WIDTH):
    """Find contiguous stretches of viable lookbacks.

    Returns list of dicts: [{start, end, width, avg_sharpe, avg_cagr,
                             avg_max_dd, avg_trades}]
    """
    viable = surface_df["viable"].values
    lookbacks = surface_df.index.values
    ranges = []
    i = 0
    while i < len(viable):
        if viable[i]:
            j = i
            while j < len(viable) and viable[j]:
                j += 1
            width = j - i
            if width >= min_width:
                lb_start = int(lookbacks[i])
                lb_end = int(lookbacks[j - 1])
                chunk = surface_df.iloc[i:j]
                ranges.append({
                    "start": lb_start,
                    "end": lb_end,
                    "width": width,
                    "avg_sharpe": chunk["sharpe"].mean(),
                    "avg_cagr": chunk["cagr"].mean(),
                    "avg_max_dd": chunk["max_dd_pct"].mean(),
                    "avg_trades": chunk["trades"].mean(),
                })
            i = j
        else:
            i += 1
    return ranges


# ===================================================================
# Walk-forward validation
# ===================================================================

def run_walk_forward_validation(cot_df, markets,
                                windows=None,
                                lookbacks=None,
                                verbose=True):
    """Walk-forward test of range selection.

    For each window:
      Train → identify robust ranges
      Test  → compare:
        (A) Range portfolio   (equal-weight lookbacks inside ranges)
        (B) Best single LB    (highest Sharpe in training)
        (C) All-LB portfolio  (equal-weight all lookbacks)

    Returns a list of per-window result dicts + aggregate summary.
    """
    windows = windows or WF_WINDOWS
    lookbacks = lookbacks or ALL_LOOKBACKS

    if verbose:
        print("Pre-computing indicators …")
    cache = _build_indicator_cache(cot_df, markets)

    results = []

    for w in windows:
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"{w['name']}  Train → {w['train_end']}  "
                  f"Test {w['test_start']} → {w['test_end']}")
            print(f"{'=' * 50}")

        # --- Train ---
        train_metrics = {}
        train_pnl = {}
        for lb in lookbacks:
            pnl, trades = _run_single_lookback(
                cache, markets, lb, w["train_start"], w["train_end"]
            )
            m = _compute_metrics(pnl, trades)
            if not pnl.empty:
                ci = _block_bootstrap_ci(pnl)
                m.update(ci)
            else:
                m.update({"sharpe_lo": np.nan, "sharpe_hi": np.nan,
                           "cagr_lo": np.nan, "cagr_hi": np.nan})
            m["viable"] = m.get("cagr_lo", 0) >= MIN_CAGR_TARGET
            train_metrics[lb] = m
            train_pnl[lb] = pnl

        train_df = pd.DataFrame(train_metrics).T
        train_df.index.name = "lookback"

        ranges = find_robust_ranges(train_df)
        range_lbs = set()
        for r in ranges:
            range_lbs.update(range(r["start"], r["end"] + 1))
        range_lbs = sorted(range_lbs.intersection(lookbacks))

        best_train_lb = int(
            train_df["sharpe"].idxmax()
        ) if not train_df.empty else lookbacks[0]

        if verbose:
            print(f"  Train ranges: {ranges if ranges else 'NONE'}")
            print(f"  Best single LB (train Sharpe): {best_train_lb}d")

        # --- Test ---
        test_pnl = {}
        test_trades = {}
        for lb in lookbacks:
            pnl, trades = _run_single_lookback(
                cache, markets, lb, w["test_start"], w["test_end"]
            )
            test_pnl[lb] = pnl
            test_trades[lb] = trades

        def _portfolio_pnl(lb_subset):
            parts = [test_pnl[lb] for lb in lb_subset
                     if lb in test_pnl and not test_pnl[lb].empty]
            if not parts:
                return pd.Series(dtype=float)
            return (
                pd.concat(parts, axis=1).fillna(0.0).sum(axis=1).sort_index()
            )

        def _portfolio_trades(lb_subset):
            parts = [test_trades[lb] for lb in lb_subset
                     if lb in test_trades and not test_trades[lb].empty]
            return (
                pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame()
            )

        n_range = len(range_lbs)
        cap_range = INITIAL_CAPITAL * max(n_range, 1)
        cap_all = INITIAL_CAPITAL * len(lookbacks)
        cap_best = INITIAL_CAPITAL

        pnl_range = _portfolio_pnl(range_lbs) if range_lbs else pd.Series(dtype=float)
        pnl_best = test_pnl.get(best_train_lb, pd.Series(dtype=float))
        pnl_all = _portfolio_pnl(lookbacks)

        trades_range = _portfolio_trades(range_lbs)
        trades_best = test_trades.get(best_train_lb, pd.DataFrame())
        trades_all = _portfolio_trades(lookbacks)

        m_range = _compute_metrics(pnl_range, trades_range, capital=cap_range)
        m_best = _compute_metrics(pnl_best, trades_best, capital=cap_best)
        m_all = _compute_metrics(pnl_all, trades_all, capital=cap_all)

        wf_result = {
            "window": w["name"],
            "test_start": w["test_start"],
            "test_end": w["test_end"],
            "train_ranges": ranges,
            "range_lookbacks": range_lbs,
            "best_single_lb": best_train_lb,
            "portfolio_A_range": m_range,
            "portfolio_B_best": m_best,
            "portfolio_C_all": m_all,
            "pnl_range": pnl_range,
            "pnl_best": pnl_best,
            "pnl_all": pnl_all,
        }
        results.append(wf_result)

        if verbose:
            print(f"  OOS Sharpe  →  Range: {m_range['sharpe']:.2f}  "
                  f"Best-{best_train_lb}d: {m_best['sharpe']:.2f}  "
                  f"All: {m_all['sharpe']:.2f}")
            print(f"  OOS CAGR    →  Range: {m_range['cagr']:.1f}%  "
                  f"Best-{best_train_lb}d: {m_best['cagr']:.1f}%  "
                  f"All: {m_all['cagr']:.1f}%")

    return results
