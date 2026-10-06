"""
COT Commercial Index Hypothesis Test — Hansen's SPA Framework

Tests H0: No COT filter configuration improves the risk-adjusted performance
of the NR3 + ORB base strategy.  Uses walk-forward OOS periods with multiple-
comparison correction via Hansen's Superior Predictive Ability (SPA) test.

Performance: pre-computes breakout signals once per market per window, then
applies vectorised COT filters — avoids redundant indicator/intraday work.

Usage:
    from hypothesis_test import run_full_hypothesis_test, build_cot_grid
    results = run_full_hypothesis_test(cot_df, markets)
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

from backtest_engine.data import load_cot_data, load_intraday_cache
from backtest_engine.backtester import prepare_data, run_backtest
from backtest_engine.stops import STOP_REGISTRY
from backtest_engine.metrics import calculate_performance_metrics

# ---------------------------------------------------------------------------
# Walk-forward windows — non-overlapping OOS periods, expanding training
# ---------------------------------------------------------------------------

WALK_FORWARD_WINDOWS = [
    {
        "train_start": "2022-01-01", "train_end": "2023-12-31",
        "test_start": "2024-01-01", "test_end": "2024-09-30",
        "name": "Window 1 (Jan–Sep 2024)",
    },
    {
        "train_start": "2022-01-01", "train_end": "2024-09-30",
        "test_start": "2024-10-01", "test_end": "2025-06-30",
        "name": "Window 2 (Oct 2024–Jun 2025)",
    },
    {
        "train_start": "2022-01-01", "train_end": "2025-06-30",
        "test_start": "2025-07-01", "test_end": "2026-04-02",
        "name": "Window 3 (Jul 2025–Apr 2026)",
    },
]

# ---------------------------------------------------------------------------
# Fixed strategy parameters — NR3 + ORB 60 min + Two-Phase ATR trail
# ---------------------------------------------------------------------------

BASE_SETUP_KEY = "narrowing_range"
BASE_SETUP_PARAMS = {"n_days": 3}
BASE_ENTRY_KEY = "orb_breakout"
BASE_STOP_KEY = "two_phase_atr"
BASE_STOP_PARAMS = {"trailing_atr_mult": 2.0}
INITIAL_CAPITAL = 30_000
RISK_PCT = 1.0

# Entry params for the base config (all COT off)
_BASE_ENTRY_PARAMS = {
    "or_type": "60m",
    "rsi_filter": False,
    "cot_filter": False,
    "cot_direction_filter": False,
    "cot_roc_filter": False,
    "cot_roc_threshold": 10,
    "cot_long": 70,
    "cot_short": 30,
}


# ===================================================================
# COT permutation grid
# ===================================================================

def build_cot_grid():
    """Return the 50-element COT filter permutation grid.

    Index 0 is always the base configuration (all COT filters off).
    Indices 1–49 are the 49 COT-enhanced variants.
    """
    cot_level_options = [
        {"cot_filter": False, "cot_long": 70, "cot_short": 30},
        {"cot_filter": True,  "cot_long": 60, "cot_short": 40},
        {"cot_filter": True,  "cot_long": 70, "cot_short": 30},
        {"cot_filter": True,  "cot_long": 80, "cot_short": 20},
        {"cot_filter": True,  "cot_long": 90, "cot_short": 10},
    ]
    cot_direction_options = [
        {"cot_direction_filter": False},
        {"cot_direction_filter": True},
    ]
    cot_roc_options = [
        {"cot_roc_filter": False, "cot_roc_threshold": 10},
        {"cot_roc_filter": True,  "cot_roc_threshold": 5},
        {"cot_roc_filter": True,  "cot_roc_threshold": 10},
        {"cot_roc_filter": True,  "cot_roc_threshold": 15},
        {"cot_roc_filter": True,  "cot_roc_threshold": 20},
    ]

    grid = []
    for level, direction, roc in itertools.product(
        cot_level_options, cot_direction_options, cot_roc_options
    ):
        entry_params = {
            "or_type": "60m",
            "rsi_filter": False,
            **level, **direction, **roc,
        }
        grid.append(entry_params)

    return grid


def grid_label(params):
    """Human-readable label for a grid configuration."""
    parts = []
    if params.get("cot_filter"):
        parts.append(f"Lvl {params['cot_long']}/{params['cot_short']}")
    if params.get("cot_direction_filter"):
        parts.append("Dir")
    if params.get("cot_roc_filter"):
        parts.append(f"ROC≥{params['cot_roc_threshold']}")
    return " + ".join(parts) if parts else "Base (No COT)"


def grid_to_dataframe(grid):
    """Convert the grid to a readable DataFrame for display."""
    rows = []
    for idx, p in enumerate(grid):
        rows.append({
            "idx": idx,
            "label": grid_label(p),
            "cot_level": f"{p['cot_long']}/{p['cot_short']}" if p.get("cot_filter") else "Off",
            "cot_direction": "On" if p.get("cot_direction_filter") else "Off",
            "cot_roc": str(p["cot_roc_threshold"]) if p.get("cot_roc_filter") else "Off",
        })
    return pd.DataFrame(rows)


# ===================================================================
# Metric helpers
# ===================================================================

def max_consecutive_losers(trades_df):
    """Longest streak of consecutive losing trades."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0
    losers = (trades_df["pnl"] < 0).values
    best, cur = 0, 0
    for is_loss in losers:
        cur = cur + 1 if is_loss else 0
        best = max(best, cur)
    return best


def compute_aggregate_metrics(trades_df, daily_pnl,
                              initial_capital=INITIAL_CAPITAL):
    """Portfolio-level metrics from pooled trades and daily PnL."""
    out = {
        "total_trades": 0, "sharpe": 0.0, "max_dd_pct": 0.0,
        "max_consec_losers": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "net_profit": 0.0, "return_pct": 0.0,
    }
    if trades_df.empty:
        return out

    winning = trades_df[trades_df["pnl"] > 0]
    losing  = trades_df[trades_df["pnl"] < 0]

    out["total_trades"] = len(trades_df)
    out["win_rate"] = len(winning) / len(trades_df) * 100

    gp = winning["pnl"].sum() if not winning.empty else 0.0
    gl = abs(losing["pnl"].sum()) if not losing.empty else 0.0
    out["profit_factor"] = gp / gl if gl > 0 else float("inf")
    out["net_profit"] = gp - gl
    out["return_pct"] = out["net_profit"] / initial_capital * 100

    if len(daily_pnl) > 1:
        dr = daily_pnl / initial_capital
        std = dr.std()
        if std > 0:
            out["sharpe"] = (dr.mean() / std) * np.sqrt(252)

    cum = daily_pnl.cumsum()
    eq  = initial_capital + cum
    peak = eq.cummax()
    dd = (eq - peak) / peak
    out["max_dd_pct"] = abs(dd.min()) * 100 if len(dd) > 0 else 0.0

    out["max_consec_losers"] = max_consecutive_losers(trades_df)
    return out


# ===================================================================
# Pre-computation & vectorised COT filter
# ===================================================================

def _precompute_base_signals(cot_df, markets, start_date, end_date,
                             intraday_cache=None):
    """Run the base strategy (no COT filter) once per market.

    Returns dict {market: DataFrame} with columns including signal,
    entry_price, or_high_signal, or_low_signal, Commercial_Index, etc.
    The expensive intraday-cache lookup happens here only once.
    """
    cache = {}
    for market in markets:
        data = prepare_data(
            cot_df, market, BASE_SETUP_KEY, BASE_ENTRY_KEY,
            start_date=start_date, end_date=end_date,
            setup_params=BASE_SETUP_PARAMS,
            entry_params=_BASE_ENTRY_PARAMS,
            intraday_cache=intraday_cache,
        )
        if not data.empty:
            cache[market] = data
    return cache


def _apply_cot_filter(data, params):
    """Vectorised COT filter on pre-computed breakout signals.

    Returns a copy with blocked signals zeroed out.
    """
    out = data.copy()
    sig = out["signal"]
    has_signal = sig != 0

    if not has_signal.any():
        return out

    block = pd.Series(False, index=out.index)

    ci = out.get("Commercial_Index")
    cc = out.get("COT_Change")
    cr = out.get("COT_ROC")

    if params.get("cot_filter") and ci is not None:
        cot_long  = params.get("cot_long", 70)
        cot_short = params.get("cot_short", 30)
        block |= has_signal & (sig == 1) & ci.notna() & (ci < cot_long)
        block |= has_signal & (sig == -1) & ci.notna() & (ci > cot_short)

    if params.get("cot_direction_filter") and cc is not None:
        block |= has_signal & (sig == 1) & cc.notna() & (cc < 0)
        block |= has_signal & (sig == -1) & cc.notna() & (cc > 0)

    if params.get("cot_roc_filter") and cr is not None:
        thr = params.get("cot_roc_threshold", 10)
        block |= has_signal & (sig == 1) & cr.notna() & (cr < thr)
        block |= has_signal & (sig == -1) & cr.notna() & (cr > -thr)

    out.loc[block, "signal"] = 0
    out.loc[block, "entry_price"] = np.nan

    return out


# ===================================================================
# Backtest runners (optimised)
# ===================================================================

def _extract_daily_pnl(data, result):
    """Convert a single-market backtest result to a date-indexed daily PnL Series."""
    if data.empty:
        return pd.Series(dtype=float)

    dates = data["Date"].values
    eq = result["equity_curve"]
    n = len(dates)
    eq_arr = np.array(eq[: n + 1], dtype=float)
    pnl = np.diff(eq_arr)

    if len(eq) > n + 1:
        pnl[-1] += eq[-1] - eq[n]

    return pd.Series(pnl, index=pd.DatetimeIndex(dates), name="pnl")


def _run_one_config(precomputed, markets, entry_params):
    """Run one config across all markets using pre-computed base signals.

    Returns (portfolio_daily_pnl, pooled_trades, per_market_metrics).
    """
    stop_cls = STOP_REGISTRY[BASE_STOP_KEY]["cls"]
    stop_strategy = stop_cls(**BASE_STOP_PARAMS)

    is_base = not (
        entry_params.get("cot_filter")
        or entry_params.get("cot_direction_filter")
        or entry_params.get("cot_roc_filter")
    )

    pnl_parts = []
    trade_frames = []
    per_market = {}

    for market in markets:
        if market not in precomputed:
            continue

        if is_base:
            data = precomputed[market]
        else:
            data = _apply_cot_filter(precomputed[market], entry_params)

        result = run_backtest(
            data, market, stop_strategy,
            initial_capital=INITIAL_CAPITAL, risk_pct=RISK_PCT,
        )

        pnl_s = _extract_daily_pnl(data, result)
        if not pnl_s.empty:
            pnl_parts.append(pnl_s)

        if not result["trades"].empty:
            trade_frames.append(result["trades"])

        m = calculate_performance_metrics(
            result["trades"], result["equity_curve"], INITIAL_CAPITAL
        )
        m["max_consec_losers"] = max_consecutive_losers(result["trades"])
        per_market[market] = m

    if pnl_parts:
        portfolio_pnl = (
            pd.concat(pnl_parts, axis=1)
            .fillna(0.0)
            .sum(axis=1)
            .sort_index()
        )
    else:
        portfolio_pnl = pd.Series(dtype=float)

    trades_df = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames else pd.DataFrame()
    )
    return portfolio_pnl, trades_df, per_market


# ===================================================================
# Walk-forward orchestration
# ===================================================================

def run_grid_for_window(cot_df, markets, grid,
                        start_date, end_date,
                        intraday_cache=None, verbose=True):
    """Run all grid configs for a single OOS window (optimised).

    Returns dict  {config_idx: {daily_pnl, trades, per_market, label, params}}
    """
    if verbose:
        print("  Pre-computing base breakout signals …")
    precomputed = _precompute_base_signals(
        cot_df, markets, start_date, end_date, intraday_cache
    )
    if verbose:
        print(f"  {len(precomputed)} markets with data")

    results = {}
    total = len(grid)
    for idx, params in enumerate(grid):
        label = grid_label(params)
        if verbose and (idx % 10 == 0 or idx == total - 1):
            print(f"  [{idx + 1:>3}/{total}] {label}")
        pnl, trades, pm = _run_one_config(precomputed, markets, params)
        results[idx] = {
            "daily_pnl": pnl,
            "trades": trades,
            "per_market": pm,
            "label": label,
            "params": params,
        }
    return results


def build_aligned_pnl_matrix(window_results_list):
    """Pool OOS daily PnL across walk-forward windows into aligned arrays.

    Returns (base_pnl, variant_pnl, date_index, variant_indices).
    """
    per_config: dict[int, list[pd.Series]] = {}
    for wr in window_results_list:
        for idx, res in wr.items():
            per_config.setdefault(idx, []).append(res["daily_pnl"])

    merged = {}
    for idx, parts in per_config.items():
        non_empty = [s for s in parts if not s.empty]
        merged[idx] = (
            pd.concat(non_empty).sort_index() if non_empty
            else pd.Series(dtype=float)
        )

    all_dates = sorted(
        set().union(*(s.index for s in merged.values() if not s.empty))
    )
    date_index = pd.DatetimeIndex(all_dates)

    aligned = {
        idx: s.reindex(date_index, fill_value=0.0).values
        for idx, s in merged.items()
    }

    base_pnl = aligned[0]
    variant_indices = sorted(i for i in aligned if i != 0)
    variant_pnl = np.column_stack([aligned[i] for i in variant_indices])

    return base_pnl, variant_pnl, date_index, variant_indices


# ===================================================================
# SPA test
# ===================================================================

def run_spa_test(base_losses, model_losses, n_bootstrap=1000):
    """Run Hansen's SPA test via the *arch* library.

    Parameters
    ----------
    base_losses  : (T,) array — losses from the benchmark  (= −pnl_base)
    model_losses : (T, K) array — losses from K alternative models
    n_bootstrap  : bootstrap replications

    Returns dict with pvalue, reject_h0 flag, n_bootstrap.
    """
    from arch.bootstrap import SPA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spa = SPA(
            base_losses, model_losses,
            block_size=None,
            reps=n_bootstrap,
            bootstrap="stationary",
        )
        spa.compute()

    pvals = spa.pvalues  # Series: lower, consistent, upper
    p_consistent = float(pvals["consistent"])

    return {
        "pvalue": p_consistent,
        "pvalue_lower": float(pvals["lower"]),
        "pvalue_upper": float(pvals["upper"]),
        "reject_h0": p_consistent < 0.05,
        "n_bootstrap": n_bootstrap,
    }


# ===================================================================
# Main entry point
# ===================================================================

def run_full_hypothesis_test(cot_df, markets,
                             windows=None, grid=None,
                             n_bootstrap=1000, verbose=True):
    """Orchestrate the full walk-forward SPA hypothesis test.

    Returns a results dict containing:
      spa, metrics_df, base_pnl, variant_pnl, dates,
      grid, variant_indices, per_window_best, windows.
    """
    windows = windows or WALK_FORWARD_WINDOWS
    grid = grid or build_cot_grid()

    if verbose:
        print("Loading intraday cache …")
    intraday_cache = load_intraday_cache(
        os.path.join(PROJECT_ROOT, "ORB_intraday_data.json")
    )

    # ---- Run each OOS window ----
    all_window_results = []
    for w in windows:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"{w['name']}  |  OOS: {w['test_start']} → {w['test_end']}")
            print(f"{'=' * 60}")
        wr = run_grid_for_window(
            cot_df, markets, grid,
            start_date=w["test_start"], end_date=w["test_end"],
            intraday_cache=intraday_cache, verbose=verbose,
        )
        all_window_results.append(wr)

    # ---- Align PnL matrix ----
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
    base_losses = -base_pnl
    model_losses = -variant_pnl
    spa_result = run_spa_test(base_losses, model_losses, n_bootstrap)
    if verbose:
        verdict = "REJECT H₀" if spa_result["reject_h0"] else "FAIL TO REJECT H₀"
        print(f"  p-value = {spa_result['pvalue']:.4f}  →  {verdict}")

    # ---- Per-variant aggregate metrics ----
    if verbose:
        print("\nComputing per-variant metrics …")

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
    base_m = compute_aggregate_metrics(base_trades, base_pnl_s)
    base_m.update({"label": "Base (No COT)", "config_idx": 0, "sharpe_diff": 0.0})
    rows.append(base_m)

    for vi in variant_indices:
        t, p = _pool(vi)
        m = compute_aggregate_metrics(t, p)
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
            dr = res["daily_pnl"] / INITIAL_CAPITAL
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
        "grid": grid,
        "variant_indices": variant_indices,
        "per_window_best": per_window_best,
        "all_window_results": all_window_results,
        "windows": windows,
    }
