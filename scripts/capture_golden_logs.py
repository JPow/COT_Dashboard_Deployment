#!/usr/bin/env python3
"""Capture trade logs for parity checks before/after engine consolidation."""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GOLDEN_START = "2023-01-01"
GOLDEN_END = "2026-12-31"
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "golden")


def _save_trades(name: str, all_results: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for market, bundle in all_results.items():
        trades = bundle.get("results", {}).get("trades")
        if trades is None or (isinstance(trades, pd.DataFrame) and trades.empty):
            continue
        t = trades.copy()
        t["market"] = market
        frames.append(t)
    if not frames:
        pd.DataFrame().to_pickle(os.path.join(OUT_DIR, f"{name}.pkl"))
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_pickle(os.path.join(OUT_DIR, f"{name}.pkl"))
    print(f"  wrote {name}.pkl ({len(out)} trades)")


def capture_unified():
    from backtest_engine.data import load_cot_data
    from backtest_engine.backtester import run_all_markets

    cot = load_cot_data()
    markets = sorted(cot["Market"].unique().tolist())
    all_results, _ = run_all_markets(
        cot,
        markets,
        setup_key="narrowing_range",
        entry_key="orb_breakout",
        stop_key="two_phase_atr",
        setup_params={"n_days": 3},
        entry_params={"or_type": "60m"},
        stop_params={"trailing_atr_mult": 2.0},
        start_date=GOLDEN_START,
        end_date=GOLDEN_END,
    )
    _save_trades("unified_backtest_app", all_results)


def capture_tf():
    os.environ["COT_SKIP_PRECOMPUTE"] = "1"
    import tf_backtest_app as tf

    cot = tf.load_cot_data()
    markets = sorted(cot["Market"].unique().tolist())
    all_results, _, _ = tf.run_all_markets_tf(
        cot,
        markets,
        lookback=tf.DEFAULT_LOOKBACK,
        start_date=GOLDEN_START,
        end_date=GOLDEN_END,
    )
    _save_trades("tf_backtest_app", all_results)


def capture_orb():
    os.environ["COT_SKIP_PRECOMPUTE"] = "1"
    import ORB_backtest as orb

    cot = orb.load_cot_data()
    markets = sorted(cot["Market"].unique().tolist())
    all_results, _ = orb.run_all_backtests(
        cot,
        markets,
        or_type="60m",
        n_narrowing_days=orb.DEFAULT_NARROWING_DAYS,
        initial_capital=orb.DEFAULT_CAPITAL,
        risk_pct=orb.DEFAULT_RISK_PCT,
        trailing_atr_mult=orb.DEFAULT_TRAILING_ATR_MULT,
        fast_atr_period=orb.DEFAULT_FAST_ATR,
        slow_atr_period=orb.DEFAULT_SLOW_ATR,
        start_date=GOLDEN_START,
        end_date=GOLDEN_END,
    )
    _save_trades("ORB_backtest", all_results)


def capture_inside():
    os.environ["COT_SKIP_PRECOMPUTE"] = "1"
    import Inside_day_backtest as idb

    cot = idb.load_cot_data()
    markets = sorted(cot["Market"].unique().tolist())
    all_results, _ = idb.run_all_backtests(
        cot,
        markets,
        n_inside_days=idb.DEFAULT_INSIDE_DAYS,
        initial_capital=idb.DEFAULT_CAPITAL,
        risk_pct=idb.DEFAULT_RISK_PCT,
        trailing_atr_mult=idb.DEFAULT_TRAILING_ATR_MULT,
        fast_atr_period=idb.DEFAULT_FAST_ATR,
        slow_atr_period=idb.DEFAULT_SLOW_ATR,
        start_date=GOLDEN_START,
        end_date=GOLDEN_END,
    )
    _save_trades("Inside_day_backtest", all_results)


def capture_cot_rsi():
    os.environ["COT_SKIP_PRECOMPUTE"] = "1"
    import backtest_app as ba

    all_results, _ = ba.run_all_backtests(
        start_date=GOLDEN_START, end_date=GOLDEN_END, ma_period=0
    )
    _save_trades("backtest_app", all_results)


def main():
    print(f"Golden window: {GOLDEN_START} → {GOLDEN_END}")
    print("Capturing unified…")
    capture_unified()
    print("Capturing tf…")
    capture_tf()
    print("Capturing ORB…")
    capture_orb()
    print("Capturing inside day…")
    capture_inside()
    print("Capturing COT+RSI…")
    capture_cot_rsi()
    print("Done.")


if __name__ == "__main__":
    main()
