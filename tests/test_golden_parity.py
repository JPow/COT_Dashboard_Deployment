"""Compare engine backtests to captured golden trade logs (if present)."""

import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GOLDEN_DIR = os.path.join(PROJECT_ROOT, "data", "golden")
GOLDEN_START = "2023-01-01"
GOLDEN_END = "2026-12-31"


def _trade_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in ["market", "entry_date", "exit_date", "direction", "pnl"] if c in df.columns]
    out = df[cols].copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.strftime("%Y-%m-%d")
    out["exit_date"] = pd.to_datetime(out["exit_date"]).dt.strftime("%Y-%m-%d")
    out["pnl"] = out["pnl"].round(2)
    return out.sort_values(cols).reset_index(drop=True)


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(GOLDEN_DIR, "unified_backtest_app.pkl")),
    reason="Golden logs not captured — run scripts/capture_golden_logs.py",
)
def test_unified_matches_golden():
    from backtest_engine.data import load_cot_data
    from backtest_engine.backtester import run_all_markets

    golden = pd.read_pickle(os.path.join(GOLDEN_DIR, "unified_backtest_app.pkl"))
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
    frames = []
    for market, bundle in all_results.items():
        t = bundle["results"]["trades"]
        if not t.empty:
            t = t.copy()
            t["market"] = market
            frames.append(t)
    current = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    pd.testing.assert_frame_equal(
        _trade_key(golden), _trade_key(current), check_dtype=False,
    )
