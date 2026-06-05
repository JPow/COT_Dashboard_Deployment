"""Synthetic tests for true opening-range ORB entry logic."""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.data import (
    ET,
    build_opening_range_day,
    find_post_or_breakout,
    get_session_times,
    load_contract_specs,
)
from backtest_engine.entries import apply_orb_breakout
from backtest_engine.stops import TwoPhaseATRStop

UTC = ZoneInfo('UTC')


def _bar(market, interval, dt_et, o, h, l, c):
    dt_utc = dt_et.astimezone(UTC).replace(tzinfo=None)
    return {
        'symbol': market,
        'interval': interval,
        'datetime': dt_utc,
        'open': o,
        'high': h,
        'low': l,
        'close': c,
        'volume': 100,
    }


def _make_intraday(market, interval, bars):
    return pd.DataFrame(bars)


def test_specs_load():
    specs = load_contract_specs()
    assert isinstance(specs, dict)
    assert len(specs) > 40
    assert 'AUSTRALIAN DOLLAR' in specs
    assert specs['AUSTRALIAN DOLLAR']['rth_open'] == '08:30'


def test_aud_30m_long_breakout():
    market = 'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE'
    trade_day = datetime(2024, 6, 3, tzinfo=ET)
    bars = [
        _bar(market, '30m', datetime(2024, 6, 3, 8, 30, tzinfo=ET), 0.6600, 0.6610, 0.6595, 0.6605),
        _bar(market, '30m', datetime(2024, 6, 3, 9, 0, tzinfo=ET), 0.6610, 0.6625, 0.6608, 0.6620),
        _bar(market, '30m', datetime(2024, 6, 3, 9, 30, tzinfo=ET), 0.6620, 0.6635, 0.6618, 0.6630),
    ]
    intra = _make_intraday(market, '30m', bars)
    or_info = build_opening_range_day(intra, market, trade_day, '30m')
    assert or_info['valid']
    assert or_info['or_high'] == 0.6610
    assert or_info['or_low'] == 0.6595

    bo = find_post_or_breakout(intra, market, trade_day, '30m')
    assert bo is not None
    assert bo['direction'] == 1
    assert bo['entry_price'] == 0.6610


def test_es_30m_no_breakout():
    market = 'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE'
    trade_day = datetime(2024, 6, 3, tzinfo=ET)
    bars = [
        _bar(market, '30m', datetime(2024, 6, 3, 9, 30, tzinfo=ET), 5300, 5310, 5295, 5305),
        _bar(market, '30m', datetime(2024, 6, 3, 10, 0, tzinfo=ET), 5305, 5308, 5298, 5300),
        _bar(market, '30m', datetime(2024, 6, 3, 10, 30, tzinfo=ET), 5300, 5306, 5296, 5302),
    ]
    intra = _make_intraday(market, '30m', bars)
    assert find_post_or_breakout(intra, market, trade_day, '30m') is None


def test_sugar_early_session():
    market = 'SUGAR NO. 11 - ICE FUTURES U.S.'
    trade_day = datetime(2024, 6, 3, tzinfo=ET)
    times = get_session_times(market, '30m')
    assert times['rth_open_str'] == '03:30'
    bars = [
        _bar(market, '30m', datetime(2024, 6, 3, 3, 30, tzinfo=ET), 20.0, 20.2, 19.9, 20.1),
        _bar(market, '30m', datetime(2024, 6, 3, 4, 0, tzinfo=ET), 20.1, 20.15, 19.85, 19.9),
        _bar(market, '30m', datetime(2024, 6, 3, 4, 30, tzinfo=ET), 19.9, 20.0, 19.7, 19.75),
    ]
    intra = _make_intraday(market, '30m', bars)
    bo = find_post_or_breakout(intra, market, trade_day, '30m')
    assert bo is not None
    assert bo['direction'] == -1


def test_dual_breakout_midpoint_tiebreaker():
    market = 'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE'
    trade_day = datetime(2024, 6, 4, tzinfo=ET)
    bars = [
        _bar(market, '30m', datetime(2024, 6, 4, 8, 30, tzinfo=ET), 0.6700, 0.6710, 0.6690, 0.6705),
        _bar(market, '30m', datetime(2024, 6, 4, 9, 0, tzinfo=ET), 0.6715, 0.6720, 0.6685, 0.6695),
    ]
    intra = _make_intraday(market, '30m', bars)
    bo = find_post_or_breakout(intra, market, trade_day, '30m')
    assert bo is not None
    assert bo['direction'] == 1  # or_open 0.6700 >= midpoint 0.6700


def test_apply_orb_breakout_slippage():
    market = 'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE'
    trade_day = pd.Timestamp('2024-06-03')
    bars = [
        _bar(market, '30m', datetime(2024, 6, 3, 8, 30, tzinfo=ET), 0.6600, 0.6610, 0.6595, 0.6605),
        _bar(market, '30m', datetime(2024, 6, 3, 9, 0, tzinfo=ET), 0.6610, 0.6625, 0.6608, 0.6620),
    ]
    intra = _make_intraday(market, '30m', bars)
    df = pd.DataFrame({
        'Date': [trade_day],
        'setup': [True],
        'Open': [0.6600],
        'Commercial_Index': [np.nan],
        'RSI': [50],
    })
    out = apply_orb_breakout(df, market_name=market, or_type='30m', intraday_cache=intra)
    tick = load_contract_specs()['AUSTRALIAN DOLLAR']['tick_size']
    assert out.iloc[0]['signal'] == 1
    assert round(out.iloc[0]['entry_price'], 6) == round(0.6610 + 2 * tick, 6)


def test_stop_buffer_is_one_tick():
    tick = 0.00005  # FX-scale tick
    row = pd.Series({
        'or_low_signal': 0.6600,
        'or_high_signal': 0.6610,
        'tick_size': tick,
    })
    stop = TwoPhaseATRStop()
    assert round(stop.initial_stop(1, 0.6611, row), 8) == round(0.6600 - tick, 8)
    assert round(stop.initial_stop(-1, 0.6599, row), 8) == round(0.6610 + tick, 8)
    assert round(stop.stop_distance(1, 0.6611, row), 8) == round(0.0010 + tick, 8)


def test_missing_or_window_skips():
    market = 'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE'
    trade_day = datetime(2024, 6, 5, tzinfo=ET)
    intra = _make_intraday(market, '30m', [])
    assert build_opening_range_day(intra, market, trade_day, '30m')['valid'] is False


if __name__ == '__main__':
    test_specs_load()
    test_aud_30m_long_breakout()
    test_es_30m_no_breakout()
    test_sugar_early_session()
    test_dual_breakout_midpoint_tiebreaker()
    test_apply_orb_breakout_slippage()
    test_stop_buffer_is_one_tick()
    test_missing_or_window_skips()
    print('All orb_entry_test checks passed.')
