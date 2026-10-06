"""Unit tests for liquidity-based continuous roll schedule."""

from datetime import date, timedelta

from ib_continuous import (
    build_liquidity_roll_schedule,
    make_bar,
    score_contract_volume,
    series_quality_metrics,
    stitch_contract_bars,
)


def _sessions(start: date, n: int):
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(dates, closes, volumes):
    return [
        make_bar(d, c, c + 1, c - 1, c, v)
        for d, c, v in zip(dates, closes, volumes)
    ]


def test_score_contract_volume_lookback():
    dates = _sessions(date(2024, 1, 2), 15)
    bars = _bars(dates, [100] * 15, list(range(1, 16)))
    # Last 10 sessions ending at last date: volumes 6..15 = 105
    assert score_contract_volume(bars, dates[-1], lookback=10) == sum(range(6, 16))


def test_liquidity_skips_dead_month():
    """Thin month between two liquid contracts should never become active."""
    # Liquid near: Jan expiry (F), liquid deferred: Apr (J), thin: Feb (G)
    near_dates = _sessions(date(2024, 1, 2), 25)
    thin_dates = _sessions(date(2024, 1, 15), 20)
    def_dates = _sessions(date(2024, 1, 10), 40)

    near = _bars(near_dates, [100] * 25, [1000] * 20 + [50] * 5)  # volume fades
    thin = _bars(thin_dates, [101] * 20, [0] * 20)  # flat stubs
    # Force equal OHLC on thin
    thin = [make_bar(d, 101, 101, 101, 101, 0) for d in thin_dates]
    deferred = _bars(def_dates, [102] * 40, [10] * 10 + [2000] * 30)

    contract_bars = {
        'PAF4': (None, date(2024, 1, 29), near),
        'PAG4': (None, date(2024, 2, 26), thin),
        'PAJ4': (None, date(2024, 4, 26), deferred),
    }
    active, events = build_liquidity_roll_schedule(
        contract_bars, lookback=5, confirm_days=3, start_date=date(2024, 1, 2),
    )
    used = set(active.values())
    assert 'PAG4' not in used
    assert 'PAF4' in used
    assert 'PAJ4' in used
    assert any(e['to'] == 'PAJ4' for e in events)


def test_stub_escape_switches_immediately():
    dates = _sessions(date(2024, 3, 1), 10)
    # Contract A listed but vol=0; B has trades from day 1
    a = [make_bar(d, 50, 50, 50, 50, 0) for d in dates]
    b = [make_bar(d, 51, 52, 50, 51, 500) for d in dates]
    contract_bars = {
        'PLA': (None, date(2024, 3, 20), a),
        'PLB': (None, date(2024, 6, 20), b),
    }
    active, events = build_liquidity_roll_schedule(
        contract_bars, lookback=3, confirm_days=3, start_date=dates[0],
    )
    assert all(v == 'PLB' for v in active.values())
    assert events == [] or events[0].get('reason') in (None, 'stub_escape', 'volume_handoff')


def test_hysteresis_requires_confirm_days():
    dates = _sessions(date(2024, 5, 1), 20)
    # A dominates first 10 days; B spikes one day then fades — should NOT roll
    vols_a = [1000] * 10 + [900] * 10
    vols_b = [10] * 10 + [950, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    a = _bars(dates, [100] * 20, vols_a)
    b = _bars(dates, [101] * 20, vols_b)
    contract_bars = {
        'A': (None, date(2024, 5, 30), a),
        'B': (None, date(2024, 8, 30), b),
    }
    active, events = build_liquidity_roll_schedule(
        contract_bars, lookback=3, confirm_days=3, start_date=dates[0],
    )
    # Single-day spike should not produce a lasting roll to B for all remaining days
    assert sum(1 for d, ls in active.items() if ls == 'B') < 5


def test_stitch_liquidity_reduces_flats():
    dates = _sessions(date(2024, 6, 3), 30)
    liquid = _bars(dates, [200 + i for i in range(30)], [1000] * 30)
    thin_dates = dates[5:20]
    thin = [make_bar(d, 210, 210, 210, 210, 0) for d in thin_dates]
    contract_bars = {
        'THIN': (None, date(2024, 7, 15), thin),
        'LIQ': (None, date(2024, 9, 15), liquid),
    }
    bars, roll_log, status = stitch_contract_bars(
        contract_bars,
        start_date=dates[0],
        roll_mode='liquidity',
        lookback=5,
        confirm_days=2,
    )
    assert status == 'ok'
    m = series_quality_metrics(bars)
    assert m['flat_pct'] < 5.0
    assert m['vol0_pct'] < 5.0
    assert all(r['contract_local_symbol'] == 'LIQ' for r in bars)


def test_expiry_buffer_still_default_path():
    dates_a = _sessions(date(2024, 1, 2), 40)
    dates_b = _sessions(date(2024, 2, 1), 40)
    a = _bars(dates_a, [10] * 40, [100] * 40)
    b = _bars(dates_b, [11] * 40, [100] * 40)
    contract_bars = {
        'A': (None, date(2024, 2, 15), a),
        'B': (None, date(2024, 4, 15), b),
    }
    bars, roll_log, status = stitch_contract_bars(
        contract_bars,
        start_date=date(2024, 1, 2),
        roll_mode='expiry_buffer',
        roll_buffer_business_days=5,
        today=date(2024, 6, 1),
    )
    assert status == 'ok'
    assert len(bars) > 0
    assert any(e.get('actual_roll') for e in roll_log)
