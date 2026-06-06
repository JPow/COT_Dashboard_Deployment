#!/usr/bin/env python3
"""
Backward-fill IBKR 30m / 60m history into ORB_intraday_data.json.

IB only returns ~30 calendar days of intraday bars per request. To extend history
beyond the current window, query *expired* contracts with endDateTime anchored to
the gap (the contract's active period), Panama-adjust, and prepend — without
replacing bars already in the archive.

Requires: ib_insync, pandas, and a running IB Gateway (same as the notebook).

Usage (with IB Gateway on port 4001):
    python intraday_data_audit.py
    python ib_intraday_backfill.py --trading-days 30
    python ib_intraday_backfill.py --market "GOLD - COMMODITY EXCHANGE INC." --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd

from backtest_engine.data import (
    INTRADAY_CACHE_FILE,
    INTRADAY_STATE_FILE,
    load_intraday_cache,
    load_intraday_state,
    summarize_intraday_ranges,
)

try:
    from ib_insync import Future, IB, util
except ImportError as exc:
    raise SystemExit(
        'ib_insync is required. Install in your notebook env: pip install ib_insync'
    ) from exc


ROLL_BUFFER_BDAYS = 5
PACING_SECONDS = 8.0
IB_HOST = '127.0.0.1'
IB_PORT = 4001
CLIENT_ID = 77
FETCH_RETRIES = 2
FETCH_TIMEOUT = 45
MAX_BACKFILL_CONTRACTS = 2
HMDS_WAKE_ATTEMPTS = 5
HMDS_WAKE_WAIT = 15


def trading_days_to_calendar_days(trading_days: int) -> int:
    """Approximate calendar days for N trading days (Mon–Fri)."""
    return int(trading_days * 7 / 5) + 5


def _normalize_end_dt(end_dt: datetime | pd.Timestamp) -> datetime:
    """Return naive UTC-stripped datetime for IB endDateTime and comparisons."""
    ts = pd.Timestamp(end_dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    return ts.to_pydatetime()


def _format_end_dt(end_dt: datetime | pd.Timestamp | None) -> str:
    """Format cache boundary for IB (UTC-naive cache → UTC endDateTime string)."""
    if end_dt is None:
        return ''
    dt = _normalize_end_dt(end_dt)
    return dt.strftime('%Y%m%d-%H:%M:%S')


def wake_hmds_farm(ib: IB, *, attempts: int = HMDS_WAKE_ATTEMPTS) -> bool:
    """Wake IB's Historical Data (HMDS) farm before bulk intraday pulls.

    The farm goes inactive when idle. A lightweight ES probe request usually
    triggers reconnection (IB message 2106). We retry with pauses because the
    first request often times out while the farm is still spinning up.
    """
    hmds_farms: list[str] = []

    def _on_error(req_id, error_code, error_string, contract):
        if error_code == 2106 and 'HMDS' in error_string:
            hmds_farms.append(error_string)
            print(f'  ✓ {error_string}', flush=True)
        elif error_code in (2104, 2158):
            print(f'  · {error_string}', flush=True)

    ib.errorEvent += _on_error
    # Fully specified front month — bare ES template is ambiguous on qualify.
    probe = Future(
        symbol='ES', exchange='CME', currency='USD',
        lastTradeDateOrContractMonth='20260618', localSymbol='ESM6',
    )
    try:
        qualified = ib.qualifyContracts(probe)
        if not qualified:
            raise RuntimeError('qualifyContracts returned empty')
        probe = qualified[0]
    except Exception as exc:
        print(f'  ✗ probe qualify failed: {exc}', flush=True)
        ib.errorEvent -= _on_error
        return False

    _sleep = getattr(ib, 'sleep', None) or time.sleep
    print('Waking HMDS historical data farm...', flush=True)
    for attempt in range(1, attempts + 1):
        ib.reqCurrentTime()
        _sleep(3)
        print(f'  probe {attempt}/{attempts}...', flush=True)
        try:
            bars = ib.reqHistoricalData(
                probe,
                endDateTime='',
                durationStr='3 D',
                barSizeSetting='1 hour',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
                timeout=FETCH_TIMEOUT,
            )
        except Exception:
            bars = []
        n = len(bars or [])
        if n > 0:
            print(f'  ✓ HMDS ready — probe returned {n} bars', flush=True)
            ib.errorEvent -= _on_error
            return True
        if hmds_farms:
            _sleep(HMDS_WAKE_WAIT)
            try:
                bars = ib.reqHistoricalData(
                    probe,
                    endDateTime='',
                    durationStr='3 D',
                    barSizeSetting='1 hour',
                    whatToShow='TRADES',
                    useRTH=False,
                    formatDate=1,
                    timeout=FETCH_TIMEOUT,
                )
            except Exception:
                bars = []
            if bars:
                print(f'  ✓ HMDS ready after farm connect — {len(bars)} bars', flush=True)
                ib.errorEvent -= _on_error
                return True
        print(f'  … farm not ready, waiting {HMDS_WAKE_WAIT}s', flush=True)
        _sleep(HMDS_WAKE_WAIT)

    ib.errorEvent -= _on_error
    print('  ✗ HMDS farm did not respond — re-login to Gateway and retry immediately', flush=True)
    return False


def connect_ib(host: str = IB_HOST, port: int = IB_PORT, client_id: int = CLIENT_ID) -> IB:
    """Connect and ensure HMDS is active."""
    util.startLoop()
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=15)
    if not ib.isConnected():
        raise SystemExit('Could not connect to IB Gateway')
    print(f'Connected to IB Gateway ({host}:{port}, clientId={client_id})', flush=True)
    if not wake_hmds_farm(ib):
        print(
            '  ⚠ HMDS probe did not return bars yet — continuing anyway; '
            'first market request often wakes the farm after login.',
            flush=True,
        )
    return ib


def _bday_offset(d: date, offset_days: int) -> date:
    if offset_days == 0:
        return d
    sign = 1 if offset_days > 0 else -1
    n = abs(offset_days)
    cur = d
    while n > 0:
        cur = cur + timedelta(days=sign)
        if cur.weekday() < 5:
            n -= 1
    return cur


def _bar_date(b) -> date:
    bd = b.date
    return bd.date() if hasattr(bd, 'date') else bd


def _last_close_on(bars, target_date: date):
    on_day = [b for b in bars if _bar_date(b) == target_date]
    return on_day[-1].close if on_day else None


def _parse_expiry(contract) -> date | None:
    s = contract.lastTradeDateOrContractMonth or ''
    try:
        if len(s) >= 8:
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        if len(s) == 6:
            return date(int(s[:4]), int(s[4:6]), 28)
    except (TypeError, ValueError):
        return None
    return None


def merge_intraday_rebuild(
    cache_df: pd.DataFrame,
    new_df: pd.DataFrame,
    market_name: str,
    interval: str,
) -> pd.DataFrame:
    """Archive-safe merge: keep bars older than the rebuild, replace overlap+tail."""
    if cache_df.empty:
        return new_df.sort_values('datetime').reset_index(drop=True)

    series_mask = (cache_df['symbol'] == market_name) & (cache_df['interval'] == interval)
    archived = cache_df[series_mask & (cache_df['datetime'] < new_df['datetime'].min())]
    other = cache_df[~series_mask]
    merged = pd.concat([other, archived, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=['symbol', 'interval', 'datetime'], keep='last')
    return merged.sort_values(['symbol', 'interval', 'datetime']).reset_index(drop=True)


def _fetch_contract_bars(
    ib: IB,
    contract,
    *,
    bar_size: str,
    end_dt: datetime | None,
    duration: str = '30 D',
    what_to_show: str = 'TRADES',
    use_rth: bool = False,
    pacing_seconds: float = PACING_SECONDS,
) -> list:
    """Fetch intraday bars with retries and IB pacing."""
    end_arg = _format_end_dt(end_dt)
    _sleep = getattr(ib, 'sleep', None) or time.sleep
    for attempt in range(FETCH_RETRIES):
        _sleep(pacing_seconds)
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_arg,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
                timeout=FETCH_TIMEOUT,
            )
        except Exception:
            bars = []
        blist = list(bars or [])
        if blist:
            return blist
        _sleep(3.0 + attempt * 3.0)
    return []


def _select_backfill_contracts(
    contracts: list[tuple[date, object]],
    start_date: date,
    end_day: date,
    max_contracts: int = MAX_BACKFILL_CONTRACTS,
) -> list[tuple[date, object]]:
    """Pick listings that were trading at ``end_day`` and are still queryable today.

    IB HMDS times out on intraday pulls for contracts already expired *as of
    today*. We therefore skip listings whose expiry is in the past, and anchor
    ``endDateTime`` at ``end_day`` on contracts whose expiry is still ahead of
    today (they were deferred months actively trading at the boundary).
    """
    today = date.today()
    eligible = []
    for exp, c in contracts:
        if exp < end_day - timedelta(days=45):
            continue
        if exp < today:
            continue
        if exp > end_day + timedelta(days=120):
            continue
        eligible.append((exp, c))
    if not eligible:
        return []

    eligible.sort(key=lambda x: x[0])
    # Prefer the nearest expiry after the cache boundary (was the front or next
    # deferred month when the gap starts).
    after = [x for x in eligible if x[0] >= end_day]
    picked = after[:max_contracts] if after else eligible[:max_contracts]
    return picked


def build_backadjusted_window(
    ib: IB,
    *,
    future_symbol: str,
    exchange: str,
    currency: str,
    trading_class: str = '',
    start_date: date,
    end_dt: datetime,
    bar_size: str,
    what_to_show: str = 'TRADES',
    use_rth: bool = False,
    roll_buffer_business_days: int = ROLL_BUFFER_BDAYS,
    pacing_seconds: float = PACING_SECONDS,
):
    """Panama-adjusted intraday stitch for a bounded window (backfill use).

    Unlike the notebook rebuild, expired contracts are included when their expiry
    falls inside [start_date, end_dt] so older intraday can be retrieved.
    """
    template = Future(symbol=future_symbol, exchange=exchange, currency=currency)
    if trading_class:
        template.tradingClass = trading_class
    template.includeExpired = True

    try:
        details = ib.reqContractDetails(template)
    except Exception as e:
        return [], [], f'reqContractDetails error: {e}'
    if not details:
        return [], [], 'no contracts found'

    time.sleep(3.0)
    end_dt = _normalize_end_dt(end_dt)
    end_day = end_dt.date()

    all_contracts = []
    for d in details:
        c = d.contract
        exp = _parse_expiry(c)
        if exp is not None:
            all_contracts.append((exp, c))

    seen = set()
    deduped = []
    for exp, c in sorted(all_contracts, key=lambda x: x[0]):
        key = c.localSymbol or (c.symbol, c.lastTradeDateOrContractMonth)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((exp, c))

    contracts = _select_backfill_contracts(deduped, start_date, end_day)
    if not contracts:
        return [], [], 'no contracts in window'

    duration = '30 D' if bar_size in ('30 mins', '1 hour') else '1 Y'
    contract_bars = {}
    for exp, c in contracts:
        local = c.localSymbol
        print(f'     fetching {local} (exp {exp})...', flush=True)
        bars = _fetch_contract_bars(
            ib, c, bar_size=bar_size, end_dt=end_dt,
            duration=duration, what_to_show=what_to_show, use_rth=use_rth,
            pacing_seconds=pacing_seconds,
        )
        contract_bars[local] = (c, exp, bars)
        print(f'       → {len(bars)} bars', flush=True)

    sorted_locals = sorted(
        [ls for ls, (_, _, bars) in contract_bars.items() if bars],
        key=lambda ls: contract_bars[ls][1],
    )
    if not sorted_locals:
        return [], [], 'no bars returned for any contract'

    offset_per_local = {sorted_locals[-1]: 0.0}
    roll_log = []

    for i in range(len(sorted_locals) - 2, -1, -1):
        cur_local = sorted_locals[i]
        nxt_local = sorted_locals[i + 1]
        _, cur_exp, cur_bars = contract_bars[cur_local]
        _, _, nxt_bars = contract_bars[nxt_local]

        nominal = _bday_offset(cur_exp, -roll_buffer_business_days)
        if nominal > date.today():
            offset_per_local[cur_local] = offset_per_local[nxt_local]
            roll_log.append({
                'from': cur_local, 'to': nxt_local,
                'nominal_roll': nominal, 'actual_roll': None,
                'gap': 0.0, 'note': 'nominal in future; skipped',
            })
            continue

        actual = None
        for k in range(15):
            cand = _bday_offset(nominal, -k)
            if (_last_close_on(cur_bars, cand) is not None
                    and _last_close_on(nxt_bars, cand) is not None):
                actual = cand
                break

        if actual is None:
            cur_dates = {_bar_date(b) for b in cur_bars}
            nxt_dates = {_bar_date(b) for b in nxt_bars}
            common = sorted(cur_dates & nxt_dates)
            if not common:
                offset_per_local[cur_local] = offset_per_local[nxt_local]
                roll_log.append({
                    'from': cur_local, 'to': nxt_local,
                    'nominal_roll': nominal, 'actual_roll': None,
                    'gap': 0.0, 'note': 'no overlap; offset propagated',
                })
                continue
            actual = (
                max(d for d in common if d <= nominal)
                if any(d <= nominal for d in common)
                else common[-1]
            )

        gap = _last_close_on(nxt_bars, actual) - _last_close_on(cur_bars, actual)
        offset_per_local[cur_local] = offset_per_local[nxt_local] + gap
        roll_log.append({
            'from': cur_local, 'to': nxt_local,
            'nominal_roll': nominal, 'actual_roll': actual, 'gap': gap,
        })

    locals_to_roll = {entry['from']: entry['actual_roll'] for entry in roll_log}
    out = []
    prev_roll = None
    for local in sorted_locals:
        _, _, bars = contract_bars[local]
        my_roll = locals_to_roll.get(local)
        offset = offset_per_local[local]
        for b in bars:
            d = _bar_date(b)
            bt = pd.Timestamp(b.date)
            if bt.tzinfo is not None:
                bt = bt.tz_convert('UTC').tz_localize(None)
            if bt.to_pydatetime() >= end_dt:
                continue
            if prev_roll is not None and d <= prev_roll:
                continue
            if my_roll is not None and d > my_roll:
                continue
            if d < start_date:
                continue
            out.append({
                'date': b.date,
                'datetime': bt,
                'open': b.open + offset,
                'high': b.high + offset,
                'low': b.low + offset,
                'close': b.close + offset,
                'volume': b.volume,
                'contract_local_symbol': local,
            })
        if my_roll is not None:
            prev_roll = my_roll

    out.sort(key=lambda x: x['datetime'])
    return out, roll_log, 'ok'


def _bars_to_df(rows: list, market_name: str, interval: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        'symbol': market_name,
        'interval': interval,
        'datetime': r['datetime'],
        'open': r['open'],
        'high': r['high'],
        'low': r['low'],
        'close': r['close'],
        'volume': r['volume'],
        'contract': r['contract_local_symbol'],
    } for r in rows])
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    return df.sort_values('datetime').reset_index(drop=True)


def backfill_series(
    ib: IB,
    *,
    market_name: str,
    display_sym: str,
    exchange: str,
    currency: str,
    trading_class: str,
    interval: str,
    cache_df: pd.DataFrame,
    state: dict,
    trading_days: int = 30,
    dry_run: bool = False,
    pending_only: bool = False,
) -> tuple[pd.DataFrame, dict, int]:
    """Prepend up to ``trading_days`` of older bars for one market + interval."""
    bar_size = '30 mins' if interval == '30m' else '1 hour'
    state_key = f'{market_name}|{interval}'
    meta = state.get(state_key, {})

    sub = cache_df[
        (cache_df['symbol'] == market_name) & (cache_df['interval'] == interval)
    ].sort_values('datetime')
    if sub.empty:
        print(f'  ⏭ {market_name} {interval}: no existing cache to extend')
        return cache_df, state, 0

    first_dt = sub['datetime'].min()
    cal_days = trading_days_to_calendar_days(trading_days)
    target_dt = first_dt - timedelta(days=cal_days)
    target_date = target_dt.date()

    if pending_only and meta.get('backfill_target'):
        print(f'  ⏭ {market_name} {interval}: already backfilled (target {meta["backfill_target"]})')
        return cache_df, state, 0

    print(
        f'  {market_name} {interval}: backfill {target_date} → {first_dt.date()} '
        f'({trading_days} TD / {cal_days} cal days)',
        flush=True,
    )
    if dry_run:
        return cache_df, state, 0

    bars, roll_log, status = build_backadjusted_window(
        ib,
        future_symbol=display_sym,
        exchange=exchange,
        currency=currency,
        trading_class=trading_class,
        start_date=target_date,
        end_dt=_normalize_end_dt(first_dt),
        bar_size=bar_size,
        use_rth=False,
    )
    if status != 'ok' or not bars:
        print(f'     ✗ {status}')
        return cache_df, state, 0

    new_df = _bars_to_df(bars, market_name, interval)
    new_df = new_df[new_df['datetime'] < first_dt]
    if new_df.empty:
        print('     ✗ no bars before existing first timestamp')
        return cache_df, state, 0

    series_mask = (cache_df['symbol'] == market_name) & (cache_df['interval'] == interval)
    other = cache_df[~series_mask]
    existing = cache_df[series_mask]
    merged_series = pd.concat([new_df, existing], ignore_index=True)
    merged_series = merged_series.drop_duplicates(
        subset=['symbol', 'interval', 'datetime'], keep='last',
    )
    cache_df = pd.concat([other, merged_series], ignore_index=True)
    cache_df = cache_df.sort_values(['symbol', 'interval', 'datetime']).reset_index(drop=True)

    meta = dict(meta)
    new_first = cache_df[
        (cache_df['symbol'] == market_name) & (cache_df['interval'] == interval)
    ]['datetime'].min()
    meta['first'] = str(new_first)
    meta['backfill_target'] = str(target_dt)
    meta['last_backfill_at'] = datetime.now().isoformat(timespec='seconds')
    meta['backfill_bars_added'] = meta.get('backfill_bars_added', 0) + len(new_df)
    state[state_key] = meta

    print(f'     ✓ +{len(new_df)} bars ({len(roll_log)} rolls), new first {meta["first"]}')
    return cache_df, state, len(new_df)


def persist_intraday(cache_df: pd.DataFrame, state: dict) -> None:
    save = cache_df.copy()
    save['datetime'] = save['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    with open(INTRADAY_CACHE_FILE, 'w') as f:
        json.dump(save.to_dict('records'), f)
    with open(INTRADAY_STATE_FILE, 'w') as f:
        json.dump(state, f, default=str, indent=2)


# Minimal IB mapping (same markets as COT IBRK Data Grabber.ipynb)
IB_MAPPING: dict[str, tuple] = {
    'GOLD - COMMODITY EXCHANGE INC.': ('GC', 'COMEX', 'USD'),
    'PLATINUM - NEW YORK MERCANTILE EXCHANGE': ('PL', 'NYMEX', 'USD'),
    'PALLADIUM - NEW YORK MERCANTILE EXCHANGE': ('PA', 'NYMEX', 'USD'),
    'COPPER- #1 - COMMODITY EXCHANGE INC.': ('HG', 'COMEX', 'USD'),
    'MICRO GOLD - COMMODITY EXCHANGE INC.': ('MGC', 'COMEX', 'USD'),
    'MICRO SILVER - COMMODITY EXCHANGE INC.': ('QI', 'COMEX', 'USD'),
    'MICRO COPPER - COMMODITY EXCHANGE INC.': ('MHG', 'COMEX', 'USD'),
    'WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE': ('CL', 'NYMEX', 'USD'),
    'GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE': ('RB', 'NYMEX', 'USD'),
    'NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE': ('NG', 'NYMEX', 'USD'),
    'E-MINI NATURAL GAS - NEW YORK MERCANTILE EXCHANGE': ('QG', 'NYMEX', 'USD'),
    'CORN - CHICAGO BOARD OF TRADE': ('ZC', 'CBOT', 'USD'),
    'OATS - CHICAGO BOARD OF TRADE': ('ZO', 'CBOT', 'USD'),
    'WHEAT-SRW - CHICAGO BOARD OF TRADE': ('ZW', 'CBOT', 'USD'),
    'SOYBEANS - CHICAGO BOARD OF TRADE': ('ZS', 'CBOT', 'USD'),
    'SOYBEAN MEAL - CHICAGO BOARD OF TRADE': ('ZM', 'CBOT', 'USD'),
    'SOYBEAN OIL - CHICAGO BOARD OF TRADE': ('ZL', 'CBOT', 'USD'),
    'MINI SOYBEANS - CHICAGO BOARD OF TRADE': ('ZS', 'CBOT', 'USD'),
    'COCOA - ICE FUTURES U.S.': ('CC', 'NYBOT', 'USD'),
    'COFFEE C - ICE FUTURES U.S.': ('KC', 'NYBOT', 'USD'),
    'COTTON NO. 2 - ICE FUTURES U.S.': ('CT', 'NYBOT', 'USD'),
    'SUGAR NO. 11 - ICE FUTURES U.S.': ('SB', 'NYBOT', 'USD'),
    'FEEDER CATTLE - CHICAGO MERCANTILE EXCHANGE': ('GF', 'CME', 'USD'),
    'LEAN HOGS - CHICAGO MERCANTILE EXCHANGE': ('HE', 'CME', 'USD'),
    'LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE': ('LE', 'CME', 'USD'),
    'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE': ('ES', 'CME', 'USD'),
    'RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE': ('RTY', 'CME', 'USD'),
    'MICRO E-MINI NASDAQ-100 INDEX - CHICAGO MERCANTILE EXCHANGE': ('MNQ', 'CME', 'USD'),
    'NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE': ('NKD', 'CME', 'USD'),
    'EMINI RUSSELL 1000 GROWTH - CHICAGO MERCANTILE EXCHANGE': ('RTY', 'CME', 'USD'),
    'VIX FUTURES - CBOE FUTURES EXCHANGE': ('VIX', 'CFE', 'USD'),
    'BITCOIN - CHICAGO MERCANTILE EXCHANGE': ('BRR', 'CME', 'USD'),
    'MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE': ('MBT', 'CME', 'USD'),
    'MICRO ETHER - CHICAGO MERCANTILE EXCHANGE': ('MET', 'CME', 'USD'),
    'UST 2Y NOTE - CHICAGO BOARD OF TRADE': ('ZT', 'CBOT', 'USD'),
    'UST 5Y NOTE - CHICAGO BOARD OF TRADE': ('ZF', 'CBOT', 'USD'),
    'UST 10Y NOTE - CHICAGO BOARD OF TRADE': ('ZN', 'CBOT', 'USD'),
    'UST BOND - CHICAGO BOARD OF TRADE': ('ZB', 'CBOT', 'USD'),
    'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE': ('AUD', 'CME', 'USD', '6A'),
    'BRITISH POUND - CHICAGO MERCANTILE EXCHANGE': ('GBP', 'CME', 'USD', '6B'),
    'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE': ('CAD', 'CME', 'USD', '6C'),
    'EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE': ('EUR', 'CME', 'USD', '6E'),
    'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE': ('JPY', 'CME', 'USD', '6J'),
    'MEXICAN PESO - CHICAGO MERCANTILE EXCHANGE': ('MXP', 'CME', 'USD', '6M'),
    'NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE': ('NZD', 'CME', 'USD', '6N'),
    'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE': ('CHF', 'CME', 'USD', '6S'),
    'SO AFRICAN RAND - CHICAGO MERCANTILE EXCHANGE': ('ZAR', 'CME', 'USD', '6Z'),
    'BRAZILIAN REAL - CHICAGO MERCANTILE EXCHANGE': ('BRL', 'CME', 'USD', '6L'),
}


def _mapping_row(market_name: str) -> dict | None:
    spec = IB_MAPPING.get(market_name)
    if not spec:
        return None
    return {
        'IB_Symbol': spec[3] if len(spec) > 3 else spec[0],
        'IB_Exchange': spec[1],
        'IB_Currency': spec[2],
        'IB_TradingClass': spec[3] if len(spec) > 3 else '',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trading-days', type=int, default=30,
                        help='Trading days to add behind current earliest bar')
    parser.add_argument('--market', action='append', help='Limit to specific COT market name(s)')
    parser.add_argument('--interval', choices=['30m', '60m', 'both'], default='both')
    parser.add_argument('--dry-run', action='store_true', help='Print plan only, no IB calls')
    parser.add_argument('--wake-only', action='store_true',
                        help='Connect and wake HMDS farm only (connectivity smoke test)')
    parser.add_argument('--pending-only', action='store_true',
                        help='Skip series that already have backfill_target in roll state')
    parser.add_argument('--host', default=IB_HOST)
    parser.add_argument('--port', type=int, default=IB_PORT)
    parser.add_argument('--client-id', type=int, default=CLIENT_ID)
    args = parser.parse_args()

    cache_df = load_intraday_cache()
    state = load_intraday_state()
    summary = summarize_intraday_ranges(cache_df, state)
    if summary.empty:
        raise SystemExit('No intraday cache found. Run the notebook intraday cell first.')

    markets = args.market or sorted(summary['market'].unique())
    intervals = ['30m', '60m'] if args.interval == 'both' else [args.interval]

    if args.dry_run:
        cal = trading_days_to_calendar_days(args.trading_days)
        print(f'DRY RUN — would backfill {args.trading_days} TD (~{cal} cal days) per series')
        for m in markets:
            for iv in intervals:
                row = summary[(summary['market'] == m) & (summary['interval'] == iv)]
                if row.empty:
                    print(f'  ⏭ {m} {iv}: no cache')
                    continue
                first = row.iloc[0]['first']
                print(f'  {m} {iv}: {first.date() - timedelta(days=cal)} → {first.date()}')
        return

    if args.wake_only:
        ib = connect_ib(args.host, args.port, args.client_id)
        ib.disconnect()
        print('HMDS wake test passed.')
        return

    ib = connect_ib(args.host, args.port, args.client_id)

    total_added = 0
    try:
        for market_name in markets:
            row = _mapping_row(market_name)
            if not row:
                print(f'  ⚠ No IB mapping for {market_name}')
                continue
            for interval in intervals:
                cache_df, state, n = backfill_series(
                    ib,
                    market_name=market_name,
                    display_sym=row['IB_Symbol'],
                    exchange=row['IB_Exchange'],
                    currency=row['IB_Currency'],
                    trading_class=row['IB_TradingClass'],
                    interval=interval,
                    cache_df=cache_df,
                    state=state,
                    trading_days=args.trading_days,
                    pending_only=args.pending_only,
                )
                total_added += n
                if n:
                    persist_intraday(cache_df, state)
                time.sleep(1.5)
    finally:
        ib.disconnect()

    print(f'\n✓ Backfill complete: +{total_added} bars → {INTRADAY_CACHE_FILE}')


if __name__ == '__main__':
    main()
