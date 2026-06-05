"""
Data loading for COT daily/weekly prices and IBKR intraday cache.
"""

import os
import json
from datetime import date, time, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

ET = ZoneInfo('America/New_York')
SESSION_TZ = 'America/New_York'

_MONTH_CODES = {
    'F': 'Jan', 'G': 'Feb', 'H': 'Mar', 'J': 'Apr', 'K': 'May', 'M': 'Jun',
    'N': 'Jul', 'Q': 'Aug', 'U': 'Sep', 'V': 'Oct', 'X': 'Nov', 'Z': 'Dec',
}


def describe_contract(local_symbol, ref_date=None):
    """Map IB localSymbol like 'QIN6' to 'Jul 2026 (QIN6)' for chart badges.

    Falls back to raw ``local_symbol`` if the suffix is not a standard month+year.
    ``ref_date`` resolves single-digit year to the correct decade (default: today).
    """
    if not local_symbol or not isinstance(local_symbol, str):
        return local_symbol or ''
    s = local_symbol.strip()
    if len(s) < 3:
        return s
    month_char = s[-2]
    year_char = s[-1]
    if month_char not in _MONTH_CODES or not year_char.isdigit():
        return s
    ref = ref_date or date.today()
    if hasattr(ref, 'date'):
        ref = ref.date()
    ref_year = ref.year
    decade = (ref_year // 10) * 10
    year = decade + int(year_char)
    if year < ref_year - 2:
        year += 10
    return f"{_MONTH_CODES[month_char]} {year} ({s})"

COT_DATA_FILE = 'cot_data.json'
INTRADAY_CACHE_FILE = 'ORB_intraday_data.json'
CONTRACT_SPECS_FILE = 'ORB_contract_specs.json'

_CONTRACT_SPECS_CACHE = None


def load_cot_data(path=None):
    """Load daily OHLCV + weekly COT data from the shared JSON bundle.

    Returns a DataFrame with columns including Date, Market, data_type,
    Open, High, Low, Close, Volume, Commercial_Index, RSI, etc.
    """
    path = path or COT_DATA_FILE
    try:
        with open(path, 'r') as f:
            data_raw = json.load(f)
        df = pd.DataFrame(data_raw)
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        return df
    except Exception as e:
        print(f"Error loading COT data from {path}: {e}")
        return pd.DataFrame()


def load_contract_specs(path=None):
    """Load futures contract specifications (tick_size, tick_value, point_value).

    Specs are cached after first load. Keys are the short market name
    (the portion before ' - ' in the COT Market column). Optional ``_meta``
    holds shared defaults (timezone, session notes).
    """
    global _CONTRACT_SPECS_CACHE
    if _CONTRACT_SPECS_CACHE is not None:
        return _CONTRACT_SPECS_CACHE
    path = path or CONTRACT_SPECS_FILE
    try:
        with open(path, 'r') as f:
            _CONTRACT_SPECS_CACHE = json.load(f)
    except FileNotFoundError:
        print(f"Warning: contract specs not found at {path}")
        _CONTRACT_SPECS_CACHE = {}
    return _CONTRACT_SPECS_CACHE


def get_contract_spec(market_name):
    """Return the spec dict for a market, or None if not found.

    Matches on the short name (before ' - '). Ignores the ``_meta`` key.
    """
    specs = load_contract_specs()
    short = market_name.split(" - ")[0].strip()
    spec = specs.get(short)
    if spec is None or not isinstance(spec, dict) or 'tick_size' not in spec:
        return None
    return spec


def get_contract_specs_meta():
    """Return shared metadata from ORB_contract_specs.json (timezone, notes)."""
    return load_contract_specs().get('_meta', {})


def load_intraday_cache(path=None):
    """Load cached IBKR intraday bars from ORB_intraday_data.json."""
    path = path or INTRADAY_CACHE_FILE
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        print(f"Error loading intraday cache from {path}: {e}")
        return pd.DataFrame()


def get_intraday_for_symbol(symbol, interval='30m', cache_df=None):
    """Return intraday rows for one symbol + interval from the cache."""
    if cache_df is None:
        cache_df = load_intraday_cache()
    if cache_df.empty:
        return pd.DataFrame()
    match = cache_df[(cache_df['symbol'] == symbol) & (cache_df['interval'] == interval)]
    return match.reset_index(drop=True) if not match.empty else pd.DataFrame()


def _parse_hhmm(hhmm):
    """Parse 'HH:MM' session time string."""
    parts = str(hhmm).strip().split(':')
    return time(int(parts[0]), int(parts[1]))


def get_session_times(market_name, or_type='30m'):
    """Return RTH open and OR window end from contract specs (ET)."""
    spec = get_contract_spec(market_name)
    if not spec or 'rth_open' not in spec:
        return None
    end_key = '30_close' if or_type == '30m' else '60_close'
    if end_key not in spec:
        return None
    return {
        'rth_open': _parse_hhmm(spec['rth_open']),
        'or_end': _parse_hhmm(spec[end_key]),
        'timezone': SESSION_TZ,
        'rth_open_str': spec['rth_open'],
        'or_end_str': spec[end_key],
    }


def normalize_intraday_to_et(intraday_df):
    """Convert IB cache timestamps (UTC-naive) to US/Eastern for session matching."""
    if intraday_df.empty:
        return intraday_df
    df = intraday_df.copy()
    dt = pd.to_datetime(df['datetime'])
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize('UTC')
    df['datetime_et'] = dt.dt.tz_convert(ET)
    df['session_date'] = df['datetime_et'].dt.date
    df['time_et'] = df['datetime_et'].dt.time
    return df


def prepare_intraday_sessions(intraday_df):
    """Normalize intraday bars and index by ET session date."""
    norm = normalize_intraday_to_et(intraday_df)
    if norm.empty:
        return norm, {}
    sessions = {
        d: g.sort_values('datetime_et').reset_index(drop=True)
        for d, g in norm.groupby('session_date')
    }
    return norm, sessions


def _session_datetime_et(session_date, t):
    """Combine ET calendar date and time into a timezone-aware timestamp."""
    return datetime.combine(session_date, t, tzinfo=ET)


def build_opening_range_day(intraday_df, market_name, trade_date, or_type='30m',
                            sessions=None):
    """Build today's opening range from intraday bars in [rth_open, or_end)."""
    times = get_session_times(market_name, or_type)
    if not times:
        return {'valid': False, 'reason': 'missing session spec'}

    trade_day = pd.Timestamp(trade_date).date()
    if sessions is None:
        _, sessions = prepare_intraday_sessions(intraday_df)
    day_bars = sessions.get(trade_day)
    if day_bars is None or day_bars.empty:
        return {'valid': False, 'reason': 'no intraday bars for session date'}

    rth_open = times['rth_open']
    or_end = times['or_end']
    or_bars = day_bars[
        (day_bars['time_et'] >= rth_open) &
        (day_bars['time_et'] < or_end)
    ]
    if or_bars.empty:
        return {'valid': False, 'reason': 'no bars in OR window'}

    return {
        'valid': True,
        'or_high': float(or_bars['high'].max()),
        'or_low': float(or_bars['low'].min()),
        'or_open': float(or_bars.iloc[0]['open']),
        'window_start': _session_datetime_et(trade_day, rth_open),
        'window_end': _session_datetime_et(trade_day, or_end),
        'reason': None,
    }


def find_post_or_breakout(intraday_df, market_name, trade_date, or_type='30m',
                          sessions=None):
    """First post-OR bar that breaks today's opening range."""
    times = get_session_times(market_name, or_type)
    if not times:
        return None

    or_info = build_opening_range_day(
        intraday_df, market_name, trade_date, or_type, sessions=sessions,
    )
    if not or_info['valid']:
        return None

    trade_day = pd.Timestamp(trade_date).date()
    if sessions is None:
        _, sessions = prepare_intraday_sessions(intraday_df)
    day_bars = sessions.get(trade_day)
    if day_bars is None or day_bars.empty:
        return None

    or_end = times['or_end']
    or_high = or_info['or_high']
    or_low = or_info['or_low']
    or_open = or_info['or_open']
    post_bars = day_bars[day_bars['time_et'] >= or_end].sort_values('datetime_et')

    for _, bar in post_bars.iterrows():
        long_bo = bar['high'] > or_high
        short_bo = bar['low'] < or_low
        direction = 0
        if long_bo and short_bo:
            mid = (or_high + or_low) / 2
            direction = 1 if or_open >= mid else -1
        elif long_bo:
            direction = 1
        elif short_bo:
            direction = -1
        if direction == 0:
            continue
        entry_level = or_high if direction == 1 else or_low
        return {
            'direction': direction,
            'entry_price': float(entry_level),
            'entry_datetime': bar['datetime_et'],
            'or_high': or_high,
            'or_low': or_low,
            'or_open': or_open,
            'window_start': or_info['window_start'],
            'window_end': or_info['window_end'],
        }
    return None


def get_first_candle_per_day(intraday_df):
    """Extract the first candle of each trading day from intraday data.

    Returns dict: {Timestamp(date) -> {'high', 'low', 'open', 'close'}}
    """
    if intraday_df.empty:
        return {}
    df = intraday_df.copy()
    df['date'] = df['datetime'].dt.date
    first_candles = {}
    for date, group in df.groupby('date'):
        group = group.sort_values('datetime')
        if len(group) == 0:
            continue
        first = group.iloc[0]
        first_candles[pd.Timestamp(date)] = {
            'high': first['high'],
            'low': first['low'],
            'open': first['open'],
            'close': first['close'],
        }
    return first_candles


def extract_market_data(cot_df, market_name):
    """Split a market's rows into daily price and weekly COT DataFrames.

    Returns (price_daily, cot_weekly) — both may be empty.
    """
    market_data = cot_df[cot_df['Market'] == market_name].copy()
    cot_weekly = market_data[market_data['data_type'] == 'weekly_cot'].copy()
    price_daily = market_data[market_data['data_type'] == 'daily_price'].copy()
    return price_daily, cot_weekly


def prepare_base_data(cot_df, market_name):
    """Build a clean daily OHLCV DataFrame for a single market.

    * Drops rows missing Close/High/Low
    * Removes weekends for non-crypto markets
    * Merges forward-filled Commercial_Index from weekly COT
    """
    price_daily, cot_weekly = extract_market_data(cot_df, market_name)
    if price_daily.empty:
        return pd.DataFrame()

    price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    available = [c for c in price_cols if c in price_daily.columns]
    if 'Contract' in price_daily.columns:
        available.append('Contract')
    df = price_daily[available].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['Close', 'High', 'Low'])

    is_crypto = 'BITCOIN' in market_name.upper() or 'ETHER' in market_name.upper()
    if not is_crypto:
        df = df[df['Date'].dt.dayofweek < 5].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # Merge COT + week-on-week change + 3-week ROC
    if 'Commercial_Index' in cot_weekly.columns:
        cot_merge = (cot_weekly[['Date', 'Commercial_Index']]
                     .dropna(subset=['Commercial_Index'])
                     .sort_values('Date')
                     .reset_index(drop=True))
        cot_merge['COT_Change'] = cot_merge['Commercial_Index'].diff()
        cot_merge['COT_ROC'] = cot_merge['Commercial_Index'].diff(3)
        df = df.sort_values('Date').reset_index(drop=True)
        df = pd.merge_asof(df, cot_merge, on='Date', direction='backward')
    else:
        df['Commercial_Index'] = np.nan
        df['COT_Change'] = np.nan
        df['COT_ROC'] = np.nan

    return df
