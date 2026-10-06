"""
Entry filters — decide *how* to enter once a setup is detected.

Each filter takes a row (or DataFrame) with ``setup == True`` and returns
signal direction (+1 long, -1 short, 0 skip) plus entry_price.

All filters share the signature:
    apply_<name>(df, **params) -> DataFrame   (adds signal, entry_price, or_high_signal, or_low_signal)
"""

import pandas as pd
import numpy as np

from .data import (get_intraday_for_symbol, get_contract_spec,
                   prepare_intraday_sessions, find_post_or_breakout)
from .filters import apply_entry_filters


# ---------------------------------------------------------------------------
# ORB Breakout (intraday first-candle confirmation)
# ---------------------------------------------------------------------------

def apply_orb_breakout(df, market_name='', or_type='30m',
                       cot_filter=False, cot_long=70, cot_short=30,
                       rsi_filter=False, rsi_long_max=70, rsi_short_min=30,
                       cot_direction_filter=False,
                       cot_roc_filter=False, cot_roc_threshold=10,
                       intraday_cache=None, **_kw):
    """Enter on first post-opening-range breakout (30m or 60m window per spec).

    Requires a boolean ``setup`` column. Uses IBKR intraday cache and
    ORB_contract_specs.json session times. Entry fill includes 2-tick slippage.
    """
    out = df.copy()
    out['signal'] = 0
    out['entry_price'] = np.nan
    out['or_high_signal'] = np.nan
    out['or_low_signal'] = np.nan
    out['entry_datetime'] = pd.Series([pd.NaT] * len(out), dtype='datetime64[ns]')
    out['or_window_start'] = pd.Series([pd.NaT] * len(out), dtype='datetime64[ns]')
    out['or_window_end'] = pd.Series([pd.NaT] * len(out), dtype='datetime64[ns]')
    out['tick_size'] = np.nan

    if not market_name:
        return out

    spec = get_contract_spec(market_name)
    if not spec or 'rth_open' not in spec:
        return out
    tick_size = float(spec.get('tick_size', 0.01))
    slippage = 2 * tick_size

    interval = '30m' if or_type == '30m' else '60m'
    intraday = get_intraday_for_symbol(market_name, interval=interval,
                                       cache_df=intraday_cache)
    if intraday.empty:
        return out

    _, sessions = prepare_intraday_sessions(intraday)

    for i in range(len(out)):
        row = out.iloc[i]
        if not row.get('setup', False):
            continue

        breakout = find_post_or_breakout(
            intraday, market_name, row['Date'], or_type, sessions=sessions,
        )
        if not breakout:
            continue

        direction = breakout['direction']
        direction = _apply_filters(row, direction, cot_filter, cot_long,
                                   cot_short, rsi_filter, rsi_long_max,
                                   rsi_short_min, cot_direction_filter,
                                   cot_roc_filter, cot_roc_threshold)
        if direction == 0:
            continue

        entry = breakout['entry_price']
        if direction == 1:
            entry += slippage
        else:
            entry -= slippage

        idx = out.index[i]
        out.at[idx, 'signal'] = direction
        out.at[idx, 'entry_price'] = entry
        out.at[idx, 'or_high_signal'] = breakout['or_high']
        out.at[idx, 'or_low_signal'] = breakout['or_low']
        out.at[idx, 'entry_datetime'] = _naive_et(breakout['entry_datetime'])
        out.at[idx, 'or_window_start'] = _naive_et(breakout['window_start'])
        out.at[idx, 'or_window_end'] = _naive_et(breakout['window_end'])
        out.at[idx, 'tick_size'] = tick_size

    return out


# ---------------------------------------------------------------------------
# Daily Breakout (no intraday data needed)
# ---------------------------------------------------------------------------

def apply_daily_breakout(df, cot_filter=False, cot_long=70, cot_short=30,
                         rsi_filter=False, rsi_long_max=70, rsi_short_min=30,
                         cot_direction_filter=False,
                         cot_roc_filter=False, cot_roc_threshold=10,
                         **_kw):
    """Entry when daily bar breaks yesterday's High/Low.

    Long  if today's High > yesterday's High
    Short if today's Low  < yesterday's Low
    """
    out = df.copy()
    out['signal'] = 0
    out['entry_price'] = np.nan
    out['or_high_signal'] = np.nan
    out['or_low_signal'] = np.nan

    if 'OR_High' not in out.columns:
        out['OR_High'] = out['High'].shift(1)
        out['OR_Low'] = out['Low'].shift(1)
        out['OR_Range'] = out['OR_High'] - out['OR_Low']

    for i in range(1, len(out)):
        row = out.iloc[i]
        if not row.get('setup', False):
            continue

        or_high = row.get('OR_High', np.nan)
        or_low = row.get('OR_Low', np.nan)
        if pd.isna(or_high) or pd.isna(or_low) or (or_high - or_low) <= 0:
            continue

        long_bo = row['High'] > or_high
        short_bo = row['Low'] < or_low

        direction = 0
        if long_bo and short_bo:
            mid = (or_high + or_low) / 2
            direction = 1 if row.get('Open', mid) >= mid else -1
        elif long_bo:
            direction = 1
        elif short_bo:
            direction = -1

        if direction == 0:
            continue

        direction = _apply_filters(row, direction, cot_filter, cot_long,
                                   cot_short, rsi_filter, rsi_long_max,
                                   rsi_short_min, cot_direction_filter,
                                   cot_roc_filter, cot_roc_threshold)
        if direction == 0:
            continue

        idx = out.index[i]
        out.at[idx, 'signal'] = direction
        out.at[idx, 'entry_price'] = or_high if direction == 1 else or_low
        out.at[idx, 'or_high_signal'] = or_high
        out.at[idx, 'or_low_signal'] = or_low

    return out


# ---------------------------------------------------------------------------
# Market-on-Close entry (for COT+RSI — enter at close on signal day)
# ---------------------------------------------------------------------------

def apply_close_entry(df, **_kw):
    """Enter at the Close on the setup day.  Direction comes from
    ``setup_direction`` set by the COT+RSI setup detector.
    """
    out = df.copy()
    out['signal'] = 0
    out['entry_price'] = np.nan
    out['or_high_signal'] = np.nan
    out['or_low_signal'] = np.nan

    for i in range(len(out)):
        row = out.iloc[i]
        if not row.get('setup', False):
            continue
        direction = int(row.get('setup_direction', 0))
        if direction == 0:
            continue

        idx = out.index[i]
        out.at[idx, 'signal'] = direction
        out.at[idx, 'entry_price'] = row['Close']

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _naive_et(ts):
    """Store timezone-aware ET timestamps as naive local times for pandas."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return pd.NaT
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        from .data import ET
        return t.tz_convert(ET).tz_localize(None)
    return t


def _apply_filters(row, direction, cot_filter, cot_long, cot_short,
                   rsi_filter, rsi_long_max, rsi_short_min,
                   cot_direction_filter=False,
                   cot_roc_filter=False, cot_roc_threshold=10,
                   ma_trend_filter=False, ma_period=0):
    return apply_entry_filters(
        row, direction,
        cot_filter=cot_filter, cot_long=cot_long, cot_short=cot_short,
        rsi_filter=rsi_filter, rsi_long_max=rsi_long_max, rsi_short_min=rsi_short_min,
        cot_direction_filter=cot_direction_filter,
        cot_roc_filter=cot_roc_filter, cot_roc_threshold=cot_roc_threshold,
        ma_trend_filter=ma_trend_filter, ma_period=ma_period,
    )


def apply_nday_breakout(df, lookback=20, **_kw):
    """N-day high/low breakout entry at the breakout level."""
    from .indicators import add_nday_breakout_bands

    out = add_nday_breakout_bands(df, lookback=lookback)
    out['signal'] = 0
    out['entry_price'] = np.nan

    long_bo = out['High'] > out['n_day_high']
    short_bo = out['Low'] < out['n_day_low']
    long_only = long_bo & ~short_bo
    short_only = short_bo & ~long_bo
    both = long_bo & short_bo

    out.loc[long_only, 'signal'] = 1
    out.loc[long_only, 'entry_price'] = out.loc[long_only, 'n_day_high']
    out.loc[short_only, 'signal'] = -1
    out.loc[short_only, 'entry_price'] = out.loc[short_only, 'n_day_low']
    if both.any():
        mid = (out['n_day_high'] + out['n_day_low']) / 2
        long_both = both & (out['Open'] >= mid)
        short_both = both & ~long_both
        out.loc[long_both, 'signal'] = 1
        out.loc[long_both, 'entry_price'] = out.loc[long_both, 'n_day_high']
        out.loc[short_both, 'signal'] = -1
        out.loc[short_both, 'entry_price'] = out.loc[short_both, 'n_day_low']
    return out


def apply_next_open(df, **_kw):
    """Enter at next bar open when setup fired on prior bar (uses setup + setup_direction)."""
    out = df.copy()
    out['signal'] = 0
    out['entry_price'] = np.nan
    for i in range(1, len(out)):
        prev = out.iloc[i - 1]
        if not prev.get('setup', False):
            continue
        direction = int(prev.get('setup_direction', 0))
        if direction == 0:
            continue
        row = out.iloc[i]
        out.iloc[i, out.columns.get_loc('signal')] = direction
        out.iloc[i, out.columns.get_loc('entry_price')] = row['Open']
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ENTRY_REGISTRY = {
    'orb_breakout': {
        'fn': apply_orb_breakout,
        'label': 'ORB Breakout (Intraday)',
        'params': {
            'or_type': {'type': str, 'default': '60m',
                        'options': ['30m', '60m'], 'label': 'Opening Range'},
        },
    },
    'daily_breakout': {
        'fn': apply_daily_breakout,
        'label': 'Daily Breakout (Prev Day H/L)',
        'params': {},
    },
    'close_entry': {
        'fn': apply_close_entry,
        'label': 'Market-on-Close (Signal Day)',
        'params': {},
    },
    'nday_breakout': {
        'fn': apply_nday_breakout,
        'label': 'N-Day High/Low Breakout',
        'params': {'lookback': {'type': int, 'default': 20, 'min': 5, 'max': 200,
                                 'label': 'Lookback Days'}},
    },
    'next_open': {
        'fn': apply_next_open,
        'label': 'Next Session Open (after setup)',
        'params': {},
    },
}
