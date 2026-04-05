"""
Entry filters — decide *how* to enter once a setup is detected.

Each filter takes a row (or DataFrame) with ``setup == True`` and returns
signal direction (+1 long, -1 short, 0 skip) plus entry_price.

All filters share the signature:
    apply_<name>(df, **params) -> DataFrame   (adds signal, entry_price, or_high_signal, or_low_signal)
"""

import pandas as pd
import numpy as np

from .data import get_intraday_for_symbol, get_first_candle_per_day


# ---------------------------------------------------------------------------
# ORB Breakout (intraday first-candle confirmation)
# ---------------------------------------------------------------------------

def apply_orb_breakout(df, market_name='', or_type='30m',
                       cot_filter=False, cot_long=70, cot_short=30,
                       rsi_filter=False, rsi_long_max=70, rsi_short_min=30,
                       cot_direction_filter=False,
                       cot_roc_filter=False, cot_roc_threshold=10,
                       intraday_cache=None, **_kw):
    """Entry only if the first intraday candle breaks yesterday's High/Low.

    Requires OR_High and OR_Low columns (previous day's H/L).
    Uses IBKR intraday cache for first-candle data.
    """
    out = df.copy()
    out['signal'] = 0
    out['entry_price'] = np.nan
    out['or_high_signal'] = np.nan
    out['or_low_signal'] = np.nan

    # Add OR levels if missing
    if 'OR_High' not in out.columns:
        out['OR_High'] = out['High'].shift(1)
        out['OR_Low'] = out['Low'].shift(1)
        out['OR_Range'] = out['OR_High'] - out['OR_Low']

    # Load first-candle data from IBKR cache
    interval = '30m' if or_type == '30m' else '60m'
    intraday = get_intraday_for_symbol(market_name, interval=interval,
                                       cache_df=intraday_cache)
    first_candles = get_first_candle_per_day(intraday) if not intraday.empty else {}

    out['or_candle_high'] = out['Date'].map(
        lambda d: first_candles.get(d, {}).get('high', np.nan))
    out['or_candle_low'] = out['Date'].map(
        lambda d: first_candles.get(d, {}).get('low', np.nan))

    for i in range(1, len(out)):
        row = out.iloc[i]
        if not row.get('setup', False):
            continue

        or_high = row.get('OR_High', np.nan)
        or_low = row.get('OR_Low', np.nan)
        if pd.isna(or_high) or pd.isna(or_low) or (or_high - or_low) <= 0:
            continue

        candle_high = row.get('or_candle_high', np.nan)
        candle_low = row.get('or_candle_low', np.nan)
        if pd.isna(candle_high) or pd.isna(candle_low):
            continue

        long_bo = candle_high > or_high
        short_bo = candle_low < or_low

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

def _apply_filters(row, direction, cot_filter, cot_long, cot_short,
                   rsi_filter, rsi_long_max, rsi_short_min,
                   cot_direction_filter=False,
                   cot_roc_filter=False, cot_roc_threshold=10):
    """Apply optional COT, RSI, COT-direction, and COT-ROC filters; returns 0 if blocked."""
    if cot_filter and not pd.isna(row.get('Commercial_Index')):
        if direction == 1 and row['Commercial_Index'] < cot_long:
            return 0
        if direction == -1 and row['Commercial_Index'] > cot_short:
            return 0
    if cot_direction_filter and not pd.isna(row.get('COT_Change')):
        if direction == 1 and row['COT_Change'] < 0:
            return 0
        if direction == -1 and row['COT_Change'] > 0:
            return 0
    if cot_roc_filter and not pd.isna(row.get('COT_ROC')):
        if direction == 1 and row['COT_ROC'] < cot_roc_threshold:
            return 0
        if direction == -1 and row['COT_ROC'] > -cot_roc_threshold:
            return 0
    if rsi_filter and not pd.isna(row.get('RSI')):
        if direction == 1 and row['RSI'] >= rsi_long_max:
            return 0
        if direction == -1 and row['RSI'] <= rsi_short_min:
            return 0
    return direction


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
}
