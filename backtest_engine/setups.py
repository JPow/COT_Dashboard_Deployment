"""
Setup detectors — identify candidate trade setups on daily data.

Each detector takes a prepared DataFrame (with indicators already computed)
and returns the same DataFrame with a boolean ``setup`` column plus any
setup-specific metadata columns.

All detectors share the signature:
    detect_<name>(df, **params) -> DataFrame
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Narrowing Range
# ---------------------------------------------------------------------------

def detect_narrowing_range(df, n_days=3, **_kw):
    """Setup fires when *previous day* ended a streak of >= n_days
    consecutive narrowing ranges.

    Requires ``consecutive_narrowing`` column (from indicators.calculate_narrowing_ranges).
    """
    from .indicators import calculate_narrowing_ranges

    out = df.copy()
    if 'consecutive_narrowing' not in out.columns:
        out = calculate_narrowing_ranges(out)

    out['setup'] = False
    for i in range(1, len(out)):
        if out.iloc[i - 1].get('consecutive_narrowing', 0) >= n_days:
            out.iloc[i, out.columns.get_loc('setup')] = True
    return out


# ---------------------------------------------------------------------------
# Inside Days
# ---------------------------------------------------------------------------

def detect_inside_days(df, n_days=3, **_kw):
    """Setup fires when *previous day* ended a streak of >= n_days
    consecutive inside days (relative to the control/mother bar).

    Requires ``consecutive_inside_days`` column (from indicators.calculate_inside_days).
    """
    from .indicators import calculate_inside_days

    out = df.copy()
    if 'consecutive_inside_days' not in out.columns:
        out = calculate_inside_days(out)

    out['setup'] = False
    for i in range(1, len(out)):
        if out.iloc[i - 1].get('consecutive_inside_days', 0) >= n_days:
            out.iloc[i, out.columns.get_loc('setup')] = True
    return out


# ---------------------------------------------------------------------------
# COT + RSI Extremes
# ---------------------------------------------------------------------------

def detect_cot_rsi(df, commercial_long=80, commercial_short=20,
                   rsi_oversold=30, rsi_overbought=70, ma_period=0, **_kw):
    """Setup fires when COT Commercial Index and RSI hit extreme levels.

    Long  : Commercial_Index >= commercial_long AND RSI < rsi_oversold
    Short : Commercial_Index <= commercial_short AND RSI > rsi_overbought

    Optional MA trend filter (ma_period > 0):
        Long also requires Close > MA, Short requires Close < MA.

    Adds columns: setup (bool) and setup_direction (+1 / -1 / 0).
    """
    out = df.copy()

    rsi_col = 'RSI' if 'RSI' in out.columns else None
    ci_col = 'Commercial_Index' if 'Commercial_Index' in out.columns else None
    if rsi_col is None or ci_col is None:
        out['setup'] = False
        out['setup_direction'] = 0
        return out

    use_ma = ma_period > 0
    ma_col = f'MA_{ma_period}'
    if use_ma and ma_col not in out.columns:
        out[ma_col] = out['Close'].rolling(window=ma_period).mean()

    long_cond = (out[ci_col] >= commercial_long) & (out[rsi_col] < rsi_oversold)
    short_cond = (out[ci_col] <= commercial_short) & (out[rsi_col] > rsi_overbought)

    if use_ma:
        long_cond = long_cond & (out['Close'] > out[ma_col]) & out[ma_col].notna()
        short_cond = short_cond & (out['Close'] < out[ma_col]) & out[ma_col].notna()

    out['setup'] = long_cond | short_cond
    out['setup_direction'] = 0
    out.loc[long_cond, 'setup_direction'] = 1
    out.loc[short_cond, 'setup_direction'] = -1

    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SETUP_REGISTRY = {
    'narrowing_range': {
        'fn': detect_narrowing_range,
        'label': 'Narrowing Range',
        'params': {'n_days': {'type': int, 'default': 3, 'min': 2, 'max': 10,
                              'label': 'Narrowing Days'}},
    },
    'inside_days': {
        'fn': detect_inside_days,
        'label': 'Inside Days',
        'params': {'n_days': {'type': int, 'default': 3, 'min': 2, 'max': 10,
                              'label': 'Inside Days'}},
    },
    'cot_rsi': {
        'fn': detect_cot_rsi,
        'label': 'COT + RSI Extremes',
        'params': {
            'commercial_long':  {'type': int, 'default': 80, 'min': 50, 'max': 100, 'label': 'COT Long >='},
            'commercial_short': {'type': int, 'default': 20, 'min': 0,  'max': 50,  'label': 'COT Short <='},
            'rsi_oversold':     {'type': int, 'default': 30, 'min': 10, 'max': 50,  'label': 'RSI Oversold <'},
            'rsi_overbought':   {'type': int, 'default': 70, 'min': 50, 'max': 90,  'label': 'RSI Overbought >'},
            'ma_period':        {'type': int, 'default': 0,  'min': 0,  'max': 200, 'label': 'MA Trend Filter'},
        },
    },
}
