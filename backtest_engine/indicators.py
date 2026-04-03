"""
Technical indicators used across all strategy models.
"""

import pandas as pd
import numpy as np


def calculate_atr(data, period=10, col_name='ATR'):
    """Average True Range (SMA of True Range).

    TR[i] = max(H-L, |H - prevClose|, |L - prevClose|)
    ATR   = SMA(TR, period)

    Falls back to 1.5 × rolling std(Close) when High/Low are absent.
    """
    df = data.copy()
    if 'High' not in df.columns or 'Low' not in df.columns:
        df[col_name] = df['Close'].rolling(window=period).std() * 1.5
        return df

    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[col_name] = df['TR'].rolling(window=period).mean()
    df.drop(columns=['TR'], inplace=True, errors='ignore')
    return df


def calculate_rsi(data, period=10, col_name='RSI'):
    """Relative Strength Index.

    RSI = 100 - 100 / (1 + RS)
    RS  = SMA(gain, period) / SMA(loss, period)
    """
    df = data.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    df[col_name] = 100 - (100 / (1 + rs))
    return df


def calculate_moving_averages(data, periods=None):
    """Simple Moving Averages on Close.  Adds columns MA_10, MA_20, …"""
    if periods is None:
        periods = [10, 20, 50, 100, 150, 200]
    df = data.copy()
    for p in periods:
        df[f'MA_{p}'] = df['Close'].rolling(window=p).mean()
    return df


def calculate_narrowing_ranges(data):
    """Count consecutive days where daily range shrinks vs. previous day.

    Adds columns: daily_range, narrowing_day, consecutive_narrowing.
    """
    df = data.copy()
    df['daily_range'] = df['High'] - df['Low']
    df['prev_range'] = df['daily_range'].shift(1)
    df['narrowing_day'] = df['daily_range'] <= df['prev_range']

    consecutive = [0] * len(df)
    count = 0
    for i in range(len(df)):
        if df.iloc[i]['narrowing_day']:
            count += 1
        else:
            count = 0
        consecutive[i] = count
    df['consecutive_narrowing'] = consecutive
    df.drop(columns=['prev_range'], inplace=True, errors='ignore')
    return df


def calculate_inside_days(data):
    """Identify inside days relative to the control (mother) bar.

    Inside_Day[i] = (H[i] <= control_H) AND (L[i] >= control_L)
    where control bar is the bar before the inside-day sequence started.
    Equal highs/lows count as inside (<=, >=).

    Adds columns: consecutive_inside_days, inside_day.
    """
    df = data.copy()
    n = len(df)
    consecutive = [0] * n
    control_high = None
    control_low = None
    count = 0

    for i in range(1, n):
        curr_high = df.iloc[i]['High']
        curr_low = df.iloc[i]['Low']

        if count == 0:
            prev_high = df.iloc[i - 1]['High']
            prev_low = df.iloc[i - 1]['Low']
            if curr_high <= prev_high and curr_low >= prev_low:
                control_high = prev_high
                control_low = prev_low
                count = 1
        else:
            if curr_high <= control_high and curr_low >= control_low:
                count += 1
            else:
                prev_high = df.iloc[i - 1]['High']
                prev_low = df.iloc[i - 1]['Low']
                if curr_high <= prev_high and curr_low >= prev_low:
                    control_high = prev_high
                    control_low = prev_low
                    count = 1
                else:
                    count = 0
                    control_high = None
                    control_low = None

        consecutive[i] = count

    df['consecutive_inside_days'] = consecutive
    df['inside_day'] = [c > 0 for c in consecutive]
    return df


def add_standard_indicators(df, fast_atr=10, slow_atr=25, rsi_period=10,
                            ma_periods=None):
    """Convenience: attach the full indicator suite to a daily DataFrame."""
    df = calculate_rsi(df, period=rsi_period)
    df = calculate_atr(df, period=fast_atr, col_name='fast_ATR')
    df = calculate_atr(df, period=slow_atr, col_name='slow_ATR')
    df = calculate_moving_averages(df, periods=ma_periods)
    return df
