"""
Data loading for COT daily/weekly prices and IBKR intraday cache.
"""

import os
import json
import pandas as pd
import numpy as np

COT_DATA_FILE = 'cot_data.json'
INTRADAY_CACHE_FILE = 'ORB_intraday_data.json'


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
