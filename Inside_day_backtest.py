"""
Inside Day Breakout Strategy Backtest Dashboard
=================================================
Dash app for backtesting Inside Day Breakout strategies with:
- 3 OR types: 30-min, 60-min, Previous Day Range
- Inside day consolidation filter (N consecutive inside days)
- Two-phase stop management: fixed → breakeven → trailing
- COT and RSI filters (toggleable)
- Fast/Slow ATR, 6 Moving Averages, RSI indicators

Data:
- Daily OHLCV + COT data from cot_data.json (shared with COT dashboard)
- Intraday 30m/60m data cached in ORB_intraday_data.json
"""

import pandas as pd
import numpy as np
import json
import os
import time
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None
    print("WARNING: yfinance not installed. Intraday data download disabled.")


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_START_DATE = '2023-01-01'
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
DEFAULT_CAPITAL = 30000
DEFAULT_RISK_PCT = 1.0
DEFAULT_TRAILING_ATR_MULT = 2.0
DEFAULT_FAST_ATR = 10
DEFAULT_SLOW_ATR = 25
DEFAULT_INSIDE_DAYS = 3
STOP_BUFFER = 0.01  # $0.01 buffer on stops

INTRADAY_CACHE_FILE = 'ORB_intraday_data.json'

# Yahoo Finance symbol mapping (Market name -> YF symbol)
MARKET_YF_SYMBOLS = {
    'GOLD': 'GC=F', 'SILVER': 'SI=F', 'PLATINUM': 'PL=F', 'PALLADIUM': 'PA=F',
    'COPPER-GRADE #1': 'HG=F', 'MICRO GOLD': 'MGC=F',
    'WTI-PHYSICAL': 'CL=F', 'GASOLINE RBOB': 'RB=F', 'NAT GAS NYME': 'NG=F',
    'HEATING OIL': 'HO=F', 'BRENT LAST DAY': 'BZ=F',
    'CORN': 'ZC=F', 'OATS': 'ZO=F', 'WHEAT-SRW': 'ZW=F',
    'SOYBEANS': 'ZS=F', 'SOYBEAN MEAL': 'ZM=F', 'SOYBEAN OIL': 'ZL=F',
    'COCOA': 'CC=F', 'COFFEE C': 'KC=F', 'COTTON NO. 2': 'CT=F',
    'SUGAR NO. 11': 'SB=F', 'ORANGE JUICE': 'OJ=F',
    'FEEDER CATTLE': 'GF=F', 'LEAN HOGS': 'HE=F', 'LIVE CATTLE': 'LE=F',
    'AUSTRALIAN DOLLAR': '6A=F', 'BRITISH POUND': '6B=F', 'CANADIAN DOLLAR': '6C=F',
    'EURO FX': '6E=F', 'JAPANESE YEN': '6J=F', 'MEXICAN PESO': '6M=F',
    'NEW ZEALAND DOLLAR': '6N=F', 'SWISS FRANC': '6S=F',
    'SO AFRICAN RAND': '6Z=F', 'BRAZILIAN REAL': '6L=F',
    'E-MINI S&P 500': 'ES=F', 'NASDAQ MINI': 'NQ=F',
    'DOW JONES INDUSTRIAL AVG': 'YM=F', 'RUSSELL E-MINI': 'RTY=F',
    'NIKKEI STOCK AVERAGE': 'NKD=F', 'VIX FUTURES': 'VX=F',
    'BITCOIN': 'BTC=F', 'MICRO BITCOIN': 'BTC-USD',
    'ETHER CASH SETTLED': 'ETH=F', 'MICRO ETHER': 'ETH-USD',
    'UST 2Y NOTE': 'ZT=F', 'UST 5Y NOTE': 'ZF=F',
    'UST 10Y NOTE': 'ZN=F', 'UST BOND': 'ZB=F',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cot_data():
    """Load daily OHLCV + weekly COT data from shared JSON file."""
    try:
        with open('cot_data.json', 'r') as f:
            data_raw = json.load(f)
        df = pd.DataFrame(data_raw)
        df["Date"] = pd.to_datetime(df["Date"], unit='ms')
        return df
    except Exception as e:
        print(f"Error loading COT data: {e}")
        return pd.DataFrame()


def load_intraday_cache():
    """Load cached intraday data from ORB_intraday_data.json."""
    if not os.path.exists(INTRADAY_CACHE_FILE):
        return pd.DataFrame()
    try:
        with open(INTRADAY_CACHE_FILE, 'r') as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        print(f"Error loading intraday cache: {e}")
        return pd.DataFrame()


def save_intraday_cache(df):
    """Save intraday data to ORB_intraday_data.json."""
    try:
        save_df = df.copy()
        save_df['datetime'] = save_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        with open(INTRADAY_CACHE_FILE, 'w') as f:
            json.dump(save_df.to_dict('records'), f, indent=2)
    except Exception as e:
        print(f"Error saving intraday cache: {e}")


def download_intraday_data(symbol, yf_symbol, interval='30m'):
    """
    Download intraday data from Yahoo Finance and merge with cache.
    
    Limits:
      - 30m: max 60 days back
      - 60m: max 730 days back
    
    Returns DataFrame with columns: symbol, interval, datetime, open, high, low, close, volume
    """
    if yf is None:
        print("yfinance not available, cannot download intraday data")
        return pd.DataFrame()

    # Load existing cache
    cache = load_intraday_cache()

    # Check what we already have for this symbol+interval
    if not cache.empty:
        existing = cache[(cache['symbol'] == symbol) & (cache['interval'] == interval)]
        if not existing.empty:
            last_cached = existing['datetime'].max()
            # If cache is recent enough (within last trading day), skip download
            if last_cached >= pd.Timestamp.now() - pd.Timedelta(days=2):
                print(f"  Cache hit for {symbol} ({interval}), last data: {last_cached}")
                return existing.reset_index(drop=True)

    # Determine download period based on interval
    if interval == '30m':
        period = '60d'  # Max for 30-min
    elif interval == '60m':
        period = '730d'  # Max for 60-min
    else:
        period = '60d'

    try:
        print(f"  Downloading {interval} data for {yf_symbol} (period={period})...")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        time.sleep(1)  # Rate limit protection

        if df.empty:
            print(f"  No data returned for {yf_symbol}")
            return pd.DataFrame()

        df = df.reset_index()
        # yfinance returns 'Datetime' for intraday
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        
        result = pd.DataFrame({
            'symbol': symbol,
            'interval': interval,
            'datetime': pd.to_datetime(df[date_col]).dt.tz_localize(None),
            'open': df['Open'].values,
            'high': df['High'].values,
            'low': df['Low'].values,
            'close': df['Close'].values,
            'volume': df['Volume'].values,
        })

        # Merge with cache (remove old entries for this symbol+interval, add new)
        if not cache.empty:
            cache = cache[~((cache['symbol'] == symbol) & (cache['interval'] == interval))]
            cache = pd.concat([cache, result], ignore_index=True)
        else:
            cache = result.copy()

        save_intraday_cache(cache)
        print(f"  Downloaded {len(result)} candles for {yf_symbol}")
        return result.reset_index(drop=True)

    except Exception as e:
        print(f"  Error downloading {yf_symbol}: {e}")
        # Fall back to cache if available
        if not cache.empty:
            existing = cache[(cache['symbol'] == symbol) & (cache['interval'] == interval)]
            if not existing.empty:
                return existing.reset_index(drop=True)
        return pd.DataFrame()


def get_yf_symbol(market_name):
    """Get Yahoo Finance symbol for a market, trying direct match then partial."""
    if market_name in MARKET_YF_SYMBOLS:
        return MARKET_YF_SYMBOLS[market_name]
    # Try partial match
    for key, val in MARKET_YF_SYMBOLS.items():
        if key in market_name or market_name in key:
            return val
    return None


# =============================================================================
# INDICATORS
# =============================================================================

def calculate_atr(data, period=10, col_name='ATR'):
    """
    Calculate Average True Range.
    
    True_Range[i] = max(High[i]-Low[i], |High[i]-Close[i-1]|, |Low[i]-Close[i-1]|)
    ATR = SMA(True_Range, period)
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


def calculate_moving_averages(data, periods=None):
    """
    Calculate Simple Moving Averages.
    MA_N = SMA(Close, N) for N in {10, 20, 50, 100, 150, 200}
    """
    if periods is None:
        periods = [10, 20, 50, 100, 150, 200]
    df = data.copy()
    for p in periods:
        df[f'MA_{p}'] = df['Close'].rolling(window=p).mean()
    return df


def calculate_rsi(data, period=10):
    """
    Calculate RSI (Relative Strength Index).
    
    RSI = 100 - (100 / (1 + RS))
    RS = SMA(gain, period) / SMA(loss, period)
    """
    df = data.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_inside_days(data):
    """
    Identify inside days relative to the CONTROL candle (mother bar).
    
    The control candle is the bar immediately before the inside day sequence.
    All inside days in the sequence must stay within the control candle's range.
    
    Inside_Day[i] = (High[i] <= Control_High) AND (Low[i] >= Control_Low)
    
    Equal highs/lows count as inside (<=, >=, not strict <, >).
    
    Example: If Day 0 has H=3.885, L=3.797, and the next 3 days all have
    highs <= 3.885 and lows >= 3.797, that's 3 consecutive inside days.
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
            # No active sequence — check against previous bar (potential control candle)
            prev_high = df.iloc[i - 1]['High']
            prev_low = df.iloc[i - 1]['Low']
            if curr_high <= prev_high and curr_low >= prev_low:
                # Inside day: previous bar becomes the control candle
                control_high = prev_high
                control_low = prev_low
                count = 1
        else:
            # Active sequence — check against the CONTROL candle, not previous bar
            if curr_high <= control_high and curr_low >= control_low:
                count += 1
            else:
                # Broke out of control range — reset
                # Check if this bar starts a new sequence vs. previous bar
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


# =============================================================================
# OPENING RANGE CALCULATION
# =============================================================================

def calculate_opening_range_intraday(intraday_df, interval='30m'):
    """
    Extract OR_High and OR_Low for each trading day from intraday data.
    
    30-min OR: High/Low of first 30-min candle of each session
    60-min OR: High/Low of first 60-min candle of each session
    
    Returns DataFrame with columns: Date, OR_High, OR_Low, OR_Range
    """
    if intraday_df.empty:
        return pd.DataFrame(columns=['Date', 'OR_High', 'OR_Low', 'OR_Range'])

    df = intraday_df.copy()
    df['date'] = df['datetime'].dt.date

    # Group by date, take the first candle of each day (= opening range)
    or_data = []
    for date, group in df.groupby('date'):
        group = group.sort_values('datetime')
        if len(group) == 0:
            continue
        first_candle = group.iloc[0]
        or_data.append({
            'Date': pd.Timestamp(date),
            'OR_High': first_candle['high'],
            'OR_Low': first_candle['low'],
            'OR_Range': first_candle['high'] - first_candle['low']
        })

    return pd.DataFrame(or_data)


def calculate_opening_range_prev_day(daily_data):
    """
    Calculate Opening Range from Previous Day's Range.
    
    OR_High = Yesterday's High
    OR_Low = Yesterday's Low
    OR_Range = OR_High - OR_Low
    """
    df = daily_data.copy()
    df['OR_High'] = df['High'].shift(1)
    df['OR_Low'] = df['Low'].shift(1)
    df['OR_Range'] = df['OR_High'] - df['OR_Low']
    return df


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_orb_data(cot_df, market_name, or_type='prev_day',
                     fast_atr_period=10, slow_atr_period=25,
                     start_date=None, end_date=None):
    """
    Prepare complete dataset for ORB backtesting.
    
    Merges daily OHLCV, COT data, indicators, and opening range.
    """
    market_data = cot_df[cot_df['Market'] == market_name].copy()
    cot_weekly = market_data[market_data['data_type'] == 'weekly_cot'].copy()
    price_daily = market_data[market_data['data_type'] == 'daily_price'].copy()

    if price_daily.empty:
        return pd.DataFrame()

    # Build strategy data from price columns
    price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in price_cols if col in price_daily.columns]
    strategy_data = price_daily[available_cols].copy()
    strategy_data = strategy_data.sort_values('Date').reset_index(drop=True)

    # Drop rows with missing essential data
    strategy_data = strategy_data.dropna(subset=['Close', 'High', 'Low'])

    # Remove weekends (Sat=5, Sun=6) for all markets except crypto (trades 24/7)
    is_crypto = 'BITCOIN' in market_name.upper() or 'ETHER' in market_name.upper()
    if not is_crypto:
        strategy_data = strategy_data[strategy_data['Date'].dt.dayofweek < 5]
        strategy_data = strategy_data.reset_index(drop=True)

    if strategy_data.empty:
        return pd.DataFrame()

    # --- Calculate indicators on FULL data (before date filtering) ---

    # RSI
    strategy_data = calculate_rsi(strategy_data, period=10)

    # Fast and Slow ATR
    strategy_data = calculate_atr(strategy_data, period=fast_atr_period, col_name='fast_ATR')
    strategy_data = calculate_atr(strategy_data, period=slow_atr_period, col_name='slow_ATR')

    # Moving Averages
    strategy_data = calculate_moving_averages(strategy_data)

    # Inside Days
    strategy_data = calculate_inside_days(strategy_data)

    # --- Opening Range ---
    if or_type == 'prev_day':
        strategy_data = calculate_opening_range_prev_day(strategy_data)
    else:
        # Intraday OR: download/load from cache
        yf_symbol = get_yf_symbol(market_name)
        if yf_symbol is None:
            # Try getting symbol from data
            yf_col = market_data['YF_Symbol'].dropna()
            if not yf_col.empty:
                yf_symbol = yf_col.iloc[0]

        if yf_symbol:
            interval = '30m' if or_type == '30m' else '60m'
            intraday = download_intraday_data(market_name, yf_symbol, interval=interval)

            if not intraday.empty:
                or_df = calculate_opening_range_intraday(intraday, interval=interval)
                # Merge OR data with daily data
                strategy_data = strategy_data.merge(or_df, on='Date', how='left')
            else:
                # No intraday data available, set OR columns to NaN
                strategy_data['OR_High'] = np.nan
                strategy_data['OR_Low'] = np.nan
                strategy_data['OR_Range'] = np.nan
        else:
            strategy_data['OR_High'] = np.nan
            strategy_data['OR_Low'] = np.nan
            strategy_data['OR_Range'] = np.nan

    # --- Merge COT data ---
    cot_cols = ['Date', 'Commercial_Index']
    if 'Commercial_Index' in cot_weekly.columns:
        cot_for_merge = cot_weekly[cot_cols].dropna(subset=['Commercial_Index']).copy()
        cot_for_merge = cot_for_merge.sort_values('Date').reset_index(drop=True)
        strategy_data = strategy_data.sort_values('Date').reset_index(drop=True)
        strategy_data = pd.merge_asof(
            strategy_data, cot_for_merge, on='Date', direction='backward'
        )
    else:
        strategy_data['Commercial_Index'] = np.nan

    # --- Filter to date range ---
    if start_date:
        strategy_data = strategy_data[strategy_data['Date'] >= pd.Timestamp(start_date)]
    if end_date:
        strategy_data = strategy_data[strategy_data['Date'] <= pd.Timestamp(end_date)]

    strategy_data = strategy_data.reset_index(drop=True)
    return strategy_data


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_orb_signals(data, n_inside_days=3,
                         cot_filter=False, cot_long_threshold=70, cot_short_threshold=30,
                         rsi_filter=False, rsi_long_max=70, rsi_short_min=30):
    """
    Generate ORB entry signals.
    
    Conditions for a valid signal:
    1. At least N consecutive inside days have occurred (setup)
    2. Price breaks above OR_High (long) or below OR_Low (short) on the NEXT day
    3. Optional COT filter: Commercial_Index >= 70 (long) or <= 30 (short)
    4. Optional RSI filter: RSI < 70 (long) or RSI > 30 (short)
    """
    df = data.copy()
    df['signal'] = 0
    df['entry_price'] = np.nan
    df['or_high_signal'] = np.nan
    df['or_low_signal'] = np.nan

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Check if we have valid OR data
        if pd.isna(row.get('OR_High')) or pd.isna(row.get('OR_Low')):
            continue
        if pd.isna(row.get('OR_Range')) or row['OR_Range'] <= 0:
            continue

        # Check inside day setup: previous day must have ended a streak of N inside days
        # The setup day is the day after N inside days. The inside day count is measured
        # on the day BEFORE the breakout day.
        if prev.get('consecutive_inside_days', 0) < n_inside_days:
            continue

        or_high = row['OR_High']
        or_low = row['OR_Low']
        or_midpoint = (or_high + or_low) / 2

        # Check for breakout
        long_breakout = row['High'] > or_high
        short_breakout = row['Low'] < or_low

        # Determine direction
        direction = 0
        if long_breakout and short_breakout:
            # Both sides breached — favour the side closer to the Open
            if row['Open'] >= or_midpoint:
                direction = 1  # Long
            else:
                direction = -1  # Short
        elif long_breakout:
            direction = 1
        elif short_breakout:
            direction = -1

        if direction == 0:
            continue

        # Apply COT filter
        if cot_filter and not pd.isna(row.get('Commercial_Index')):
            if direction == 1 and row['Commercial_Index'] < cot_long_threshold:
                continue
            if direction == -1 and row['Commercial_Index'] > cot_short_threshold:
                continue

        # Apply RSI filter (block if already at extreme)
        if rsi_filter and not pd.isna(row.get('RSI')):
            if direction == 1 and row['RSI'] >= rsi_long_max:
                continue  # Already overbought, skip long
            if direction == -1 and row['RSI'] <= rsi_short_min:
                continue  # Already oversold, skip short

        df.iloc[i, df.columns.get_loc('signal')] = direction
        # Entry price is at the OR boundary
        entry = or_high if direction == 1 else or_low
        df.iloc[i, df.columns.get_loc('entry_price')] = entry
        df.iloc[i, df.columns.get_loc('or_high_signal')] = or_high
        df.iloc[i, df.columns.get_loc('or_low_signal')] = or_low

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class ORBBacktester:
    """
    Inside Day Breakout Backtester with two-phase stop management.
    
    Phase 1 (Fixed Stop):
      Stop at OR_Low - 0.01 (long) or OR_High + 0.01 (short).
      Stays fixed until 1:1 R/R is reached.
      
    Phase 2 (Breakeven + Trailing):
      Stop moves to entry (breakeven), then trails at slow_ATR * trailing_mult
      behind the daily close. Only moves in favourable direction.
    """

    def __init__(self, initial_capital=30000, risk_pct=1.0,
                 trailing_atr_mult=2.0, atr_source='slow_ATR'):
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct / 100.0  # Convert percentage to decimal
        self.trailing_atr_mult = trailing_atr_mult
        self.atr_source = atr_source  # 'slow_ATR' or 'fast_ATR'

    def calculate_position_size(self, risk_amount, stop_distance, market_name=""):
        """
        Position_Size = Risk_Amount / Stop_Distance
        
        For crypto: allow fractional units (round to 4 decimals)
        For all other assets: floor to whole units
        """
        if stop_distance <= 0:
            return 0, "Invalid stop distance"

        raw_units = risk_amount / stop_distance

        allows_fractional = 'BITCOIN' in market_name.upper() or 'ETHER' in market_name.upper()

        if allows_fractional:
            if raw_units < 0.0001:
                return 0, f"Position too small ({raw_units:.6f} units)"
            return round(raw_units, 4), None
        else:
            if raw_units < 1:
                return 0, f"Can only afford {raw_units:.4f} units (need >= 1)"
            return int(raw_units), None

    def backtest(self, data, market_name="Unknown"):
        """
        Run the ORB backtest on prepared data with signals.
        
        Returns dict with: trades, missed_trades, equity_curve, final_capital, total_return
        """
        df = data.copy().reset_index(drop=True)
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'signal', 'fast_ATR', 'slow_ATR']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"  Missing columns for {market_name}: {missing}")
            return {
                'trades': pd.DataFrame(), 'missed_trades': pd.DataFrame(),
                'equity_curve': [self.initial_capital],
                'final_capital': self.initial_capital, 'total_return': 0
            }

        in_position = False
        position_direction = 0
        entry_price = 0
        entry_date = None
        entry_idx = 0
        stop_loss = 0
        stop_distance = 0
        risk_amount = 0
        units = 0
        phase = 1  # 1 = fixed stop, 2 = breakeven + trailing
        or_high_trade = 0
        or_low_trade = 0

        trades = []
        missed_trades = []
        current_capital = self.initial_capital
        equity = [current_capital]

        # Track stop history for visualization
        stop_history = {}

        for i in range(len(df)):
            row = df.iloc[i]
            date = row['Date']
            close = row['Close']
            high = row.get('High', close)
            low = row.get('Low', close)
            signal = row.get('signal', 0)
            atr_val = row.get(self.atr_source, np.nan)

            if pd.isna(close):
                equity.append(current_capital)
                continue

            # --- EXIT LOGIC (if in position) ---
            if in_position:
                days_held = i - entry_idx
                exit_reason = None
                exit_price = None

                # Check stop hit
                if position_direction == 1:  # Long
                    if low <= stop_loss:
                        exit_reason = f"Stop (Phase {phase})"
                        exit_price = stop_loss
                elif position_direction == -1:  # Short
                    if high >= stop_loss:
                        exit_reason = f"Stop (Phase {phase})"
                        exit_price = stop_loss

                if exit_reason:
                    pnl = (exit_price - entry_price) * units if position_direction == 1 \
                        else (entry_price - exit_price) * units
                    pnl_pct = (pnl / (entry_price * units)) * 100 if (entry_price * units) > 0 else 0
                    current_capital += pnl

                    trades.append({
                        'market': market_name,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'direction': 'Long' if position_direction == 1 else 'Short',
                        'entry_price': round(entry_price, 4),
                        'exit_price': round(exit_price, 4),
                        'units': units,
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 2),
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                        'or_high': or_high_trade,
                        'or_low': or_low_trade,
                    })
                    in_position = False
                    position_direction = 0
                    phase = 1
                else:
                    # --- STOP MANAGEMENT (no exit this bar) ---
                    if phase == 1:
                        # Check if 1:1 R/R achieved
                        if position_direction == 1:
                            unrealized = close - entry_price
                        else:
                            unrealized = entry_price - close

                        if unrealized >= stop_distance:
                            # Move to Phase 2: breakeven first
                            phase = 2
                            stop_loss = entry_price  # Breakeven

                    if phase == 2 and not pd.isna(atr_val) and atr_val > 0:
                        # Trailing stop calculation
                        if position_direction == 1:
                            trail_stop = close - (self.trailing_atr_mult * atr_val)
                            stop_loss = max(stop_loss, trail_stop)  # Only moves up
                        else:
                            trail_stop = close + (self.trailing_atr_mult * atr_val)
                            stop_loss = min(stop_loss, trail_stop)  # Only moves down

                # Record stop level for visualization
                stop_history[date] = stop_loss

            # --- ENTRY LOGIC (if not in position and signal exists) ---
            if not in_position and signal != 0:
                entry_signal_price = row.get('entry_price', np.nan)
                or_h = row.get('or_high_signal', np.nan)
                or_l = row.get('or_low_signal', np.nan)

                if pd.isna(entry_signal_price) or pd.isna(or_h) or pd.isna(or_l):
                    equity.append(current_capital)
                    continue

                or_range = or_h - or_l
                if or_range <= 0:
                    equity.append(current_capital)
                    continue

                # Calculate stop distance
                stop_dist = or_range + STOP_BUFFER

                # Calculate position size
                risk_amt = current_capital * self.risk_pct
                pos_units, error = self.calculate_position_size(
                    risk_amt, stop_dist, market_name
                )

                if error:
                    missed_trades.append({
                        'market': market_name, 'date': date,
                        'direction': 'Long' if signal == 1 else 'Short',
                        'price': entry_signal_price,
                        'or_range': or_range, 'reason': error
                    })
                    equity.append(current_capital)
                    continue

                # Set entry
                entry_price = entry_signal_price
                entry_date = date
                entry_idx = i
                position_direction = signal
                units = pos_units
                stop_distance = stop_dist
                risk_amount = risk_amt
                or_high_trade = or_h
                or_low_trade = or_l
                phase = 1

                # Set initial stop
                if signal == 1:  # Long
                    stop_loss = or_l - STOP_BUFFER
                else:  # Short
                    stop_loss = or_h + STOP_BUFFER

                in_position = True
                stop_history[date] = stop_loss

            equity.append(current_capital)

        # --- Close any open position at end of data ---
        if in_position:
            final_close = df.iloc[-1]['Close']
            final_date = df.iloc[-1]['Date']
            days_held = len(df) - 1 - entry_idx
            pnl = (final_close - entry_price) * units if position_direction == 1 \
                else (entry_price - final_close) * units
            pnl_pct = (pnl / (entry_price * units)) * 100 if (entry_price * units) > 0 else 0
            current_capital += pnl

            trades.append({
                'market': market_name,
                'entry_date': entry_date,
                'exit_date': final_date,
                'direction': 'Long' if position_direction == 1 else 'Short',
                'entry_price': round(entry_price, 4),
                'exit_price': round(final_close, 4),
                'units': units,
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'exit_reason': 'End of Data',
                'days_held': days_held,
                'or_high': or_high_trade,
                'or_low': or_low_trade,
            })
            equity.append(current_capital)

        return {
            'trades': pd.DataFrame(trades),
            'missed_trades': pd.DataFrame(missed_trades),
            'equity_curve': equity,
            'final_capital': current_capital,
            'total_return': (current_capital - self.initial_capital) / self.initial_capital * 100,
            'stop_history': stop_history,
        }


# =============================================================================
# PERFORMANCE METRICS (reused from backtest_app.py)
# =============================================================================

def calculate_performance_metrics(trades_df, equity_curve, initial_capital=30000):
    """Calculate comprehensive performance metrics."""
    if trades_df.empty:
        return {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'total_return_pct': 0, 'cagr': 0, 'max_drawdown_pct': 0,
            'sharpe_ratio': 0, 'avg_win': 0, 'avg_loss': 0, 'net_profit': 0,
            'winning_trades': 0, 'losing_trades': 0, 'gross_profit': 0,
            'gross_loss': 0, 'avg_days_held': 0,
        }

    metrics = {}
    total_trades = len(trades_df)
    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] < 0]

    metrics['total_trades'] = total_trades
    metrics['winning_trades'] = len(winning)
    metrics['losing_trades'] = len(losing)
    metrics['win_rate'] = (len(winning) / total_trades * 100) if total_trades > 0 else 0

    total_profit = winning['pnl'].sum() if not winning.empty else 0
    total_loss = abs(losing['pnl'].sum()) if not losing.empty else 0
    metrics['gross_profit'] = total_profit
    metrics['gross_loss'] = total_loss
    metrics['net_profit'] = total_profit - total_loss
    metrics['profit_factor'] = (total_profit / total_loss) if total_loss > 0 else float('inf')

    metrics['avg_win'] = winning['pnl'].mean() if not winning.empty else 0
    metrics['avg_loss'] = losing['pnl'].mean() if not losing.empty else 0
    metrics['avg_days_held'] = trades_df['days_held'].mean() if 'days_held' in trades_df.columns else 0

    final_capital = equity_curve[-1] if equity_curve else initial_capital
    metrics['total_return_pct'] = (final_capital - initial_capital) / initial_capital * 100

    if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
        first_entry = pd.to_datetime(trades_df['entry_date']).min()
        last_exit = pd.to_datetime(trades_df['exit_date']).max()
        days = (last_exit - first_entry).days
        years = days / 365.25 if days > 0 else 1
    else:
        years = 1
    metrics['cagr'] = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    metrics['max_drawdown_pct'] = abs(drawdown.min()) * 100

    # Sharpe ratio using actual trade frequency
    if len(trades_df) > 1 and 'pnl_pct' in trades_df.columns:
        returns = trades_df['pnl_pct'] / 100
        if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
            trade_years = (pd.to_datetime(trades_df['exit_date']).max() -
                           pd.to_datetime(trades_df['entry_date']).min()).days / 365.25
        else:
            trade_years = 1
        actual_trades_per_year = len(trades_df) / trade_years if trade_years > 0 else len(trades_df)
        if returns.std() > 0 and actual_trades_per_year > 0:
            metrics['sharpe_ratio'] = (
                (returns.mean() * actual_trades_per_year) /
                (returns.std() * np.sqrt(actual_trades_per_year))
            )
        else:
            metrics['sharpe_ratio'] = 0
    else:
        metrics['sharpe_ratio'] = 0

    return metrics


# =============================================================================
# RUN BACKTEST FOR A SINGLE MARKET
# =============================================================================

def run_backtest_for_market(cot_df, market_name, or_type='prev_day',
                            n_inside_days=3, initial_capital=30000,
                            risk_pct=1.0, trailing_atr_mult=2.0,
                            fast_atr_period=10, slow_atr_period=25,
                            cot_filter=False, cot_long=70, cot_short=30,
                            rsi_filter=False, rsi_long_max=70, rsi_short_min=30,
                            start_date=None, end_date=None):
    """Run complete ORB backtest for a single market."""
    # Prepare data
    data = prepare_orb_data(
        cot_df, market_name, or_type=or_type,
        fast_atr_period=fast_atr_period, slow_atr_period=slow_atr_period,
        start_date=start_date, end_date=end_date
    )
    if data.empty:
        return None

    # Generate signals
    data = generate_orb_signals(
        data, n_inside_days=n_inside_days,
        cot_filter=cot_filter, cot_long_threshold=cot_long, cot_short_threshold=cot_short,
        rsi_filter=rsi_filter, rsi_long_max=rsi_long_max, rsi_short_min=rsi_short_min
    )

    # Run backtest
    backtester = ORBBacktester(
        initial_capital=initial_capital,
        risk_pct=risk_pct,
        trailing_atr_mult=trailing_atr_mult,
        atr_source='slow_ATR'
    )
    results = backtester.backtest(data, market_name=market_name)
    metrics = calculate_performance_metrics(results['trades'], results['equity_curve'], initial_capital)

    return {'data': data, 'results': results, 'metrics': metrics}


def run_all_backtests(cot_df, markets, or_type='prev_day',
                      n_inside_days=3, initial_capital=30000,
                      risk_pct=1.0, trailing_atr_mult=2.0,
                      fast_atr_period=10, slow_atr_period=25,
                      cot_filter=False, cot_long=70, cot_short=30,
                      rsi_filter=False, rsi_long_max=70, rsi_short_min=30,
                      start_date=None, end_date=None):
    """Run ORB backtests for all markets."""
    start = start_date or DEFAULT_START_DATE
    end = end_date or DEFAULT_END_DATE

    all_results = {}
    summary_data = []

    total_trades = 0
    total_wins = 0
    total_gross_profit = 0
    total_gross_loss = 0
    total_net_profit = 0
    max_drawdown_seen = 0
    all_pnl_pcts = []
    total_missed = 0

    for market in markets:
        print(f"Processing {market}...")
        result = run_backtest_for_market(
            cot_df, market, or_type=or_type,
            n_inside_days=n_inside_days, initial_capital=initial_capital,
            risk_pct=risk_pct, trailing_atr_mult=trailing_atr_mult,
            fast_atr_period=fast_atr_period, slow_atr_period=slow_atr_period,
            cot_filter=cot_filter, cot_long=cot_long, cot_short=cot_short,
            rsi_filter=rsi_filter, rsi_long_max=rsi_long_max, rsi_short_min=rsi_short_min,
            start_date=start, end_date=end
        )
        if result:
            all_results[market] = result
            m = result['metrics']

            missed_df = result['results']['missed_trades']
            missed_count = len(missed_df) if not missed_df.empty else 0
            total_missed += missed_count

            total_trades += m.get('total_trades', 0)
            total_wins += m.get('winning_trades', 0)
            total_gross_profit += m.get('gross_profit', 0)
            total_gross_loss += m.get('gross_loss', 0)
            total_net_profit += m.get('net_profit', 0)
            max_drawdown_seen = max(max_drawdown_seen, m.get('max_drawdown_pct', 0))

            trades_df = result['results']['trades']
            if not trades_df.empty and 'pnl_pct' in trades_df.columns:
                all_pnl_pcts.extend(trades_df['pnl_pct'].tolist())

            summary_data.append({
                'Market': market,
                'Trades': m.get('total_trades', 0),
                'Missed': missed_count,
                'Win Rate %': round(m.get('win_rate', 0), 1),
                'Avg Days': round(m.get('avg_days_held', 0), 1),
                'Return %': round(m.get('total_return_pct', 0), 2),
                'CAGR %': round(m.get('cagr', 0), 2),
                'Max DD %': round(m.get('max_drawdown_pct', 0), 2),
                'Sharpe': round(m.get('sharpe_ratio', 0), 2),
                'Profit Factor': round(m.get('profit_factor', 0), 2) if m.get('profit_factor', 0) != float('inf') else 999,
                'Net Profit': round(m.get('net_profit', 0), 2)
            })

    # --- Aggregate TOTAL row ---
    total_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    total_return_pct = (total_net_profit / initial_capital * 100) if initial_capital > 0 else 0
    total_profit_factor = (total_gross_profit / total_gross_loss) if total_gross_loss > 0 else 999

    try:
        years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    except Exception:
        years = 1

    if len(all_pnl_pcts) > 1 and years > 0:
        pnl_array = np.array(all_pnl_pcts) / 100
        actual_tpy = len(all_pnl_pcts) / years if years > 0 else len(all_pnl_pcts)
        std_ret = np.std(pnl_array)
        if std_ret > 0 and actual_tpy > 0:
            total_sharpe = (np.mean(pnl_array) * actual_tpy) / (std_ret * np.sqrt(actual_tpy))
        else:
            total_sharpe = 0
    else:
        total_sharpe = 0

    total_cagr = (
        ((initial_capital + total_net_profit) / initial_capital) ** (1 / years) - 1
    ) * 100 if years > 0 and (initial_capital + total_net_profit) > 0 else 0

    summary_data.append({
        'Market': '*** TOTAL ***',
        'Trades': total_trades,
        'Missed': total_missed,
        'Win Rate %': round(total_win_rate, 1),
        'Avg Days': 0,
        'Return %': round(total_return_pct, 2),
        'CAGR %': round(total_cagr, 2),
        'Max DD %': round(max_drawdown_seen, 2),
        'Sharpe': round(total_sharpe, 2),
        'Profit Factor': round(total_profit_factor, 2) if total_profit_factor != float('inf') else 999,
        'Net Profit': round(total_net_profit, 2)
    })

    return all_results, pd.DataFrame(summary_data)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_orb_strategy_chart(data, trades_df, market_name):
    """
    Create multi-pane strategy chart with:
    - Pane 1: Price + MAs + OR levels + trade markers
    - Pane 2: ATR (fast + slow)
    - Pane 3: Inside day count
    - Pane 4: Commercial Index (if data available)
    - Pane 5: RSI
    """
    df = data.copy()

    has_cot = 'Commercial_Index' in df.columns and df['Commercial_Index'].notna().any()
    n_rows = 5 if has_cot else 4
    heights = [0.40, 0.15, 0.15, 0.15, 0.15] if has_cot else [0.45, 0.18, 0.18, 0.19]

    subtitles = [f"Price & Opening Range: {market_name}", "ATR (Fast & Slow)", "Inside Days"]
    if has_cot:
        subtitles.append("Commercial Index (COT)")
    subtitles.append("RSI")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=heights, subplot_titles=subtitles
    )

    # --- Pane 1: Candlestick + MAs + OR levels ---
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350',
        increasing_fillcolor='#26A69A', decreasing_fillcolor='#EF5350',
    ), row=1, col=1)
    # Disable rangeslider that Candlestick adds by default
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # Moving Averages
    ma_colors = {
        'MA_10': '#FF6D00', 'MA_20': '#FFD600', 'MA_50': '#00E676',
        'MA_100': '#00BCD4', 'MA_150': '#AA00FF', 'MA_200': '#FF1744'
    }
    for ma_col, color in ma_colors.items():
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df[ma_col], name=ma_col.replace('_', ' '),
                line=dict(color=color, width=1, dash='dot'),
                visible='legendonly'  # Hidden by default, toggle in legend
            ), row=1, col=1)

    # OR levels (show where signals occurred)
    signal_rows = df[df['signal'] != 0]
    if not signal_rows.empty and 'or_high_signal' in signal_rows.columns:
        fig.add_trace(go.Scatter(
            x=signal_rows['Date'], y=signal_rows['or_high_signal'],
            name="OR High", mode='markers',
            marker=dict(symbol='line-ew', size=10, color='#00E676', line=dict(width=2))
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=signal_rows['Date'], y=signal_rows['or_low_signal'],
            name="OR Low", mode='markers',
            marker=dict(symbol='line-ew', size=10, color='#FF1744', line=dict(width=2))
        ), row=1, col=1)

    # Trade markers
    if not trades_df.empty:
        longs = trades_df[trades_df['direction'] == 'Long']
        shorts = trades_df[trades_df['direction'] == 'Short']
        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=longs['entry_date'], y=longs['entry_price'],
                mode='markers', name='Long Entry',
                marker=dict(symbol='triangle-up', size=14, color='#00C853')
            ), row=1, col=1)
        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts['entry_date'], y=shorts['entry_price'],
                mode='markers', name='Short Entry',
                marker=dict(symbol='triangle-down', size=14, color='#FF1744')
            ), row=1, col=1)
        # Exit markers
        fig.add_trace(go.Scatter(
            x=trades_df['exit_date'], y=trades_df['exit_price'],
            mode='markers', name='Exit',
            marker=dict(symbol='x', size=11, color='#FFD600', line=dict(width=2))
        ), row=1, col=1)

    # --- Pane 2: ATR ---
    if 'fast_ATR' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['fast_ATR'], name="Fast ATR",
            line=dict(color="#FF6D00", width=1.5)
        ), row=2, col=1)
    if 'slow_ATR' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['slow_ATR'], name="Slow ATR",
            line=dict(color="#00BCD4", width=1.5)
        ), row=2, col=1)

    # --- Pane 3: Inside Days ---
    if 'consecutive_inside_days' in df.columns:
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['consecutive_inside_days'], name="Inside Days",
            marker_color='#7C4DFF', opacity=0.7
        ), row=3, col=1)

    # --- Pane 4: Commercial Index (if available) ---
    row_offset = 4 if has_cot else 4
    if has_cot:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Commercial_Index'], name="Commercial Index",
            line=dict(color="#00BFA5", width=2)
        ), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="green", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="red", row=4, col=1)
        fig.update_yaxes(range=[0, 100], row=4, col=1)
        row_offset = 5

    # --- Pane 5 (or 4): RSI ---
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'], name="RSI",
            line=dict(color="#AA00FF", width=2)
        ), row=row_offset, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row_offset, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row_offset, col=1)
        fig.update_yaxes(range=[0, 100], row=row_offset, col=1)

    fig.update_layout(
        height=900 if has_cot else 800,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80)
    )
    # Remove weekend gaps from x-axis (all shared panes)
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],  # Hide Saturday to Monday
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ATR", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)

    return fig


def create_equity_curve(equity_curve, initial_capital):
    """Create equity curve chart with auto-scaled y-axis."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve, name="Equity",
        line=dict(color="#2962FF", width=2),
        fill='tonexty', fillcolor='rgba(41, 98, 255, 0.1)'
    ))
    fig.add_hline(
        y=initial_capital, line_dash="dash", line_color="gray",
        annotation_text=f"Initial: ${initial_capital:,.0f}"
    )
    # Auto-scale y-axis to the data range with some padding
    eq_min = min(equity_curve) if equity_curve else initial_capital
    eq_max = max(equity_curve) if equity_curve else initial_capital
    padding = max((eq_max - eq_min) * 0.1, initial_capital * 0.02)  # At least 2% of capital
    fig.update_layout(
        title="Equity Curve", height=350, template="plotly_white",
        yaxis_title="Equity ($)", xaxis_title="Trade #",
        yaxis=dict(range=[eq_min - padding, eq_max + padding])
    )
    return fig


# =============================================================================
# CURRENT SIGNALS DETECTION
# =============================================================================

def get_current_signals(cot_df, markets, n_inside_days=3,
                        cot_filter=False, cot_long_threshold=70, cot_short_threshold=30,
                        rsi_filter=False, rsi_long_max=70, rsi_short_min=30):
    """
    Scan all markets for current active Inside Day setups.
    
    Returns DataFrame with markets that currently have N consecutive inside days
    and are ready for a breakout trade.
    """
    signals = []
    
    for market in markets:
        # Get recent data (last 30 days to ensure we have enough for indicators)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        # Prepare data
        data = prepare_orb_data(
            cot_df, market, or_type='prev_day',
            fast_atr_period=10, slow_atr_period=25,
            start_date=start_date, end_date=end_date
        )
        
        if data.empty or len(data) < 2:
            continue
        
        # Get the most recent complete day
        latest = data.iloc[-1]
        
        # Check if we have the required inside day streak
        if latest.get('consecutive_inside_days', 0) >= n_inside_days:
            # Check COT filter if enabled
            if cot_filter and not pd.isna(latest.get('Commercial_Index')):
                cot_idx = latest['Commercial_Index']
                if cot_idx < cot_long_threshold and cot_idx > cot_short_threshold:
                    continue  # Neither bullish nor bearish COT
            
            # Get OR levels (previous day high/low)
            or_high = latest.get('OR_High', np.nan)
            or_low = latest.get('OR_Low', np.nan)
            
            if pd.isna(or_high) or pd.isna(or_low):
                continue
            
            # Determine signal bias
            bias = "Setup Ready"
            cot_value = latest.get('Commercial_Index', np.nan)
            rsi_value = latest.get('RSI', np.nan)
            
            # Apply directional bias based on filters
            cot_bullish = not pd.isna(cot_value) and cot_value >= cot_long_threshold
            cot_bearish = not pd.isna(cot_value) and cot_value <= cot_short_threshold
            
            if cot_filter:
                if cot_bullish:
                    bias = "Long Setup"
                elif cot_bearish:
                    bias = "Short Setup"
            
            signals.append({
                'Market': market,
                'Inside Days': int(latest['consecutive_inside_days']),
                'Date': latest['Date'].strftime('%Y-%m-%d'),
                'Close': round(latest['Close'], 4),
                'OR High': round(or_high, 4),
                'OR Low': round(or_low, 4),
                'Range': round(or_high - or_low, 4),
                'COT Index': round(cot_value, 1) if not pd.isna(cot_value) else 'N/A',
                'RSI': round(rsi_value, 1) if not pd.isna(rsi_value) else 'N/A',
                'Bias': bias
            })
    
    return pd.DataFrame(signals)


# =============================================================================
# CONSOLIDATION ALERTS
# =============================================================================

def check_inside_day_alerts(cot_df, markets, lookback_days=10):
    """
    Scan all markets for active inside day streaks in the last N trading days.
    
    Returns a list of dicts sorted by streak length (longest first).
    """
    alerts = []
    
    for market in markets:
        market_data = cot_df[cot_df['Market'] == market].copy()
        price_daily = market_data[market_data['data_type'] == 'daily_price'].copy()
        
        if price_daily.empty:
            continue
        
        price_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        available = [c for c in price_cols if c in price_daily.columns]
        data = price_daily[available].copy()
        data = data.sort_values('Date').reset_index(drop=True)
        data = data.dropna(subset=['Close', 'High', 'Low'])
        
        # Remove weekends except crypto
        is_crypto = 'BITCOIN' in market.upper() or 'ETHER' in market.upper()
        if not is_crypto:
            data = data[data['Date'].dt.dayofweek < 5].reset_index(drop=True)
        
        if len(data) < lookback_days + 1:
            continue
        
        # Calculate inside days on full data
        data = calculate_inside_days(data)
        
        # Check the most recent row
        last_row = data.iloc[-1]
        streak = int(last_row.get('consecutive_inside_days', 0))
        
        if streak >= 3:
            # Find the control candle (the bar before the streak started)
            streak_start_idx = len(data) - streak - 1
            if streak_start_idx >= 0:
                control = data.iloc[streak_start_idx]
                control_high = control['High']
                control_low = control['Low']
            else:
                control_high = np.nan
                control_low = np.nan
            
            alerts.append({
                'Market': market,
                'inside_days': streak,
                'control_high': round(control_high, 4) if not pd.isna(control_high) else None,
                'control_low': round(control_low, 4) if not pd.isna(control_low) else None,
                'last_date': last_row['Date'],
                'last_close': round(last_row['Close'], 4),
            })
    
    # Sort by streak length descending
    alerts = sorted(alerts, key=lambda x: x['inside_days'], reverse=True)
    return alerts


def create_consolidation_alert_panel(alerts):
    """Create the inside day consolidation alert panel."""
    if not alerts:
        return dbc.Alert(
            "No active inside day patterns (3+ days) detected",
            color="dark", className="mb-3",
            style={'backgroundColor': '#16213e', 'border': '1px solid #1a1a2e'}
        )
    
    alert_items = []
    for a in alerts[:15]:  # Show max 15
        count = a['inside_days']
        if count >= 10:
            badge_color = "danger"
        elif count >= 5:
            badge_color = "warning"
        else:
            badge_color = "info"
        
        range_text = ""
        if a['control_high'] is not None and a['control_low'] is not None:
            range_text = f" | Control: {a['control_high']} - {a['control_low']}"
        
        alert_items.append(
            html.Div([
                dbc.Badge(f"{count} Inside Days", color=badge_color, className="me-2"),
                html.Strong(a['Market'][:45]),
                html.Span(
                    f" — {a['last_date'].strftime('%Y-%m-%d')}, Close: {a['last_close']}{range_text}",
                    className="text-muted ms-2", style={'fontSize': '0.85em'}
                )
            ], className="mb-2")
        )
    
    return dbc.Alert([
        html.H5([
            html.Span("ALERT: ", style={'fontWeight': 'bold'}),
            f"{len(alerts)} Inside Day Pattern(s) Detected",
            dbc.Badge(f"{len([a for a in alerts if a['inside_days'] >= 10])} 10+", color="danger", className="ms-3"),
            dbc.Badge(f"{len([a for a in alerts if 5 <= a['inside_days'] < 10])} 5-9", color="warning", className="ms-2"),
            dbc.Badge(f"{len([a for a in alerts if a['inside_days'] < 5])} 3-4", color="info", className="ms-2"),
        ], className="alert-heading"),
        html.Hr(),
        html.Div(alert_items)
    ], color="dark", className="mb-3",
        style={'backgroundColor': '#16213e', 'border': '1px solid #1a1a2e'})


# =============================================================================
# DASH APP
# =============================================================================

# Load data at startup
print("Loading COT data...")
cot_df = load_cot_data()
markets = sorted(cot_df['Market'].unique().tolist()) if not cot_df.empty else []
print(f"Loaded {len(markets)} markets")

# Pre-compute with Previous Day Range (fastest, no download needed)
print("Pre-computing ORB backtest (Previous Day Range)...")
all_results, summary_df = run_all_backtests(
    cot_df, markets, or_type='prev_day', n_inside_days=DEFAULT_INSIDE_DAYS,
    initial_capital=DEFAULT_CAPITAL, risk_pct=DEFAULT_RISK_PCT,
    trailing_atr_mult=DEFAULT_TRAILING_ATR_MULT,
    fast_atr_period=DEFAULT_FAST_ATR, slow_atr_period=DEFAULT_SLOW_ATR,
    start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE
)
# Clean summary_df for DataTable compatibility (no NaN, no numpy types)
if not summary_df.empty:
    summary_df = summary_df.fillna(0)
    for col in summary_df.columns:
        summary_df[col] = summary_df[col].apply(
            lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x
        )
print(f"Computed results for {len(all_results)} markets")

# Check for inside day consolidation alerts
print("\nChecking for inside day alerts...")
inside_day_alerts = check_inside_day_alerts(cot_df, markets, lookback_days=10)
if inside_day_alerts:
    print(f"  ALERT: {len(inside_day_alerts)} market(s) with active inside day patterns")
    for a in inside_day_alerts:
        print(f"    {a['Market'][:40]}: {a['inside_days']} inside days (close: {a['last_close']})")
else:
    print("  No active inside day patterns")

# Define fixed column set for the summary table
SUMMARY_COLUMNS = [
    {"name": "Market", "id": "Market"},
    {"name": "Trades", "id": "Trades"},
    {"name": "Missed", "id": "Missed"},
    {"name": "Win Rate %", "id": "Win Rate %"},
    {"name": "Avg Days", "id": "Avg Days"},
    {"name": "Return %", "id": "Return %"},
    {"name": "CAGR %", "id": "CAGR %"},
    {"name": "Max DD %", "id": "Max DD %"},
    {"name": "Sharpe", "id": "Sharpe"},
    {"name": "Profit Factor", "id": "Profit Factor"},
    {"name": "Net Profit", "id": "Net Profit"},
]

TRADES_COLUMNS = [
    {"name": "Entry Date", "id": "entry_date"},
    {"name": "Exit Date", "id": "exit_date"},
    {"name": "Direction", "id": "direction"},
    {"name": "Entry Price", "id": "entry_price"},
    {"name": "Exit Price", "id": "exit_price"},
    {"name": "Units", "id": "units"},
    {"name": "PnL", "id": "pnl"},
    {"name": "Days Held", "id": "days_held"},
    {"name": "Exit Reason", "id": "exit_reason"},
]

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = dbc.Container([
    # --- Title ---
    dbc.Row([
        dbc.Col([
            html.H1("Inside Day Breakout Strategy Backtest", className="text-center my-3"),
            html.P(
                "Inside Day Consolidation Breakout + Two-Phase Stop Management",
                className="text-center text-muted"
            )
        ])
    ]),

    # --- Alert Panel ---
    dbc.Row([
        dbc.Col([
            create_consolidation_alert_panel(inside_day_alerts)
        ])
    ]),

    # --- Row 1: Strategy Parameters ---
    dbc.Row([
        dbc.Col([
            html.Label("OR Type", className="text-muted small"),
            dcc.Dropdown(
                id='orb-or-type',
                options=[
                    {'label': 'Previous Day Range', 'value': 'prev_day'},
                    {'label': '30-min Opening Range (60d max)', 'value': '30m'},
                    {'label': '60-min Opening Range (~2yr max)', 'value': '60m'},
                ],
                value='prev_day', clearable=False,
                style={'color': 'black'}
            )
        ], width=3),
        dbc.Col([
            html.Label("Inside Days Required", className="text-muted small"),
            dcc.Dropdown(
                id='orb-inside-days',
                options=[{'label': f'{n} days', 'value': n} for n in [3, 5, 7, 10, 14]],
                value=DEFAULT_INSIDE_DAYS, clearable=False,
                style={'color': 'black'}
            )
        ], width=2),
        dbc.Col([
            html.Label("Capital ($)", className="text-muted small"),
            dbc.Input(id='orb-capital', type='number', value=DEFAULT_CAPITAL, min=1000, step=1000)
        ], width=2),
        dbc.Col([
            html.Label("Risk %", className="text-muted small"),
            dbc.Input(id='orb-risk-pct', type='number', value=DEFAULT_RISK_PCT, min=0.1, max=10, step=0.1)
        ], width=1),
        dbc.Col([
            html.Label("Trail ATR Mult", className="text-muted small"),
            dbc.Input(id='orb-trail-mult', type='number', value=DEFAULT_TRAILING_ATR_MULT, min=0.5, max=10, step=0.5)
        ], width=2),
    ], className="mb-2"),

    # --- Row 2: ATR + Filters ---
    dbc.Row([
        dbc.Col([
            html.Label("Fast ATR Period", className="text-muted small"),
            dcc.Dropdown(
                id='orb-fast-atr',
                options=[{'label': str(n), 'value': n} for n in [5, 10, 15, 20]],
                value=DEFAULT_FAST_ATR, clearable=False,
                style={'color': 'black'}
            )
        ], width=2),
        dbc.Col([
            html.Label("Slow ATR Period", className="text-muted small"),
            dcc.Dropdown(
                id='orb-slow-atr',
                options=[{'label': str(n), 'value': n} for n in [25, 30, 40, 50, 100]],
                value=DEFAULT_SLOW_ATR, clearable=False,
                style={'color': 'black'}
            )
        ], width=2),
        dbc.Col([
            html.Div([
                dbc.Checklist(
                    id='orb-cot-filter',
                    options=[{'label': ' COT Filter (Long>=70, Short<=30)', 'value': 'on'}],
                    value=[], switch=True, className="mt-3"
                )
            ])
        ], width=3),
        dbc.Col([
            html.Div([
                dbc.Checklist(
                    id='orb-rsi-filter',
                    options=[{'label': ' RSI Filter (Block extremes)', 'value': 'on'}],
                    value=[], switch=True, className="mt-3"
                )
            ])
        ], width=3),
    ], className="mb-2"),

    # --- Row 3: Date Range + Run ---
    dbc.Row([
        dbc.Col([
            html.Label("Start Date", className="text-muted small"),
            dcc.DatePickerSingle(
                id='orb-start-date', date=DEFAULT_START_DATE,
                display_format='YYYY-MM-DD', className="mb-2"
            )
        ], width=2),
        dbc.Col([
            html.Label("End Date", className="text-muted small"),
            dcc.DatePickerSingle(
                id='orb-end-date', date=DEFAULT_END_DATE,
                display_format='YYYY-MM-DD', className="mb-2"
            )
        ], width=2),
        dbc.Col([
            html.Label(" ", className="text-muted small"),
            html.Br(),
            dbc.Button("Run Backtest", id="orb-run-btn", color="primary", size="lg", className="mt-1")
        ], width=2),
        dbc.Col([
            html.Div(id="orb-status", className="text-muted mt-4")
        ], width=6)
    ], className="mb-3"),

    html.Hr(),

    # --- Summary Table ---
    dbc.Row([
        dbc.Col([
            html.H4("All Markets Summary", className="mt-2"),
            dash_table.DataTable(
                id='orb-summary-table',
                columns=SUMMARY_COLUMNS,
                data=summary_df.to_dict('records') if not summary_df.empty else [],
                sort_action="native",
                filter_action="native",
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1a1a2e', 'color': 'white', 'fontWeight': 'bold'},
                style_cell={
                    'backgroundColor': '#16213e', 'color': 'white',
                    'textAlign': 'center', 'padding': '10px'
                },
                style_data_conditional=[
                    {'if': {'filter_query': '{Return %} > 0', 'column_id': 'Return %'},
                     'backgroundColor': '#1b4332', 'color': 'white'},
                    {'if': {'filter_query': '{Return %} < 0', 'column_id': 'Return %'},
                     'backgroundColor': '#4a1c1c', 'color': 'white'},
                    {'if': {'filter_query': '{Win Rate %} >= 50', 'column_id': 'Win Rate %'},
                     'backgroundColor': '#1b4332'},
                    {'if': {'filter_query': '{Win Rate %} < 40', 'column_id': 'Win Rate %'},
                     'backgroundColor': '#4a1c1c'},
                    {'if': {'filter_query': '{Market} = "*** TOTAL ***"'},
                     'backgroundColor': '#0f3460', 'fontWeight': 'bold',
                     'borderTop': '2px solid #FFD600'},
                ]
            )
        ])
    ], className="mb-4"),

    html.Hr(),

    # --- Market Detail ---
    dbc.Row([
        dbc.Col([
            html.H4("Detailed Market Analysis"),
            dcc.Dropdown(
                id='orb-market-dropdown',
                options=[{'label': m, 'value': m} for m in markets],
                value=markets[0] if markets else None,
                className="mb-3", style={'color': 'black'}
            )
        ], width=6)
    ]),

    # Metrics Cards
    dbc.Row(id='orb-metrics-cards', className="mb-4"),

    # Strategy Chart
    dbc.Row([
        dbc.Col([dcc.Graph(id='orb-strategy-chart')])
    ]),

    # Equity Curve + Trades Table
    dbc.Row([
        dbc.Col([dcc.Graph(id='orb-equity-chart')], width=6),
        dbc.Col([
            html.H5("Recent Trades"),
            dash_table.DataTable(
                id='orb-trades-table',
                columns=TRADES_COLUMNS,
                data=[],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1a1a2e', 'color': 'white'},
                style_cell={
                    'backgroundColor': '#16213e', 'color': 'white', 'textAlign': 'center'
                },
                page_size=10
            )
        ], width=6)
    ]),

    # Hidden store
    dcc.Store(id='orb-results-store'),

], fluid=True)


# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    [Output('current-signals-table', 'data'),
     Output('current-signals-table', 'columns'),
     Output('current-signals-status', 'children')],
    [Input('refresh-signals-btn', 'n_clicks'),
     Input('orb-inside-days', 'value'),
     Input('orb-cot-filter', 'value'),
     Input('orb-rsi-filter', 'value')],
    prevent_initial_call=False
)
def update_current_signals(n_clicks, inside_days, cot_filter_val, rsi_filter_val):
    """Update the current signals table based on latest data and filter settings."""
    inside_days = inside_days or DEFAULT_INSIDE_DAYS
    cot_on = 'on' in (cot_filter_val or [])
    rsi_on = 'on' in (rsi_filter_val or [])
    
    # Get current signals
    signals_df = get_current_signals(
        cot_df, markets, 
        n_inside_days=inside_days,
        cot_filter=cot_on,
        rsi_filter=rsi_on
    )
    
    if signals_df.empty:
        status = f"No active setups found (requiring {inside_days}+ inside days)"
        return [], [], status
    
    # Define columns for the table
    columns = [
        {"name": "Market", "id": "Market"},
        {"name": "Inside Days", "id": "Inside Days"},
        {"name": "Date", "id": "Date"},
        {"name": "Close", "id": "Close"},
        {"name": "OR High", "id": "OR High"},
        {"name": "OR Low", "id": "OR Low"},
        {"name": "Range", "id": "Range"},
        {"name": "COT Index", "id": "COT Index"},
        {"name": "RSI", "id": "RSI"},
        {"name": "Bias", "id": "Bias"},
    ]
    
    status = f"Found {len(signals_df)} active setup(s) with {inside_days}+ inside days"
    if cot_on:
        status += " (COT filter active)"
    
    return signals_df.to_dict('records'), columns, status


@app.callback(
    [Output('orb-results-store', 'data'),
     Output('orb-summary-table', 'data'),
     Output('orb-summary-table', 'columns'),
     Output('orb-status', 'children')],
    Input('orb-run-btn', 'n_clicks'),
    [State('orb-or-type', 'value'),
     State('orb-inside-days', 'value'),
     State('orb-capital', 'value'),
     State('orb-risk-pct', 'value'),
     State('orb-trail-mult', 'value'),
     State('orb-fast-atr', 'value'),
     State('orb-slow-atr', 'value'),
     State('orb-cot-filter', 'value'),
     State('orb-rsi-filter', 'value'),
     State('orb-start-date', 'date'),
     State('orb-end-date', 'date')],
    prevent_initial_call=True
)
def run_backtest_callback(n_clicks, or_type, inside_days, capital, risk_pct,
                          trail_mult, fast_atr, slow_atr, cot_filter_val,
                          rsi_filter_val, start_date, end_date):
    """Re-run backtests when Run button is clicked."""
    global all_results, summary_df

    if not start_date or not end_date:
        return dash.no_update, dash.no_update, dash.no_update, "Please select both dates"

    capital = capital or DEFAULT_CAPITAL
    risk_pct = risk_pct or DEFAULT_RISK_PCT
    trail_mult = trail_mult or DEFAULT_TRAILING_ATR_MULT
    fast_atr = fast_atr or DEFAULT_FAST_ATR
    slow_atr = slow_atr or DEFAULT_SLOW_ATR
    inside_days = inside_days or DEFAULT_INSIDE_DAYS

    cot_on = 'on' in (cot_filter_val or [])
    rsi_on = 'on' in (rsi_filter_val or [])

    or_labels = {'prev_day': 'Previous Day', '30m': '30-min', '60m': '60-min'}

    all_results, summary_df = run_all_backtests(
        cot_df, markets, or_type=or_type,
        n_inside_days=inside_days, initial_capital=capital,
        risk_pct=risk_pct, trailing_atr_mult=trail_mult,
        fast_atr_period=fast_atr, slow_atr_period=slow_atr,
        cot_filter=cot_on, rsi_filter=rsi_on,
        start_date=start_date, end_date=end_date
    )

    # Clean summary_df for DataTable compatibility
    if not summary_df.empty:
        summary_df = summary_df.fillna(0)
        for col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x
            )

    cot_label = "COT ON" if cot_on else "COT OFF"
    rsi_label = "RSI ON" if rsi_on else "RSI OFF"
    status = (
        f"Backtest complete: {or_labels.get(or_type, or_type)} OR, "
        f"{inside_days} inside days, {cot_label}, {rsi_label}, "
        f"${capital:,.0f} capital, {risk_pct}% risk "
        f"({len(all_results)} markets)"
    )

    return (
        {'timestamp': datetime.now().isoformat()},
        summary_df.to_dict('records') if not summary_df.empty else [],
        SUMMARY_COLUMNS,
        status
    )


@app.callback(
    [Output('orb-metrics-cards', 'children'),
     Output('orb-strategy-chart', 'figure'),
     Output('orb-equity-chart', 'figure'),
     Output('orb-trades-table', 'data'),
     Output('orb-trades-table', 'columns')],
    [Input('orb-market-dropdown', 'value'),
     Input('orb-results-store', 'data')]
)
def update_market_view(selected_market, _store):
    """Update detail view when market is selected or backtest results change."""
    if not selected_market or selected_market not in all_results:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white")
        return [], empty_fig, empty_fig, [], []

    result = all_results[selected_market]
    metrics = result['metrics']
    data = result['data']
    trades_df = result['results']['trades']
    equity = result['results']['equity_curve']

    # Determine capital from first equity value
    capital = equity[0] if equity else DEFAULT_CAPITAL

    # --- Metric Cards ---
    def metric_card(title, value, color_class=""):
        return dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6(title, className="card-subtitle text-muted mb-1", style={'fontSize': '0.8em'}),
                html.H4(value, className=f"card-title mb-0 {color_class}")
            ])
        ]), width=2)

    ret_pct = metrics.get('total_return_pct', 0)
    ret_color = "text-success" if ret_pct > 0 else "text-danger"

    cards = [
        metric_card("Total Return", f"{ret_pct:.2f}%", ret_color),
        metric_card("Win Rate", f"{metrics.get('win_rate', 0):.1f}%"),
        metric_card("Trades", f"{metrics.get('total_trades', 0)}"),
        metric_card("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}"),
        metric_card("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%", "text-warning"),
        metric_card("Profit Factor",
                     f"{metrics.get('profit_factor', 0):.2f}"
                     if metrics.get('profit_factor', 0) != float('inf') else "---"),
    ]

    # --- Strategy Chart ---
    strategy_fig = create_orb_strategy_chart(data, trades_df, selected_market)

    # --- Equity Chart ---
    equity_fig = create_equity_curve(equity, capital)

    # --- Trades Table ---
    if not trades_df.empty:
        display_trades = trades_df.copy()
        display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
        display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
        display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"${x:,.4f}")
        display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f"${x:,.4f}")
        display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f"${x:,.2f}")
        display_trades['units'] = display_trades['units'].apply(
            lambda x: str(int(x)) if isinstance(x, (int, float)) and x == int(x) else str(x)
        )
        display_trades['days_held'] = display_trades['days_held'].apply(
            lambda x: str(int(x)) if not pd.isna(x) else "0"
        )

        table_cols = [
            'entry_date', 'exit_date', 'direction', 'entry_price',
            'exit_price', 'units', 'pnl', 'days_held', 'exit_reason'
        ]
        available = [c for c in table_cols if c in display_trades.columns]
        table_data = display_trades[available].to_dict('records')
    else:
        table_data = []

    return cards, strategy_fig, equity_fig, table_data, TRADES_COLUMNS


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8052)
