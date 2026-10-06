"""COT + RSI strategy helpers shared by notebooks and legacy Dash apps."""

import pandas as pd
import numpy as np

from .indicators import calculate_atr
from .metrics import calculate_performance_metrics

def prepare_strategy_data(df, market_name, start_date=None, end_date=None, ma_period=0):
    """Prepare strategy data for a specific market with optional date filtering.
    
    Args:
        df: DataFrame with COT and price data
        market_name: Name of the market to filter
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        ma_period: Moving average period for trend filter (0 = no MA filter)
    """
    market_data = df[df['Market'] == market_name].copy()
    cot_weekly = market_data[market_data['data_type'] == 'weekly_cot'].copy()
    price_daily = market_data[market_data['data_type'] == 'daily_price'].copy()

    if price_daily.empty:
        return pd.DataFrame()

    # Only get raw price data - we'll calculate MA on-the-fly
    price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'RSI']
    available_cols = [col for col in price_cols if col in price_daily.columns]
    strategy_data = price_daily[available_cols].copy()
    
    # Sort by date first for proper rolling calculations
    strategy_data = strategy_data.sort_values('Date').reset_index(drop=True)
    
    # Calculate TrendMA on-the-fly with user-selected period (0 = no MA filter)
    if ma_period and ma_period > 0:
        strategy_data['TrendMA'] = strategy_data['Close'].rolling(window=ma_period).mean()
    else:
        strategy_data['TrendMA'] = None  # No MA filter

    cot_cols = ['Net Commercial Position', 'OI', 'Commercial_Index']
    cot_for_merge = cot_weekly[['Date'] + cot_cols].copy()
    
    strategy_data = pd.merge(strategy_data, cot_for_merge, on='Date', how='left')
    strategy_data[cot_cols] = strategy_data[cot_cols].ffill()
    strategy_data = strategy_data.dropna(subset=['Close', 'Commercial_Index'])
    strategy_data = strategy_data.sort_values('Date').reset_index(drop=True)
    
    # Filter to backtest period if dates provided
    if start_date:
        strategy_data = strategy_data[strategy_data['Date'] >= pd.Timestamp(start_date)]
    if end_date:
        strategy_data = strategy_data[strategy_data['Date'] <= pd.Timestamp(end_date)]
    
    return strategy_data


def generate_signals(data, commercial_long=80, commercial_short=20, rsi_oversold=30, rsi_overbought=70):
    """Generate trading signals with optional TrendMA filter."""
    df = data.copy()
    df['signal'] = 0
    
    # Check if MA filter is enabled (TrendMA has actual values)
    use_ma_filter = 'TrendMA' in df.columns and df['TrendMA'].notna().any()
    
    if use_ma_filter:
        # With MA filter: Long requires uptrend, Short requires downtrend
        long_condition = (
            (df['Commercial_Index'] >= commercial_long) & 
            (df['RSI'] < rsi_oversold) &
            (df['Close'] > df['TrendMA']) &
            (df['TrendMA'].notna())
        )
        short_condition = (
            (df['Commercial_Index'] <= commercial_short) & 
            (df['RSI'] > rsi_overbought) &
            (df['Close'] < df['TrendMA']) &
            (df['TrendMA'].notna())
        )
    else:
        # No MA filter: Pure COT + RSI signals
        long_condition = (
            (df['Commercial_Index'] >= commercial_long) & 
            (df['RSI'] < rsi_oversold)
        )
        short_condition = (
            (df['Commercial_Index'] <= commercial_short) & 
            (df['RSI'] > rsi_overbought)
        )
    
    df.loc[long_condition, 'signal'] = 1
    df.loc[short_condition, 'signal'] = -1
    
    return df


class COTRSIBacktester:
    """Backtester for COT + RSI trading strategy."""
    
    def __init__(self, initial_capital=30000, risk_per_trade=0.01, 
                 atr_stop_mult=2, atr_target_mult=3, max_hold_days=20, rsi_exit=60):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_hold_days = max_hold_days
        self.rsi_exit = rsi_exit
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []
        
    def calculate_position_size(self, entry_price, atr, market_name=""):
        """
        Position sizing based on 1% risk rule.
        
        Formula:
        - Risk amount = capital × 0.01
        - Stop distance = 2 × ATR
        - Max units = risk_amount / stop_distance
        - If units < 1: missed trade (can't afford 1 unit)
        - If units >= 1: round to nearest integer
        
        Exception: Bitcoin and Ether allow fractional units
        """
        if pd.isna(atr) or atr <= 0:
            return 0, "Invalid ATR"
        
        risk_amount = self.capital * self.risk_per_trade  # 1% of capital
        stop_distance = self.atr_stop_mult * atr          # 2 × ATR
        
        raw_units = risk_amount / stop_distance
        
        # Bitcoin and Ether can be traded as fractional units
        allows_fractional = 'BITCOIN' in market_name.upper() or 'ETHER' in market_name.upper()
        
        if allows_fractional:
            # Allow fractional units for crypto (round to 4 decimal places)
            if raw_units < 0.0001:
                return 0, f"Missed trade: Position too small ({raw_units:.6f} units)"
            return round(raw_units, 4), None
        else:
            # Standard assets require whole units
            if raw_units < 1:
                return 0, f"Missed trade: Can only afford {raw_units:.2f} units"
            return round(raw_units), None
    
    def backtest(self, data, market_name="Unknown"):
        df = data.copy().reset_index(drop=True)
        required = ['Date', 'Open', 'Close', 'RSI', 'ATR', 'signal']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return {'trades': pd.DataFrame(), 'equity_curve': [self.initial_capital], 
                    'final_capital': self.initial_capital, 'total_return': 0}
        
        in_position = False
        position_direction = 0
        entry_price = entry_date = entry_idx = stop_loss = take_profit = units = 0
        trades = []
        missed_trades = []  # Track signals we couldn't afford
        equity = [self.initial_capital]
        current_capital = self.initial_capital
        
        for i in range(len(df)):
            row = df.iloc[i]
            date, close, rsi, atr, signal = row['Date'], row['Close'], row['RSI'], row['ATR'], row['signal']
            high = row.get('High', close)
            low = row.get('Low', close)
            
            if pd.isna(close) or pd.isna(rsi):
                equity.append(current_capital)
                continue
            
            if in_position:
                days_held = i - entry_idx
                exit_reason = exit_price = None
                
                if position_direction == 1:
                    if low <= stop_loss:
                        exit_reason, exit_price = "Stop Loss", stop_loss
                    elif high >= take_profit:
                        exit_reason, exit_price = "Take Profit", take_profit
                    elif rsi >= self.rsi_exit:
                        exit_reason, exit_price = f"RSI Exit", close
                    elif days_held >= self.max_hold_days:
                        exit_reason, exit_price = "Max Hold", close
                else:
                    if high >= stop_loss:
                        exit_reason, exit_price = "Stop Loss", stop_loss
                    elif low <= take_profit:
                        exit_reason, exit_price = "Take Profit", take_profit
                    elif rsi <= self.rsi_exit:
                        exit_reason, exit_price = f"RSI Exit", close
                    elif days_held >= self.max_hold_days:
                        exit_reason, exit_price = "Max Hold", close
                
                if exit_reason:
                    pnl = (exit_price - entry_price) * units if position_direction == 1 else (entry_price - exit_price) * units
                    pnl_pct = (pnl / (entry_price * units)) * 100 if units > 0 else 0
                    current_capital += pnl
                    trades.append({
                        'market': market_name, 'entry_date': entry_date, 'exit_date': date,
                        'direction': 'Long' if position_direction == 1 else 'Short',
                        'entry_price': entry_price, 'exit_price': exit_price, 'units': units,
                        'pnl': pnl, 'pnl_pct': pnl_pct, 'exit_reason': exit_reason, 'days_held': days_held
                    })
                    in_position = False
                    position_direction = 0
            
            if not in_position and signal != 0 and not pd.isna(atr):
                entry_price, entry_date, entry_idx, position_direction = close, date, i, signal
                units, error = self.calculate_position_size(entry_price, atr, market_name)
                if error:
                    # Log missed trade
                    missed_trades.append({
                        'market': market_name,
                        'date': date,
                        'direction': 'Long' if signal == 1 else 'Short',
                        'price': close,
                        'atr': atr,
                        'reason': error
                    })
                    continue
                if signal == 1:
                    stop_loss = entry_price - (self.atr_stop_mult * atr)
                    take_profit = entry_price + (self.atr_target_mult * atr)
                else:
                    stop_loss = entry_price + (self.atr_stop_mult * atr)
                    take_profit = entry_price - (self.atr_target_mult * atr)
                in_position = True
            
            equity.append(current_capital)
        
        if in_position:
            final_close, final_date = df.iloc[-1]['Close'], df.iloc[-1]['Date']
            days_held = len(df) - 1 - entry_idx
            pnl = (final_close - entry_price) * units if position_direction == 1 else (entry_price - final_close) * units
            pnl_pct = (pnl / (entry_price * units)) * 100 if units > 0 else 0
            current_capital += pnl
            trades.append({
                'market': market_name, 'entry_date': entry_date, 'exit_date': final_date,
                'direction': 'Long' if position_direction == 1 else 'Short',
                'entry_price': entry_price, 'exit_price': final_close, 'units': units,
                'pnl': pnl, 'pnl_pct': pnl_pct, 'exit_reason': 'End of Data', 'days_held': days_held
            })
            equity.append(current_capital)
        
        self.trades = trades
        self.missed_trades = missed_trades
        self.equity_curve = equity
        self.capital = current_capital
        
        return {
            'trades': pd.DataFrame(trades),
            'missed_trades': pd.DataFrame(missed_trades),
            'equity_curve': equity,
            'final_capital': current_capital,
            'total_return': (current_capital - self.initial_capital) / self.initial_capital * 100
        }


