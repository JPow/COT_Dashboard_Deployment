"""
COT + RSI Trading Strategy Backtest Dashboard
Deployable Dash app for Render

Shares cot_data.json with main COT dashboard app.
"""

import pandas as pd
import numpy as np
import json
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare data from shared JSON file."""
    try:
        with open('cot_data.json', 'r') as f:
            data_raw = json.load(f)
        df = pd.DataFrame(data_raw)
        df["Date"] = pd.to_datetime(df["Date"], unit='ms')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# =============================================================================
# STRATEGY FUNCTIONS (from notebook)
# =============================================================================

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


def calculate_atr(data, period=10):
    """Calculate Average True Range."""
    df = data.copy()
    if 'High' not in df.columns or 'Low' not in df.columns:
        df['ATR'] = df['Close'].rolling(window=period).std() * 1.5  # Fallback
        return df
    
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['prev_close'])
    df['tr3'] = abs(df['Low'] - df['prev_close'])
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(columns=['prev_close', 'tr1', 'tr2', 'tr3', 'TR'], inplace=True, errors='ignore')
    return df


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


def calculate_performance_metrics(trades_df, equity_curve, initial_capital=30000):
    """Calculate comprehensive performance metrics."""
    if trades_df.empty:
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'total_return_pct': 0,
                'cagr': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0, 'avg_win': 0, 
                'avg_loss': 0, 'net_profit': 0}
    
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
    
    final_capital = equity_curve[-1] if equity_curve else initial_capital
    metrics['total_return_pct'] = (final_capital - initial_capital) / initial_capital * 100
    
    if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
        days = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days
        years = days / 365.25 if days > 0 else 1
    else:
        years = 1
    metrics['cagr'] = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    equity_series = pd.Series(equity_curve)
    drawdown = (equity_series - equity_series.cummax()) / equity_series.cummax()
    metrics['max_drawdown_pct'] = abs(drawdown.min()) * 100
    
    if len(trades_df) > 1 and 'pnl_pct' in trades_df.columns:
        returns = trades_df['pnl_pct'] / 100
        avg_days = trades_df['days_held'].mean() if 'days_held' in trades_df.columns else 5
        trades_per_year = 252 / avg_days if avg_days > 0 else 20
        metrics['sharpe_ratio'] = (returns.mean() * trades_per_year) / (returns.std() * np.sqrt(trades_per_year)) if returns.std() > 0 else 0
    else:
        metrics['sharpe_ratio'] = 0
    
    return metrics


def run_backtest_for_market(df, market_name, initial_capital=30000, start_date=None, end_date=None, ma_period=0):
    """Run complete backtest for a single market (ma_period=0 means no MA filter)."""
    data = prepare_strategy_data(df, market_name, start_date, end_date, ma_period)
    if data.empty:
        return None
    
    data = calculate_atr(data, period=10)
    data = generate_signals(data)
    
    backtester = COTRSIBacktester(initial_capital=initial_capital)
    results = backtester.backtest(data, market_name=market_name)
    metrics = calculate_performance_metrics(results['trades'], results['equity_curve'], initial_capital)
    
    return {'data': data, 'results': results, 'metrics': metrics}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_strategy_chart(data, trades_df, market_name):
    """Create 3-pane strategy chart."""
    df = data.copy()
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f"Price: {market_name}", "Commercial Index (COT)", "RSI")
    )
    
    # Price
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Price", line=dict(color="#2962FF", width=1.5)), row=1, col=1)
    
    # Trade markers
    if not trades_df.empty:
        longs = trades_df[trades_df['direction'] == 'Long']
        shorts = trades_df[trades_df['direction'] == 'Short']
        if not longs.empty:
            fig.add_trace(go.Scatter(x=longs['entry_date'], y=longs['entry_price'], mode='markers', name='Long Entry',
                                     marker=dict(symbol='triangle-up', size=12, color='#00C853')), row=1, col=1)
        if not shorts.empty:
            fig.add_trace(go.Scatter(x=shorts['entry_date'], y=shorts['entry_price'], mode='markers', name='Short Entry',
                                     marker=dict(symbol='triangle-down', size=12, color='#FF1744')), row=1, col=1)
        fig.add_trace(go.Scatter(x=trades_df['exit_date'], y=trades_df['exit_price'], mode='markers', name='Exit',
                                 marker=dict(symbol='x', size=10, color='#FFD600')), row=1, col=1)
    
    # Commercial Index
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Commercial_Index'], name="Commercial Index", line=dict(color="#00BFA5", width=2)), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="orange", row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color="#AA00FF", width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=60, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(height=800, hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Index", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    
    return fig


def create_equity_curve(equity_curve, initial_capital):
    """Create equity curve chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity_curve, name="Equity", line=dict(color="#2962FF", width=2),
                             fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)'))
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                  annotation_text=f"Initial: ${initial_capital:,.0f}")
    fig.update_layout(title="Equity Curve", height=350, template="plotly_white",
                      yaxis_title="Equity ($)", xaxis_title="Trade #")
    return fig


# =============================================================================
# DASH APP
# =============================================================================

# Load data
df = load_data()
markets = sorted(df['Market'].unique().tolist()) if not df.empty else []

# Default backtest period
DEFAULT_START_DATE = '2023-01-01'
DEFAULT_END_DATE = '2025-12-31'

def run_all_backtests(start_date=None, end_date=None, ma_period=0):
    """Run backtests for all markets with given date range and MA period (0=no MA filter)."""
    start = start_date or DEFAULT_START_DATE
    end = end_date or DEFAULT_END_DATE
    
    all_results = {}
    summary_data = []
    
    # Track totals for aggregate calculations
    total_trades = 0
    total_wins = 0
    total_gross_profit = 0
    total_gross_loss = 0
    total_net_profit = 0
    max_drawdown_seen = 0
    all_pnl_pcts = []
    total_missed = 0
    
    for market in markets:
        result = run_backtest_for_market(df, market, start_date=start, end_date=end, ma_period=ma_period)
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
                'Return %': round(m.get('total_return_pct', 0), 2),
                'CAGR %': round(m.get('cagr', 0), 2),
                'Max DD %': round(m.get('max_drawdown_pct', 0), 2),
                'Sharpe': round(m.get('sharpe_ratio', 0), 2),
                'Profit Factor': round(m.get('profit_factor', 0), 2) if m.get('profit_factor', 0) != float('inf') else 999,
                'Net Profit': round(m.get('net_profit', 0), 2)
            })
    
    # Calculate aggregate metrics for TOTAL row
    initial_capital = 30000
    total_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    total_return_pct = (total_net_profit / initial_capital * 100) if initial_capital > 0 else 0
    total_profit_factor = (total_gross_profit / total_gross_loss) if total_gross_loss > 0 else 999
    
    if len(all_pnl_pcts) > 1:
        pnl_array = np.array(all_pnl_pcts) / 100
        avg_return = np.mean(pnl_array)
        std_return = np.std(pnl_array)
        trades_per_year = 252 / 5
        total_sharpe = (avg_return * trades_per_year) / (std_return * np.sqrt(trades_per_year)) if std_return > 0 else 0
    else:
        total_sharpe = 0
    
    # Calculate years from date range
    try:
        years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    except:
        years = 2
    total_cagr = (((initial_capital + total_net_profit) / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    summary_data.append({
        'Market': '*** TOTAL ***',
        'Trades': total_trades,
        'Missed': total_missed,
        'Win Rate %': round(total_win_rate, 1),
        'Return %': round(total_return_pct, 2),
        'CAGR %': round(total_cagr, 2),
        'Max DD %': round(max_drawdown_seen, 2),
        'Sharpe': round(total_sharpe, 2),
        'Profit Factor': round(total_profit_factor, 2) if total_profit_factor != float('inf') else 999,
        'Net Profit': round(total_net_profit, 2)
    })
    
    return all_results, pd.DataFrame(summary_data)

# Pre-compute with default dates
print("Pre-computing backtest results for all markets...")
all_results, summary_df = run_all_backtests(DEFAULT_START_DATE, DEFAULT_END_DATE)
print(f"Computed results for {len(all_results)} markets")

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server  # For Render deployment

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("COT + RSI Strategy Backtest", className="text-center my-4"),
            html.P("Commercial Index + RSI Mean Reversion Strategy", className="text-center text-muted")
        ])
    ]),
    
    # Date Range Selector and MA Period
    dbc.Row([
        dbc.Col([
            html.Label("Backtest Start Date", className="text-muted"),
            dcc.DatePickerSingle(
                id='start-date-picker',
                date=DEFAULT_START_DATE,
                display_format='YYYY-MM-DD',
                className="mb-2"
            )
        ], width=2),
        dbc.Col([
            html.Label("Backtest End Date", className="text-muted"),
            dcc.DatePickerSingle(
                id='end-date-picker',
                date=DEFAULT_END_DATE,
                display_format='YYYY-MM-DD',
                className="mb-2"
            )
        ], width=2),
        dbc.Col([
            html.Label("Trend MA Filter", className="text-muted"),
            dcc.Dropdown(
                id='ma-period-dropdown',
                options=[
                    {'label': 'No MA Filter', 'value': 0},
                    {'label': '6-day MA', 'value': 6},
                    {'label': '12-day MA', 'value': 12},
                    {'label': '18-day MA', 'value': 18},
                    {'label': '36-day MA', 'value': 36},
                    {'label': '60-day MA', 'value': 60},
                    {'label': '120-day MA', 'value': 120},
                    {'label': '200-day MA', 'value': 200},
                ],
                value=0,
                clearable=False,
                style={'color': 'black'}
            )
        ], width=2),
        dbc.Col([
            html.Label(" ", className="text-muted"),  # Spacer
            html.Br(),
            dbc.Button("Run Backtest", id="run-backtest-btn", color="primary", className="mt-1")
        ], width=2),
        dbc.Col([
            html.Div(id="backtest-status", className="text-muted mt-4")
        ], width=4)
    ], className="mb-3"),
    
    # Summary Table
    dbc.Row([
        dbc.Col([
            html.H4("All Markets Summary", className="mt-3"),
            dash_table.DataTable(
                id='summary-table',
                columns=[{"name": col, "id": col} for col in summary_df.columns],
                data=summary_df.to_dict('records'),
                sort_action="native",
                filter_action="native",
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1a1a2e', 'color': 'white', 'fontWeight': 'bold'},
                style_cell={'backgroundColor': '#16213e', 'color': 'white', 'textAlign': 'center', 'padding': '10px'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Return %} > 0', 'column_id': 'Return %'}, 'backgroundColor': '#1b4332', 'color': 'white'},
                    {'if': {'filter_query': '{Return %} < 0', 'column_id': 'Return %'}, 'backgroundColor': '#4a1c1c', 'color': 'white'},
                    {'if': {'filter_query': '{Win Rate %} >= 50', 'column_id': 'Win Rate %'}, 'backgroundColor': '#1b4332'},
                    {'if': {'filter_query': '{Win Rate %} < 40', 'column_id': 'Win Rate %'}, 'backgroundColor': '#4a1c1c'},
                    # Highlight TOTAL row
                    {'if': {'filter_query': '{Market} = "*** TOTAL ***"'}, 'backgroundColor': '#0f3460', 'fontWeight': 'bold', 'borderTop': '2px solid #FFD600'},
                ]
            )
        ])
    ], className="mb-4"),
    
    html.Hr(),
    
    # Market Selector
    dbc.Row([
        dbc.Col([
            html.H4("Detailed Market Analysis"),
            dcc.Dropdown(
                id='market-dropdown',
                options=[{'label': m, 'value': m} for m in markets],
                value=markets[0] if markets else None,
                className="mb-3",
                style={'color': 'black'}
            )
        ], width=6)
    ]),
    
    # Metrics Cards
    dbc.Row(id='metrics-cards', className="mb-4"),
    
    # Strategy Chart
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='strategy-chart')
        ])
    ]),
    
    # Equity Curve and Trade Table
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='equity-chart')
        ], width=6),
        dbc.Col([
            html.H5("Recent Trades"),
            dash_table.DataTable(
                id='trades-table',
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1a1a2e', 'color': 'white'},
                style_cell={'backgroundColor': '#16213e', 'color': 'white', 'textAlign': 'center'},
                page_size=10
            )
        ], width=6)
    ]),
    
    # Hidden stores for data
    dcc.Store(id='results-store', data={'all_results': {}, 'summary': summary_df.to_dict('records')}),
    
], fluid=True)


# Callback to run backtest when button clicked
@app.callback(
    [Output('results-store', 'data'),
     Output('summary-table', 'data'),
     Output('backtest-status', 'children')],
    Input('run-backtest-btn', 'n_clicks'),
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('ma-period-dropdown', 'value')],
    prevent_initial_call=True
)
def update_backtest(n_clicks, start_date, end_date, ma_period):
    """Re-run backtests when button is clicked."""
    global all_results
    
    if not start_date or not end_date:
        return dash.no_update, dash.no_update, "Please select both dates"
    
    # ma_period=0 means no MA filter, None means use default (0)
    if ma_period is None:
        ma_period = 0
    
    # Run backtests with new dates and MA period
    all_results, new_summary_df = run_all_backtests(start_date, end_date, ma_period)
    
    ma_label = "No MA" if ma_period == 0 else f"MA={ma_period}"
    status = f"✓ Backtest complete: {start_date} to {end_date}, {ma_label} ({len(all_results)} markets)"
    
    return (
        {'summary': new_summary_df.to_dict('records')},
        new_summary_df.to_dict('records'),
        status
    )


# Callbacks
@app.callback(
    [Output('metrics-cards', 'children'),
     Output('strategy-chart', 'figure'),
     Output('equity-chart', 'figure'),
     Output('trades-table', 'data'),
     Output('trades-table', 'columns')],
    Input('market-dropdown', 'value')
)
def update_market_view(selected_market):
    if not selected_market or selected_market not in all_results:
        empty_fig = go.Figure()
        return [], empty_fig, empty_fig, [], []
    
    result = all_results[selected_market]
    metrics = result['metrics']
    data = result['data']
    trades_df = result['results']['trades']
    equity = result['results']['equity_curve']
    
    # Metrics cards
    cards = [
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Total Return", className="card-subtitle text-muted"),
                html.H4(f"{metrics.get('total_return_pct', 0):.2f}%", 
                       className="card-title text-success" if metrics.get('total_return_pct', 0) > 0 else "card-title text-danger")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Win Rate", className="card-subtitle text-muted"),
                html.H4(f"{metrics.get('win_rate', 0):.1f}%", className="card-title")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Trades", className="card-subtitle text-muted"),
                html.H4(f"{metrics.get('total_trades', 0)}", className="card-title")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Sharpe Ratio", className="card-subtitle text-muted"),
                html.H4(f"{metrics.get('sharpe_ratio', 0):.2f}", className="card-title")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Max Drawdown", className="card-subtitle text-muted"),
                html.H4(f"{metrics.get('max_drawdown_pct', 0):.2f}%", className="card-title text-warning")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Profit Factor", className="card-subtitle text-muted"),
                html.H4(f"{metrics.get('profit_factor', 0):.2f}" if metrics.get('profit_factor', 0) != float('inf') else "∞", className="card-title")
            ])
        ]), width=2),
    ]
    
    # Strategy chart
    strategy_fig = create_strategy_chart(data, trades_df, selected_market)
    
    # Equity chart
    equity_fig = create_equity_curve(equity, 30000)
    
    # Trades table
    if not trades_df.empty:
        display_trades = trades_df.tail(10).copy()
        display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
        display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
        display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"${x:,.2f}")
        display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f"${x:,.2f}")
        display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f"${x:,.2f}")
        
        table_cols = ['entry_date', 'exit_date', 'market', 'direction', 'entry_price', 'exit_price', 'pnl', 'exit_reason']
        columns = [{"name": col.replace('_', ' ').title(), "id": col} for col in table_cols]
        table_data = display_trades[table_cols].to_dict('records')
    else:
        columns = []
        table_data = []
    
    return cards, strategy_fig, equity_fig, table_data, columns


if __name__ == '__main__':
    app.run(debug=True, port=8051)

