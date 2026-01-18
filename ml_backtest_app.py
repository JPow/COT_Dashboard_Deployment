"""
ML Strategy Validation Dashboard
Walk-forward validation for COT + RSI trading strategy

Tests parameter robustness across:
- RSI Exit: 50, 55, 60, 65, 70
- MA Filter: 0, 50, 100, 150, 200
- ATR Entry Filter: 0, 10, 20, 30, 40, 50
"""

import pandas as pd
import numpy as np
import json
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
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
# CORE STRATEGY FUNCTIONS
# =============================================================================

def prepare_strategy_data(df, market_name, start_date=None, end_date=None, ma_period=0):
    """Prepare strategy data for a specific market."""
    market_data = df[df['Market'] == market_name].copy()
    cot_weekly = market_data[market_data['data_type'] == 'weekly_cot'].copy()
    price_daily = market_data[market_data['data_type'] == 'daily_price'].copy()

    if price_daily.empty:
        return pd.DataFrame()

    price_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'RSI']
    available_cols = [col for col in price_cols if col in price_daily.columns]
    strategy_data = price_daily[available_cols].copy()
    
    strategy_data = strategy_data.sort_values('Date').reset_index(drop=True)
    
    if ma_period and ma_period > 0:
        strategy_data['TrendMA'] = strategy_data['Close'].rolling(window=ma_period).mean()
    else:
        strategy_data['TrendMA'] = None

    cot_cols = ['Net Commercial Position', 'OI', 'Commercial_Index']
    cot_for_merge = cot_weekly[['Date'] + cot_cols].copy()
    
    strategy_data = pd.merge(strategy_data, cot_for_merge, on='Date', how='left')
    strategy_data[cot_cols] = strategy_data[cot_cols].ffill()
    strategy_data = strategy_data.dropna(subset=['Close', 'Commercial_Index'])
    strategy_data = strategy_data.sort_values('Date').reset_index(drop=True)
    
    if start_date:
        strategy_data = strategy_data[strategy_data['Date'] >= pd.Timestamp(start_date)]
    if end_date:
        strategy_data = strategy_data[strategy_data['Date'] <= pd.Timestamp(end_date)]
    
    return strategy_data


def calculate_atr(data, period=10):
    """Calculate Average True Range."""
    df = data.copy()
    if 'High' not in df.columns or 'Low' not in df.columns:
        df['ATR'] = df['Close'].rolling(window=period).std() * 1.5
        return df
    
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['prev_close'])
    df['tr3'] = abs(df['Low'] - df['prev_close'])
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(columns=['prev_close', 'tr1', 'tr2', 'tr3', 'TR'], inplace=True, errors='ignore')
    return df


def calculate_atr_ma(data, atr_ma_period):
    """Calculate moving average of ATR for entry filter."""
    df = data.copy()
    if atr_ma_period and atr_ma_period > 0 and 'ATR' in df.columns:
        df['ATR_MA'] = df['ATR'].rolling(window=atr_ma_period).mean()
    else:
        df['ATR_MA'] = None
    return df


def generate_signals_ml(data, commercial_long=80, commercial_short=20, 
                        rsi_oversold=30, rsi_overbought=70, use_atr_filter=False):
    """Generate trading signals with optional filters."""
    df = data.copy()
    df['signal'] = 0
    
    base_long = (df['Commercial_Index'] >= commercial_long) & (df['RSI'] < rsi_oversold)
    base_short = (df['Commercial_Index'] <= commercial_short) & (df['RSI'] > rsi_overbought)
    
    use_ma_filter = 'TrendMA' in df.columns and df['TrendMA'].notna().any()
    if use_ma_filter:
        ma_long = (df['Close'] > df['TrendMA']) & (df['TrendMA'].notna())
        ma_short = (df['Close'] < df['TrendMA']) & (df['TrendMA'].notna())
    else:
        ma_long = True
        ma_short = True
    
    if use_atr_filter and 'ATR_MA' in df.columns and df['ATR_MA'].notna().any():
        atr_filter = (df['ATR'] < df['ATR_MA']) & (df['ATR_MA'].notna())
    else:
        atr_filter = True
    
    long_condition = base_long & ma_long & atr_filter
    short_condition = base_short & ma_short & atr_filter
    
    df.loc[long_condition, 'signal'] = 1
    df.loc[short_condition, 'signal'] = -1
    
    return df


# =============================================================================
# BACKTESTER CLASS
# =============================================================================

class MLBacktester:
    """Backtester for ML validation."""
    
    def __init__(self, initial_capital=30000, risk_per_trade=0.01, 
                 atr_stop_mult=2, atr_target_mult=3, max_hold_days=20, rsi_exit=60):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_hold_days = max_hold_days
        self.rsi_exit = rsi_exit
        
    def calculate_position_size(self, entry_price, atr, market_name=""):
        if pd.isna(atr) or atr <= 0:
            return 0, "Invalid ATR"
        
        risk_amount = self.initial_capital * self.risk_per_trade
        stop_distance = self.atr_stop_mult * atr
        raw_units = risk_amount / stop_distance
        
        allows_fractional = 'BITCOIN' in market_name.upper() or 'ETHER' in market_name.upper()
        
        if allows_fractional:
            if raw_units < 0.0001:
                return 0, "Position too small"
            return round(raw_units, 4), None
        else:
            if raw_units < 1:
                return 0, "Position too small"
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
        equity = [self.initial_capital]
        current_capital = self.initial_capital
        
        for i in range(len(df)):
            row = df.iloc[i]
            date, close, rsi, atr, signal = row['Date'], row['Close'], row['RSI'], row['ATR'], row['signal']
            high = row.get('High', close)
            low = row.get('Low', close)
            
            if pd.isna(close) or pd.isna(rsi) or pd.isna(atr):
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
                        exit_reason, exit_price = "RSI Exit", close
                    elif days_held >= self.max_hold_days:
                        exit_reason, exit_price = "Max Hold", close
                else:
                    if high >= stop_loss:
                        exit_reason, exit_price = "Stop Loss", stop_loss
                    elif low <= take_profit:
                        exit_reason, exit_price = "Take Profit", take_profit
                    elif rsi <= (100 - self.rsi_exit):
                        exit_reason, exit_price = "RSI Exit", close
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
            
            if not in_position and signal != 0:
                units, error = self.calculate_position_size(close, atr, market_name)
                if error:
                    continue
                entry_price, entry_date, entry_idx, position_direction = close, date, i, signal
                if signal == 1:
                    stop_loss = entry_price - (self.atr_stop_mult * atr)
                    take_profit = entry_price + (self.atr_target_mult * atr)
                else:
                    stop_loss = entry_price + (self.atr_stop_mult * atr)
                    take_profit = entry_price - (self.atr_target_mult * atr)
                in_position = True
            
            equity.append(current_capital)
        
        if in_position and len(df) > 0:
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
        
        return {
            'trades': pd.DataFrame(trades),
            'equity_curve': equity,
            'final_capital': current_capital,
            'total_return': (current_capital - self.initial_capital) / self.initial_capital * 100
        }


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_performance_metrics(trades_df, equity_curve, initial_capital=30000):
    """Calculate comprehensive performance metrics."""
    if trades_df.empty:
        return {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 
            'total_return_pct': 0, 'sharpe_ratio': 0, 'max_drawdown_pct': 0,
            'net_profit': 0, 'gross_profit': 0, 'gross_loss': 0
        }
    
    metrics = {}
    total_trades = len(trades_df)
    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] < 0]
    
    metrics['total_trades'] = total_trades
    metrics['win_rate'] = (len(winning) / total_trades * 100) if total_trades > 0 else 0
    
    total_profit = winning['pnl'].sum() if not winning.empty else 0
    total_loss = abs(losing['pnl'].sum()) if not losing.empty else 0
    metrics['gross_profit'] = total_profit
    metrics['gross_loss'] = total_loss
    metrics['net_profit'] = total_profit - total_loss
    metrics['profit_factor'] = (total_profit / total_loss) if total_loss > 0 else float('inf')
    
    final_capital = equity_curve[-1] if equity_curve else initial_capital
    metrics['total_return_pct'] = (final_capital - initial_capital) / initial_capital * 100
    
    equity_series = pd.Series(equity_curve)
    drawdown = (equity_series - equity_series.cummax()) / equity_series.cummax()
    metrics['max_drawdown_pct'] = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
    
    if len(trades_df) > 1 and 'pnl_pct' in trades_df.columns:
        returns = trades_df['pnl_pct'] / 100
        if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
            trade_years = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25
            trade_years = max(trade_years, 0.1)
        else:
            trade_years = 1
        actual_trades_per_year = len(trades_df) / trade_years
        if returns.std() > 0 and actual_trades_per_year > 0:
            metrics['sharpe_ratio'] = (returns.mean() * actual_trades_per_year) / (returns.std() * np.sqrt(actual_trades_per_year))
        else:
            metrics['sharpe_ratio'] = 0
    else:
        metrics['sharpe_ratio'] = 0
    
    return metrics


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

WALK_FORWARD_WINDOWS = [
    {'train_start': '2022-01-01', 'train_end': '2023-06-30', 'test_start': '2023-07-01', 'test_end': '2023-12-31', 'name': 'Window 1'},
    {'train_start': '2022-07-01', 'train_end': '2024-01-31', 'test_start': '2024-02-01', 'test_end': '2024-07-31', 'name': 'Window 2'},
    {'train_start': '2023-01-01', 'train_end': '2024-07-31', 'test_start': '2024-08-01', 'test_end': '2025-01-31', 'name': 'Window 3'},
    {'train_start': '2023-07-01', 'train_end': '2025-01-31', 'test_start': '2025-02-01', 'test_end': '2025-07-31', 'name': 'Window 4'},
    {'train_start': '2024-01-01', 'train_end': '2025-07-31', 'test_start': '2025-08-01', 'test_end': '2026-01-31', 'name': 'Window 5'},
]

RSI_EXIT_VALUES = [50, 55, 60, 65, 70]
MA_FILTER_VALUES = [0, 50, 100, 150, 200]
ATR_FILTER_VALUES = [0, 10, 20, 30, 40, 50]


def run_backtest_with_params(df, markets, start_date, end_date, 
                              rsi_exit=60, ma_period=0, atr_filter_period=0):
    """Run backtest across all markets with specific parameters."""
    all_trades = []
    
    for market in markets:
        data = prepare_strategy_data(df, market, start_date, end_date, ma_period)
        if data.empty:
            continue
        
        data = calculate_atr(data, period=10)
        data = calculate_atr_ma(data, atr_filter_period)
        
        use_atr_filter = atr_filter_period > 0
        data = generate_signals_ml(data, use_atr_filter=use_atr_filter)
        
        backtester = MLBacktester(rsi_exit=rsi_exit)
        results = backtester.backtest(data, market_name=market)
        
        if not results['trades'].empty:
            all_trades.append(results['trades'])
    
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades = combined_trades.sort_values('entry_date')
        equity = [30000]
        current = 30000
        for _, trade in combined_trades.iterrows():
            current += trade['pnl']
            equity.append(current)
        metrics = calculate_performance_metrics(combined_trades, equity)
    else:
        combined_trades = pd.DataFrame()
        equity = [30000]
        metrics = calculate_performance_metrics(pd.DataFrame(), equity)
    
    return {'trades': combined_trades, 'equity_curve': equity, 'metrics': metrics}


def run_walk_forward_validation(df, markets, param_grid, windows, progress_callback=None):
    """Run walk-forward validation across all parameter combinations."""
    results = []
    total_combos = len(param_grid)
    
    for idx, (rsi_exit, ma_period, atr_filter) in enumerate(param_grid):
        if progress_callback:
            progress_callback(idx + 1, total_combos)
        
        window_metrics = []
        
        for window in windows:
            result = run_backtest_with_params(
                df, markets,
                start_date=window['test_start'],
                end_date=window['test_end'],
                rsi_exit=rsi_exit,
                ma_period=ma_period,
                atr_filter_period=atr_filter
            )
            window_metrics.append(result['metrics'])
        
        if window_metrics:
            sharpes = [m['sharpe_ratio'] for m in window_metrics]
            win_rates = [m['win_rate'] for m in window_metrics]
            profit_factors = [m['profit_factor'] for m in window_metrics if m['profit_factor'] != float('inf')]
            total_trades = sum([m['total_trades'] for m in window_metrics])
            total_profit = sum([m['net_profit'] for m in window_metrics])
            max_dd = max([m['max_drawdown_pct'] for m in window_metrics]) if window_metrics else 0
            
            results.append({
                'rsi_exit': rsi_exit,
                'ma_period': ma_period,
                'atr_filter': atr_filter,
                'mean_sharpe': np.mean(sharpes),
                'std_sharpe': np.std(sharpes),
                'min_sharpe': np.min(sharpes),
                'max_sharpe': np.max(sharpes),
                'mean_win_rate': np.mean(win_rates),
                'mean_profit_factor': np.mean(profit_factors) if profit_factors else 0,
                'total_trades': total_trades,
                'total_profit': total_profit,
                'max_drawdown': max_dd,
                'windows_with_sharpe_gt_05': sum([1 for s in sharpes if s > 0.5]),
            })
    
    return pd.DataFrame(results)


# =============================================================================
# LOAD DATA AND INITIALIZE
# =============================================================================

print("Loading data...")
df = load_data()
markets = sorted(df['Market'].unique().tolist()) if not df.empty else []
print(f"Loaded {len(df)} rows across {len(markets)} markets")

# Pre-compute validation results
print("Running walk-forward validation (this may take a few minutes)...")
param_grid = list(product(RSI_EXIT_VALUES, MA_FILTER_VALUES, ATR_FILTER_VALUES))
validation_results = run_walk_forward_validation(df, markets, param_grid, WALK_FORWARD_WINDOWS)
print(f"Validation complete: {len(validation_results)} parameter combinations tested")


# =============================================================================
# DASH APP
# =============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("ML Strategy Validation", className="text-center my-4"),
            html.P("Walk-Forward Parameter Robustness Analysis", className="text-center text-muted")
        ])
    ]),
    
    # Summary Stats
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Total Combinations", className="card-subtitle text-muted"),
                html.H4(f"{len(validation_results)}", className="card-title")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Robust (Sharpe > 0.5)", className="card-subtitle text-muted"),
                html.H4(f"{(validation_results['mean_sharpe'] > 0.5).sum()}", className="card-title text-success")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Best Sharpe", className="card-subtitle text-muted"),
                html.H4(f"{validation_results['mean_sharpe'].max():.2f}", className="card-title text-success")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Worst Sharpe", className="card-subtitle text-muted"),
                html.H4(f"{validation_results['mean_sharpe'].min():.2f}", className="card-title text-danger")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Sharpe Variance", className="card-subtitle text-muted"),
                html.H4(f"{validation_results['mean_sharpe'].var():.4f}", className="card-title")
            ])
        ]), width=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Markets Tested", className="card-subtitle text-muted"),
                html.H4(f"{len(markets)}", className="card-title")
            ])
        ]), width=2),
    ], className="mb-4"),
    
    html.Hr(),
    
    # Heatmap Selection
    dbc.Row([
        dbc.Col([
            html.H4("Parameter Sensitivity Heatmaps"),
            dcc.Dropdown(
                id='heatmap-selector',
                options=[
                    {'label': 'RSI Exit vs MA Filter', 'value': 'rsi_ma'},
                    {'label': 'RSI Exit vs ATR Filter', 'value': 'rsi_atr'},
                    {'label': 'MA Filter vs ATR Filter', 'value': 'ma_atr'},
                ],
                value='rsi_ma',
                clearable=False,
                style={'color': 'black'}
            )
        ], width=4)
    ], className="mb-3"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='heatmap-chart')
        ])
    ]),
    
    html.Hr(),
    
    # Sensitivity Bar Charts
    dbc.Row([
        dbc.Col([
            html.H4("Parameter Sensitivity Analysis"),
            dcc.Graph(id='sensitivity-chart')
        ])
    ]),
    
    html.Hr(),
    
    # Top 20 Table
    dbc.Row([
        dbc.Col([
            html.H4("Top 20 Parameter Combinations (by Mean Sharpe)"),
            dash_table.DataTable(
                id='top-20-table',
                columns=[
                    {"name": "RSI Exit", "id": "rsi_exit"},
                    {"name": "MA Period", "id": "ma_period"},
                    {"name": "ATR Filter", "id": "atr_filter"},
                    {"name": "Mean Sharpe", "id": "mean_sharpe"},
                    {"name": "Std Sharpe", "id": "std_sharpe"},
                    {"name": "Win Rate %", "id": "mean_win_rate"},
                    {"name": "Profit Factor", "id": "mean_profit_factor"},
                    {"name": "Total Trades", "id": "total_trades"},
                    {"name": "Net Profit", "id": "total_profit"},
                ],
                data=validation_results.sort_values('mean_sharpe', ascending=False).head(20).round(3).to_dict('records'),
                sort_action="native",
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1a1a2e', 'color': 'white', 'fontWeight': 'bold'},
                style_cell={'backgroundColor': '#16213e', 'color': 'white', 'textAlign': 'center', 'padding': '10px'},
                style_data_conditional=[
                    {'if': {'filter_query': '{mean_sharpe} > 0.5', 'column_id': 'mean_sharpe'}, 'backgroundColor': '#1b4332', 'color': 'white'},
                    {'if': {'filter_query': '{mean_sharpe} < 0', 'column_id': 'mean_sharpe'}, 'backgroundColor': '#4a1c1c', 'color': 'white'},
                ]
            )
        ])
    ], className="mb-4"),
    
    html.Hr(),
    
    # Full Results (Collapsed)
    dbc.Row([
        dbc.Col([
            dbc.Button("Show Full Results Table", id="collapse-button", className="mb-3", color="secondary"),
            dbc.Collapse(
                dash_table.DataTable(
                    id='full-results-table',
                    columns=[{"name": col, "id": col} for col in validation_results.columns],
                    data=validation_results.sort_values('mean_sharpe', ascending=False).round(3).to_dict('records'),
                    sort_action="native",
                    filter_action="native",
                    page_size=20,
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#1a1a2e', 'color': 'white', 'fontWeight': 'bold'},
                    style_cell={'backgroundColor': '#16213e', 'color': 'white', 'textAlign': 'center', 'padding': '10px'},
                ),
                id="collapse-results",
                is_open=False,
            )
        ])
    ]),
    
], fluid=True)


# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    Output('heatmap-chart', 'figure'),
    Input('heatmap-selector', 'value')
)
def update_heatmap(heatmap_type):
    if heatmap_type == 'rsi_ma':
        heatmap_data = validation_results.groupby(['rsi_exit', 'ma_period'])['mean_sharpe'].mean().unstack()
        title = 'Mean Sharpe: RSI Exit vs MA Filter (averaged over ATR filter)'
        x_label = 'MA Filter Period'
        y_label = 'RSI Exit'
    elif heatmap_type == 'rsi_atr':
        heatmap_data = validation_results.groupby(['rsi_exit', 'atr_filter'])['mean_sharpe'].mean().unstack()
        title = 'Mean Sharpe: RSI Exit vs ATR Filter (averaged over MA filter)'
        x_label = 'ATR Filter Period'
        y_label = 'RSI Exit'
    else:
        heatmap_data = validation_results.groupby(['ma_period', 'atr_filter'])['mean_sharpe'].mean().unstack()
        title = 'Mean Sharpe: MA Filter vs ATR Filter (averaged over RSI exit)'
        x_label = 'ATR Filter Period'
        y_label = 'MA Filter Period'
    
    fig = px.imshow(
        heatmap_data.values,
        labels=dict(x=x_label, y=y_label, color="Mean Sharpe"),
        x=heatmap_data.columns.astype(str),
        y=heatmap_data.index.astype(str),
        color_continuous_scale='RdYlGn',
        title=title,
        aspect='auto'
    )
    fig.update_layout(height=500, template='plotly_dark')
    return fig


@app.callback(
    Output('sensitivity-chart', 'figure'),
    Input('heatmap-selector', 'value')  # Just need a trigger, any input works
)
def update_sensitivity(_):
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        'RSI Exit Sensitivity', 'MA Filter Sensitivity', 'ATR Filter Sensitivity'
    ))
    
    rsi_sens = validation_results.groupby('rsi_exit')['mean_sharpe'].agg(['mean', 'std']).reset_index()
    fig.add_trace(go.Bar(x=rsi_sens['rsi_exit'], y=rsi_sens['mean'], 
                         error_y=dict(type='data', array=rsi_sens['std']),
                         name='RSI Exit', marker_color='#00BFA5'), row=1, col=1)
    
    ma_sens = validation_results.groupby('ma_period')['mean_sharpe'].agg(['mean', 'std']).reset_index()
    fig.add_trace(go.Bar(x=ma_sens['ma_period'], y=ma_sens['mean'],
                         error_y=dict(type='data', array=ma_sens['std']),
                         name='MA Filter', marker_color='#2962FF'), row=1, col=2)
    
    atr_sens = validation_results.groupby('atr_filter')['mean_sharpe'].agg(['mean', 'std']).reset_index()
    fig.add_trace(go.Bar(x=atr_sens['atr_filter'], y=atr_sens['mean'],
                         error_y=dict(type='data', array=atr_sens['std']),
                         name='ATR Filter', marker_color='#AA00FF'), row=1, col=3)
    
    fig.update_layout(height=400, showlegend=False, template='plotly_dark',
                      title='Mean Sharpe ± Std Across Parameter Values')
    return fig


@app.callback(
    Output("collapse-results", "is_open"),
    Input("collapse-button", "n_clicks"),
    State("collapse-results", "is_open"),
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run(debug=True, port=8052)

