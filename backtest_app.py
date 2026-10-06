"""
COT + RSI Trading Strategy Backtest Dashboard
Deployable Dash app for Render

Shares cot_data.json with main COT dashboard app.
"""

import os
import pandas as pd
import numpy as np
import json
import dash

from backtest_engine.data import load_cot_data as _load_cot_data
from backtest_engine.indicators import calculate_atr
from backtest_engine.metrics import calculate_performance_metrics
from backtest_engine.charts import create_equity_curve

_SKIP_PRECOMPUTE = os.environ.get("COT_SKIP_PRECOMPUTE") == "1"
from dash import Dash, html, dcc, callback, Output, Input, dash_table, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare data from shared JSON file."""
    return _load_cot_data()

# =============================================================================
# RECENT SIGNALS ALERT SYSTEM
# =============================================================================

def check_recent_signals(df, days=7, commercial_long=80, commercial_short=20, rsi_oversold=30, rsi_overbought=70):
    """
    Check for trading signals that occurred in the last N days.
    
    Returns a list of dictionaries with signal details.
    """
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_signals = []
    
    markets = df['Market'].unique()
    
    for market in markets:
        market_data = df[df['Market'] == market].copy()
        price_daily = market_data[market_data['data_type'] == 'daily_price'].copy()
        cot_weekly = market_data[market_data['data_type'] == 'weekly_cot'].copy()
        
        if price_daily.empty or cot_weekly.empty:
            continue
        
        # Check if Commercial_Index exists in COT data
        if 'Commercial_Index' not in cot_weekly.columns:
            continue
            
        # Sort both by Date for merge_asof
        price_daily = price_daily[['Date', 'Close', 'RSI']].copy().sort_values('Date').reset_index(drop=True)
        cot_for_merge = cot_weekly[['Date', 'Commercial_Index']].dropna(subset=['Commercial_Index']).copy()
        cot_for_merge = cot_for_merge.sort_values('Date').reset_index(drop=True)
        
        if cot_for_merge.empty:
            continue
        
        # Use merge_asof to carry forward most recent COT data to each daily price row
        # direction='backward' means: find most recent COT date <= each price date
        merged = pd.merge_asof(price_daily, cot_for_merge, on='Date', direction='backward')
        
        # Drop rows with missing required data
        merged = merged.dropna(subset=['Close', 'RSI', 'Commercial_Index'])
        
        # Filter to recent days
        merged = merged[merged['Date'] >= cutoff_date]
        
        if merged.empty:
            continue
        
        # Check for signals (no MA filter - pure COT + RSI)
        for _, row in merged.iterrows():
            signal_type = None
            
            # Long signal: Commercial >= 80 AND RSI < 30
            if row['Commercial_Index'] >= commercial_long and row['RSI'] < rsi_oversold:
                signal_type = 'LONG'
            # Short signal: Commercial <= 20 AND RSI > 70
            elif row['Commercial_Index'] <= commercial_short and row['RSI'] > rsi_overbought:
                signal_type = 'SHORT'
            
            if signal_type:
                recent_signals.append({
                    'Market': market,
                    'Date': row['Date'],
                    'Signal': signal_type,
                    'Commercial_Index': round(row['Commercial_Index'], 1),
                    'RSI': round(row['RSI'], 1),
                    'Close': round(row['Close'], 2)
                })
    
    # Sort by date descending (most recent first)
    recent_signals = sorted(recent_signals, key=lambda x: x['Date'], reverse=True)
    
    return recent_signals


# =============================================================================
# STRATEGY FUNCTIONS (from notebook)
# =============================================================================

from backtest_engine.cot_rsi import (
    prepare_strategy_data,
    generate_signals,
    COTRSIBacktester,
)

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


# =============================================================================
# DASH APP
# =============================================================================

# Load data
df = load_data()
markets = sorted(df['Market'].unique().tolist()) if not df.empty else []

# Default backtest period
DEFAULT_START_DATE = '2023-01-01'
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')

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
    
    # Calculate years from date range (needed for both Sharpe and CAGR)
    try:
        years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    except:
        years = 1
    
    # Calculate TOTAL Sharpe ratio using actual trade frequency (no assumptions)
    if len(all_pnl_pcts) > 1 and years > 0:
        pnl_array = np.array(all_pnl_pcts) / 100
        avg_return = np.mean(pnl_array)
        std_return = np.std(pnl_array)
        
        # Calculate actual trades per year from data
        actual_trades_per_year = len(all_pnl_pcts) / years if years > 0 else len(all_pnl_pcts)
        
        # Annualized Sharpe: (mean return * N) / (std * sqrt(N))
        if std_return > 0 and actual_trades_per_year > 0:
            total_sharpe = (avg_return * actual_trades_per_year) / (std_return * np.sqrt(actual_trades_per_year))
        else:
            total_sharpe = 0
    else:
        total_sharpe = 0
    
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
if _SKIP_PRECOMPUTE:
    all_results, summary_df = {}, pd.DataFrame()
else:
    print("Pre-computing backtest results for all markets...")
    all_results, summary_df = run_all_backtests(DEFAULT_START_DATE, DEFAULT_END_DATE)
    print(f"Computed results for {len(all_results)} markets")

# Check for recent signals (last 7 days)
print("\n🔔 Checking for recent signals...")
recent_signals = check_recent_signals(df, days=7)
if recent_signals:
    print(f"⚠️  ALERT: {len(recent_signals)} signal(s) in the last 7 days!")
    for sig in recent_signals:
        print(f"   {sig['Signal']} - {sig['Market'][:40]} on {sig['Date'].strftime('%Y-%m-%d')} (COT: {sig['Commercial_Index']}, RSI: {sig['RSI']})")
else:
    print("✓ No new signals in the last 7 days")

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server  # For Render deployment

# Layout
def create_alert_panel(signals):
    """Create the recent signals alert panel."""
    if not signals:
        return dbc.Alert(
            "✓ No new signals in the last 7 days",
            color="dark",
            className="mb-3",
            style={'backgroundColor': '#16213e', 'border': '1px solid #1a1a2e'}
        )
    
    # Group signals by type
    long_signals = [s for s in signals if s['Signal'] == 'LONG']
    short_signals = [s for s in signals if s['Signal'] == 'SHORT']
    
    alert_items = []
    
    for sig in signals[:10]:  # Show max 10 most recent
        badge_color = "success" if sig['Signal'] == 'LONG' else "danger"
        alert_items.append(
            html.Div([
                dbc.Badge(sig['Signal'], color=badge_color, className="me-2"),
                html.Strong(sig['Market'][:35]),
                html.Span(f" — {sig['Date'].strftime('%Y-%m-%d')}", className="text-muted ms-2"),
                html.Span(f" (COT: {sig['Commercial_Index']}, RSI: {sig['RSI']}, Price: ${sig['Close']:,.2f})", 
                         className="text-muted ms-1", style={'fontSize': '0.85em'})
            ], className="mb-2")
        )
    
    return dbc.Alert([
        html.H5([
            html.Span("🔔 ", style={'fontSize': '1.2em'}),
            f"ALERT: {len(signals)} Signal(s) in Last 7 Days",
            dbc.Badge(f"{len(long_signals)} Long", color="success", className="ms-3"),
            dbc.Badge(f"{len(short_signals)} Short", color="danger", className="ms-2"),
        ], className="alert-heading"),
        html.Hr(),
        html.Div(alert_items)
    ], color="dark", className="mb-3", style={'backgroundColor': '#16213e', 'border': '1px solid #1a1a2e'})

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("COT + RSI Strategy Backtest", className="text-center my-4"),
            html.P("Commercial Index + RSI Mean Reversion Strategy", className="text-center text-muted")
        ])
    ]),
    
    # Recent Signals Alert Panel
    dbc.Row([
        dbc.Col([
            create_alert_panel(recent_signals)
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
    print("Deprecated standalone port — use: python -m app.main (Backtest page uses unified engine)")
    from app.main import app as multi_app
    multi_app.run(debug=True, port=8050)

