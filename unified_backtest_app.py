"""
Unified Strategy Backtest Dashboard
====================================
Single Dash app for backtesting multiple strategy models:
  - Setup:  Narrowing Range / Inside Days / COT+RSI Extremes
  - Entry:  ORB Breakout (30m/60m) / Daily Breakout / Market-on-Close
  - Stop:   Two-Phase ATR Trail / ATR Stop+Target

All shared logic lives in the backtest_engine/ package.
"""

import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, dash_table, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime

from backtest_engine.data import load_cot_data
from backtest_engine.backtester import run_all_markets
from backtest_engine.charts import create_strategy_chart, create_equity_curve
from backtest_engine.setups import SETUP_REGISTRY
from backtest_engine.entries import ENTRY_REGISTRY
from backtest_engine.stops import STOP_REGISTRY


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_START = '2023-01-01'
DEFAULT_END = datetime.now().strftime('%Y-%m-%d')
DEFAULT_CAPITAL = 30000
DEFAULT_RISK = 1.0
DEFAULT_ATR_PERIOD = 10

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


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading COT data...")
cot_df = load_cot_data()
markets = sorted(cot_df['Market'].unique().tolist()) if not cot_df.empty else []
print(f"Loaded {len(markets)} markets")

# Initial backtest with defaults
print("Running initial backtest (Narrowing Range + ORB 60m + Two-Phase ATR)...")
all_results, summary_df = run_all_markets(
    cot_df, markets,
    setup_key='narrowing_range', entry_key='orb_breakout', stop_key='two_phase_atr',
    setup_params={'n_days': 3},
    entry_params={'or_type': '60m'},
    stop_params={'trailing_atr_mult': 2.0},
    atr_period=DEFAULT_ATR_PERIOD,
    initial_capital=DEFAULT_CAPITAL, risk_pct=DEFAULT_RISK,
    start_date=DEFAULT_START, end_date=DEFAULT_END,
)
if not summary_df.empty:
    summary_df = summary_df.fillna(0)
    for col in summary_df.columns:
        summary_df[col] = summary_df[col].apply(
            lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x)
print(f"Initial backtest: {len(all_results)} markets")


# =============================================================================
# DASH APP
# =============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([

    # --- Title ---
    dbc.Row([dbc.Col([
        html.H1("Unified Strategy Backtest", className="text-center my-3"),
        html.P("Mix-and-match: Setup + Entry + Stop Management",
               className="text-center text-muted"),
    ])]),

    # === Row 1: Strategy Selection ===
    dbc.Row([
        dbc.Col([
            html.Label("Setup Model", className="text-muted small"),
            dcc.Dropdown(
                id='setup-dropdown',
                options=[{'label': v['label'], 'value': k}
                         for k, v in SETUP_REGISTRY.items()],
                value='narrowing_range', clearable=False,
                style={'color': 'black'},
            ),
        ], width=3),
        dbc.Col([
            html.Label("Entry Filter", className="text-muted small"),
            dcc.Dropdown(
                id='entry-dropdown',
                options=[{'label': v['label'], 'value': k}
                         for k, v in ENTRY_REGISTRY.items()],
                value='orb_breakout', clearable=False,
                style={'color': 'black'},
            ),
        ], width=3),
        dbc.Col([
            html.Label("Stop Strategy", className="text-muted small"),
            dcc.Dropdown(
                id='stop-dropdown',
                options=[{'label': v['label'], 'value': k}
                         for k, v in STOP_REGISTRY.items()],
                value='two_phase_atr', clearable=False,
                style={'color': 'black'},
            ),
        ], width=3),
    ], className="mb-2"),

    # === Row 2: Setup-specific params (dynamic) ===
    dbc.Row(id='setup-params-row', className="mb-2"),

    # === Row 3: Entry-specific params ===
    dbc.Row([
        dbc.Col([
            html.Label("Opening Range", className="text-muted small"),
            dcc.Dropdown(
                id='or-type-dropdown',
                options=[
                    {'label': '30-min (60d history)', 'value': '30m'},
                    {'label': '60-min (~1yr history)', 'value': '60m'},
                ],
                value='60m', clearable=False,
                style={'color': 'black'},
            ),
        ], width=2),
        dbc.Col([
            dbc.Checklist(id='cot-filter-toggle',
                          options=[{'label': ' COT Level 70/30', 'value': 'on'}],
                          value=[], switch=True, className="mt-3"),
        ], width=2),
        dbc.Col([
            dbc.Checklist(id='cot-direction-toggle',
                          options=[{'label': ' COT Direction (WoW)', 'value': 'on'}],
                          value=[], switch=True, className="mt-3"),
        ], width=2),
        dbc.Col([
            dbc.Checklist(id='cot-roc-toggle',
                          options=[{'label': ' COT ROC (10pts / 3wk)', 'value': 'on'}],
                          value=[], switch=True, className="mt-3"),
        ], width=2),
        dbc.Col([
            dbc.Checklist(id='rsi-filter-toggle',
                          options=[{'label': " Don't Trade RSI Extremes 70/30", 'value': 'on'}],
                          value=[], switch=True, className="mt-3"),
        ], width=2),
    ], className="mb-2"),

    # === Row 4: Stop params ===
    dbc.Row(id='stop-params-row', className="mb-2"),

    # === Row 5: Capital / ATR / Dates / Run ===
    dbc.Row([
        dbc.Col([
            html.Label("Capital ($)", className="text-muted small"),
            dbc.Input(id='capital-input', type='number', value=DEFAULT_CAPITAL,
                      min=1000, step=1000),
        ], width=2),
        dbc.Col([
            html.Label("Risk %", className="text-muted small"),
            dbc.Input(id='risk-input', type='number', value=DEFAULT_RISK,
                      min=0.1, max=10, step=0.1),
        ], width=1),
        dbc.Col([
            html.Label("Start", className="text-muted small"),
            dcc.DatePickerSingle(id='start-date', date=DEFAULT_START,
                                 display_format='YYYY-MM-DD'),
        ], width=2),
        dbc.Col([
            html.Label("End", className="text-muted small"),
            dcc.DatePickerSingle(id='end-date', date=DEFAULT_END,
                                 display_format='YYYY-MM-DD'),
        ], width=2),
        dbc.Col([
            html.Label(" ", className="text-muted small"), html.Br(),
            dbc.Button("Run Backtest", id="run-btn", color="primary",
                       size="lg", className="mt-1"),
        ], width=2),
    ], className="mb-2"),

    dbc.Row([dbc.Col(html.Div(id="status-text", className="text-muted mt-1"))]),

    html.Hr(),

    # === Summary Table ===
    dbc.Row([dbc.Col([
        html.H4("All Markets Summary", className="mt-2"),
        dash_table.DataTable(
            id='summary-table',
            columns=SUMMARY_COLUMNS,
            data=summary_df.to_dict('records') if not summary_df.empty else [],
            sort_action="native", filter_action="native", page_size=15,
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': '#1a1a2e', 'color': 'white',
                          'fontWeight': 'bold'},
            style_cell={'backgroundColor': '#16213e', 'color': 'white',
                        'textAlign': 'center', 'padding': '10px'},
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
            ],
        ),
    ])], className="mb-4"),

    html.Hr(),

    # === Market Detail ===
    dbc.Row([dbc.Col([
        html.H4("Detailed Market Analysis"),
        dcc.Dropdown(
            id='market-dropdown',
            options=[{'label': m, 'value': m} for m in markets],
            value=markets[0] if markets else None,
            className="mb-3", style={'color': 'black'},
        ),
    ], width=6)]),

    dbc.Row(id='metrics-cards', className="mb-4"),
    dbc.Row([dbc.Col([dcc.Graph(id='strategy-chart')])]),
    dbc.Row([
        dbc.Col([dcc.Graph(id='equity-chart')], width=6),
        dbc.Col([
            html.H5("Recent Trades"),
            dash_table.DataTable(
                id='trades-table', columns=TRADES_COLUMNS, data=[],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1a1a2e', 'color': 'white'},
                style_cell={'backgroundColor': '#16213e', 'color': 'white',
                            'textAlign': 'center'},
                page_size=10,
            ),
        ], width=6),
    ]),

    dcc.Store(id='results-store'),

], fluid=True)


# =============================================================================
# DYNAMIC PARAMETER ROWS
# =============================================================================

@app.callback(
    Output('setup-params-row', 'children'),
    Input('setup-dropdown', 'value'),
)
def render_setup_params(setup_key):
    if not setup_key or setup_key not in SETUP_REGISTRY:
        return []
    params = SETUP_REGISTRY[setup_key].get('params', {})
    cols = []
    for pname, pinfo in params.items():
        if pinfo['type'] == int:
            cols.append(dbc.Col([
                html.Label(pinfo['label'], className="text-muted small"),
                dbc.Input(id={'type': 'setup-param', 'name': pname},
                          type='number', value=pinfo['default'],
                          min=pinfo.get('min', 0), max=pinfo.get('max', 999),
                          step=1),
            ], width=2))
    return cols


@app.callback(
    Output('stop-params-row', 'children'),
    Input('stop-dropdown', 'value'),
)
def render_stop_params(stop_key):
    if not stop_key or stop_key not in STOP_REGISTRY:
        return []
    params = STOP_REGISTRY[stop_key].get('params', {})
    cols = []
    for pname, pinfo in params.items():
        if 'options' in pinfo:
            cols.append(dbc.Col([
                html.Label(pinfo['label'], className="text-muted small"),
                dcc.Dropdown(
                    id={'type': 'stop-param', 'name': pname},
                    options=[{'label': o, 'value': o} for o in pinfo['options']],
                    value=pinfo['default'], clearable=False,
                    style={'color': 'black'},
                ),
            ], width=2))
        else:
            cols.append(dbc.Col([
                html.Label(pinfo['label'], className="text-muted small"),
                dbc.Input(id={'type': 'stop-param', 'name': pname},
                          type='number', value=pinfo['default'],
                          min=pinfo.get('min', 0), max=pinfo.get('max', 999),
                          step=pinfo.get('step', 1)),
            ], width=2))
    return cols


# =============================================================================
# RUN BACKTEST CALLBACK
# =============================================================================

@app.callback(
    [Output('results-store', 'data'),
     Output('summary-table', 'data'),
     Output('status-text', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('setup-dropdown', 'value'),
     State('entry-dropdown', 'value'),
     State('stop-dropdown', 'value'),
     State('or-type-dropdown', 'value'),
     State('cot-filter-toggle', 'value'),
     State('cot-direction-toggle', 'value'),
     State('cot-roc-toggle', 'value'),
     State('rsi-filter-toggle', 'value'),
     State('capital-input', 'value'),
     State('risk-input', 'value'),
     State('start-date', 'date'),
     State('end-date', 'date')],
    prevent_initial_call=True,
)
def run_backtest_cb(n_clicks, setup_key, entry_key, stop_key,
                    or_type, cot_toggle, cot_dir_toggle, cot_roc_toggle,
                    rsi_toggle, capital, risk,
                    start_date, end_date):
    global all_results, summary_df

    if not start_date or not end_date:
        return dash.no_update, dash.no_update, "Select both dates"

    capital = capital or DEFAULT_CAPITAL
    risk = risk or DEFAULT_RISK

    # Gather setup params from pattern-matching ids
    setup_params = _gather_params_from_layout('setup-param', setup_key, SETUP_REGISTRY)
    stop_params = _gather_params_from_layout('stop-param', stop_key, STOP_REGISTRY)

    # Entry params
    cot_on = 'on' in (cot_toggle or [])
    cot_dir_on = 'on' in (cot_dir_toggle or [])
    cot_roc_on = 'on' in (cot_roc_toggle or [])
    rsi_on = 'on' in (rsi_toggle or [])
    entry_params = {
        'or_type': or_type or '60m',
        'cot_filter': cot_on, 'cot_long': 70, 'cot_short': 30,
        'cot_direction_filter': cot_dir_on,
        'cot_roc_filter': cot_roc_on, 'cot_roc_threshold': 10,
        'rsi_filter': rsi_on, 'rsi_long_max': 70, 'rsi_short_min': 30,
    }

    all_results, summary_df = run_all_markets(
        cot_df, markets,
        setup_key=setup_key, entry_key=entry_key, stop_key=stop_key,
        setup_params=setup_params, entry_params=entry_params,
        stop_params=stop_params,
        atr_period=DEFAULT_ATR_PERIOD,
        initial_capital=capital, risk_pct=risk,
        start_date=start_date, end_date=end_date,
    )

    if not summary_df.empty:
        summary_df = summary_df.fillna(0)
        for col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x)

    setup_lbl = SETUP_REGISTRY.get(setup_key, {}).get('label', setup_key)
    entry_lbl = ENTRY_REGISTRY.get(entry_key, {}).get('label', entry_key)
    stop_lbl = STOP_REGISTRY.get(stop_key, {}).get('label', stop_key)

    status = (f"Backtest complete: {setup_lbl} + {entry_lbl} + {stop_lbl} | "
              f"${capital:,.0f}, {risk}% risk | "
              f"{len(all_results)} markets, {start_date} to {end_date}")

    return (
        {'timestamp': datetime.now().isoformat()},
        summary_df.to_dict('records') if not summary_df.empty else [],
        status,
    )


def _gather_params_from_layout(param_type, key, registry):
    """Fallback: use registry defaults (pattern-matching callbacks would need
    ALL_MATCHING to read dynamic ids, which adds complexity).
    For now the dynamic input values are read via defaults stored in the registry.
    This will be upgraded to read actual input values when needed.
    """
    params = {}
    if key in registry:
        for pname, pinfo in registry[key].get('params', {}).items():
            params[pname] = pinfo['default']
    return params


# =============================================================================
# MARKET DETAIL CALLBACK
# =============================================================================

@app.callback(
    [Output('metrics-cards', 'children'),
     Output('strategy-chart', 'figure'),
     Output('equity-chart', 'figure'),
     Output('trades-table', 'data'),
     Output('trades-table', 'columns')],
    [Input('market-dropdown', 'value'),
     Input('results-store', 'data')],
)
def update_market_detail(market, _store):
    if not market or market not in all_results:
        empty = go.Figure()
        empty.update_layout(template="plotly_white")
        return [], empty, empty, [], TRADES_COLUMNS

    res = all_results[market]
    m = res['metrics']
    data = res['data']
    trades_df = res['results']['trades']
    equity = res['results']['equity_curve']
    cap = equity[0] if equity else DEFAULT_CAPITAL

    # Metric cards
    def card(title, value, color=""):
        return dbc.Col(dbc.Card([dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted mb-1",
                     style={'fontSize': '0.8em'}),
            html.H4(value, className=f"card-title mb-0 {color}"),
        ])]), width=2)

    ret = m.get('total_return_pct', 0)
    cards = [
        card("Total Return", f"{ret:.2f}%",
             "text-success" if ret > 0 else "text-danger"),
        card("Win Rate", f"{m.get('win_rate', 0):.1f}%"),
        card("Trades", f"{m.get('total_trades', 0)}"),
        card("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}"),
        card("Max Drawdown", f"{m.get('max_drawdown_pct', 0):.2f}%", "text-warning"),
        card("Profit Factor",
             f"{m.get('profit_factor', 0):.2f}"
             if m.get('profit_factor', 0) != float('inf') else "---"),
    ]

    # Determine setup key from data columns
    setup_key = 'narrowing_range' if 'consecutive_narrowing' in data.columns else \
                'inside_days' if 'consecutive_inside_days' in data.columns else 'cot_rsi'

    strat_fig = create_strategy_chart(data, trades_df, market, setup_key=setup_key)
    eq_fig = create_equity_curve(equity, cap)

    # Format trades for display
    if not trades_df.empty:
        disp = trades_df.copy()
        disp['entry_date'] = pd.to_datetime(disp['entry_date']).dt.strftime('%Y-%m-%d')
        disp['exit_date'] = pd.to_datetime(disp['exit_date']).dt.strftime('%Y-%m-%d')
        disp['entry_price'] = disp['entry_price'].apply(lambda x: f"${x:,.4f}")
        disp['exit_price'] = disp['exit_price'].apply(lambda x: f"${x:,.4f}")
        disp['pnl'] = disp['pnl'].apply(lambda x: f"${x:,.2f}")
        disp['units'] = disp['units'].apply(
            lambda x: str(int(x)) if isinstance(x, (int, float)) and x == int(x)
            else str(x))
        disp['days_held'] = disp['days_held'].apply(
            lambda x: str(int(x)) if not pd.isna(x) else "0")
        table_cols = ['entry_date', 'exit_date', 'direction', 'entry_price',
                      'exit_price', 'units', 'pnl', 'days_held', 'exit_reason']
        avail = [c for c in table_cols if c in disp.columns]
        table_data = disp[avail].to_dict('records')
    else:
        table_data = []

    return cards, strat_fig, eq_fig, table_data, TRADES_COLUMNS


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8054)
