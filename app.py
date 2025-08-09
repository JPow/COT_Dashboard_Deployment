import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from datetime import date, datetime, timedelta
import json

external_stylesheets = [dbc.themes.CYBORG]

app = dash.Dash(__name__, title="Interactive Dashboard", external_stylesheets=external_stylesheets)

server = app.server

# Load data from JSON file
try:
    with open('cot_data.json', 'r') as f:
        data = json.load(f)
    df6 = pd.DataFrame(data)
    # Convert timestamps to datetime - handle both string and numeric formats
    try:
        # First try assuming it's in milliseconds (pandas default JSON export)
        df6['Date'] = pd.to_datetime(df6['Date'], unit='ms')
    except (ValueError, TypeError):
        # If that fails, try standard datetime parsing
        df6['Date'] = pd.to_datetime(df6['Date'])
    
    # For compatibility with existing code, if data_type column doesn't exist, treat all as weekly_cot
    if 'data_type' not in df6.columns:
        df6['data_type'] = 'weekly_cot'
    
    # Get week dates for the date picker (use weekly COT data)
    weekly_cot_data = df6[df6['data_type'] == 'weekly_cot']
    week_dates = weekly_cot_data['Date'].dt.strftime('%Y-%m-%d').unique().tolist() if not weekly_cot_data.empty else df6['Date'].dt.strftime('%Y-%m-%d').unique().tolist()
    
except Exception as e:
    print(f"Error loading data: {e}")
    df6 = pd.DataFrame()
    week_dates = []

app.layout = dbc.Container([
    # Title Row
    dbc.Row([
        dbc.Col(
            html.H1("Commodity Prices and Open Interest Dashboard",
                   className="text-center mb-4 mt-3",
                   style={'color': 'white'})
        )
    ]),

    # Price & Open Interest Analysis Section
    dbc.Row([
        dbc.Col([
            html.Div("Price & Open Interest Analysis",
                style={'textAlign': 'center', 'color': 'white', 'fontSize': 20, 'marginBottom': '20px'}),
        # Container for dropdown and label
            html.Div([
                html.Label('Select Market:', style={'color': 'white', 'fontSize': 16}),
                dcc.Dropdown(
                id='commodity-dropdown',
                options=[{'label': market, 'value': market} for market in df6['Market'].unique()] if not df6.empty else [],
                value=df6['Market'].unique()[0] if not df6.empty else None,
                style={'width': '300px', 'margin': '0 auto'}  # Fixed width and centered
            )
        ], style={'textAlign': 'center', 'marginBottom': '30px'}),  # Added spacing
        # Container for the graph
            html.Div([
                dcc.Graph(id='combined-graph')
            ], style={'width': '90%', 'margin': '0 auto'})  # Centered with some margin
        ])
    ]),

    # Buy and Sell Indications Section
    dbc.Row([
        dbc.Col([
            html.Div("Buy and Sell Indications", 
                    style={'textAlign': 'center', 'color': 'white', 'fontSize': 20, 'marginBottom': '20px'}),
            dash_table.DataTable(
                id='table',
                data=[],
                columns=[],
                style_table={'maxWidth': '800px', 'margin': 'auto', 'overflowx': 'auto'},  # Increased maxWidth
                style_cell={'textAlign': 'center', 'color': 'black', 'fontSize': 12, 'padding': '10px', 
                           'minWidth': '70px', 'width': '100px', 'maxWidth': '180px'},
                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold', 
                            'textAlign': 'center', 'fontSize': 14},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{OI_Index} >= 80', 'column_id': 'OI_Index'},
                        'backgroundColor': '#4D9F6B',
                        'color': 'white'
                    },
                    {
                        'if': {'filter_query': '{OI_Index} <= 20', 'column_id': 'OI_Index'},
                        'backgroundColor': '#C46A33',
                        'color': 'white'
                    },
                    {
                        'if': {'filter_query': '{Retail_Index} >= 80', 'column_id': 'Retail_Index'},
                        'backgroundColor': '#4D9F6B',
                        'color': 'white'
                    },
                    {
                        'if': {'filter_query': '{Retail_Index} <= 20', 'column_id': 'Retail_Index'},
                        'backgroundColor': '#C46A33',
                        'color': 'white'
                    },
                    {
                        'if': {'filter_query': '{Commercial_Index} >= 80', 'column_id': 'Commercial_Index'},
                        'backgroundColor': '#4D9F6B',
                        'color': 'white'
                    },
                    {
                        'if': {'filter_query': '{Commercial_Index} <= 20', 'column_id': 'Commercial_Index'},
                        'backgroundColor': '#C46A33',
                        'color': 'white'
                    }
                ]
            )
        ])
    ]),

    # Bubble Chart Section
    dbc.Row([
        dbc.Col([
            html.Div("Retail vs Commercial Positioning",
                    style={'textAlign': 'center', 'color': 'white', 'fontSize': 20, 
                           'marginTop': '20px', 'marginBottom': '20px'}),
            html.Label("Select A Tuesday Date", style={'color': 'white'}),
            dcc.DatePickerSingle(
                id='my-date-picker-single',
                min_date_allowed=min(week_dates) if week_dates else None,
                max_date_allowed=max(week_dates) if week_dates else None,
                initial_visible_month=max(week_dates) if week_dates else None,
                date=max(week_dates) if week_dates else None,
                style={'fontSize': 14}
            ),
            dcc.Graph(id='bubble-graph')
        ])
    ]),

    # Open Interest Graph Section
    dbc.Row([
        dbc.Col([
            html.Div("Open Interest Index",
                    style={'textAlign': 'center', 'color': 'white', 'fontSize': 20, 
                           'marginTop': '20px', 'marginBottom': '20px'}),
            dcc.Graph(id='open-interest-graph')
        ])
    ])
])

# Callback for bubble graph
@app.callback(
    Output('bubble-graph', 'figure'),
    Input('my-date-picker-single', 'date')
)
def update_bubble(week_selected):
    if df6.empty or not week_selected:
        return {}
        
    # Use weekly COT data for bubble chart
    weekly_data = df6[df6['data_type'] == 'weekly_cot'].copy() if 'data_type' in df6.columns else df6.copy()
    
    # Convert selected week to datetime
    week_selected = pd.to_datetime(week_selected).date()
    
    # Filter for last 12 weeks of data
    twelve_weeks_ago = week_selected - pd.Timedelta(weeks=12)
    date_selected = weekly_data[
        (weekly_data["Date"].dt.date <= week_selected) & 
        (weekly_data["Date"].dt.date >= twelve_weeks_ago)
    ].copy()  # Explicit copy to avoid SettingWithCopyWarning
    
    if date_selected.empty:
        return {}
    
    # Format dates for hover data and animation
    date_selected.loc[:, 'Formatted_Date'] = date_selected['Date'].dt.strftime('%d-%m-%Y')
    date_selected.loc[:, 'Animation_Date'] = date_selected['Date'].dt.strftime('%d-%m-%Y')
    
    fig = px.scatter(date_selected, 
                    x='Retail_Index', 
                    y='Commercial_Index', 
                    hover_data={'Market': True, 'Formatted_Date': True, 'Date': False},
                    color='group' if 'group' in date_selected.columns else None, 
                    size='OI', 
                    size_max=40,
                    animation_frame='Animation_Date',
                    animation_group='Market',
                    labels={'Retail_Index': 'Retail', 
                           'Commercial_Index': 'Commercial Index'}
                   )

    # Set x-axis and y-axis ranges for each subplot
    fig.update_xaxes(range=[0, 125])
    fig.update_yaxes(range=[0, 125])
    
    # Update layout for better animation display
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="middle",
            y=1.10,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="lightgrey",
            borderwidth=1
        ),
        margin=dict(r=150),  
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {"frame": {"duration": 500, "redraw": True},
                           "fromcurrent": True}]
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], {"frame": {"duration": 0, "redraw": True},
                             "mode": "immediate",
                             "transition": {"duration": 0}}]
            )]
        )]
    )
    
    return fig

# Callback to update open interest graph
@app.callback(
    Output('open-interest-graph', 'figure'),
    [Input('commodity-dropdown', 'value'),
     Input('combined-graph', 'relayoutData')]
)
def update_open_interest_graph(selected_commodities, relayout_data):
    if df6.empty or not selected_commodities:
        return {}
    
    # Use weekly COT data for open interest
    if 'data_type' in df6.columns:
        data = df6[(df6['data_type'] == 'weekly_cot') & (df6['Market'] == selected_commodities)]
    else:
        data = df6[df6['Market'] == selected_commodities]
    
    if data.empty:
        return {}
        
    x = data['Date']
    y = data['OI_Index']
    
    fig = go.Figure()
    
    # Add the main trace
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Open Interest Index',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add horizontal lines
    high_value = 80
    low_value = 20
    
    fig.update_layout(
        shapes=[
            dict(
                type='line',
                x0=min(x),
                x1=max(x),
                y0=high_value,
                y1=high_value,
                line=dict(color='orange', dash='dash')
            ),
            dict(
                type='line',
                x0=min(x),
                x1=max(x),
                y0=low_value,
                y1=low_value,
                line=dict(color='orange', dash='dash')
            )
        ],
        title=f'Open Interest Index: {selected_commodities}',
        xaxis_title='Date',
        yaxis_title='Index Value',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis=dict(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ],
                y=0.95,
                yanchor="bottom",
                x=0.5,
                xanchor="center",
                bgcolor="white",
                bordercolor="lightgray",
                borderwidth=1,
                font=dict(size=10)
            )
        )
    )
    
    return fig

# New callback for the combined price and open interest graph
@app.callback(
    Output('combined-graph', 'figure'),
    Input('commodity-dropdown', 'value')
)
def update_combined_graph(selected_commodities):
    if df6.empty or not selected_commodities:
        return {}
    
    # Filter data for the selected market
    market_data = df6[df6['Market'] == selected_commodities].copy()
    
    if market_data.empty:
        return {}
    
    # Check if we have the new combined data structure
    if 'data_type' in market_data.columns:
        # Separate daily price data and weekly COT data
        daily_data = market_data[market_data['data_type'] == 'daily_price'].copy()
        weekly_data = market_data[market_data['data_type'] == 'weekly_cot'].copy()
        print(f"DEBUG: Using new data structure. Daily records: {len(daily_data)}, Weekly records: {len(weekly_data)}")
    else:
        # Fallback to old structure - use all data for everything
        daily_data = market_data.copy()
        weekly_data = market_data.copy()
        print(f"DEBUG: Using old data structure. Total records: {len(market_data)}")
    
    # Create figure with three rows (panes)
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=(f"Price: {selected_commodities}", 
                                      "Commercial & Retail Open Interest Indices",
                                      "RSI Technical Indicator"))

    # Add price trace on top pane (use daily data if available, otherwise weekly)
    price_data = daily_data if not daily_data.empty else weekly_data
    if not price_data.empty and 'Close' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data['Date'],
                y=price_data['Close'],
                name="Daily Price" if not daily_data.empty else "Price",
                line=dict(color="#1f77b4", width=2)
            ),
            row=1, col=1
        )
        
        # Add 200-day moving average if it exists
        if '200MA' in price_data.columns and not price_data['200MA'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=price_data['Date'],
                    y=price_data['200MA'],
                    name="200-day MA",
                    line=dict(color="orange", width=1.5)
                ),
                row=1, col=1
            )

    # Add COT indices traces on middle pane (use weekly data if available, otherwise all data)
    cot_data = weekly_data if not weekly_data.empty else market_data
    if not cot_data.empty:
        if 'Commercial_Index' in cot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=cot_data['Date'],
                    y=cot_data['Commercial_Index'],
                    name="Commercial Index",
                    line=dict(color="#2ca02c", width=2)
                ),
                row=2, col=1
            )

        if 'Retail_Index' in cot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=cot_data['Date'],
                    y=cot_data['Retail_Index'],
                    name="Retail Index",
                    line=dict(color="red", width=2)
                ),
                row=2, col=1
            )

    # Add RSI trace on bottom pane (use daily data if available, otherwise weekly)
    rsi_data = daily_data if not daily_data.empty else weekly_data
    if not rsi_data.empty and 'RSI' in rsi_data.columns and not rsi_data['RSI'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=rsi_data['Date'],
                y=rsi_data['RSI'],
                name="Daily RSI" if not daily_data.empty else "RSI",
                line=dict(color="purple", width=2)
            ),
            row=3, col=1
        )

    # Add horizontal lines at 20 and 80 for the indices on middle pane
    if not cot_data.empty and not cot_data['Date'].empty:
        fig.add_shape(
            type="line", line=dict(color="orange", width=1, dash="dash"),
            y0=20, y1=20, x0=cot_data['Date'].min(), x1=cot_data['Date'].max(),
            row=2, col=1
        )
        fig.add_shape(
            type="line", line=dict(color="orange", width=1, dash="dash"),
            y0=80, y1=80, x0=cot_data['Date'].min(), x1=cot_data['Date'].max(),
            row=2, col=1
        )

    # Add horizontal lines at 30 and 70 for RSI on bottom pane
    if not rsi_data.empty and not rsi_data['Date'].empty:
        fig.add_shape(
            type="line", line=dict(color="gray", width=1, dash="dash"),
            y0=30, y1=30, x0=rsi_data['Date'].min(), x1=rsi_data['Date'].max(),
            row=3, col=1
        )
        fig.add_shape(
            type="line", line=dict(color="gray", width=1, dash="dash"),
            y0=70, y1=70, x0=rsi_data['Date'].min(), x1=rsi_data['Date'].max(),
            row=3, col=1
        )

    # Set y-axes titles and ranges
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Index Value", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    # Update x-axis with range selector on bottom pane
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]),
            y=0.95,  # Position at bottom
            yanchor="bottom",  # Anchor to bottom
            x=0.5,  # Center horizontally
            xanchor="center",  # Anchor to center
            bgcolor="white",  # White background
            bordercolor="lightgray",  # Light gray border
            borderwidth=1,  # Border width
            font=dict(size=10)  # Smaller font size
        ),
        row=3, col=1
    )

    # Update x-axis title on bottom pane
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Set overall layout
    fig.update_layout(
        height=1000,
        width=1000,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    return fig

# Callback to update the summary table
@app.callback(
    Output('table', 'data'),
    Output('table', 'columns'),
    Input('commodity-dropdown', 'value')
)
def update_table(selected_commodity):
    if df6.empty:
        return [], []
    
    # Use weekly COT data for the table
    table_data = df6[df6['data_type'] == 'weekly_cot'].copy() if 'data_type' in df6.columns else df6.copy()
    
    # Filter the data to show only rows where at least one of the indices (OI, Retail, or Commercial) is extreme (≥ 80 or ≤ 20)
    latest_data = table_data[
        (table_data['Date'] == table_data['Date'].max()) & 
        (((table_data['OI_Index'] >= 80) | (table_data['OI_Index'] <= 20)) |
         ((table_data['Retail_Index'] >= 80) | (table_data['Retail_Index'] <= 20)) |
         ((table_data['Commercial_Index'] >= 80) | (table_data['Commercial_Index'] <= 20)))
    ]
    
    if latest_data.empty:
        return [], []
    
    # Select only the relevant columns
    columns_to_show = ['Market', 'OI_Index', 'Retail_Index', 'Commercial_Index']
    latest_data = latest_data[columns_to_show]
    
    # Create columns for the table
    columns = [{'name': col, 'id': col, 'type': 'numeric', 'format': {'specifier': '.0f'}}
               if latest_data[col].dtype in ['float64', 'float32']
               else {'name': col, 'id': col}
               for col in latest_data.columns]
    
    # Convert data to dict for the table
    data = latest_data.to_dict('records')
    
    return data, columns

if __name__ == '__main__':
    app.run_server(debug=True)