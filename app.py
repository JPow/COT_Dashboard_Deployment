# Import required libraries
import dash
from dash import Dash, html, dcc, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os

# Initialize the Dash app with a dark theme
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Load your data from the notebook
try:
    # Read the data from your notebook
    df = pd.read_html('COT Cursor Experiments.ipynb')[0]  # Assuming the table is the first table in the notebook
    df.columns = ['Market', 'OI_Index', 'Retail_Index', 'Commercial_Index']
    data_loaded = True
except Exception as e:
    print(f"Error loading data: {e}")
    data_loaded = False
    df = pd.DataFrame()

# Create the WSGI application
app = app.server

# Create week_dates list for the date picker
week_dates = sorted(df['Date'].dt.date.unique())

# Define the layout
app.layout = html.Div([
    html.H1("COT Futures Dashboard", className="text-center mb-4"),
    
    # Error message if data failed to load
    html.Div([
        html.H3("Error: Data file not found", className="text-danger"),
        html.P("Please ensure data/df6.json exists and is properly formatted.")
    ], style={'display': 'none' if data_loaded else 'block'}),
    
    # Main dashboard content
    html.Div([
        # Date Picker
        html.Div([
            html.Label("Select Date:"),
            dcc.DatePickerSingle(
                id='date-picker',
                date=week_dates[-1] if week_dates else None,
                min_date_allowed=week_dates[0] if week_dates else None,
                max_date_allowed=week_dates[-1] if week_dates else None,
                display_format='YYYY-MM-DD'
            )
        ], className="mb-4"),
        
        # Commodity Selection
        html.Div([
            html.Label("Select Commodities:"),
            dcc.Dropdown(
                id='commodity-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns if col not in ['Date']],
                value=['Gold', 'Silver', 'Copper'],
                multi=True
            )
        ], className="mb-4"),
        
        # Bubble Graph
        dcc.Graph(id='bubble-graph'),
        
        # Open Interest Graph
        dcc.Graph(id='open-interest-graph'),
        
        # Combined Graph
        dcc.Graph(id='combined-graph')
    ], style={'display': 'block' if data_loaded else 'none'})
])

# Callback for bubble graph
@app.callback(
    Output('bubble-graph', 'figure'),
    Input('date-picker', 'date')
)
def update_bubble(week_selected):
    if not data_loaded:
        return go.Figure()
        
    if isinstance(week_selected, str):
        week_selected = datetime.strptime(week_selected, '%Y-%m-%d').date()
    
    date_selected = df[df["Date"].dt.date <= week_selected]
    
    fig = px.scatter(date_selected, 
                    x='Retail_Index', 
                    y='Commercial_Index',
                    hover_data=['Date'],
                    color='Date',
                    size='Open_Interest',
                    animation_frame='Date',
                    animation_group='Date',
                    title='Retail vs Commercial Index Bubble Chart',
                    labels={'Retail_Index': 'Retail Index',
                           'Commercial_Index': 'Commercial Index',
                           'Open_Interest': 'Open Interest'})
    
    fig.update_layout(
        height=800,
        width=1200,
        xaxis_range=[-100, 100],
        yaxis_range=[-100, 100],
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]
                    )
                ]
            )
        ]
    )
    
    return fig

# Callback for open interest graph
@app.callback(
    Output('open-interest-graph', 'figure'),
    [Input('commodity-dropdown', 'value'),
     Input('date-picker', 'date')]
)
def update_open_interest_graph(selected_commodities, week_selected):
    if not data_loaded:
        return go.Figure()
        
    if not selected_commodities:
        return go.Figure()
    
    if isinstance(week_selected, str):
        week_selected = datetime.strptime(week_selected, '%Y-%m-%d').date()
    
    date_selected = df[df["Date"].dt.date <= week_selected]
    
    fig = go.Figure()
    
    for commodity in selected_commodities:
        fig.add_trace(go.Scatter(
            x=date_selected['Date'],
            y=date_selected[f'{commodity}_Open_Interest_Index'],
            name=f'{commodity} Open Interest',
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Open Interest Index Over Time',
        xaxis_title='Date',
        yaxis_title='Open Interest Index',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Callback for combined graph
@app.callback(
    Output('combined-graph', 'figure'),
    [Input('commodity-dropdown', 'value'),
     Input('date-picker', 'date')]
)
def update_combined_graph(selected_commodities, week_selected):
    if not data_loaded:
        return go.Figure()
        
    if not selected_commodities:
        return go.Figure()
    
    if isinstance(week_selected, str):
        week_selected = datetime.strptime(week_selected, '%Y-%m-%d').date()
    
    date_selected = df[df["Date"].dt.date <= week_selected]
    
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('Price', 'Open Interest Index'))
    
    for commodity in selected_commodities:
        # Price subplot
        fig.add_trace(go.Scatter(
            x=date_selected['Date'],
            y=date_selected[f'{commodity}_Price'],
            name=f'{commodity} Price',
            mode='lines+markers'
        ), row=1, col=1)
        
        # Open Interest subplot
        fig.add_trace(go.Scatter(
            x=date_selected['Date'],
            y=date_selected[f'{commodity}_Open_Interest_Index'],
            name=f'{commodity} Open Interest',
            mode='lines+markers',
            showlegend=False
        ), row=2, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        title_text="Price and Open Interest Index Over Time"
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Open Interest Index", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True) 