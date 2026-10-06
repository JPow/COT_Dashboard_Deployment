"""
Multi-page COT dashboard: home (COT charts), backtest, research, position sizing.
"""

import os
import sys

import dash
from dash import Dash, html, page_container
import dash_bootstrap_components as dbc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=os.path.join(os.path.dirname(__file__), "pages"),
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY],
    title="COT Lab",
)
server = app.server

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="COT Lab",
        brand_href="/",
        children=[
            dbc.NavItem(dbc.NavLink("Dashboard", href="/")),
            dbc.NavItem(dbc.NavLink("Backtest", href="/backtest")),
            dbc.NavItem(dbc.NavLink("Research", href="/research")),
            dbc.NavItem(dbc.NavLink("Sizing", href="/sizing")),
        ],
        color="dark",
        dark=True,
        className="mb-3",
    ),
    page_container,
])


if __name__ == "__main__":
    app.run(debug=True, port=8050)
