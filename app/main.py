"""
Multi-page COT dashboard: home (COT charts), backtest, research, position sizing.
"""

import os
import sys

from dash import Dash, html, page_container, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BRAND = "Commitment of Traders Lab"

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=os.path.join(os.path.dirname(__file__), "pages"),
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY],
    title=BRAND,
)
server = app.server

app.layout = html.Div([
    dcc.Location(id="cot-url", refresh=False),
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(BRAND, href="/"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Dashboard", href="/", id="nav-home")),
                dbc.NavItem(dbc.NavLink("Backtest", href="/backtest", id="nav-backtest")),
                dbc.NavItem(dbc.NavLink("Research", href="/research", id="nav-research")),
                dbc.NavItem(dbc.NavLink("Sizing", href="/sizing", id="nav-sizing")),
            ], navbar=True, className="ms-auto"),
        ], fluid=True),
        dark=True,
        className="cot-navbar mb-0",
    ),
    page_container,
])


@callback(
    Output("nav-home", "active"),
    Output("nav-backtest", "active"),
    Output("nav-research", "active"),
    Output("nav-sizing", "active"),
    Input("cot-url", "pathname"),
)
def _highlight_nav(pathname):
    path = pathname or "/"
    return (
        path == "/",
        path.startswith("/backtest"),
        path.startswith("/research"),
        path.startswith("/sizing"),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
