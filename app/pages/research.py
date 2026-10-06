"""Walk-forward parameter grid (from ml_backtest_app) as a Dash page."""

import os
import sys

from dash import register_page, html, dcc
import dash_bootstrap_components as dbc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

register_page(__name__, path="/research", title="Research", order=2)

layout = dbc.Container([
    html.H3("Strategy research"),
    html.P(
        "Walk-forward grids and SPA tests run in notebooks and the research/ package. "
        "Use COT_trading_strategy.ipynb for strategy exploration, and "
        "tests/COT_Hypothesis_Test.ipynb, tests/COT_TrendFollowing_Test.ipynb, "
        "tests/Lookback_Robustness.ipynb for formal tests."
    ),
    dbc.Alert(
        "To run the ML parameter grid locally: python ml_backtest_app.py (legacy port 8052) "
        "until the grid is ported into this page.",
        color="info",
    ),
], fluid=True)
