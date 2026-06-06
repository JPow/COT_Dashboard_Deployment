"""
Futures Position Size Calculator
================================
Dash tool for sizing trades from entry, stop, and fixed dollar risk.

Uses ORB_contract_specs.json for point values and contract metadata.
Run: python position.py
"""

import os
import sys

import dash
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest_engine.data import load_contract_specs

STOCKS_KEY = "STOCKS"
STOCK_SPEC = {
    "traded_contract": "Stocks (shares)",
    "point_value": 1.0,
}


def _is_stocks(commodity):
    return commodity == STOCKS_KEY


def _allows_fractional(commodity):
    name = commodity.upper()
    return "BITCOIN" in name or "ETHER" in name


def _format_contract_label(market_key, spec):
    traded = spec.get("traded_contract", market_key)
    return f"{market_key} — {traded}"


def _stop_distance(entry_price, stop_price, direction):
    """Price distance from entry to stop in points."""
    if direction == "long":
        return entry_price - stop_price
    return stop_price - entry_price


def compute_position(commodity, direction, entry_price, stop_price, risk_usd):
    """Return sizing result dict or error message."""
    if _is_stocks(commodity):
        spec = STOCK_SPEC
    else:
        specs = load_contract_specs()
        spec = specs.get(commodity)
        if spec is None:
            return {"error": f"No contract spec found for {commodity}."}

    try:
        entry = float(entry_price)
        stop = float(stop_price)
        risk = float(risk_usd)
    except (TypeError, ValueError):
        return {"error": "Enter valid numbers for entry, stop, and risk."}

    if entry <= 0 or stop <= 0:
        return {"error": "Entry and stop prices must be greater than zero."}
    if risk <= 0:
        return {"error": "Risk must be greater than zero."}

    stop_dist = _stop_distance(entry, stop, direction)
    if stop_dist <= 0:
        if direction == "long":
            return {"error": "For a long, stop must be below entry."}
        return {"error": "For a short, stop must be above entry."}

    point_value = spec["point_value"]
    dollar_risk_per_contract = stop_dist * point_value
    contracts = round(risk / dollar_risk_per_contract, 2)
    is_stock = _is_stocks(commodity)
    fractional = _allows_fractional(commodity)
    tradeable_contracts = (
        contracts if fractional else round(int(contracts), 2)
    )
    notional = entry * contracts * point_value

    return {
        "error": None,
        "commodity": commodity,
        "traded_contract": spec.get("traded_contract", commodity),
        "direction": direction,
        "entry_price": entry,
        "stop_price": stop,
        "risk_usd": risk,
        "stop_distance": stop_dist,
        "point_value": point_value,
        "dollar_risk_per_contract": dollar_risk_per_contract,
        "contracts": contracts,
        "tradeable_contracts": tradeable_contracts,
        "fractional_allowed": fractional,
        "amount_to_invest": notional,
        "is_stock": is_stock,
    }


def _format_contracts(value):
    return f"{value:,.2f}"


def _result_cards(result):
    if result.get("error"):
        return dbc.Alert(result["error"], color="danger", className="mb-0")

    contracts = result["contracts"]
    contracts_text = _format_contracts(contracts)
    unit = "Shares" if result.get("is_stock") else "Contracts"
    per_unit = "share" if result.get("is_stock") else "contract"

    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6(f"{unit} to Buy", className="card-subtitle text-muted mb-1"),
            html.H2(contracts_text, className="card-title text-success mb-0 result-value"),
        ]), className="result-card h-100"), xs=12, md=4, className="mb-2 mb-md-0"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Amount to Invest", className="card-subtitle text-muted mb-1"),
            html.H2(
                f"${result['amount_to_invest']:,.2f}",
                className="card-title text-info mb-0 result-value",
            ),
            html.Small("Notional exposure at entry", className="text-muted"),
        ]), className="result-card h-100"), xs=12, md=4, className="mb-2 mb-md-0"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Risk if Stopped", className="card-subtitle text-muted mb-1"),
            html.H2(
                f"${result['contracts'] * result['dollar_risk_per_contract']:,.2f}",
                className="card-title text-warning mb-0 result-value",
            ),
            html.Small(
                f"${result['dollar_risk_per_contract']:,.2f} per {per_unit}",
                className="text-muted",
            ),
        ]), className="result-card h-100"), xs=12, md=4),
    ], className="g-2 mb-2")

    if not result["fractional_allowed"] and contracts < 1:
        min_risk = result["dollar_risk_per_contract"]
        unit = "share" if result.get("is_stock") else "contract"
        cards = html.Div([
            cards,
            dbc.Alert(
                f"Less than 1 {unit} at this risk. "
                f"Increase risk to ${min_risk:,.2f} for 1 {unit}, "
                f"or adjust your stop.",
                color="info",
                className="mt-3 mb-0",
            ),
        ])

    return cards


def _detail_table(result):
    if result.get("error"):
        return html.Div()

    is_stock = result.get("is_stock")
    tradable_label = "Tradable shares" if is_stock else "Tradable contracts"
    rows = [
        ("Instrument", result["traded_contract"]),
        ("Direction", result["direction"].upper()),
        ("Entry", f"${result['entry_price']:,.2f}"),
        ("Stop", f"${result['stop_price']:,.2f}"),
        (
            "Stop distance",
            f"${result['stop_distance']:,.2f} per share"
            if is_stock
            else f"{result['stop_distance']:,.4f} points",
        ),
        (
            "Point value",
            "$1.00 per $1 move"
            if is_stock
            else f"${result['point_value']:,.2f} / point",
        ),
        ("Target risk", f"${result['risk_usd']:,.2f}"),
        (tradable_label, _format_contracts(result["tradeable_contracts"])),
    ]
    if not result["fractional_allowed"] and result["contracts"] < 1:
        unit = "share" if is_stock else "contract"
        rows.append((
            f"Risk for 1 {unit}",
            f"${result['dollar_risk_per_contract']:,.2f}",
        ))
    return dbc.Card(dbc.CardBody([
        html.Div([
            html.Div([
                html.Span(label, className="text-muted"),
                html.Span(value, className="fw-semibold text-end"),
            ], className="d-flex justify-content-between align-items-start gap-3 detail-row")
            for label, value in rows
        ]),
    ]), className="mb-0")


specs = load_contract_specs()
commodity_options = [
    {"label": "Stocks — shares (USD)", "value": STOCKS_KEY},
] + [
    {"label": _format_contract_label(key, spec), "value": key}
    for key, spec in sorted(specs.items())
]
default_commodity = "MICRO GOLD" if "MICRO GOLD" in specs else commodity_options[0]["value"]

MOBILE_CSS = """
@media (max-width: 576px) {
    .page-title { font-size: 1.5rem; margin-top: 0.75rem !important; }
    .page-subtitle { font-size: 0.9rem; }
    .result-value { font-size: 1.75rem; }
    .result-card .card-body { padding: 1rem; }
    .mobile-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
}
@media (min-width: 577px) {
    .result-value { font-size: 2rem; }
}
.position-input { font-size: 1.1rem; min-height: 3rem; }
.direction-group { width: 100%; }
.direction-group .btn { flex: 1; min-height: 3rem; font-size: 1rem; }

#commodity .Select-control { min-height: 3rem; }
#commodity .Select-placeholder,
#commodity .Select-value-label { line-height: 2.75rem !important; }
#commodity .Select-menu-outer { max-height: 240px; }

/* dcc.Dropdown font color only (options menu is portaled to body) */
#commodity .Select-value-label,
#commodity .Select-input > input,
#commodity div[class*="-singleValue"],
#commodity div[class*="-Input"] input,
#commodity div[class*="-placeholder"] {
    color: #000 !important;
}
.Select-menu-outer,
.Select-menu-outer *,
div[class*="-menu"] div[class*="-option"],
div[class*="-menu"] div[class*="-option"] * {
    color: #000 !important;
}

.detail-row { padding: 0.65rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.detail-row:last-child { border-bottom: none; }
"""

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1"},
    ],
)
server = app.server

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>""" + MOBILE_CSS + """</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = dbc.Container([
    dbc.Row([dbc.Col([
        html.H1(
            "Position Size Calculator",
            className="text-center my-3 page-title",
        ),
        html.P(
            "Size futures and stock trades from entry, stop, and fixed dollar risk.",
            className="text-center text-muted page-subtitle",
        ),
    ], xs=12)]),

    dbc.Row([
        dbc.Col([
            html.Label("Commodity", className="text-muted small fw-semibold"),
            dcc.Dropdown(
                id="commodity",
                options=commodity_options,
                value=default_commodity,
                clearable=False,
            ),
        ], xs=12, className="mb-3 mb-md-0"),
        dbc.Col([
            html.Label("Direction", className="text-muted small fw-semibold"),
            dbc.RadioItems(
                id="direction",
                options=[
                    {"label": "Long", "value": "long"},
                    {"label": "Short", "value": "short"},
                ],
                value="long",
                inline=True,
                inputClassName="btn-check",
                labelClassName="btn btn-outline-light",
                labelCheckedClassName="active",
                className="btn-group direction-group mt-1",
            ),
        ], xs=12),
    ], className="mb-3 g-2"),

    dbc.Row([
        dbc.Col([
            html.Label("Entry Price", className="text-muted small fw-semibold"),
            dbc.Input(
                id="entry-price",
                type="number",
                value=2000,
                step="any",
                inputMode="decimal",
                className="position-input",
            ),
        ], xs=12, sm=6, md=4, className="mb-3 mb-md-0"),
        dbc.Col([
            html.Label("Stop Price", className="text-muted small fw-semibold"),
            dbc.Input(
                id="stop-price",
                type="number",
                value=1980,
                step="any",
                inputMode="decimal",
                className="position-input",
            ),
        ], xs=12, sm=6, md=4, className="mb-3 mb-md-0"),
        dbc.Col([
            html.Label("Risk (USD)", className="text-muted small fw-semibold"),
            dbc.InputGroup([
                dbc.InputGroupText("$"),
                dbc.Input(
                    id="risk-usd",
                    type="number",
                    value=100,
                    min=1,
                    step="any",
                    inputMode="decimal",
                    className="position-input",
                ),
            ]),
        ], xs=12, md=4),
    ], className="mb-2 g-2"),

    dbc.Row([dbc.Col(
        html.Small(
            "Stop = price where you exit if wrong.",
            className="text-muted d-block mb-3",
        ),
        xs=12,
    )]),

    html.Hr(className="my-3"),

    html.Div(id="results"),
    html.Div(id="details", className="mt-2 mt-md-3"),
], fluid=True, className="py-2 py-md-3 mobile-container")


@callback(
    Output("results", "children"),
    Output("details", "children"),
    Input("commodity", "value"),
    Input("direction", "value"),
    Input("entry-price", "value"),
    Input("stop-price", "value"),
    Input("risk-usd", "value"),
)
def update_position(commodity, direction, entry_price, stop_price, risk_usd):
    if not commodity:
        return dbc.Alert("Select a commodity.", color="warning"), html.Div()

    result = compute_position(
        commodity, direction, entry_price, stop_price, risk_usd
    )
    return _result_cards(result), _detail_table(result)


if __name__ == "__main__":
    import socket

    port = 8056
    host = "0.0.0.0"  # reachable from phone on same Wi‑Fi

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except OSError:
        local_ip = "YOUR_COMPUTER_IP"

    print(f"On this computer:  http://127.0.0.1:{port}")
    print(f"On your phone:     http://{local_ip}:{port}")
    print("(Phone must be on the same Wi‑Fi as this computer.)")
    app.run(debug=True, host=host, port=port)
