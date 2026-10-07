"""
Commitment of Traders Lab — home page.

Weekly Commitment of Traders positioning vs daily futures prices.
Indices (OI / Commercial / Retail / Traders) are recomputed over a selectable
rolling window so "last 52 weeks" means exactly that, with no look-ahead.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import (
    ALL, Input, Output, State, callback, ctx, dash_table, dcc, html,
    no_update, register_page,
)
import dash_bootstrap_components as dbc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest_engine.data import load_cot_data  # noqa: E402

register_page(__name__, path="/", title="Commitment of Traders Lab", order=0)

# =============================================================================
# Rules and visual constants
# =============================================================================

RULES = {
    "long":  {"oi_max": 30, "comm_min": 80, "retail_max": 50},
    "short": {"oi_min": 70, "comm_max": 20, "retail_min": 50},
}

WINDOW_OPTIONS = [
    {"label": "26w", "value": 26},
    {"label": "52w", "value": 52},
    {"label": "156w", "value": 156},
    {"label": "All", "value": 0},
]
DEFAULT_WINDOW = 52

C = {
    "long": "#2ecc71",
    "short": "#e74c3c",
    "watch": "#95a5a6",
    "comm": "#1abc9c",
    "retail": "#f39c12",
    "oi": "#3498db",
    "traders": "#bdc3c7",
    "price": "#ecf0f1",
    "ma50": "#9b59b6",
    "ma200": "#e67e22",
    "rsi": "#af7ac5",
    "paper": "rgba(0,0,0,0)",
    "plot": "#14181d",
    "grid": "#2a3038",
    "text": "#e8edf2",
}

GROUP_COLORS = {
    "Currencies": "#5dade2",
    "Metals": "#f5b041",
    "Grains": "#58d68d",
    "Softs": "#eb984e",
    "Energies": "#ec7063",
    "Livestock": "#af7ac5",
    "Indices": "#48c9b0",
    "Financials": "#aab7b8",
    "Other": "#85929e",
}

BIAS_LABEL = {"Long": "LONG", "Short": "SHORT", "Watch L": "WATCH · long", "Watch S": "WATCH · short", "": "—"}
BIAS_COLOR = {"Long": C["long"], "Short": C["short"], "Watch L": C["watch"], "Watch S": C["watch"], "": C["watch"]}

# =============================================================================
# Data
# =============================================================================

_RAW = load_cot_data()
if _RAW.empty:
    WEEKLY = pd.DataFrame()
    DAILY = pd.DataFrame()
else:
    _RAW["group"] = _RAW.get("group", pd.Series(index=_RAW.index, dtype=object)).fillna("Other")
    WEEKLY = _RAW[_RAW["data_type"] == "weekly_cot"].sort_values(["Market", "Date"]).reset_index(drop=True)
    DAILY = _RAW[_RAW["data_type"] == "daily_price"].sort_values(["Market", "Date"]).reset_index(drop=True)

MARKETS = sorted(WEEKLY["Market"].unique().tolist()) if not WEEKLY.empty else []
MARKET_GROUP = (WEEKLY.groupby("Market")["group"].first().to_dict() if not WEEKLY.empty else {})
WEEK_DATES = (sorted(WEEKLY["Date"].dt.normalize().unique().tolist()) if not WEEKLY.empty else [])
LATEST_WEEK = WEEK_DATES[-1] if WEEK_DATES else None
LATEST_PRICE_DATE = DAILY["Date"].max() if not DAILY.empty else None


def short_name(market: str) -> str:
    return market.split(" - ")[0].strip()


_INDEX_CACHE: dict[int, pd.DataFrame] = {}


def indexed_weekly(window: int) -> pd.DataFrame:
    """Weekly frame with OI_idx / Comm_idx / Retail_idx / Traders_idx for the window.

    window = 0 reuses the Grabber's full-sample columns ("All").
    Otherwise a rolling min/max over ``window`` weeks per market.
    """
    if window in _INDEX_CACHE:
        return _INDEX_CACHE[window]
    if WEEKLY.empty:
        return WEEKLY

    w = WEEKLY.copy()
    pairs = [
        ("OI", "OI_idx", "OI_Index"),
        ("Net Commercial Position", "Comm_idx", "Commercial_Index"),
        ("Net Retail Position", "Retail_idx", "Retail_Index"),
        ("Net Traders Position", "Traders_idx", "Traders_Index"),
    ]
    if window == 0:
        for _, dst, grabber_col in pairs:
            w[dst] = w[grabber_col]
    else:
        min_periods = max(13, window // 2)
        for src, dst, _ in pairs:
            g = w.groupby("Market")[src]
            lo = g.transform(lambda s: s.rolling(window, min_periods=min_periods).min())
            hi = g.transform(lambda s: s.rolling(window, min_periods=min_periods).max())
            rng = (hi - lo).replace(0, np.nan)
            w[dst] = ((w[src] - lo) / rng * 100).clip(0, 100)

    w["Comm_chg"] = w.groupby("Market")["Comm_idx"].diff()

    oi, comm, ret = w["OI_idx"], w["Comm_idx"], w["Retail_idx"]
    l_hits = (oi <= RULES["long"]["oi_max"]).astype(int) + (comm > RULES["long"]["comm_min"]).astype(int) \
        + (ret < RULES["long"]["retail_max"]).astype(int)
    s_hits = (oi >= RULES["short"]["oi_min"]).astype(int) + (comm < RULES["short"]["comm_max"]).astype(int) \
        + (ret > RULES["short"]["retail_min"]).astype(int)
    valid = oi.notna() & comm.notna() & ret.notna()
    w["Bias"] = np.select(
        [valid & (l_hits == 3), valid & (s_hits == 3), valid & (l_hits == 2), valid & (s_hits == 2)],
        ["Long", "Short", "Watch L", "Watch S"],
        default="",
    )
    w["Strength"] = ((comm - 50).abs() + (oi - 50).abs()).round(1)

    _INDEX_CACHE[window] = w
    return w


def latest_snapshot(window: int) -> pd.DataFrame:
    """One row per market for the most recent week, joined with latest RSI/Close."""
    w = indexed_weekly(window)
    if w.empty:
        return w
    last = w.groupby("Market").tail(1).copy()
    if LATEST_WEEK is not None:
        last = last[last["Date"] >= LATEST_WEEK - pd.Timedelta(days=10)]
    if not DAILY.empty:
        px_last = DAILY.groupby("Market").tail(1)[["Market", "RSI", "Close"]].rename(
            columns={"RSI": "RSI_last", "Close": "Close_last"})
        last = last.merge(px_last, on="Market", how="left")
    last["Short"] = last["Market"].map(short_name)
    return last.reset_index(drop=True)


# =============================================================================
# Layout helpers
# =============================================================================

def _fmt(v, nd=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def _base_layout(fig: go.Figure, height: int, uirev: str):
    fig.update_layout(
        autosize=False,
        height=height,
        template="plotly_dark",
        paper_bgcolor=C["paper"],
        plot_bgcolor=C["plot"],
        font=dict(color=C["text"], size=12),
        margin=dict(l=56, r=24, t=48, b=40),
        uirevision=uirev,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    return fig


def setup_card(row: pd.Series) -> html.Div:
    bias = row["Bias"]
    color = BIAS_COLOR.get(bias, C["watch"])
    chg = row.get("Comm_chg", np.nan)
    chg_txt = "" if pd.isna(chg) else f"Δ Comm {chg:+.0f}"
    card = dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(row["Short"], className="fw-semibold", style={"fontSize": 14}),
                html.Span(BIAS_LABEL.get(bias, bias), className="ms-2 badge",
                          style={"backgroundColor": color, "color": "#111", "fontSize": 10}),
            ], className="d-flex justify-content-between align-items-center mb-2"),
            html.Div([
                html.Span([html.Small("OI ", className="text-muted"), _fmt(row["OI_idx"])],
                          style={"color": C["oi"]}),
                html.Span([html.Small("Comm ", className="text-muted"), _fmt(row["Comm_idx"])],
                          style={"color": C["comm"]}),
                html.Span([html.Small("Retail ", className="text-muted"), _fmt(row["Retail_idx"])],
                          style={"color": C["retail"]}),
            ], className="d-flex justify-content-between", style={"fontSize": 13}),
            html.Div([
                html.Small(f"RSI {_fmt(row.get('RSI_last'))}", className="text-muted"),
                html.Small(chg_txt, className="text-muted"),
            ], className="d-flex justify-content-between mt-1"),
        ], className="py-2 px-3"),
        className="cot-setup-card",
        style={"borderLeft": f"4px solid {color}"},
    )
    return html.Div(card, id={"type": "setup-card", "market": row["Market"]}, n_clicks=0,
                    className="mb-2", style={"cursor": "pointer"})


def setups_column(title: str, color: str, rows: pd.DataFrame, empty_text: str):
    cards = [setup_card(r) for _, r in rows.iterrows()] if not rows.empty else [
        html.Div(empty_text, className="text-muted small fst-italic py-3")
    ]
    return dbc.Col([
        html.Div([
            html.Span(title, className="text-uppercase small fw-semibold", style={"color": color, "letterSpacing": 1}),
            html.Span(f"{len(rows)}", className="badge bg-secondary ms-2") if not rows.empty else None,
        ], className="mb-2"),
        *cards,
    ], md=4)


def kpi(label: str, value: str, color: str | None = None, sub: str | None = None):
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div(label, className="cot-kpi-label"),
            html.Div(value, className="cot-kpi-value", style={"color": color or C["text"]}),
            html.Div(sub or "", className="cot-kpi-sub"),
        ]), className="cot-kpi-card"),
        xs=6, md=2,
    )


def rules_popover():
    L, S = RULES["long"], RULES["short"]
    return html.Span([
        dbc.Button("Rules", id="rules-btn", size="sm", outline=True,
                   color="secondary", className="cot-rules-btn ms-1"),
        dbc.Popover([
            dbc.PopoverHeader("Setup rules"),
            dbc.PopoverBody([
                html.Div([html.B("Long", style={"color": C["long"]}),
                          f": OI ≤ {L['oi_max']}, Commercial > {L['comm_min']}, Retail < {L['retail_max']}"]),
                html.Div([html.B("Short", style={"color": C["short"]}),
                          f": OI ≥ {S['oi_min']}, Commercial < {S['comm_max']}, Retail > {S['retail_min']}"],
                         className="mt-1"),
                html.Div("Watch: two of the three conditions in one direction.", className="mt-1 text-muted small"),
                html.Hr(className="my-2"),
                html.Div("Indices are (value − min) / (max − min) × 100 over the selected window of weekly "
                         "reports, per market. 'All' uses the full history as produced by the Grabber.",
                         className="small text-muted"),
            ]),
        ], target="rules-btn", trigger="click", placement="bottom"),
    ])


# =============================================================================
# Layout
# =============================================================================

market_options = [
    {"label": f"{short_name(m)}  ·  {MARKET_GROUP.get(m, 'Other')}", "value": m} for m in MARKETS
]

as_of_parts = []
if LATEST_WEEK is not None:
    as_of_parts.append(f"COT week ending {LATEST_WEEK:%a %d %b %Y}")
if LATEST_PRICE_DATE is not None:
    as_of_parts.append(f"prices to {LATEST_PRICE_DATE:%d %b %Y}")
as_of_text = "  ·  ".join(as_of_parts)

control_bar = html.Div(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div("Dashboard", className="fw-semibold", style={"fontSize": 15}),
                html.Div("Weekly positioning vs daily futures prices", className="cot-panel-sub"),
            ], lg=3, md=12, className="mb-2 mb-lg-0"),
            dbc.Col(
                dcc.Dropdown(
                    id="commodity-dropdown",
                    options=market_options,
                    value=MARKETS[0] if MARKETS else None,
                    clearable=False,
                    placeholder="Search market…",
                    style={"color": "#111"},
                ),
                lg=4, md=12, className="mb-2 mb-lg-0",
            ),
            dbc.Col([
                html.Div([
                    html.Span("Index window", className="cot-panel-sub me-2"),
                    dbc.RadioItems(
                        id="index-window",
                        options=WINDOW_OPTIONS,
                        value=DEFAULT_WINDOW,
                        inline=True,
                        className="btn-group btn-group-sm cot-pills",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-light",
                        labelCheckedClassName="active",
                    ),
                    rules_popover(),
                ], className="d-flex flex-wrap align-items-center gap-1"),
            ], lg=5, md=12),
        ], className="align-items-center g-2"),
        dbc.Row([
            dbc.Col(
                html.Div(as_of_text, id="as-of-text", className="cot-asof-chip mt-2"),
                width="auto", className="ms-lg-auto",
            ),
        ], className="g-0"),
    ], fluid=True, className="py-2"),
    className="cot-control-bar",
)

layout = html.Div([
    control_bar,
    dbc.Container([
        # --- Setups strip -------------------------------------------------
        html.Div([
            html.Div([
                html.H5("Setups this week", className="cot-panel-title"),
                html.Span(id="setups-subtitle", className="cot-panel-sub ms-3"),
            ], className="d-flex align-items-baseline mb-3"),
            dbc.Row(id="setups-strip", className="g-3"),
        ], className="cot-panel mt-3"),

        # --- Selected market ---------------------------------------------
        html.Div([
            html.Div([
                html.H5(id="market-title", className="cot-panel-title"),
                html.Span(id="market-group", className="badge bg-secondary ms-2"),
            ], className="d-flex align-items-center mb-3"),
            dbc.Row(id="kpi-strip", className="g-2 mb-3"),
            html.Div(
                dcc.Graph(id="combined-graph", style={"height": "700px"},
                          config={"responsive": False, "displayModeBar": True, "displaylogo": False}),
                className="cot-chart-shell",
                style={"height": "710px", "overflow": "hidden"},
            ),
        ], className="cot-panel"),

        # --- Positioning map + OI/Commercial ------------------------------
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5("Positioning map", className="cot-panel-title"),
                        html.Span("Retail vs Commercial · bubble size = OI index",
                                  className="cot-panel-sub ms-3"),
                    ], className="d-flex align-items-baseline mb-2"),
                    html.Div([
                        html.Span("As of", className="cot-panel-sub me-2"),
                        dcc.DatePickerSingle(
                            id="my-date-picker-single",
                            min_date_allowed=WEEK_DATES[0] if WEEK_DATES else None,
                            max_date_allowed=WEEK_DATES[-1] if WEEK_DATES else None,
                            initial_visible_month=WEEK_DATES[-1] if WEEK_DATES else None,
                            date=WEEK_DATES[-1] if WEEK_DATES else None,
                            display_format="DD MMM YYYY",
                            style={"fontSize": 12},
                        ),
                    ], className="d-flex align-items-center mb-2"),
                    html.Div(
                        dcc.Graph(id="bubble-graph", style={"height": "640px"},
                                  config={"responsive": False, "displayModeBar": True, "displaylogo": False}),
                        className="cot-chart-shell",
                        style={"height": "650px", "overflow": "hidden"},
                    ),
                ], className="cot-panel h-100"),
            ], lg=7),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5("OI and Commercial index", className="cot-panel-title"),
                        html.Span("selected market · same window", className="cot-panel-sub ms-3"),
                    ], className="d-flex align-items-baseline mb-2"),
                    html.Div(style={"height": 32}),
                    html.Div(
                        dcc.Graph(id="open-interest-graph", style={"height": "640px"},
                                  config={"responsive": False, "displayModeBar": False}),
                        className="cot-chart-shell",
                        style={"height": "650px", "overflow": "hidden"},
                    ),
                ], className="cot-panel h-100"),
            ], lg=5),
        ], className="g-3"),

        # --- All markets table (collapsed) --------------------------------
        html.Div([
            dbc.Accordion([
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col(dbc.RadioItems(
                            id="table-bias-filter",
                            options=[{"label": "All", "value": "all"},
                                     {"label": "Long", "value": "Long"},
                                     {"label": "Short", "value": "Short"},
                                     {"label": "Watch", "value": "watch"}],
                            value="all", inline=True,
                            className="btn-group btn-group-sm cot-pills",
                            inputClassName="btn-check",
                            labelClassName="btn btn-outline-light",
                            labelCheckedClassName="active",
                        ), md="auto"),
                        dbc.Col(dcc.Dropdown(
                            id="table-group-filter",
                            options=[{"label": g, "value": g} for g in sorted(set(MARKET_GROUP.values()))],
                            placeholder="All groups", clearable=True, style={"color": "#111", "minWidth": 180},
                        ), md=3),
                        dbc.Col(html.Small("Click a row to select the market. Sort or filter any column.",
                                           className="cot-panel-sub"), className="d-flex align-items-center"),
                    ], className="g-2 mb-3 align-items-center"),
                    dash_table.DataTable(
                        id="all-table",
                        data=[],
                        columns=[
                            {"name": "Market", "id": "Short"},
                            {"name": "Group", "id": "group"},
                            {"name": "Bias", "id": "BiasLabel"},
                            {"name": "OI", "id": "OI_idx", "type": "numeric", "format": {"specifier": ".0f"}},
                            {"name": "Commercial", "id": "Comm_idx", "type": "numeric", "format": {"specifier": ".0f"}},
                            {"name": "Retail", "id": "Retail_idx", "type": "numeric", "format": {"specifier": ".0f"}},
                            {"name": "Traders", "id": "Traders_idx", "type": "numeric", "format": {"specifier": ".0f"}},
                            {"name": "RSI", "id": "RSI_last", "type": "numeric", "format": {"specifier": ".0f"}},
                            {"name": "Δ Comm", "id": "Comm_chg", "type": "numeric", "format": {"specifier": "+.0f"}},
                            {"name": "Strength", "id": "Strength", "type": "numeric", "format": {"specifier": ".0f"}},
                        ],
                        sort_action="native",
                        filter_action="native",
                        page_action="none",
                        style_as_list_view=True,
                        style_table={"overflowX": "auto"},
                        style_cell={"backgroundColor": "#14181d", "color": C["text"], "border": "0",
                                    "fontSize": 13, "padding": "8px 10px", "textAlign": "right"},
                        style_cell_conditional=[
                            {"if": {"column_id": c}, "textAlign": "left"} for c in ["Short", "group", "BiasLabel"]
                        ],
                        style_header={"backgroundColor": "#1a1f26", "color": "#8b95a1", "fontWeight": 600,
                                      "border": "0", "textTransform": "uppercase", "fontSize": 11},
                        style_filter={"backgroundColor": "#1a1f26", "color": C["text"]},
                        style_data_conditional=[
                            {"if": {"filter_query": '{BiasLabel} = "LONG"', "column_id": "BiasLabel"},
                             "color": C["long"], "fontWeight": 600},
                            {"if": {"filter_query": '{BiasLabel} = "SHORT"', "column_id": "BiasLabel"},
                             "color": C["short"], "fontWeight": 600},
                            {"if": {"filter_query": '{BiasLabel} contains "WATCH"', "column_id": "BiasLabel"},
                             "color": C["watch"]},
                            {"if": {"state": "active"}, "backgroundColor": "#2a3038", "border": "0"},
                        ],
                    ),
                ], title="All markets", item_id="table"),
            ], start_collapsed=True, flush=True, className="cot-accordion mt-2 mb-5"),
        ]),
    ], fluid=True, className="pb-4"),
], style={"backgroundColor": "#0b0d10", "minHeight": "100vh"})


# =============================================================================
# Callbacks
# =============================================================================

@callback(
    Output("commodity-dropdown", "value"),
    Input({"type": "setup-card", "market": ALL}, "n_clicks"),
    Input("bubble-graph", "clickData"),
    Input("all-table", "active_cell"),
    State("all-table", "derived_viewport_data"),
    prevent_initial_call=True,
)
def select_market(card_clicks, bubble_click, active_cell, viewport):
    trig = ctx.triggered_id
    if trig is None:
        return no_update
    if isinstance(trig, dict) and trig.get("type") == "setup-card":
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return no_update
        return trig["market"]
    if trig == "bubble-graph":
        try:
            return bubble_click["points"][0]["customdata"][0]
        except (KeyError, IndexError, TypeError):
            return no_update
    if trig == "all-table":
        if not active_cell or not viewport:
            return no_update
        row = active_cell.get("row")
        if row is None or row >= len(viewport):
            return no_update
        return viewport[row].get("Market", no_update)
    return no_update


@callback(
    Output("setups-strip", "children"),
    Output("setups-subtitle", "children"),
    Input("index-window", "value"),
)
def render_setups(window):
    snap = latest_snapshot(window)
    if snap.empty:
        return [dbc.Col(html.Div("No data loaded.", className="text-muted"))], ""

    snap = snap.sort_values("Strength", ascending=False)
    longs = snap[snap["Bias"] == "Long"]
    shorts = snap[snap["Bias"] == "Short"]
    watch = snap[snap["Bias"].str.startswith("Watch")]

    n_missing = int(snap[["OI_idx", "Comm_idx", "Retail_idx"]].isna().any(axis=1).sum())
    win_txt = "full history" if window == 0 else f"{window}-week window"
    subtitle = f"{len(longs)} long · {len(shorts)} short · {len(watch)} watch  —  {win_txt}"
    if n_missing:
        subtitle += f"  ·  {n_missing} market(s) lack enough history for this window"

    cols = [
        setups_column("Long setups", C["long"], longs, "No market meets all three long conditions."),
        setups_column("Short setups", C["short"], shorts, "No market meets all three short conditions."),
        setups_column("Watch · two of three", C["watch"], watch.head(8), "Nothing close this week."),
    ]
    return cols, subtitle


@callback(
    Output("market-title", "children"),
    Output("market-group", "children"),
    Output("kpi-strip", "children"),
    Input("commodity-dropdown", "value"),
    Input("index-window", "value"),
)
def render_kpis(market, window):
    if not market:
        return "", "", []
    snap = latest_snapshot(window)
    row = snap[snap["Market"] == market]
    title = short_name(market)
    group = MARKET_GROUP.get(market, "Other")
    if row.empty:
        return title, group, [dbc.Col(html.Div("No weekly data for this market.", className="text-muted"))]
    r = row.iloc[0]
    bias = r["Bias"]
    chg = r.get("Comm_chg", np.nan)
    chips = [
        kpi("Bias", BIAS_LABEL.get(bias, "—"), BIAS_COLOR.get(bias), f"strength {_fmt(r['Strength'])}"),
        kpi("OI index", _fmt(r["OI_idx"]), C["oi"], f"rule: ≤{RULES['long']['oi_max']} long · ≥{RULES['short']['oi_min']} short"),
        kpi("Commercial", _fmt(r["Comm_idx"]), C["comm"],
            ("" if pd.isna(chg) else f"{chg:+.0f} vs last week")),
        kpi("Retail", _fmt(r["Retail_idx"]), C["retail"], "< 50 long · > 50 short"),
        kpi("Traders (specs)", _fmt(r["Traders_idx"]), C["traders"], "large speculators"),
        kpi("RSI (10)", _fmt(r.get("RSI_last")), C["rsi"],
            f"close {r['Close_last']:,.4g}" if not pd.isna(r.get("Close_last", np.nan)) else ""),
    ]
    return title, group, chips


def _xrange_for_window(window: int, end: pd.Timestamp):
    if window == 0:
        return None
    months = {26: 6, 52: 12, 156: 36}.get(window, 12)
    return [end - pd.DateOffset(months=months), end + pd.Timedelta(days=3)]


@callback(
    Output("combined-graph", "figure"),
    Input("commodity-dropdown", "value"),
    Input("index-window", "value"),
)
def render_main_chart(market, window):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.55, 0.27, 0.18],
        subplot_titles=("Price", "Commercial and Retail index", "RSI (10)"),
    )
    _base_layout(fig, 700, f"main-{market}")
    if not market or DAILY.empty:
        return fig

    d = DAILY[DAILY["Market"] == market]
    w = indexed_weekly(window)
    w = w[w["Market"] == market]

    if not d.empty:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], name="Close", line=dict(color=C["price"], width=1.6)),
                      row=1, col=1)
        if len(d) >= 50:
            fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"].rolling(50).mean(), name="50d MA",
                                     line=dict(color=C["ma50"], width=1)), row=1, col=1)
        if len(d) >= 200:
            fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"].rolling(200).mean(), name="200d MA",
                                     line=dict(color=C["ma200"], width=1)), row=1, col=1)
        if d["RSI"].notna().any():
            fig.add_trace(go.Scatter(x=d["Date"], y=d["RSI"], name="RSI", line=dict(color=C["rsi"], width=1.2),
                                     showlegend=False), row=3, col=1)

    if not w.empty:
        fig.add_trace(go.Scatter(x=w["Date"], y=w["Comm_idx"], name="Commercial", line=dict(color=C["comm"], width=1.8),
                                 mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=w["Date"], y=w["Retail_idx"], name="Retail", line=dict(color=C["retail"], width=1.8),
                                 mode="lines"), row=2, col=1)

    # Threshold guides: index pane 20/80, RSI pane 30/70
    guide = dict(color="#6c757d", width=1, dash="dash")
    fig.add_hline(y=RULES["long"]["comm_min"], line=guide, row=2, col=1)
    fig.add_hline(y=RULES["short"]["comm_max"], line=guide, row=2, col=1)
    fig.add_hline(y=70, line=guide, row=3, col=1)
    fig.add_hline(y=30, line=guide, row=3, col=1)

    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[0, 20, 50, 80, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=3, col=1)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[dict(count=3, label="3m", step="month", stepmode="backward"),
                     dict(count=6, label="6m", step="month", stepmode="backward"),
                     dict(count=1, label="1y", step="year", stepmode="backward"),
                     dict(step="all", label="All")],
            bgcolor="#1f2327", activecolor="#495057", font=dict(size=11, color=C["text"]),
            x=1, xanchor="right", y=1.08, yanchor="bottom",
        ),
        row=1, col=1,
    )
    end = d["Date"].max() if not d.empty else (w["Date"].max() if not w.empty else None)
    if end is not None:
        rng = _xrange_for_window(window, end)
        if rng:
            fig.update_xaxes(range=rng)
    fig.update_layout(hovermode="x unified")
    for ann in fig.layout.annotations:
        ann.update(x=0, xanchor="left", font=dict(size=12, color="#aab7b8"))
    return fig


@callback(
    Output("bubble-graph", "figure"),
    Input("my-date-picker-single", "date"),
    Input("index-window", "value"),
    Input("commodity-dropdown", "value"),
)
def render_bubble(as_of, window, selected):
    fig = go.Figure()
    _base_layout(fig, 640, f"bubble-{window}")
    w = indexed_weekly(window)
    if w.empty or not as_of:
        return fig

    as_of = pd.Timestamp(as_of).normalize()
    dates = [dt for dt in WEEK_DATES if dt <= as_of]
    if not dates:
        return fig
    cur_date = dates[-1]
    frame_dates = dates[-12:]

    # Fixed market order per group across every frame so Plotly can tween positions
    span = w[(w["Date"] >= frame_dates[0] - pd.Timedelta(days=6)) & (w["Date"] <= cur_date)]
    span = span.dropna(subset=["OI_idx", "Comm_idx", "Retail_idx"])
    group_markets = {g: sorted(span.loc[span["group"] == g, "Market"].unique()) for g in GROUP_COLORS}

    def week_rows(dt):
        """One row per market for the week ending ``dt``, reindexed to the fixed market list."""
        rows = w[(w["Date"] >= dt - pd.Timedelta(days=6)) & (w["Date"] <= dt)]
        rows = rows.groupby("Market").tail(1).set_index("Market")
        rows = rows.reindex(span["Market"].unique())
        rows["Bias"] = rows["Bias"].fillna("")
        rows["group"] = rows.index.map(MARKET_GROUP).fillna("Other")
        return rows.reset_index()

    def size_of(s):
        return (9 + 0.30 * s.fillna(0)).tolist()

    def group_traces(rows, showlegend):
        traces = []
        for g in GROUP_COLORS:
            sub = rows.set_index("Market").reindex(group_markets[g]).reset_index()
            sub["Bias"] = sub["Bias"].fillna("")
            border = [BIAS_COLOR[b] if b in ("Long", "Short") else "rgba(0,0,0,0.45)" for b in sub["Bias"]]
            width = [3 if b in ("Long", "Short") else 1 for b in sub["Bias"]]
            traces.append(go.Scatter(
                x=sub["Retail_idx"], y=sub["Comm_idx"], mode="markers", name=g,
                legendgroup=g, showlegend=showlegend,
                customdata=np.stack([sub["Market"], sub["Market"].map(short_name),
                                     sub["OI_idx"].fillna(0), sub["Bias"].replace("", "—")], axis=1) if len(sub) else None,
                marker=dict(size=size_of(sub["OI_idx"]), color=GROUP_COLORS[g], opacity=0.8,
                            line=dict(width=width, color=border)),
                hovertemplate="<b>%{customdata[1]}</b><br>Commercial %{y:.0f} · Retail %{x:.0f} · OI %{customdata[2]:.0f}"
                              "<br>%{customdata[3]}<extra></extra>",
            ))
        return traces

    def highlight_trace(rows):
        sel = rows[rows["Market"] == selected] if selected else rows.iloc[0:0]
        return go.Scatter(
            x=sel["Retail_idx"], y=sel["Comm_idx"], mode="markers+text",
            text=sel["Market"].map(short_name), textposition="top center",
            textfont=dict(size=11, color="#ffffff"),
            marker=dict(size=[s + 10 for s in size_of(sel["OI_idx"])], symbol="circle-open",
                        line=dict(width=2, color="#ffffff")),
            hoverinfo="skip", showlegend=False, name="selected",
        )

    cur = week_rows(cur_date)
    for tr in group_traces(cur, True) + [highlight_trace(cur)]:
        fig.add_trace(tr)
    n_traces = len(GROUP_COLORS) + 1

    # Rule zones: subtle fill, small corner labels
    L, S = RULES["long"], RULES["short"]
    fig.add_shape(type="rect", x0=0, x1=L["retail_max"], y0=L["comm_min"], y1=100, line_width=0,
                  fillcolor=C["long"], opacity=0.07, layer="below")
    fig.add_shape(type="rect", x0=S["retail_min"], x1=100, y0=0, y1=S["comm_max"], line_width=0,
                  fillcolor=C["short"], opacity=0.07, layer="below")
    fig.add_annotation(x=1, y=99, text="Long zone", showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=10, color=C["long"]), opacity=0.9)
    fig.add_annotation(x=99, y=1, text="Short zone", showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=10, color=C["short"]), opacity=0.9)
    fig.add_vline(x=50, line=dict(color=C["grid"], width=1))
    fig.add_hline(y=50, line=dict(color=C["grid"], width=1))

    # Animation: one frame per week, Play/Pause + slider
    frames, steps = [], []
    for dt in frame_dates:
        rows = week_rows(dt)
        name = f"{dt:%Y-%m-%d}"
        frames.append(go.Frame(data=group_traces(rows, False) + [highlight_trace(rows)],
                               traces=list(range(n_traces)), name=name))
        steps.append(dict(method="animate", label=f"{dt:%d %b}",
                          args=[[name], {"frame": {"duration": 400, "redraw": False}, "mode": "immediate",
                                         "transition": {"duration": 400, "easing": "cubic-in-out"}}]))
    fig.frames = frames

    fig.update_layout(
        xaxis=dict(title="Retail index  →  more bullish", range=[0, 100], autorange=False,
                   tickvals=[0, 20, 50, 80, 100]),
        yaxis=dict(title="Commercial index  →  more bullish", range=[0, 100], autorange=False,
                   tickvals=[0, 20, 50, 80, 100]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left", title_text="",
                    itemsizing="constant", font=dict(size=11),
                    itemclick="toggleothers", itemdoubleclick="toggle"),
        hoverlabel=dict(bgcolor="#1f2327", bordercolor="#495057", font=dict(color=C["text"], size=12)),
        margin=dict(l=56, r=24, t=40, b=120),
        updatemenus=[dict(
            type="buttons", showactive=False, direction="left",
            x=0, xanchor="left", y=-0.14, yanchor="top", pad=dict(r=10, t=0),
            bgcolor="#1f2327", bordercolor="#495057", font=dict(size=12, color=C["text"]),
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, {"frame": {"duration": 1100, "redraw": False}, "fromcurrent": True,
                                  "mode": "immediate", "transition": {"duration": 900, "easing": "cubic-in-out"}}]),
                dict(label="❚❚  Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            active=len(steps) - 1, steps=steps,
            x=0.22, len=0.78, y=-0.14, yanchor="top", pad=dict(t=6, b=0),
            currentvalue=dict(prefix="Week ending ", visible=True, xanchor="left",
                              font=dict(size=12, color="#aab7b8")),
            bgcolor="#1f2327", bordercolor="#495057", activebgcolor="#aab7b8",
            tickcolor="#495057", font=dict(size=10, color="#aab7b8"),
        )],
    )
    return fig


@callback(
    Output("open-interest-graph", "figure"),
    Input("commodity-dropdown", "value"),
    Input("index-window", "value"),
)
def render_oi_chart(market, window):
    L, S = RULES["long"], RULES["short"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, row_heights=[0.5, 0.5],
        subplot_titles=(f"Commercial index   ·   long > {L['comm_min']}  /  short < {S['comm_max']}",
                        f"OI index   ·   long ≤ {L['oi_max']}  /  short ≥ {S['oi_min']}"),
    )
    _base_layout(fig, 640, f"oi-{market}")
    for ann in fig.layout.annotations:
        ann.update(x=0, xanchor="left", font=dict(size=12, color="#aab7b8"))
    fig.update_layout(showlegend=False, hovermode="x unified", margin=dict(l=48, r=24, t=40, b=40))
    if not market:
        return fig
    w = indexed_weekly(window)
    w = w[w["Market"] == market]
    if w.empty:
        return fig

    fig.add_trace(go.Scatter(x=w["Date"], y=w["Comm_idx"], name="Commercial",
                             line=dict(color=C["comm"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w["Date"], y=w["OI_idx"], name="OI",
                             line=dict(color=C["oi"], width=2)), row=2, col=1)

    guide = dict(color="#6c757d", width=1, dash="dash")
    fig.add_hline(y=L["comm_min"], line=guide, row=1, col=1)
    fig.add_hline(y=S["comm_max"], line=guide, row=1, col=1)
    fig.add_hline(y=S["oi_min"], line=guide, row=2, col=1)
    fig.add_hline(y=L["oi_max"], line=guide, row=2, col=1)

    # Weeks where the full rule fired: faint vertical band across both panes
    for bias, color in (("Long", C["long"]), ("Short", C["short"])):
        for dt in w.loc[w["Bias"] == bias, "Date"]:
            fig.add_vrect(x0=dt - pd.Timedelta(days=3), x1=dt + pd.Timedelta(days=3),
                          fillcolor=color, opacity=0.18, line_width=0, layer="below")

    fig.update_yaxes(range=[0, 100], tickvals=[0, S["comm_max"], 50, L["comm_min"], 100], row=1, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[0, L["oi_max"], 50, S["oi_min"], 100], row=2, col=1)
    rng = _xrange_for_window(window, w["Date"].max())
    if rng:
        fig.update_xaxes(range=rng)
    return fig


@callback(
    Output("all-table", "data"),
    Input("index-window", "value"),
    Input("table-bias-filter", "value"),
    Input("table-group-filter", "value"),
)
def render_table(window, bias_filter, group_filter):
    snap = latest_snapshot(window)
    if snap.empty:
        return []
    if bias_filter == "watch":
        snap = snap[snap["Bias"].str.startswith("Watch")]
    elif bias_filter in ("Long", "Short"):
        snap = snap[snap["Bias"] == bias_filter]
    if group_filter:
        snap = snap[snap["group"] == group_filter]
    snap = snap.copy()
    snap["_order"] = snap["Bias"].map({"Long": 0, "Short": 1, "Watch L": 2, "Watch S": 3}).fillna(4)
    snap = snap.sort_values(["_order", "Strength"], ascending=[True, False])
    snap["BiasLabel"] = snap["Bias"].map(BIAS_LABEL)
    cols = ["Market", "Short", "group", "Bias", "BiasLabel", "OI_idx", "Comm_idx", "Retail_idx",
            "Traders_idx", "RSI_last", "Comm_chg", "Strength"]
    cols = [c for c in cols if c in snap.columns]
    out = snap[cols].round(1)
    return out.to_dict("records")
