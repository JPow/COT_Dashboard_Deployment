"""
Trend-Following Breakout Dashboard
====================================
N-day high/low breakout strategy with realistic transaction costs.

Strategy:
  Long:  High breaks above N-day highest high → enter at breakout level
  Short: Low breaks below N-day lowest low    → enter at breakout level
  Stop:  2×ATR initial → breakeven at 1:1 R/R → trail 2×ATR

Costs:
  - Commission: $10 per completed trade
  - Slippage: 2 ticks adverse on entry (market-specific tick sizes)
"""

import sys
import os
import pandas as pd
import numpy as np
import dash
from dash import Dash, html, dcc, dash_table, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest_engine.data import (load_cot_data, prepare_base_data,
                                  load_contract_specs, get_contract_spec)
from backtest_engine.indicators import add_standard_indicators
from backtest_engine.backtester import calculate_position_size
from backtest_engine.metrics import calculate_performance_metrics

# ============================================================================
# Constants
# ============================================================================

DEFAULT_LOOKBACK = 20
DEFAULT_START = "2022-01-01"
DEFAULT_END = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 30_000
RISK_PCT = 1.0
COMMISSION_PER_TRADE = 10.0
SLIPPAGE_TICKS = 2
ATR_STOP_MULT = 2.0
ATR_TRAIL_MULT = 2.0
STOP_BUFFER = 0.01

DEFAULT_TICK_SIZE = 0.01

SUMMARY_COLUMNS = [
    {"name": "Market",        "id": "Market"},
    {"name": "Trades",        "id": "Trades"},
    {"name": "Missed",        "id": "Missed"},
    {"name": "Win Rate %",    "id": "Win Rate %"},
    {"name": "Avg Days",      "id": "Avg Days"},
    {"name": "Return %",      "id": "Return %"},
    {"name": "CAGR %",        "id": "CAGR %"},
    {"name": "Max DD %",      "id": "Max DD %"},
    {"name": "Sharpe",        "id": "Sharpe"},
    {"name": "Profit Factor", "id": "Profit Factor"},
    {"name": "Net Profit",    "id": "Net Profit"},
    {"name": "Total Costs",   "id": "Total Costs"},
]

TRADES_COLUMNS = [
    {"name": "Entry Date",   "id": "entry_date"},
    {"name": "Exit Date",    "id": "exit_date"},
    {"name": "Direction",    "id": "direction"},
    {"name": "Entry Price",  "id": "entry_price"},
    {"name": "Exit Price",   "id": "exit_price"},
    {"name": "Units",        "id": "units"},
    {"name": "PnL",          "id": "pnl"},
    {"name": "Commission",   "id": "commission"},
    {"name": "Days Held",    "id": "days_held"},
    {"name": "Exit Reason",  "id": "exit_reason"},
]


def _short_name(market):
    return market.split(" - ")[0].strip()


def _get_spec_value(market_name, key, default):
    spec = get_contract_spec(market_name)
    return spec[key] if spec and key in spec else default


def get_tick_size(market_name):
    return _get_spec_value(market_name, "tick_size", DEFAULT_TICK_SIZE)


def get_point_value(market_name):
    return _get_spec_value(market_name, "point_value", 1.0)


# ============================================================================
# ATR Two-Phase Stop
# ============================================================================

class ATRTwoPhaseStop:
    """Phase 1: fixed stop at entry +/- N*ATR.
    At 1:1 R/R → breakeven. Phase 2: trail with ATR."""

    def __init__(self, atr_stop_mult=ATR_STOP_MULT, atr_trail_mult=ATR_TRAIL_MULT):
        self.atr_stop_mult = atr_stop_mult
        self.atr_trail_mult = atr_trail_mult

    def initial_stop(self, direction, entry_price, row, **_kw):
        atr = row.get("ATR", np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = abs(entry_price) * 0.02
        if direction == 1:
            return entry_price - self.atr_stop_mult * atr - STOP_BUFFER
        return entry_price + self.atr_stop_mult * atr + STOP_BUFFER

    def stop_distance(self, direction, entry_price, row, **_kw):
        atr = row.get("ATR", np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = abs(entry_price) * 0.02
        return self.atr_stop_mult * atr + STOP_BUFFER

    def update(self, direction, entry_price, stop_loss, row, bar_idx,
               entry_idx, phase=1, stop_distance=0, **_kw):
        close = row["Close"]
        high = row.get("High", close)
        low = row.get("Low", close)
        atr = row.get("ATR", np.nan)

        exit_reason = None
        exit_price = None

        if direction == 1 and low <= stop_loss:
            exit_reason = f"Stop (Phase {phase})"
            exit_price = stop_loss
        elif direction == -1 and high >= stop_loss:
            exit_reason = f"Stop (Phase {phase})"
            exit_price = stop_loss

        if exit_reason:
            return stop_loss, exit_reason, exit_price, {"phase": 1}

        if phase == 1:
            unrealised = (
                (close - entry_price) if direction == 1
                else (entry_price - close)
            )
            if stop_distance > 0 and unrealised >= stop_distance:
                phase = 2
                stop_loss = entry_price

        if phase == 2 and not pd.isna(atr) and atr > 0:
            if direction == 1:
                trail = close - self.atr_trail_mult * atr
                stop_loss = max(stop_loss, trail)
            else:
                trail = close + self.atr_trail_mult * atr
                stop_loss = min(stop_loss, trail)

        return stop_loss, None, None, {
            "phase": phase, "stop_distance": stop_distance
        }


# ============================================================================
# Breakout signals
# ============================================================================

def add_breakout_signals(df, lookback):
    """N-day breakout signals. Entry at the breakout level (N-day high/low)."""
    out = df.copy()
    out["n_day_high"] = (
        out["High"].shift(1).rolling(lookback, min_periods=lookback).max()
    )
    out["n_day_low"] = (
        out["Low"].shift(1).rolling(lookback, min_periods=lookback).min()
    )
    out["signal"] = 0
    out["entry_price"] = np.nan

    long_bo = out["High"] > out["n_day_high"]
    short_bo = out["Low"] < out["n_day_low"]

    long_only = long_bo & ~short_bo
    short_only = short_bo & ~long_bo
    both = long_bo & short_bo

    out.loc[long_only, "signal"] = 1
    out.loc[long_only, "entry_price"] = out.loc[long_only, "n_day_high"]

    out.loc[short_only, "signal"] = -1
    out.loc[short_only, "entry_price"] = out.loc[short_only, "n_day_low"]

    if both.any():
        mid = (out["n_day_high"] + out["n_day_low"]) / 2
        long_both = both & (out["Open"] >= mid)
        short_both = both & ~long_both
        out.loc[long_both, "signal"] = 1
        out.loc[long_both, "entry_price"] = out.loc[long_both, "n_day_high"]
        out.loc[short_both, "signal"] = -1
        out.loc[short_both, "entry_price"] = out.loc[short_both, "n_day_low"]

    return out


# ============================================================================
# Backtest with transaction costs
# ============================================================================

def _calc_pnl(direction, entry, exit_price, units, point_value=1.0):
    raw = (exit_price - entry) if direction == 1 else (entry - exit_price)
    return raw * units * point_value


def _empty_result(capital):
    return {
        "trades": pd.DataFrame(), "missed_trades": pd.DataFrame(),
        "equity_curve": [capital], "final_capital": capital,
        "total_return": 0, "total_commission": 0.0, "total_slippage_cost": 0.0,
    }


def run_backtest_with_costs(data, market_name, stop_strategy,
                            initial_capital=INITIAL_CAPITAL,
                            risk_pct=RISK_PCT,
                            commission=COMMISSION_PER_TRADE,
                            slippage_ticks=SLIPPAGE_TICKS):
    """Bar-by-bar backtest loop with commission, slippage, and point_value."""
    df = data.copy().reset_index(drop=True)
    tick = get_tick_size(market_name)
    pv = get_point_value(market_name)
    slip = slippage_ticks * tick

    required = ["Date", "Close", "signal"]
    if any(c not in df.columns for c in required):
        return _empty_result(initial_capital)

    in_position = False
    direction = 0
    entry_price = 0.0
    entry_date = None
    entry_idx = 0
    stop_loss = 0.0
    units = 0
    stop_state = {}

    trades = []
    missed_trades = []
    current_capital = initial_capital
    equity = [current_capital]
    total_commission = 0.0
    total_slippage_cost = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        date = row["Date"]
        close = row["Close"]
        signal = row.get("signal", 0)

        if pd.isna(close):
            equity.append(current_capital)
            continue

        # ---- EXIT ----
        if in_position:
            stop_loss, exit_reason, exit_price, stop_state = stop_strategy.update(
                direction, entry_price, stop_loss, row, i, entry_idx,
                **stop_state,
            )
            if exit_reason:
                raw_pnl = _calc_pnl(direction, entry_price, exit_price,
                                    units, pv)
                net_pnl = raw_pnl - commission
                notional = entry_price * units * pv
                pnl_pct = (
                    (net_pnl / notional * 100) if notional > 0 else 0
                )
                current_capital += net_pnl
                total_commission += commission
                slip_dollar = slip * units * pv
                trades.append({
                    "market": market_name,
                    "entry_date": entry_date, "exit_date": date,
                    "direction": "Long" if direction == 1 else "Short",
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "units": units,
                    "pnl": round(net_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "commission": commission,
                    "slippage_cost": round(slip_dollar, 4),
                    "exit_reason": exit_reason,
                    "days_held": i - entry_idx,
                })
                in_position = False
                direction = 0
                stop_state = {}

        # ---- ENTRY ----
        if not in_position and signal != 0:
            ep = row.get("entry_price", np.nan)
            if pd.isna(ep):
                equity.append(current_capital)
                continue

            ep = ep + slip if signal == 1 else ep - slip

            stop_dist = stop_strategy.stop_distance(signal, ep, row)
            if stop_dist <= 0:
                equity.append(current_capital)
                continue

            risk_amt = current_capital * (risk_pct / 100.0)
            pos_units, error = calculate_position_size(
                risk_amt, stop_dist, market_name, pv,
            )
            if error:
                missed_trades.append({
                    "market": market_name, "date": date,
                    "direction": "Long" if signal == 1 else "Short",
                    "price": ep, "reason": error,
                })
                equity.append(current_capital)
                continue

            entry_price = ep
            entry_date = date
            entry_idx = i
            direction = signal
            units = pos_units
            stop_loss = stop_strategy.initial_stop(direction, entry_price, row)
            stop_state = {"stop_distance": stop_dist, "phase": 1}
            in_position = True
            total_slippage_cost += slip * units * pv

        equity.append(current_capital)

    # close open position at end of data
    if in_position:
        final = df.iloc[-1]
        raw_pnl = _calc_pnl(direction, entry_price, final["Close"],
                             units, pv)
        net_pnl = raw_pnl - commission
        notional = entry_price * units * pv
        pnl_pct = (
            (net_pnl / notional * 100) if notional > 0 else 0
        )
        current_capital += net_pnl
        total_commission += commission
        slip_dollar = slip * units * pv
        trades.append({
            "market": market_name, "entry_date": entry_date,
            "exit_date": final["Date"],
            "direction": "Long" if direction == 1 else "Short",
            "entry_price": round(entry_price, 6),
            "exit_price": round(final["Close"], 6),
            "units": units,
            "pnl": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "commission": commission,
            "slippage_cost": round(slip_dollar, 4),
            "exit_reason": "End of Data",
            "days_held": len(df) - 1 - entry_idx,
        })
        equity.append(current_capital)

    return {
        "trades": pd.DataFrame(trades),
        "missed_trades": pd.DataFrame(missed_trades),
        "equity_curve": equity,
        "final_capital": current_capital,
        "total_return": (
            (current_capital - initial_capital) / initial_capital * 100
        ),
        "total_commission": total_commission,
        "total_slippage_cost": total_slippage_cost,
    }


# ============================================================================
# Aggregate helpers
# ============================================================================

def max_consecutive_losers(trades_df):
    if trades_df.empty:
        return 0
    streak = max_streak = 0
    for pnl in trades_df["pnl"]:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _total_row(totals, start_date, end_date):
    t = totals["trades"]
    wr = (totals["wins"] / t * 100) if t > 0 else 0
    ret = (totals["net_profit"] / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL > 0 else 0
    pf = (
        (totals["gross_profit"] / totals["gross_loss"])
        if totals["gross_loss"] > 0 else 999
    )
    try:
        years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    except Exception:
        years = 1
    years = max(years, 0.01)

    final = INITIAL_CAPITAL + totals["net_profit"]
    cagr = (
        ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
        if final > 0 else 0
    )

    sharpe = 0.0
    if len(totals["all_pnl"]) > 1:
        arr = np.array(totals["all_pnl"]) / 100
        tpy = len(arr) / years
        if np.std(arr) > 0 and tpy > 0:
            sharpe = (np.mean(arr) * tpy) / (np.std(arr) * np.sqrt(tpy))

    return {
        "Market": "*** TOTAL ***",
        "Trades": t,
        "Missed": totals["missed"],
        "Win Rate %": round(wr, 1),
        "Avg Days": 0,
        "Return %": round(ret, 2),
        "CAGR %": round(cagr, 2),
        "Max DD %": round(totals["max_dd"], 2),
        "Sharpe": round(sharpe, 2),
        "Profit Factor": round(min(pf, 999), 2),
        "Net Profit": round(totals["net_profit"], 2),
        "Total Costs": round(
            totals["total_commission"] + totals["total_slippage"], 2
        ),
    }


def run_all_markets_tf(cot_df, markets, lookback, start_date, end_date):
    """Run the breakout strategy across every market and collect results."""
    stop = ATRTwoPhaseStop()
    all_results = {}
    summary_rows = []
    totals = {
        "trades": 0, "wins": 0, "gross_profit": 0, "gross_loss": 0,
        "net_profit": 0, "max_dd": 0, "missed": 0, "all_pnl": [],
        "total_commission": 0.0, "total_slippage": 0.0,
    }

    for market in markets:
        df = prepare_base_data(cot_df, market)
        if df.empty:
            continue
        df = add_standard_indicators(df, atr_period=10)
        df = add_breakout_signals(df, lookback)

        if start_date:
            df = df[df["Date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.Timestamp(end_date)]
        df = df.reset_index(drop=True)
        if df.empty:
            continue

        result = run_backtest_with_costs(df, market, stop)
        metrics = calculate_performance_metrics(
            result["trades"], result["equity_curve"], INITIAL_CAPITAL,
        )

        all_results[market] = {
            "data": df, "results": result, "metrics": metrics,
        }

        missed_n = (
            len(result["missed_trades"])
            if not result["missed_trades"].empty else 0
        )
        totals["trades"] += metrics.get("total_trades", 0)
        totals["wins"] += metrics.get("winning_trades", 0)
        totals["gross_profit"] += metrics.get("gross_profit", 0)
        totals["gross_loss"] += metrics.get("gross_loss", 0)
        totals["net_profit"] += metrics.get("net_profit", 0)
        totals["max_dd"] = max(
            totals["max_dd"], metrics.get("max_drawdown_pct", 0),
        )
        totals["missed"] += missed_n
        totals["total_commission"] += result.get("total_commission", 0)
        totals["total_slippage"] += result.get("total_slippage_cost", 0)
        if not result["trades"].empty and "pnl_pct" in result["trades"].columns:
            totals["all_pnl"].extend(result["trades"]["pnl_pct"].tolist())

        trade_costs = (
            result.get("total_commission", 0)
            + result.get("total_slippage_cost", 0)
        )
        summary_rows.append({
            "Market": _short_name(market),
            "Trades": metrics.get("total_trades", 0),
            "Missed": missed_n,
            "Win Rate %": round(metrics.get("win_rate", 0), 1),
            "Avg Days": round(metrics.get("avg_days_held", 0), 1),
            "Return %": round(metrics.get("total_return_pct", 0), 2),
            "CAGR %": round(metrics.get("cagr", 0), 2),
            "Max DD %": round(metrics.get("max_drawdown_pct", 0), 2),
            "Sharpe": round(metrics.get("sharpe_ratio", 0), 2),
            "Profit Factor": round(
                min(metrics.get("profit_factor", 0), 999), 2,
            ),
            "Net Profit": round(metrics.get("net_profit", 0), 2),
            "Total Costs": round(trade_costs, 2),
        })

    summary_rows.append(_total_row(totals, start_date, end_date))
    summary_df = pd.DataFrame(summary_rows)
    return all_results, summary_df, totals


# ============================================================================
# Charts
# ============================================================================

def create_tf_chart(data, trades_df, market_name):
    """Candlestick chart with N-day bands and trade markers."""
    df = data.copy()
    has_bands = "n_day_high" in df.columns

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.78, 0.22],
        subplot_titles=[f"Price: {_short_name(market_name)}", "ATR"],
    )

    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
        increasing_fillcolor="#26A69A", decreasing_fillcolor="#EF5350",
    ), row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    if has_bands:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["n_day_high"], name="N-Day High",
            line=dict(color="rgba(0,200,83,0.5)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["n_day_low"], name="N-Day Low",
            line=dict(color="rgba(255,23,68,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(100,100,100,0.06)",
        ), row=1, col=1)

    if not trades_df.empty:
        longs = trades_df[trades_df["direction"] == "Long"]
        shorts = trades_df[trades_df["direction"] == "Short"]
        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=longs["entry_date"], y=longs["entry_price"],
                mode="markers", name="Long Entry",
                marker=dict(symbol="triangle-up", size=12, color="#00C853"),
            ), row=1, col=1)
        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts["entry_date"], y=shorts["entry_price"],
                mode="markers", name="Short Entry",
                marker=dict(symbol="triangle-down", size=12, color="#FF1744"),
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trades_df["exit_date"], y=trades_df["exit_price"],
            mode="markers", name="Exit",
            marker=dict(symbol="x", size=10, color="#FFD600",
                        line=dict(width=2)),
        ), row=1, col=1)

    if "ATR" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["ATR"], name="ATR (10)",
            line=dict(color="#FF6D00", width=1.5),
        ), row=2, col=1)

    fig.update_layout(
        height=620, hovermode="x unified", template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        margin=dict(t=60, b=30),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ATR", row=2, col=1)
    return fig


def create_equity_curve(equity_curve, initial_capital):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve, name="Equity",
        line=dict(color="#2962FF", width=2),
        fill="tonexty", fillcolor="rgba(41,98,255,0.15)",
    ))
    fig.add_hline(
        y=initial_capital, line_dash="dash", line_color="gray",
        annotation_text=f"Initial: ${initial_capital:,.0f}",
    )
    eq_min = min(equity_curve) if equity_curve else initial_capital
    eq_max = max(equity_curve) if equity_curve else initial_capital
    padding = max((eq_max - eq_min) * 0.1, initial_capital * 0.02)
    fig.update_layout(
        title="Equity Curve", height=340, template="plotly_dark",
        yaxis_title="Equity ($)", xaxis_title="Bar #",
        yaxis=dict(range=[eq_min - padding, eq_max + padding]),
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
    )
    return fig


# ============================================================================
# Load data
# ============================================================================

print("Loading COT data ...")
cot_df = load_cot_data()
markets = sorted(cot_df["Market"].unique().tolist()) if not cot_df.empty else []
print(f"  {len(markets)} markets loaded")

print(f"Running initial backtest (lookback={DEFAULT_LOOKBACK}d) ...")
all_results, summary_df, agg_totals = run_all_markets_tf(
    cot_df, markets, DEFAULT_LOOKBACK, DEFAULT_START, DEFAULT_END,
)
if not summary_df.empty:
    summary_df = summary_df.fillna(0)
    for col in summary_df.columns:
        summary_df[col] = summary_df[col].apply(
            lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x
        )
print(f"  {len(all_results)} markets with results")


# ============================================================================
# Dash app
# ============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container([

    # ---- Title ----
    dbc.Row([dbc.Col([
        html.H1("Trend-Following Breakout Backtest",
                className="text-center my-3"),
        html.P(
            "N-day high/low breakout · 2×ATR stop → breakeven → trail · "
            "With $10 commission + 2-tick slippage",
            className="text-center text-muted",
        ),
    ])]),

    # ---- Cost badges ----
    dbc.Row([dbc.Col([
        dbc.Badge("$10 Commission / Trade", color="warning",
                  className="me-2 px-3 py-2"),
        dbc.Badge("2 Ticks Slippage on Entry", color="info",
                  className="me-2 px-3 py-2"),
        dbc.Badge(f"${INITIAL_CAPITAL:,.0f} Capital", color="success",
                  className="px-3 py-2"),
    ], className="text-center mb-3")]),

    # ---- Controls ----
    dbc.Row([
        dbc.Col([
            html.Label("Lookback (days)", className="text-muted small"),
            dbc.Input(id="lookback-input", type="number",
                      value=DEFAULT_LOOKBACK, min=5, max=200, step=1),
        ], width=2),
        dbc.Col([
            html.Label("Start", className="text-muted small"),
            dcc.DatePickerSingle(id="start-date", date=DEFAULT_START,
                                 display_format="YYYY-MM-DD"),
        ], width=2),
        dbc.Col([
            html.Label("End", className="text-muted small"),
            dcc.DatePickerSingle(id="end-date", date=DEFAULT_END,
                                 display_format="YYYY-MM-DD"),
        ], width=2),
        dbc.Col([
            html.Label(" ", className="text-muted small"), html.Br(),
            dbc.Button("Run Backtest", id="run-btn", color="primary",
                       size="lg", className="mt-1"),
        ], width=2),
    ], className="mb-2"),

    dbc.Row([dbc.Col(
        html.Div(id="status-text", className="text-muted mt-1"),
    )]),

    html.Hr(),

    # ---- Cost impact cards ----
    dbc.Row(id="cost-cards", className="mb-3"),

    # ---- Summary table ----
    dbc.Row([dbc.Col([
        html.H4("All Markets Summary", className="mt-2"),
        dash_table.DataTable(
            id="summary-table",
            columns=SUMMARY_COLUMNS,
            data=summary_df.to_dict("records") if not summary_df.empty else [],
            sort_action="native", filter_action="native", page_size=20,
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#1a1a2e", "color": "white",
                "fontWeight": "bold",
            },
            style_cell={
                "backgroundColor": "#16213e", "color": "white",
                "textAlign": "center", "padding": "8px",
            },
            style_data_conditional=[
                {"if": {"filter_query": "{Return %} > 0",
                         "column_id": "Return %"},
                 "backgroundColor": "#1b4332", "color": "white"},
                {"if": {"filter_query": "{Return %} < 0",
                         "column_id": "Return %"},
                 "backgroundColor": "#4a1c1c", "color": "white"},
                {"if": {"filter_query": "{Win Rate %} >= 50",
                         "column_id": "Win Rate %"},
                 "backgroundColor": "#1b4332"},
                {"if": {"filter_query": "{Win Rate %} < 40",
                         "column_id": "Win Rate %"},
                 "backgroundColor": "#4a1c1c"},
                {"if": {"filter_query": '{Market} = "*** TOTAL ***"'},
                 "backgroundColor": "#0f3460", "fontWeight": "bold",
                 "borderTop": "2px solid #FFD600"},
            ],
        ),
    ])], className="mb-4"),

    html.Hr(),

    # ---- Market detail ----
    dbc.Row([dbc.Col([
        html.H4("Market Detail"),
        dcc.Dropdown(
            id="market-dropdown",
            options=[{"label": _short_name(m), "value": m} for m in markets],
            value=markets[0] if markets else None,
            className="mb-3", style={"color": "black"},
        ),
    ], width=6)]),

    dbc.Row(id="metrics-cards", className="mb-4"),
    dbc.Row([dbc.Col([dcc.Graph(id="strategy-chart")])]),
    dbc.Row([
        dbc.Col([dcc.Graph(id="equity-chart")], width=6),
        dbc.Col([
            html.H5("Trades"),
            dash_table.DataTable(
                id="trades-table", columns=TRADES_COLUMNS, data=[],
                sort_action="native", page_size=12,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#1a1a2e", "color": "white"},
                style_cell={
                    "backgroundColor": "#16213e", "color": "white",
                    "textAlign": "center",
                },
            ),
        ], width=6),
    ]),

    dcc.Store(id="results-store"),

], fluid=True)


# ============================================================================
# Callbacks
# ============================================================================

@app.callback(
    [Output("results-store", "data"),
     Output("summary-table", "data"),
     Output("cost-cards", "children", allow_duplicate=True),
     Output("status-text", "children")],
    Input("run-btn", "n_clicks"),
    [State("lookback-input", "value"),
     State("start-date", "date"),
     State("end-date", "date")],
    prevent_initial_call=True,
)
def run_backtest_cb(n_clicks, lookback, start_date, end_date):
    global all_results, summary_df, agg_totals

    lookback = int(lookback or DEFAULT_LOOKBACK)
    if not start_date or not end_date:
        return dash.no_update, dash.no_update, dash.no_update, "Select dates"

    all_results, summary_df, agg_totals = run_all_markets_tf(
        cot_df, markets, lookback, start_date, end_date,
    )

    if not summary_df.empty:
        summary_df = summary_df.fillna(0)
        for col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: float(x)
                if isinstance(x, (np.integer, np.floating)) else x
            )

    status = (
        f"Done — {lookback}d breakout | "
        f"{len(all_results)} markets | "
        f"{start_date} → {end_date} | "
        f"$10 commission + 2-tick slippage"
    )

    return (
        {"timestamp": datetime.now().isoformat()},
        summary_df.to_dict("records") if not summary_df.empty else [],
        _build_cost_cards(agg_totals),
        status,
    )


@app.callback(
    Output("cost-cards", "children"),
    Input("results-store", "data"),
)
def update_cost_cards(_store):
    return _build_cost_cards(agg_totals)


def _build_cost_cards(totals):
    total_comm = totals.get("total_commission", 0)
    total_slip = totals.get("total_slippage", 0)
    total_costs = total_comm + total_slip
    net = totals.get("net_profit", 0)
    gross = net + total_costs

    def card(title, value, color=""):
        return dbc.Col(dbc.Card([dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted mb-1",
                     style={"fontSize": "0.8em"}),
            html.H4(value, className=f"card-title mb-0 {color}"),
        ])], className="h-100"), width=2)

    return [
        card("Total Commission",
             f"${total_comm:,.0f}",
             "text-warning"),
        card("Total Slippage Cost",
             f"${total_slip:,.2f}",
             "text-warning"),
        card("Combined Costs",
             f"${total_costs:,.2f}",
             "text-danger"),
        card("Gross PnL (no costs)",
             f"${gross:,.2f}",
             "text-info"),
        card("Net PnL (after costs)",
             f"${net:,.2f}",
             "text-success" if net > 0 else "text-danger"),
        card("Cost as % of Gross",
             f"{(total_costs / gross * 100):.1f}%"
             if gross > 0 else "N/A",
             "text-warning"),
    ]


@app.callback(
    [Output("metrics-cards", "children"),
     Output("strategy-chart", "figure"),
     Output("equity-chart", "figure"),
     Output("trades-table", "data"),
     Output("trades-table", "columns")],
    [Input("market-dropdown", "value"),
     Input("results-store", "data")],
)
def update_market_detail(market, _store):
    empty = go.Figure()
    empty.update_layout(template="plotly_dark",
                        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
    if not market or market not in all_results:
        return [], empty, empty, [], TRADES_COLUMNS

    res = all_results[market]
    m = res["metrics"]
    data = res["data"]
    trades_df = res["results"]["trades"]
    equity = res["results"]["equity_curve"]
    cap = equity[0] if equity else INITIAL_CAPITAL

    mcl = max_consecutive_losers(trades_df)

    def card(title, value, color=""):
        return dbc.Col(dbc.Card([dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted mb-1",
                     style={"fontSize": "0.8em"}),
            html.H4(value, className=f"card-title mb-0 {color}"),
        ])], className="h-100"), width=2)

    ret = m.get("total_return_pct", 0)
    cards = [
        card("Total Return",
             f"{ret:.2f}%",
             "text-success" if ret > 0 else "text-danger"),
        card("Win Rate", f"{m.get('win_rate', 0):.1f}%"),
        card("Trades", f"{m.get('total_trades', 0)}"),
        card("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}"),
        card("Max DD",
             f"{m.get('max_drawdown_pct', 0):.2f}%",
             "text-warning"),
        card("Max Consec. Losers", f"{mcl}", "text-warning"),
    ]

    strat_fig = create_tf_chart(data, trades_df, market)
    eq_fig = create_equity_curve(equity, cap)

    if not trades_df.empty:
        disp = trades_df.copy()
        disp["entry_date"] = pd.to_datetime(
            disp["entry_date"]
        ).dt.strftime("%Y-%m-%d")
        disp["exit_date"] = pd.to_datetime(
            disp["exit_date"]
        ).dt.strftime("%Y-%m-%d")
        disp["entry_price"] = disp["entry_price"].apply(
            lambda x: f"{x:,.6f}",
        )
        disp["exit_price"] = disp["exit_price"].apply(
            lambda x: f"{x:,.6f}",
        )
        disp["pnl"] = disp["pnl"].apply(lambda x: f"${x:,.2f}")
        disp["commission"] = disp["commission"].apply(
            lambda x: f"${x:,.0f}",
        )
        disp["units"] = disp["units"].apply(
            lambda x: str(int(x))
            if isinstance(x, (int, float)) and x == int(x)
            else str(x)
        )
        disp["days_held"] = disp["days_held"].apply(
            lambda x: str(int(x)) if not pd.isna(x) else "0",
        )
        avail = [
            c["id"] for c in TRADES_COLUMNS if c["id"] in disp.columns
        ]
        table_data = disp[avail].to_dict("records")
    else:
        table_data = []

    return cards, strat_fig, eq_fig, table_data, TRADES_COLUMNS


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8055)
