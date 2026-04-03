"""
Plotly chart builders — shared across all strategy models.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_strategy_chart(data, trades_df, market_name, setup_key='narrowing_range'):
    """Multi-pane strategy chart with candlesticks, indicators and trade markers."""
    df = data.copy()
    has_cot = 'Commercial_Index' in df.columns and df['Commercial_Index'].notna().any()
    has_narrowing = 'consecutive_narrowing' in df.columns
    has_inside = 'consecutive_inside_days' in df.columns
    has_setup_pane = has_narrowing or has_inside

    n_rows = 2 + int(has_setup_pane) + int(has_cot) + 1  # price + ATR + (setup) + (COT) + RSI
    heights = []
    subtitles = [f"Price: {market_name}"]

    # Build row config dynamically
    heights.append(0.40)
    subtitles.append("ATR (Fast & Slow)")
    heights.append(0.12)
    if has_setup_pane:
        label = "Narrowing Range Days" if has_narrowing else "Inside Days"
        subtitles.append(label)
        heights.append(0.12)
    if has_cot:
        subtitles.append("Commercial Index (COT)")
        heights.append(0.12)
    subtitles.append("RSI")
    heights.append(0.12)

    # Normalise heights to sum ~ 1
    total = sum(heights)
    heights = [h / total for h in heights]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=heights, subplot_titles=subtitles,
    )

    # --- Pane 1: Candlestick ---
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350',
        increasing_fillcolor='#26A69A', decreasing_fillcolor='#EF5350',
    ), row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    ma_colors = {
        'MA_10': '#FF6D00', 'MA_20': '#FFD600', 'MA_50': '#00E676',
        'MA_100': '#00BCD4', 'MA_150': '#AA00FF', 'MA_200': '#FF1744',
    }
    for col, color in ma_colors.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df[col], name=col.replace('_', ' '),
                line=dict(color=color, width=1, dash='dot'),
                visible='legendonly',
            ), row=1, col=1)

    # Trade markers
    if not trades_df.empty:
        longs = trades_df[trades_df['direction'] == 'Long']
        shorts = trades_df[trades_df['direction'] == 'Short']
        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=longs['entry_date'], y=longs['entry_price'],
                mode='markers', name='Long Entry',
                marker=dict(symbol='triangle-up', size=14, color='#00C853'),
            ), row=1, col=1)
        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts['entry_date'], y=shorts['entry_price'],
                mode='markers', name='Short Entry',
                marker=dict(symbol='triangle-down', size=14, color='#FF1744'),
            ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trades_df['exit_date'], y=trades_df['exit_price'],
            mode='markers', name='Exit',
            marker=dict(symbol='x', size=11, color='#FFD600', line=dict(width=2)),
        ), row=1, col=1)

    # --- Pane 2: ATR ---
    r = 2
    if 'fast_ATR' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['fast_ATR'], name="Fast ATR",
                                 line=dict(color="#FF6D00", width=1.5)), row=r, col=1)
    if 'slow_ATR' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['slow_ATR'], name="Slow ATR",
                                 line=dict(color="#00BCD4", width=1.5)), row=r, col=1)
    r += 1

    # --- Pane 3 (optional): Setup indicator ---
    if has_setup_pane:
        col_name = 'consecutive_narrowing' if has_narrowing else 'consecutive_inside_days'
        fig.add_trace(go.Bar(
            x=df['Date'], y=df[col_name],
            name=col_name.replace('_', ' ').title(),
            marker_color='#7C4DFF', opacity=0.7,
        ), row=r, col=1)
        r += 1

    # --- COT pane (optional) ---
    if has_cot:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Commercial_Index'], name="Commercial Index",
            line=dict(color="#00BFA5", width=2),
        ), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="green", row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="red", row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)
        r += 1

    # --- RSI pane ---
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'], name="RSI",
            line=dict(color="#AA00FF", width=2),
        ), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=r, col=1)

    fig.update_layout(
        height=max(700, 200 * n_rows),
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        margin=dict(t=80),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)

    return fig


def create_equity_curve(equity_curve, initial_capital):
    """Equity curve with auto-scaled y-axis."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve, name="Equity",
        line=dict(color="#2962FF", width=2),
        fill='tonexty', fillcolor='rgba(41, 98, 255, 0.1)',
    ))
    fig.add_hline(
        y=initial_capital, line_dash="dash", line_color="gray",
        annotation_text=f"Initial: ${initial_capital:,.0f}",
    )
    eq_min = min(equity_curve) if equity_curve else initial_capital
    eq_max = max(equity_curve) if equity_curve else initial_capital
    padding = max((eq_max - eq_min) * 0.1, initial_capital * 0.02)
    fig.update_layout(
        title="Equity Curve", height=350, template="plotly_white",
        yaxis_title="Equity ($)", xaxis_title="Trade #",
        yaxis=dict(range=[eq_min - padding, eq_max + padding]),
    )
    return fig
