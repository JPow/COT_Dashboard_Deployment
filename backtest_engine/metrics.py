"""
Performance analytics — shared across all strategy models.
"""

import pandas as pd
import numpy as np


def calculate_performance_metrics(trades_df, equity_curve, initial_capital=30000):
    """Comprehensive performance metrics from a trades DataFrame and equity curve."""
    if trades_df.empty:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'profit_factor': 0, 'total_return_pct': 0,
            'cagr': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0,
            'avg_win': 0, 'avg_loss': 0, 'net_profit': 0,
            'gross_profit': 0, 'gross_loss': 0, 'avg_days_held': 0,
        }

    m = {}
    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] < 0]

    m['total_trades'] = len(trades_df)
    m['winning_trades'] = len(winning)
    m['losing_trades'] = len(losing)
    m['win_rate'] = (len(winning) / len(trades_df) * 100) if len(trades_df) > 0 else 0

    total_profit = winning['pnl'].sum() if not winning.empty else 0
    total_loss = abs(losing['pnl'].sum()) if not losing.empty else 0
    m['gross_profit'] = total_profit
    m['gross_loss'] = total_loss
    m['net_profit'] = total_profit - total_loss
    m['profit_factor'] = (total_profit / total_loss) if total_loss > 0 else float('inf')

    m['avg_win'] = winning['pnl'].mean() if not winning.empty else 0
    m['avg_loss'] = losing['pnl'].mean() if not losing.empty else 0
    m['avg_days_held'] = trades_df['days_held'].mean() if 'days_held' in trades_df.columns else 0

    final_capital = equity_curve[-1] if equity_curve else initial_capital
    m['total_return_pct'] = (final_capital - initial_capital) / initial_capital * 100

    # CAGR
    if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
        first = pd.to_datetime(trades_df['entry_date']).min()
        last = pd.to_datetime(trades_df['exit_date']).max()
        days = (last - first).days
        years = days / 365.25 if days > 0 else 1
    else:
        years = 1
    m['cagr'] = ((final_capital / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

    # Max drawdown
    eq = pd.Series(equity_curve)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    m['max_drawdown_pct'] = abs(dd.min()) * 100

    # Sharpe (annualised using actual trade frequency)
    if len(trades_df) > 1 and 'pnl_pct' in trades_df.columns:
        returns = trades_df['pnl_pct'] / 100
        if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
            trade_years = (pd.to_datetime(trades_df['exit_date']).max() -
                           pd.to_datetime(trades_df['entry_date']).min()).days / 365.25
        else:
            trade_years = 1
        trade_years = max(trade_years, 0.01)
        tpy = len(trades_df) / trade_years
        if returns.std() > 0 and tpy > 0:
            m['sharpe_ratio'] = (returns.mean() * tpy) / (returns.std() * np.sqrt(tpy))
        else:
            m['sharpe_ratio'] = 0
    else:
        m['sharpe_ratio'] = 0

    return m
