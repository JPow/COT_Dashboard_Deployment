"""
Unified backtest engine that wires Setup → Entry → Stop.
"""

import pandas as pd
import numpy as np

from .data import (prepare_base_data, load_intraday_cache,
                   load_contract_specs, get_contract_spec)
from .indicators import (add_standard_indicators, calculate_narrowing_ranges,
                         calculate_inside_days)
from .setups import SETUP_REGISTRY
from .entries import ENTRY_REGISTRY
from .stops import STOP_REGISTRY


STOP_BUFFER = 0.01


def calculate_position_size(risk_amount, stop_distance, market_name="",
                            point_value=1.0):
    """Position_Size = Risk_Amount / (Stop_Distance * point_value).

    ``point_value`` converts a 1-point price move into dollars per contract
    (e.g. 50 for E-mini S&P, 10 for Micro Gold).

    Crypto (BITCOIN / ETHER) allows fractional units; everything else floors
    to whole contracts.
    """
    if stop_distance <= 0:
        return 0, "Invalid stop distance"
    dollar_risk_per_contract = stop_distance * point_value
    raw_units = risk_amount / dollar_risk_per_contract
    allows_fractional = 'BITCOIN' in market_name.upper() or 'ETHER' in market_name.upper()
    if allows_fractional:
        if raw_units < 0.0001:
            return 0, f"Position too small ({raw_units:.6f} units)"
        return round(raw_units, 4), None
    else:
        if raw_units < 1:
            return 0, (f"Can only afford {raw_units:.4f} units "
                       f"(risk ${risk_amount:.0f} / "
                       f"stop ${dollar_risk_per_contract:.0f} per contract)")
        return int(raw_units), None


def prepare_data(cot_df, market_name, setup_key, entry_key,
                 atr_period=10, rsi_period=10,
                 start_date=None, end_date=None,
                 setup_params=None, entry_params=None,
                 intraday_cache=None):
    """Build a fully-prepared DataFrame for one market.

    Steps:
        1. Base daily OHLCV + COT merge
        2. Standard indicators (RSI, dual ATR, MAs)
        3. Setup-specific indicators (narrowing range / inside days)
        4. Setup detector → ``setup`` column
        5. Entry filter → ``signal``, ``entry_price`` columns
    """
    setup_params = setup_params or {}
    entry_params = entry_params or {}

    df = prepare_base_data(cot_df, market_name)
    if df.empty:
        return pd.DataFrame()

    # Indicators
    df = add_standard_indicators(df, atr_period=atr_period,
                                 rsi_period=rsi_period)

    # Setup-specific pre-processing
    if setup_key in ('narrowing_range',):
        df = calculate_narrowing_ranges(df)
    elif setup_key in ('inside_days',):
        df = calculate_inside_days(df)

    # Run setup detector
    setup_fn = SETUP_REGISTRY[setup_key]['fn']
    df = setup_fn(df, **setup_params)

    # Run entry filter
    entry_fn = ENTRY_REGISTRY[entry_key]['fn']
    entry_kw = dict(entry_params)
    entry_kw['market_name'] = market_name
    entry_kw['intraday_cache'] = intraday_cache
    df = entry_fn(df, **entry_kw)

    # Date filter (after indicators so lookback periods are satisfied)
    if start_date:
        df = df[df['Date'] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.Timestamp(end_date)]

    return df.reset_index(drop=True)


def run_backtest(data, market_name, stop_strategy,
                 initial_capital=30000, risk_pct=1.0,
                 point_value=None):
    """Execute the backtest loop on prepared data.

    ``stop_strategy`` is an instance of a stop class from stops.py.
    ``point_value`` converts a 1-point price move to dollars per contract.
    If *None*, it is looked up automatically from ORB_contract_specs.json.
    """
    df = data.copy().reset_index(drop=True)
    required = ['Date', 'Close', 'signal']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _empty_result(initial_capital)

    if point_value is None:
        spec = get_contract_spec(market_name)
        point_value = spec["point_value"] if spec else 1.0

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

    for i in range(len(df)):
        row = df.iloc[i]
        date = row['Date']
        close = row['Close']
        signal = row.get('signal', 0)

        if pd.isna(close):
            equity.append(current_capital)
            continue

        # --- EXIT ---
        if in_position:
            stop_loss, exit_reason, exit_price, stop_state = stop_strategy.update(
                direction, entry_price, stop_loss, row, i, entry_idx,
                **stop_state
            )
            if exit_reason:
                pnl = _calc_pnl(direction, entry_price, exit_price,
                                units, point_value)
                notional = entry_price * units * point_value
                pnl_pct = (pnl / notional * 100) if notional > 0 else 0
                current_capital += pnl
                trades.append(_trade_record(
                    market_name, entry_date, date, direction,
                    entry_price, exit_price, units, pnl, pnl_pct,
                    exit_reason, i - entry_idx, row))
                in_position = False
                direction = 0
                stop_state = {}

        # --- ENTRY ---
        if not in_position and signal != 0:
            ep = row.get('entry_price', np.nan)
            if pd.isna(ep):
                equity.append(current_capital)
                continue

            stop_dist = stop_strategy.stop_distance(signal, ep, row)
            if stop_dist <= 0:
                equity.append(current_capital)
                continue

            risk_amt = current_capital * (risk_pct / 100.0)
            pos_units, error = calculate_position_size(
                risk_amt, stop_dist, market_name, point_value,
            )
            if error:
                missed_trades.append({
                    'market': market_name, 'date': date,
                    'direction': 'Long' if signal == 1 else 'Short',
                    'price': ep, 'reason': error,
                })
                equity.append(current_capital)
                continue

            entry_price = ep
            entry_date = date
            entry_idx = i
            direction = signal
            units = pos_units
            stop_loss = stop_strategy.initial_stop(direction, entry_price, row)
            stop_state = {'stop_distance': stop_dist, 'phase': 1}
            in_position = True

        equity.append(current_capital)

    # Close open position at end of data
    if in_position:
        final = df.iloc[-1]
        pnl = _calc_pnl(direction, entry_price, final['Close'],
                         units, point_value)
        notional = entry_price * units * point_value
        pnl_pct = (pnl / notional * 100) if notional > 0 else 0
        current_capital += pnl
        trades.append(_trade_record(
            market_name, entry_date, final['Date'], direction,
            entry_price, final['Close'], units, pnl, pnl_pct,
            'End of Data', len(df) - 1 - entry_idx, final))
        equity.append(current_capital)

    return {
        'trades': pd.DataFrame(trades),
        'missed_trades': pd.DataFrame(missed_trades),
        'equity_curve': equity,
        'final_capital': current_capital,
        'total_return': (current_capital - initial_capital) / initial_capital * 100,
    }


def run_all_markets(cot_df, markets, setup_key, entry_key, stop_key,
                    setup_params=None, entry_params=None, stop_params=None,
                    atr_period=10,
                    initial_capital=30000, risk_pct=1.0,
                    start_date=None, end_date=None):
    """Run the full pipeline for every market.  Returns (all_results, summary_df)."""
    setup_params = setup_params or {}
    entry_params = entry_params or {}
    stop_params = stop_params or {}

    stop_cls = STOP_REGISTRY[stop_key]['cls']
    stop_strategy = stop_cls(**stop_params)

    # Pre-load intraday cache once
    intraday_cache = load_intraday_cache() if entry_key == 'orb_breakout' else None

    all_results = {}
    summary_rows = []
    totals = dict(trades=0, wins=0, gross_profit=0, gross_loss=0,
                  net_profit=0, max_dd=0, missed=0, all_pnl=[])

    for market in markets:
        data = prepare_data(
            cot_df, market, setup_key, entry_key,
            atr_period=atr_period,
            start_date=start_date, end_date=end_date,
            setup_params=setup_params, entry_params=entry_params,
            intraday_cache=intraday_cache,
        )
        if data.empty:
            continue

        result = run_backtest(data, market, stop_strategy,
                              initial_capital=initial_capital,
                              risk_pct=risk_pct)

        from .metrics import calculate_performance_metrics
        metrics = calculate_performance_metrics(
            result['trades'], result['equity_curve'], initial_capital)

        all_results[market] = {'data': data, 'results': result, 'metrics': metrics}

        missed_n = len(result['missed_trades']) if not result['missed_trades'].empty else 0
        totals['trades'] += metrics.get('total_trades', 0)
        totals['wins'] += metrics.get('winning_trades', 0)
        totals['gross_profit'] += metrics.get('gross_profit', 0)
        totals['gross_loss'] += metrics.get('gross_loss', 0)
        totals['net_profit'] += metrics.get('net_profit', 0)
        totals['max_dd'] = max(totals['max_dd'], metrics.get('max_drawdown_pct', 0))
        totals['missed'] += missed_n
        if not result['trades'].empty and 'pnl_pct' in result['trades'].columns:
            totals['all_pnl'].extend(result['trades']['pnl_pct'].tolist())

        summary_rows.append({
            'Market': market,
            'Trades': metrics.get('total_trades', 0),
            'Missed': missed_n,
            'Win Rate %': round(metrics.get('win_rate', 0), 1),
            'Avg Days': round(metrics.get('avg_days_held', 0), 1),
            'Return %': round(metrics.get('total_return_pct', 0), 2),
            'CAGR %': round(metrics.get('cagr', 0), 2),
            'Max DD %': round(metrics.get('max_drawdown_pct', 0), 2),
            'Sharpe': round(metrics.get('sharpe_ratio', 0), 2),
            'Profit Factor': round(min(metrics.get('profit_factor', 0), 999), 2),
            'Net Profit': round(metrics.get('net_profit', 0), 2),
        })

    # Aggregate TOTAL row
    summary_rows.append(_total_row(totals, initial_capital, start_date, end_date))

    summary_df = pd.DataFrame(summary_rows)
    return all_results, summary_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_pnl(direction, entry, exit_price, units, point_value=1.0):
    raw = (exit_price - entry) if direction == 1 else (entry - exit_price)
    return raw * units * point_value


def _trade_record(market, entry_date, exit_date, direction, entry_price,
                  exit_price, units, pnl, pnl_pct, exit_reason, days_held, row):
    return {
        'market': market,
        'entry_date': entry_date, 'exit_date': exit_date,
        'direction': 'Long' if direction == 1 else 'Short',
        'entry_price': round(entry_price, 4),
        'exit_price': round(exit_price, 4),
        'units': units,
        'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2),
        'exit_reason': exit_reason,
        'days_held': days_held,
        'or_high': row.get('or_high_signal', np.nan),
        'or_low': row.get('or_low_signal', np.nan),
    }


def _total_row(totals, initial_capital, start_date, end_date):
    t = totals['trades']
    wr = (totals['wins'] / t * 100) if t > 0 else 0
    ret = (totals['net_profit'] / initial_capital * 100) if initial_capital > 0 else 0
    pf = (totals['gross_profit'] / totals['gross_loss']) if totals['gross_loss'] > 0 else 999

    try:
        years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    except Exception:
        years = 1
    years = max(years, 0.01)

    final = initial_capital + totals['net_profit']
    cagr = ((final / initial_capital) ** (1 / years) - 1) * 100 if final > 0 else 0

    sharpe = 0
    if len(totals['all_pnl']) > 1:
        arr = np.array(totals['all_pnl']) / 100
        tpy = len(arr) / years
        if np.std(arr) > 0 and tpy > 0:
            sharpe = (np.mean(arr) * tpy) / (np.std(arr) * np.sqrt(tpy))

    return {
        'Market': '*** TOTAL ***', 'Trades': t, 'Missed': totals['missed'],
        'Win Rate %': round(wr, 1), 'Avg Days': 0,
        'Return %': round(ret, 2), 'CAGR %': round(cagr, 2),
        'Max DD %': round(totals['max_dd'], 2), 'Sharpe': round(sharpe, 2),
        'Profit Factor': round(min(pf, 999), 2),
        'Net Profit': round(totals['net_profit'], 2),
    }


def _empty_result(capital):
    return {
        'trades': pd.DataFrame(), 'missed_trades': pd.DataFrame(),
        'equity_curve': [capital], 'final_capital': capital, 'total_return': 0,
    }
