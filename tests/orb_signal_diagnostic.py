"""Compare legacy (yesterday daily H/L + first bar) vs true ORB signal counts.

Run from repo root:
    python tests/orb_signal_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine.data import (
    get_contract_spec,
    get_first_candle_per_day,
    get_intraday_for_symbol,
    load_cot_data,
    load_intraday_cache,
    prepare_base_data,
)
from backtest_engine.entries import apply_orb_breakout
from backtest_engine.indicators import add_standard_indicators, calculate_narrowing_ranges

START = '2023-01-01'
END = '2026-06-05'
N_NARROWING = 3


def _add_setups(df, n_days=N_NARROWING):
    out = df.copy()
    out['setup'] = False
    for i in range(1, len(out)):
        if out.iloc[i - 1].get('consecutive_narrowing', 0) >= n_days:
            out.iloc[i, out.columns.get_loc('setup')] = True
    return out


def _legacy_orb_signals(df, market_name, or_type, intraday_cache):
    """Previous logic: first calendar bar vs yesterday's daily high/low."""
    out = df.copy()
    out['OR_High'] = out['High'].shift(1)
    out['OR_Low'] = out['Low'].shift(1)
    out['signal'] = 0

    interval = '30m' if or_type == '30m' else '60m'
    intraday = get_intraday_for_symbol(market_name, interval=interval, cache_df=intraday_cache)
    first_candles = get_first_candle_per_day(intraday) if not intraday.empty else {}

    for i in range(1, len(out)):
        row = out.iloc[i]
        if not row.get('setup', False):
            continue
        or_high = row.get('OR_High', np.nan)
        or_low = row.get('OR_Low', np.nan)
        if pd.isna(or_high) or pd.isna(or_low) or (or_high - or_low) <= 0:
            continue
        candle = first_candles.get(pd.Timestamp(row['Date'].date()), {})
        ch = candle.get('high', np.nan)
        cl = candle.get('low', np.nan)
        if pd.isna(ch) or pd.isna(cl):
            continue
        long_bo = ch > or_high
        short_bo = cl < or_low
        if long_bo or short_bo:
            out.iloc[i, out.columns.get_loc('signal')] = 1 if long_bo else -1
    return out


def prepare_market(cot_df, market):
    df = prepare_base_data(cot_df, market)
    if df.empty:
        return df
    df = add_standard_indicators(df)
    df = calculate_narrowing_ranges(df)
    if START:
        df = df[df['Date'] >= pd.Timestamp(START)]
    if END:
        df = df[df['Date'] <= pd.Timestamp(END)]
    return df.reset_index(drop=True)


def main():
    cot = load_cot_data()
    cache = load_intraday_cache()
    markets = sorted(cot[cot['data_type'] == 'daily_price']['Market'].unique())
    rows = []

    for market in markets:
        data = prepare_market(cot, market)
        if data.empty:
            continue
        data = _add_setups(data)
        setups = int(data['setup'].sum())
        has_spec = get_contract_spec(market) is not None

        for or_type in ('30m', '60m'):
            legacy = _legacy_orb_signals(data, market, or_type, cache)
            new = apply_orb_breakout(
                data, market_name=market, or_type=or_type, intraday_cache=cache,
            )
            rows.append({
                'market': market[:42],
                'or_type': or_type,
                'setups': setups,
                'legacy_signals': int((legacy['signal'] != 0).sum()),
                'new_signals': int((new['signal'] != 0).sum()),
                'delta': int((new['signal'] != 0).sum()) - int((legacy['signal'] != 0).sum()),
                'has_spec': has_spec,
            })

    df = pd.DataFrame(rows)
    print('=== ORB signal diagnostic: legacy vs true opening-range entry ===')
    print(f'Period: {START} → {END} | narrowing days: {N_NARROWING}')
    print()
    print(df.groupby('or_type')[['legacy_signals', 'new_signals', 'delta']].sum())
    print()

    gap = df[(df['setups'] > 0) & (df['new_signals'] == 0)].sort_values('setups', ascending=False)
    print(f'Markets with setups but zero NEW signals: {len(gap)} rows')
    if not gap.empty:
        print(gap.head(15).to_string(index=False))
    print()

    big_delta = df.reindex(df['delta'].abs().sort_values(ascending=False).index).head(10)
    print('Largest |delta| (new − legacy):')
    print(big_delta[['market', 'or_type', 'legacy_signals', 'new_signals', 'delta']].to_string(index=False))


if __name__ == '__main__':
    main()
