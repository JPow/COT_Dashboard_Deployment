#!/usr/bin/env python3
"""Print intraday cache date-range coverage for 30m / 60m ORB backtests."""

from __future__ import annotations

import argparse

import pandas as pd

from backtest_engine.data import (
    INTRADAY_CACHE_FILE,
    INTRADAY_STATE_FILE,
    load_intraday_cache,
    load_intraday_state,
    summarize_intraday_ranges,
)


def _print_summary(summary, interval: str, trading_days_target: int) -> None:
    sub = summary[summary['interval'] == interval]
    if sub.empty:
        print(f'\n## {interval} — no data')
        return

    firsts = sub['first']
    lasts = sub['last']
    print(f'\n## {interval} — {len(sub)} markets')
    print(f'  Earliest start:  {firsts.min().date()}')
    print(f'  Latest start:    {firsts.max().date()}')
    print(f'  Median start:    {firsts.median().date()}')
    print(f'  Latest end:      {lasts.max().date()}')
    print(f'  Median span:     {sub["calendar_days"].median():.0f} cal days '
          f'(~{sub["est_trading_days"].median():.0f} trading days)')
    print(f'  Total bars:      {sub["bars"].sum():,}')

    cal_backfill = int(trading_days_target * 7 / 5) + 5
    target = firsts.median() - pd.Timedelta(days=cal_backfill)
    print(f'  +{trading_days_target} TD backfill target (median): ~{target.date()}')

    print('\n  Narrowest coverage (latest starts):')
    for _, row in sub.nlargest(5, 'first').iterrows():
        print(f'    {row["first"].date()}  {row["market"][:55]}  '
              f'({row["bars"]} bars)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default=INTRADAY_CACHE_FILE)
    parser.add_argument('--state', default=INTRADAY_STATE_FILE)
    parser.add_argument('--trading-days', type=int, default=30,
                        help='Backfill horizon to report (default: 30 trading days)')
    parser.add_argument('--csv', metavar='PATH', help='Write per-market summary to CSV')
    args = parser.parse_args()

    cache = load_intraday_cache(args.cache)
    state = load_intraday_state(args.state)
    summary = summarize_intraday_ranges(cache, state)

    print('=' * 72)
    print('INTRADAY CACHE COVERAGE')
    print(f'  cache: {args.cache}  ({len(cache):,} rows)' if not cache.empty else
          f'  cache: {args.cache}  (empty / missing)')
    print('=' * 72)

    for interval in ('30m', '60m'):
        _print_summary(summary, interval, args.trading_days)

    if args.csv and not summary.empty:
        out = summary.copy()
        out['first'] = out['first'].dt.strftime('%Y-%m-%d %H:%M:%S')
        out['last'] = out['last'].dt.strftime('%Y-%m-%d %H:%M:%S')
        out.to_csv(args.csv, index=False)
        print(f'\nWrote {args.csv}')


if __name__ == '__main__':
    main()
