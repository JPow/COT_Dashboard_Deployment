"""
Back-adjusted continuous futures builder with expiry-buffer or liquidity rolls.

Panama (difference) adjustment is unchanged across roll modes; only the handoff
schedule differs.

roll_mode:
  - 'expiry_buffer': roll each contract `roll_buffer_business_days` before expiry
  - 'liquidity': stay on the highest trailing TRADES volume contract, with
    hysteresis (confirm_days) and a hard filter that rejects vol=0 stubs when
    another contract traded that day.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any


# ---------------------------------------------------------------------------
# Bar helpers (duck-typed: IB BarData or SimpleNamespace/dict-like)
# ---------------------------------------------------------------------------

def _bday_offset(d: date, offset_days: int) -> date:
    if offset_days == 0:
        return d
    sign = 1 if offset_days > 0 else -1
    n = abs(offset_days)
    cur = d
    while n > 0:
        cur = cur + timedelta(days=sign)
        if cur.weekday() < 5:
            n -= 1
    return cur


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, 'date') and callable(value.date):
        try:
            out = value.date()
            if isinstance(out, date):
                return out
        except TypeError:
            pass
    s = str(value)[:10]
    return date.fromisoformat(s)


def bar_date(b) -> date:
    if isinstance(b, dict):
        return _as_date(b['date'] if 'date' in b else b.get('Date'))
    bd = getattr(b, 'date', None)
    if bd is None and isinstance(b, dict):
        bd = b.get('date') or b.get('Date')
    return _as_date(bd)


def bar_attr(b, name: str, default=None):
    if isinstance(b, dict):
        return b.get(name, b.get(name.capitalize(), default))
    return getattr(b, name, default)


def last_close_on(bars, target_date: date):
    on_day = [b for b in bars if bar_date(b) == target_date]
    if not on_day:
        return None
    return float(bar_attr(on_day[-1], 'close'))


def bar_on(bars, target_date: date):
    on_day = [b for b in bars if bar_date(b) == target_date]
    return on_day[-1] if on_day else None


def make_bar(d, open_, high, low, close, volume=0):
    return SimpleNamespace(
        date=d, open=open_, high=high, low=low, close=close, volume=volume,
    )


# ---------------------------------------------------------------------------
# Liquidity scoring / schedule
# ---------------------------------------------------------------------------

def score_contract_volume(bars, asof_date: date, lookback: int = 10) -> float:
    """Sum of daily volume over the last `lookback` sessions on/before asof_date."""
    prior = [b for b in bars if bar_date(b) <= asof_date]
    if not prior:
        return 0.0
    prior.sort(key=bar_date)
    window = prior[-lookback:]
    return float(sum((bar_attr(b, 'volume') or 0) for b in window))


def _session_dates(contract_bars: dict) -> list[date]:
    dates = set()
    for _, _, bars in contract_bars.values():
        for b in bars:
            dates.add(bar_date(b))
    return sorted(d for d in dates if d.weekday() < 5)


def build_liquidity_roll_schedule(
    contract_bars: dict,
    *,
    lookback: int = 10,
    confirm_days: int = 3,
    start_date: date | None = None,
    forward_window_days: int = 180,
) -> tuple[dict[date, str], list[dict]]:
    """Walk the calendar and pick the active contract by trailing volume.

    Parameters
    ----------
    contract_bars : dict
        local_symbol -> (contract_or_None, expiry_date, bars)
    lookback : int
        Trailing sessions for volume score (default 10).
    confirm_days : int
        Challenger must beat the active contract for this many consecutive
        sessions before a roll (default 3). Stub escape (vol=0 while another
        traded) switches immediately.
    start_date : optional floor for schedule dates.
    forward_window_days : int
        Ignore contracts expiring more than this many days after as-of
        (default 180, matching find_liquid_front_contract).

    Returns
    -------
    active_by_date : {date: local_symbol}
    roll_events : list of {from, to, actual_roll, reason, from_vol, to_vol}
    """
    locals_by_exp = sorted(
        [ls for ls, (_, _, bars) in contract_bars.items() if bars],
        key=lambda ls: contract_bars[ls][1],
    )
    if not locals_by_exp:
        return {}, []

    sessions = _session_dates(contract_bars)
    if start_date is not None:
        sessions = [d for d in sessions if d >= start_date]
    if not sessions:
        return {}, []

    exp_of = {ls: contract_bars[ls][1] for ls in locals_by_exp}
    bars_of = {ls: contract_bars[ls][2] for ls in locals_by_exp}

    def day_volume(ls: str, d: date) -> float:
        b = bar_on(bars_of[ls], d)
        if b is None:
            return 0.0
        return float(bar_attr(b, 'volume') or 0)

    def eligible(ls: str, d: date) -> bool:
        # Must not be expired, and not deeper than the forward window.
        if exp_of[ls] < d:
            return False
        if exp_of[ls] > d + timedelta(days=forward_window_days):
            return False
        # Must have at least one bar on or before d (contract existed in feed).
        return any(bar_date(b) <= d for b in bars_of[ls])

    active = None
    challenger = None
    challenge_streak = 0
    active_by_date: dict[date, str] = {}
    roll_events: list[dict] = []

    for d in sessions:
        cands = [ls for ls in locals_by_exp if eligible(ls, d)]
        if not cands:
            continue

        scores = {ls: score_contract_volume(bars_of[ls], d, lookback) for ls in cands}
        # Prefer higher volume; tie-break nearer expiry.
        ranked = sorted(cands, key=lambda ls: (-scores[ls], exp_of[ls]))
        best = ranked[0]

        # If prior active expired / left the candidate set, seed from best.
        if active is not None and active not in cands:
            roll_events.append({
                'from': active,
                'to': best,
                'actual_roll': d,
                'reason': 'expired_active',
                'from_vol': 0.0,
                'to_vol': scores[best],
            })
            active = best
            challenger = None
            challenge_streak = 0
            active_by_date[d] = active
            continue

        # Hard filter: never stay on a vol=0 stub if a later/equal-expiry
        # contract traded that day. Never roll backward to an earlier expiry.
        traded = [ls for ls in cands if day_volume(ls, d) > 0]
        if traded and active is not None:
            forward_traded = [
                ls for ls in traded if exp_of[ls] >= exp_of[active]
            ]
            pool = forward_traded or (
                traded if active not in cands else []
            )
            if pool:
                traded_ranked = sorted(
                    pool, key=lambda ls: (-scores[ls], exp_of[ls])
                )
                active_vol_today = day_volume(active, d)
                if active_vol_today == 0 and active not in pool:
                    if traded_ranked[0] != active:
                        roll_events.append({
                            'from': active,
                            'to': traded_ranked[0],
                            'actual_roll': d,
                            'reason': 'stub_escape',
                            'from_vol': scores.get(active, 0.0),
                            'to_vol': scores[traded_ranked[0]],
                        })
                    active = traded_ranked[0]
                    challenger = None
                    challenge_streak = 0
                    active_by_date[d] = active
                    continue
            if day_volume(best, d) == 0 and forward_traded:
                best = sorted(
                    forward_traded, key=lambda ls: (-scores[ls], exp_of[ls])
                )[0]
        elif traded and active is None:
            traded_ranked = sorted(traded, key=lambda ls: (-scores[ls], exp_of[ls]))
            active = traded_ranked[0]
            active_by_date[d] = active
            continue

        if active is None:
            active = best
            active_by_date[d] = active
            continue

        if best == active:
            challenger = None
            challenge_streak = 0
            active_by_date[d] = active
            continue

        # Only roll forward to a later (or equal) expiry — never reverse.
        if exp_of[best] < exp_of[active]:
            challenger = None
            challenge_streak = 0
            active_by_date[d] = active
            continue

        if challenger == best:
            challenge_streak += 1
        else:
            challenger = best
            challenge_streak = 1

        if challenge_streak >= confirm_days:
            roll_events.append({
                'from': active,
                'to': best,
                'actual_roll': d,
                'reason': 'volume_handoff',
                'from_vol': scores.get(active, 0.0),
                'to_vol': scores[best],
            })
            active = best
            challenger = None
            challenge_streak = 0

        active_by_date[d] = active

    return active_by_date, roll_events


def _panama_offsets_from_roll_events(
    contract_bars: dict,
    roll_events: list[dict],
    newest_local: str,
) -> tuple[dict[str, float], list[dict]]:
    """Accumulate Panama offsets newest -> oldest along liquidity roll events."""
    offset_per_local = {newest_local: 0.0}
    roll_log: list[dict] = []

    # Events are chronological (old->new). Process reverse for offsets.
    for ev in reversed(roll_events):
        cur_local = ev['from']
        nxt_local = ev['to']
        actual = ev['actual_roll']
        cur_bars = contract_bars[cur_local][2]
        nxt_bars = contract_bars[nxt_local][2]

        c_close = last_close_on(cur_bars, actual)
        n_close = last_close_on(nxt_bars, actual)
        if c_close is None or n_close is None:
            # Walk back up to 15 BD for overlap
            found = None
            for k in range(15):
                cand = _bday_offset(actual, -k)
                cc = last_close_on(cur_bars, cand)
                nc = last_close_on(nxt_bars, cand)
                if cc is not None and nc is not None:
                    found = (cand, cc, nc)
                    break
            if found is None:
                offset_per_local[cur_local] = offset_per_local.get(nxt_local, 0.0)
                roll_log.append({
                    'from': cur_local, 'to': nxt_local,
                    'nominal_roll': actual, 'actual_roll': None,
                    'gap': 0.0,
                    'note': 'no overlap; offset propagated',
                    'reason': ev.get('reason'),
                })
                continue
            actual, c_close, n_close = found

        gap = n_close - c_close
        offset_per_local[cur_local] = offset_per_local.get(nxt_local, 0.0) + gap
        roll_log.append({
            'from': cur_local, 'to': nxt_local,
            'nominal_roll': actual, 'actual_roll': actual,
            'gap': gap,
            'reason': ev.get('reason'),
            'from_vol': ev.get('from_vol'),
            'to_vol': ev.get('to_vol'),
        })

    return offset_per_local, roll_log


def stitch_from_liquidity_schedule(
    contract_bars: dict,
    active_by_date: dict[date, str],
    roll_events: list[dict],
    *,
    start_date: date | None = None,
) -> tuple[list[dict], list[dict]]:
    """Apply Panama offsets and emit continuous bars for liquidity schedule."""
    if not active_by_date:
        return [], []

    # Newest active contract = last date's active
    last_d = max(active_by_date)
    newest_local = active_by_date[last_d]
    offset_per_local, roll_log = _panama_offsets_from_roll_events(
        contract_bars, roll_events, newest_local,
    )

    # Any contract that appeared but has no offset yet inherits from next event
    for ls in {active_by_date[d] for d in active_by_date}:
        offset_per_local.setdefault(ls, 0.0)

    out = []
    for d in sorted(active_by_date):
        if start_date is not None and d < start_date:
            continue
        local = active_by_date[d]
        b = bar_on(contract_bars[local][2], d)
        if b is None:
            continue
        o = float(bar_attr(b, 'open'))
        h = float(bar_attr(b, 'high'))
        l = float(bar_attr(b, 'low'))
        c = float(bar_attr(b, 'close'))
        vol = float(bar_attr(b, 'volume') or 0)
        # Drop IB settlement stubs (no trades) from the continuous series.
        if vol == 0 and o == h == l == c:
            continue
        offset = offset_per_local.get(local, 0.0)
        raw_date = b.date if not isinstance(b, dict) else b.get('date', d)
        out.append({
            'date': raw_date,
            'open': o + offset,
            'high': h + offset,
            'low': l + offset,
            'close': c + offset,
            'volume': vol,
            'contract_local_symbol': local,
        })

    out.sort(key=lambda x: bar_date(SimpleNamespace(date=x['date'])))
    return out, roll_log


def stitch_expiry_buffer(
    contract_bars: dict,
    *,
    start_date: date,
    roll_buffer_business_days: int = 5,
    today: date | None = None,
) -> tuple[list[dict], list[dict]]:
    """Original expiry−N BD Panama stitch."""
    today = today or date.today()
    sorted_locals = sorted(
        [ls for ls, (_, _, bars) in contract_bars.items() if bars],
        key=lambda ls: contract_bars[ls][1],
    )
    if not sorted_locals:
        return [], []

    offset_per_local = {sorted_locals[-1]: 0.0}
    roll_log = []

    for i in range(len(sorted_locals) - 2, -1, -1):
        cur_local = sorted_locals[i]
        nxt_local = sorted_locals[i + 1]
        _, cur_exp, cur_bars = contract_bars[cur_local]
        _, _, nxt_bars = contract_bars[nxt_local]

        nominal = _bday_offset(cur_exp, -roll_buffer_business_days)
        if nominal > today:
            offset_per_local[cur_local] = offset_per_local[nxt_local]
            roll_log.append({
                'from': cur_local, 'to': nxt_local,
                'nominal_roll': nominal, 'actual_roll': None,
                'gap': 0.0, 'note': 'nominal in future; skipped',
            })
            continue

        actual = None
        for k in range(15):
            cand = _bday_offset(nominal, -k)
            if (last_close_on(cur_bars, cand) is not None
                    and last_close_on(nxt_bars, cand) is not None):
                actual = cand
                break

        if actual is None:
            cur_dates = {bar_date(b) for b in cur_bars}
            nxt_dates = {bar_date(b) for b in nxt_bars}
            common = sorted(cur_dates & nxt_dates)
            if not common:
                offset_per_local[cur_local] = offset_per_local[nxt_local]
                roll_log.append({
                    'from': cur_local, 'to': nxt_local,
                    'nominal_roll': nominal, 'actual_roll': None,
                    'gap': 0.0, 'note': 'no overlap; offset propagated',
                })
                continue
            actual = (
                max(d for d in common if d <= nominal)
                if any(d <= nominal for d in common)
                else common[-1]
            )

        gap = last_close_on(nxt_bars, actual) - last_close_on(cur_bars, actual)
        offset_per_local[cur_local] = offset_per_local[nxt_local] + gap
        roll_log.append({
            'from': cur_local, 'to': nxt_local,
            'nominal_roll': nominal, 'actual_roll': actual, 'gap': gap,
        })

    locals_to_roll = {entry['from']: entry['actual_roll'] for entry in roll_log}
    out = []
    prev_roll = None
    for local in sorted_locals:
        _, _, bars = contract_bars[local]
        my_roll = locals_to_roll.get(local)
        offset = offset_per_local[local]
        for b in bars:
            d = bar_date(b)
            if prev_roll is not None and d <= prev_roll:
                continue
            if my_roll is not None and d > my_roll:
                continue
            if d < start_date:
                continue
            out.append({
                'date': b.date if not isinstance(b, dict) else b.get('date', d),
                'open': float(bar_attr(b, 'open')) + offset,
                'high': float(bar_attr(b, 'high')) + offset,
                'low': float(bar_attr(b, 'low')) + offset,
                'close': float(bar_attr(b, 'close')) + offset,
                'volume': bar_attr(b, 'volume') or 0,
                'contract_local_symbol': local,
            })
        if my_roll is not None:
            prev_roll = my_roll

    out.sort(key=lambda x: _as_date(x['date']))
    seen_keys = set()
    deduped = []
    for r in out:
        k = (_as_date(r['date']), r['contract_local_symbol'])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        deduped.append(r)
    return deduped, roll_log


def stitch_contract_bars(
    contract_bars: dict,
    *,
    start_date: date,
    roll_mode: str = 'expiry_buffer',
    roll_buffer_business_days: int = 5,
    lookback: int = 10,
    confirm_days: int = 3,
    forward_window_days: int = 180,
    today: date | None = None,
) -> tuple[list[dict], list[dict], str]:
    """Stitch pre-fetched per-contract bars into a continuous series.

    Returns (bars, roll_log, status).
    """
    if roll_mode == 'expiry_buffer':
        bars, roll_log = stitch_expiry_buffer(
            contract_bars,
            start_date=start_date,
            roll_buffer_business_days=roll_buffer_business_days,
            today=today,
        )
        return bars, roll_log, 'ok'

    if roll_mode == 'liquidity':
        active, events = build_liquidity_roll_schedule(
            contract_bars,
            lookback=lookback,
            confirm_days=confirm_days,
            start_date=start_date,
            forward_window_days=forward_window_days,
        )
        bars, roll_log = stitch_from_liquidity_schedule(
            contract_bars, active, events, start_date=start_date,
        )
        for entry in roll_log:
            entry.setdefault('roll_mode', 'liquidity')
            entry.setdefault('lookback', lookback)
            entry.setdefault('confirm_days', confirm_days)
            entry.setdefault('forward_window_days', forward_window_days)
        return bars, roll_log, 'ok'

    return [], [], f'unknown roll_mode: {roll_mode}'


# ---------------------------------------------------------------------------
# Quality metrics (offline validation)
# ---------------------------------------------------------------------------

def series_quality_metrics(rows: list[dict]) -> dict[str, Any]:
    """Compute flat/vol0 stats for a continuous series of bar dicts."""
    n = len(rows)
    if n == 0:
        return {'n': 0, 'flat_pct': 0.0, 'vol0_pct': 0.0, 'n_contracts': 0}

    def _o(r): return float(r.get('open', r.get('Open', 0)))
    def _h(r): return float(r.get('high', r.get('High', 0)))
    def _l(r): return float(r.get('low', r.get('Low', 0)))
    def _c(r): return float(r.get('close', r.get('Close', 0)))
    def _v(r): return float(r.get('volume', r.get('Volume', 0)) or 0)
    def _ctr(r): return r.get('contract_local_symbol') or r.get('Contract') or ''

    flat = sum(1 for r in rows if _o(r) == _h(r) == _l(r) == _c(r))
    v0 = sum(1 for r in rows if _v(r) == 0)
    contracts = []
    prev = None
    for r in sorted(rows, key=lambda x: _as_date(x.get('date') or x.get('Date'))):
        c = _ctr(r)
        if c and c != prev:
            contracts.append(c)
            prev = c

    max_abs_ret = 0.0
    sorted_rows = sorted(rows, key=lambda x: _as_date(x.get('date') or x.get('Date')))
    for a, b in zip(sorted_rows, sorted_rows[1:]):
        ca, cb = _c(a), _c(b)
        if ca:
            max_abs_ret = max(max_abs_ret, abs(cb / ca - 1.0) * 100.0)

    return {
        'n': n,
        'flat_pct': round(100.0 * flat / n, 2),
        'vol0_pct': round(100.0 * v0 / n, 2),
        'n_contracts': len(contracts),
        'contract_path': contracts,
        'max_abs_day_ret_pct': round(max_abs_ret, 2),
    }
