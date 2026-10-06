"""
Stop / exit management strategies.

Each strategy is a class that implements:
    initial_stop(direction, entry_price, row, **params) -> stop_price
    update(direction, entry_price, stop_loss, row, bar_idx, entry_idx, **state) -> (stop_loss, exit_reason, exit_price, state)

``update`` is called every bar while in a position.  It returns:
    - new stop_loss level
    - exit_reason (str or None if no exit)
    - exit_price  (float or None)
    - state dict  (carry-forward between bars; e.g. phase)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Two-Phase: Fixed → Breakeven → Trailing ATR
# ---------------------------------------------------------------------------

class TwoPhaseATRStop:
    """Phase 1: fixed stop at opposite OR boundary (OR_Low long, OR_High short).
    Phase 2: after 1:1 R/R, move to breakeven then trail with ATR.
    """

    label = 'Two-Phase ATR Trail'

    def __init__(self, trailing_atr_mult=2.0, **_kw):
        self.trailing_atr_mult = trailing_atr_mult

    def initial_stop(self, direction, entry_price, row, **_kw):
        or_low = row.get('or_low_signal', np.nan)
        or_high = row.get('or_high_signal', np.nan)
        if direction == 1:
            return or_low if not pd.isna(or_low) else entry_price * 0.98
        else:
            return or_high if not pd.isna(or_high) else entry_price * 1.02

    def stop_distance(self, direction, entry_price, row, **_kw):
        or_low = row.get('or_low_signal', np.nan)
        or_high = row.get('or_high_signal', np.nan)
        if not pd.isna(or_high) and not pd.isna(or_low):
            return or_high - or_low
        return abs(entry_price) * 0.02

    def update(self, direction, entry_price, stop_loss, row, bar_idx,
               entry_idx, phase=1, stop_distance=0, **_kw):
        close = row['Close']
        high = row.get('High', close)
        low = row.get('Low', close)
        atr_val = row.get('ATR', np.nan)

        exit_reason = None
        exit_price = None

        if direction == 1 and low <= stop_loss:
            exit_reason = f"Stop (Phase {phase})"
            exit_price = stop_loss
        elif direction == -1 and high >= stop_loss:
            exit_reason = f"Stop (Phase {phase})"
            exit_price = stop_loss

        if exit_reason:
            return stop_loss, exit_reason, exit_price, {'phase': 1}

        # Phase transition: 1 → 2 at 1:1 R/R
        if phase == 1:
            unrealised = (close - entry_price) if direction == 1 else (entry_price - close)
            if stop_distance > 0 and unrealised >= stop_distance:
                phase = 2
                stop_loss = entry_price  # breakeven

        # Trailing in phase 2
        if phase == 2 and not pd.isna(atr_val) and atr_val > 0:
            if direction == 1:
                trail = close - (self.trailing_atr_mult * atr_val)
                stop_loss = max(stop_loss, trail)
            else:
                trail = close + (self.trailing_atr_mult * atr_val)
                stop_loss = min(stop_loss, trail)

        return stop_loss, None, None, {'phase': phase, 'stop_distance': stop_distance}


# ---------------------------------------------------------------------------
# Fixed ATR Stop + ATR Target (COT+RSI style)
# ---------------------------------------------------------------------------

class ATRStopTarget:
    """Fixed stop at N×ATR, take-profit at M×ATR, with max hold."""

    label = 'ATR Stop + Target'

    def __init__(self, atr_stop_mult=2.0, atr_target_mult=3.0,
                 max_hold_days=20, **_kw):
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_hold_days = max_hold_days

    def initial_stop(self, direction, entry_price, row, **_kw):
        atr = row.get('ATR', np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = abs(entry_price) * 0.02
        if direction == 1:
            return entry_price - (self.atr_stop_mult * atr)
        else:
            return entry_price + (self.atr_stop_mult * atr)

    def stop_distance(self, direction, entry_price, row, **_kw):
        atr = row.get('ATR', np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = abs(entry_price) * 0.02
        return self.atr_stop_mult * atr

    def update(self, direction, entry_price, stop_loss, row, bar_idx,
               entry_idx, take_profit=None, **_kw):
        close = row['Close']
        high = row.get('High', close)
        low = row.get('Low', close)
        days_held = bar_idx - entry_idx

        exit_reason = None
        exit_price = None

        if take_profit is None:
            atr = row.get('ATR', np.nan)
            if pd.isna(atr) or atr <= 0:
                atr = abs(entry_price) * 0.02
            if direction == 1:
                take_profit = entry_price + (self.atr_target_mult * atr)
            else:
                take_profit = entry_price - (self.atr_target_mult * atr)

        if direction == 1:
            if low <= stop_loss:
                exit_reason, exit_price = "Stop Loss", stop_loss
            elif high >= take_profit:
                exit_reason, exit_price = "Take Profit", take_profit
            elif days_held >= self.max_hold_days:
                exit_reason, exit_price = "Max Hold", close
        else:
            if high >= stop_loss:
                exit_reason, exit_price = "Stop Loss", stop_loss
            elif low <= take_profit:
                exit_reason, exit_price = "Take Profit", take_profit
            elif days_held >= self.max_hold_days:
                exit_reason, exit_price = "Max Hold", close

        return stop_loss, exit_reason, exit_price, {'take_profit': take_profit}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ATRTwoPhaseStop:
    """N-day breakout style: fixed N×ATR stop → breakeven at 1:1 → trail N×ATR."""

    label = 'ATR Two-Phase (Breakout)'

    def __init__(self, atr_stop_mult=2.0, atr_trail_mult=2.0, stop_buffer=0.01, **_kw):
        self.atr_stop_mult = atr_stop_mult
        self.atr_trail_mult = atr_trail_mult
        self.stop_buffer = stop_buffer

    def initial_stop(self, direction, entry_price, row, **_kw):
        atr = row.get('ATR', np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = abs(entry_price) * 0.02
        if direction == 1:
            return entry_price - self.atr_stop_mult * atr - self.stop_buffer
        return entry_price + self.atr_stop_mult * atr + self.stop_buffer

    def stop_distance(self, direction, entry_price, row, **_kw):
        atr = row.get('ATR', np.nan)
        if pd.isna(atr) or atr <= 0:
            atr = abs(entry_price) * 0.02
        return self.atr_stop_mult * atr + self.stop_buffer

    def update(self, direction, entry_price, stop_loss, row, bar_idx,
               entry_idx, phase=1, stop_distance=0, **_kw):
        close = row['Close']
        high = row.get('High', close)
        low = row.get('Low', close)
        atr = row.get('ATR', np.nan)

        exit_reason = None
        exit_price = None

        if direction == 1 and low <= stop_loss:
            exit_reason = f'Stop (Phase {phase})'
            exit_price = stop_loss
        elif direction == -1 and high >= stop_loss:
            exit_reason = f'Stop (Phase {phase})'
            exit_price = stop_loss

        if exit_reason:
            return stop_loss, exit_reason, exit_price, {'phase': 1}

        if phase == 1:
            unrealised = (close - entry_price) if direction == 1 else (entry_price - close)
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

        return stop_loss, None, None, {'phase': phase, 'stop_distance': stop_distance}


class ATRStopTargetRSIExit(ATRStopTarget):
    """ATR stop + target with RSI mean-reversion exit (COT+RSI strategy)."""

    label = 'ATR Stop + Target + RSI Exit'

    def __init__(self, rsi_exit=60, **kw):
        super().__init__(**kw)
        self.rsi_exit = rsi_exit

    def update(self, direction, entry_price, stop_loss, row, bar_idx,
               entry_idx, take_profit=None, **_kw):
        rsi = row.get('RSI', np.nan)
        close = row['Close']
        days_held = bar_idx - entry_idx
        if direction == 1 and not pd.isna(rsi) and rsi >= self.rsi_exit:
            return stop_loss, 'RSI Exit', close, {'take_profit': take_profit}
        if direction == -1 and not pd.isna(rsi) and rsi <= self.rsi_exit:
            return stop_loss, 'RSI Exit', close, {'take_profit': take_profit}
        return super().update(direction, entry_price, stop_loss, row, bar_idx,
                              entry_idx, take_profit=take_profit, **_kw)


class PrevDayRangeStop(TwoPhaseATRStop):
    """Initial stop at previous day high/low (inside-day style), then two-phase trail."""

    label = 'Prev Day Range + ATR Trail'

    def initial_stop(self, direction, entry_price, row, **_kw):
        or_low = row.get('OR_Low', row.get('or_low_signal', np.nan))
        or_high = row.get('OR_High', row.get('or_high_signal', np.nan))
        if direction == 1 and not pd.isna(or_low):
            return or_low
        if direction == -1 and not pd.isna(or_high):
            return or_high
        return super().initial_stop(direction, entry_price, row)

    def stop_distance(self, direction, entry_price, row, **_kw):
        or_low = row.get('OR_Low', row.get('or_low_signal', np.nan))
        or_high = row.get('OR_High', row.get('or_high_signal', np.nan))
        if not pd.isna(or_high) and not pd.isna(or_low):
            return abs(or_high - or_low)
        return super().stop_distance(direction, entry_price, row)


STOP_REGISTRY = {
    'two_phase_atr': {
        'cls': TwoPhaseATRStop,
        'label': 'Two-Phase ATR Trail',
        'params': {
            'trailing_atr_mult': {'type': float, 'default': 2.0, 'min': 0.5,
                                  'max': 10.0, 'step': 0.5,
                                  'label': 'Trail ATR Mult'},
        },
    },
    'atr_stop_target': {
        'cls': ATRStopTarget,
        'label': 'ATR Stop + Target',
        'params': {
            'atr_stop_mult':   {'type': float, 'default': 2.0, 'min': 0.5,
                                'max': 5.0, 'step': 0.5, 'label': 'Stop ATR Mult'},
            'atr_target_mult': {'type': float, 'default': 3.0, 'min': 1.0,
                                'max': 10.0, 'step': 0.5, 'label': 'Target ATR Mult'},
            'max_hold_days':   {'type': int, 'default': 20, 'min': 5,
                                'max': 60, 'label': 'Max Hold Days'},
        },
    },
    'atr_two_phase': {
        'cls': ATRTwoPhaseStop,
        'label': 'ATR Two-Phase (Breakout)',
        'params': {
            'atr_stop_mult': {'type': float, 'default': 2.0, 'min': 0.5, 'max': 10.0,
                              'step': 0.5, 'label': 'Stop ATR Mult'},
            'atr_trail_mult': {'type': float, 'default': 2.0, 'min': 0.5, 'max': 10.0,
                               'step': 0.5, 'label': 'Trail ATR Mult'},
        },
    },
    'atr_stop_target_rsi': {
        'cls': ATRStopTargetRSIExit,
        'label': 'ATR Stop + Target + RSI Exit',
        'params': {
            'atr_stop_mult': {'type': float, 'default': 2.0, 'min': 0.5, 'max': 5.0, 'step': 0.5,
                              'label': 'Stop ATR Mult'},
            'atr_target_mult': {'type': float, 'default': 3.0, 'min': 1.0, 'max': 10.0, 'step': 0.5,
                                'label': 'Target ATR Mult'},
            'max_hold_days': {'type': int, 'default': 20, 'min': 5, 'max': 60, 'label': 'Max Hold Days'},
            'rsi_exit': {'type': int, 'default': 60, 'min': 50, 'max': 80, 'label': 'RSI Exit'},
        },
    },
    'prev_day_range': {
        'cls': PrevDayRangeStop,
        'label': 'Prev Day Range + ATR Trail',
        'params': {
            'trailing_atr_mult': {'type': float, 'default': 2.0, 'min': 0.5,
                                  'max': 10.0, 'step': 0.5, 'label': 'Trail ATR Mult'},
        },
    },
}
