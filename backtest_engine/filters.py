"""Optional signal filters applied during entry (COT, RSI, direction, ROC)."""

import pandas as pd


def apply_entry_filters(row, direction, *,
                        cot_filter=False, cot_long=70, cot_short=30,
                        rsi_filter=False, rsi_long_max=70, rsi_short_min=30,
                        cot_direction_filter=False,
                        cot_roc_filter=False, cot_roc_threshold=10,
                        ma_trend_filter=False, ma_period=0):
    """Return 0 if the trade direction is blocked by any enabled filter."""
    if direction == 0:
        return 0
    if cot_filter and not pd.isna(row.get('Commercial_Index')):
        if direction == 1 and row['Commercial_Index'] < cot_long:
            return 0
        if direction == -1 and row['Commercial_Index'] > cot_short:
            return 0
    if cot_direction_filter and not pd.isna(row.get('COT_Change')):
        if direction == 1 and row['COT_Change'] < 0:
            return 0
        if direction == -1 and row['COT_Change'] > 0:
            return 0
    if cot_roc_filter and not pd.isna(row.get('COT_ROC')):
        if direction == 1 and row['COT_ROC'] < cot_roc_threshold:
            return 0
        if direction == -1 and row['COT_ROC'] > -cot_roc_threshold:
            return 0
    if rsi_filter and not pd.isna(row.get('RSI')):
        if direction == 1 and row['RSI'] >= rsi_long_max:
            return 0
        if direction == -1 and row['RSI'] <= rsi_short_min:
            return 0
    if ma_trend_filter and ma_period > 0:
        ma_col = f'MA_{ma_period}'
        if ma_col not in row.index or pd.isna(row.get(ma_col)):
            return 0
        if direction == 1 and row['Close'] <= row[ma_col]:
            return 0
        if direction == -1 and row['Close'] >= row[ma_col]:
            return 0
    return direction
