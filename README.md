# COT Dashboard Deployment

A web-based dashboard for visualizing Commitment of Traders (COT) data, featuring interactive charts and real-time updates.

## Features

- Interactive price and open interest analysis
- Retail vs Commercial positioning visualization
- Open Interest Index tracking
- Mobile-responsive design
- Real-time data updates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JPow/COT_Dashboard_Deployment.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the COT dashboard:
```bash
python app.py
```

2. Access the dashboard at `http://localhost:8050`

## IB daily price cache (volume-based front)

The notebook `COT IBRK Data Grabber.ipynb` builds `ib_daily_cache.json` using the **most-traded** listed contract (summed daily volume over the last 10 sessions) as the front cap, then stitches historical expiries for a Panama back-adjusted series.

After changing that logic, do a **one-time full rebuild** with IB Gateway connected: in the daily-cache cell, set `FORCE_FULL_REBUILD = True`, run the cell once, then set it back to `False`.

## Backtest Dashboards

### Unified Strategy Backtest (`unified_backtest_app.py`)

A mix-and-match Dash app for backtesting any combination of setup, entry, and stop strategy across all COT markets. Runs on port 8054.

- **Setups:** Narrowing Range (NR3), Inside Days, COT+RSI Extremes
- **Entries:** ORB Breakout (30m / 60m), Daily Breakout, Market-on-Close
- **Stops:** Two-Phase ATR Trail (OR-width initial stop → breakeven → ATR trail), ATR Stop+Target
- **Optional filters:** COT level 70/30, COT direction (WoW), COT ROC, RSI extremes
- Configurable capital, risk %, date range, and ATR period
- Summary table across all markets with drill-down into per-market charts, equity curves, and trade logs

```bash
python unified_backtest_app.py
```

### ORB Narrowing Range Backtest (`ORB_backtest.py`)

Dedicated Dash app for the narrowing-range setup + true opening-range breakout entry. Runs on port 8053.

- **Setup:** N consecutive narrowing daily ranges (or NR2: two narrowest of 20 days)
- **Opening range:** High/low of intraday bars from `rth_open` → `30_close` or `60_close` ET per [`ORB_contract_specs.json`](ORB_contract_specs.json)
- **Entry:** First bar **after** the OR window that breaks OR high/low; fill at breakout level ± 2 ticks (market-specific)
- **Stop:** Opposite side of today's opening range ± 1 tick (market-specific), then two-phase ATR trail (breakeven at 1:1, slow ATR × mult)
- **30m / 60m toggle:** Changing the Opening Range dropdown re-runs the full backtest across all markets
- Intraday data from `ORB_intraday_data.json` (built by `COT IBRK Data Grabber.ipynb`); timestamps stored UTC-naive, converted to ET in code

```bash
python ORB_backtest.py
```

### Trend-Following Breakout Backtest (`tf_backtest_app.py`)

A dedicated Dash app for the N-day high/low breakout strategy with realistic transaction costs. Runs on port 8055.

- **Strategy:** Long when High breaks above the prior N-day highest high; Short when Low breaks below the prior N-day lowest low. Entry at the breakout level.
- **Stop:** 2×ATR from entry → breakeven at 1:1 R/R → trailing 2×ATR
- **Costs:** $10 commission per trade + 2-tick adverse slippage on entry (market-specific tick sizes for all 48 futures)
- **Capital:** $30,000 per market, 1% risk per trade
- Lookback period is adjustable (5–200 days)
- Cost impact summary cards (total commission, slippage, gross vs net PnL)
- Per-market detail with candlestick chart showing N-day bands, ATR subplot, equity curve, and trade log

```bash
python tf_backtest_app.py
```

## Statistical Tests (`tests/`)

### COT Hypothesis Test — ORB NR3 (`hypothesis_test.py` / `COT_Hypothesis_Test.ipynb`)

Tests whether any COT filter configuration improves the NR3 + Opening Range Breakout base strategy. Uses Hansen's Superior Predictive Ability (SPA) test with walk-forward out-of-sample windows to correct for multiple comparisons across 49 COT filter permutations.

**Conclusion:** COT filters do not significantly improve ORB NR3.

### COT Hypothesis Test — Trend Following (`trend_following_test.py` / `COT_TrendFollowing_Test.ipynb`)

Same SPA framework applied to a diversified N-day breakout portfolio (lookbacks 5–100 in steps of 5). Tests 49 COT filter permutations against the unfiltered baseline across all markets.

**Conclusion:** COT filters do not significantly improve trend-following breakouts.

### Lookback Robustness Analysis (`lookback_robustness.py` / `Lookback_Robustness.ipynb`)

Maps the full performance surface across lookbacks 5–100 (step 1) with block-bootstrap 95% confidence intervals, identifies contiguous robust ranges where the lower CI of CAGR exceeds 20%, and validates them via 4-window walk-forward testing.

Three portfolio approaches are compared out-of-sample:
- **(A) Robust Range** — equal-weight lookbacks inside the identified range
- **(B) Best Single** — highest in-sample Sharpe lookback
- **(C) All Lookbacks** — equal-weight across all 96 lookbacks

**Finding:** The 5–55 day breakout range is structurally stable across all training windows and consistently profitable out-of-sample, though returns should be discounted for transaction costs.

## Dependencies

- dash==2.16.1
- cot_reports==0.1.3
- pandas==2.1.4
- plotly==5.9.0
- dash-bootstrap-components==1.6.0
- numpy==1.23.5
- gunicorn==21.2.0

## Deployment

The dashboard is deployed using Render. Access it at: [Your Render URL]
Remember to update the price date for the most recent price data. 

## License

GPL-3.0 license
