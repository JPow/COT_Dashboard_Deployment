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

1. Run the multi-page COT app (dashboard, backtest, sizing):
```bash
python -m app.main
```

2. Access at `http://localhost:8050` (`/` dashboard, `/backtest` unified engine, `/sizing` position tool)

Legacy single-file entry: `python app.py` (redirects to the same app).

## Data source

All prices and COT fields for the dashboard and backtests come from **`COT IBRK Data Grabber.ipynb`** (Interactive Brokers Gateway + `cot_reports`). There is no Yahoo Finance pipeline. After updating data locally, run `python -m pipeline.publish` then commit and push `data/cot_data.json` (and other caches if changed) so Render serves current prices.

## Intraday ORB cache (30m / 60m archive)

`data/ORB_intraday_data.json` is the persistent intraday database for ORB backtests. It is built once and extended over time — never wiped on routine updates.

**Routine forward updates** — run the *Build Intraday Cache* cell in `COT IBRK Data Grabber.ipynb` with IB Gateway connected. This incrementally appends new bars when the front contract has not rolled.

**Extend history backward (+30 trading days)** — IB only returns ~30 calendar days of 30m/60m bars per request. To add older months (e.g. March when the cache starts in April), run the backfill script once per month while Gateway is connected:

```bash
python -m pipeline.ib_intraday_backfill --trading-days 30
# or one market:
python -m pipeline.ib_intraday_backfill --market "GOLD - COMMODITY EXCHANGE INC." --dry-run
# shim still works: python ib_intraday_backfill.py ...
```

The backfill queries expired contracts with `endDateTime` anchored to the gap, Panama-adjusts, and **prepends** bars without replacing data already in the archive. Progress is tracked in `data/ORB_intraday_roll_state.json` (`backfill_target`, `last_backfill_at`).

On full intraday rebuilds (contract roll), keep archived bars older than the rebuild window instead of deleting them — use `merge_intraday_rebuild()` from `ib_intraday_backfill.py` in the notebook cell.

## IB daily price cache (continuous futures)

The notebook `COT IBRK Data Grabber.ipynb` builds and maintains these **data** files:

| File | Role |
|---|---|
| `data/ib_daily_cache.json` | Panama back-adjusted daily OHLC per market (what backtests / `cot_data.json` consume) |
| `data/ib_daily_roll_state.json` | Per-market rebuild metadata (roll dates, gaps, `roll_mode`, etc.) |
| `data/cot_data.json` | Weekly COT + daily prices merged for the dashboard / strategies |
| `data/ORB_intraday_data.json` | 30m / 60m intraday archive for ORB backtests |
| `data/ORB_contract_specs.json` | Tick size, point value, session times |

### `liquidity_roll_cohort.json` — config, not a cache

**This file does not replace any of the JSON caches above.** It is a small allow-list of COT market names that should use **liquidity-based rolls** (`roll_mode='liquidity'`) instead of the default expiry−5BD calendar roll when the daily cache is fully rebuilt.

- **What it contains:** markets whose expiry-buffer series had high flat/vol0 rates (thin inter-months), plus validated PL/PA.
- **How it is used:** the Grabber daily-cache cell loads `LIQUIDITY_ROLL_MARKETS` from this file (fallback: platinum + palladium only). Markets **not** in the list keep `roll_mode='expiry_buffer'`.
- **When prices change:** only after an IB rebuild (or `liquidity_roll_pilot.py --promote` for validated pilots). Editing the cohort alone does not rewrite `ib_daily_cache.json`.

Shared stitch logic lives in [`pipeline/ib_continuous.py`](pipeline/ib_continuous.py) (root `ib_continuous.py` is a shim). After updating caches locally, run `python -m pipeline.publish` for git push reminders. Pilot / validate / promote / rank:

```bash
# use orklys_env
/Users/Work/NoteBooks/orklys_env/bin/python liquidity_roll_pilot.py --market platinum
/Users/Work/NoteBooks/orklys_env/bin/python liquidity_roll_pilot.py --rank-only   # refreshes ranking + suggests cohort
```

After changing roll logic, do a **one-time full rebuild** with IB Gateway connected: in the daily-cache cell, set `FORCE_FULL_REBUILD = True`, run once, then set it back to `False`. Markets in the cohort are rebuilt with liquidity rolls; others stay on expiry-buffer.

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

The dashboard is deployed using Render (`gunicorn` via `start.sh`). **`cot_data.json` is not updated by CI** — refresh it by running the Grabber with IB Gateway, then push the updated JSON to the branch Render builds from (typically `main` or your deploy branch).

## License

GPL-3.0 license
