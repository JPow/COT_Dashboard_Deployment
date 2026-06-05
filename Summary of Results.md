# Summary of Results

## 1. COT Hypothesis Test — ORB NR3

**Files:** `tests/hypothesis_test.py`, `tests/COT_Hypothesis_Test.ipynb`

**Question:** Does any COT Commercial Index filter improve the Narrowing Range 3-day (NR3) + Opening Range Breakout strategy?

**Method:** Hansen's Superior Predictive Ability (SPA) test across 49 COT filter permutations (level thresholds, direction, rate of change), evaluated on 3 walk-forward out-of-sample windows covering Jan 2024 – Apr 2026. 578 OOS trading days, 1000 bootstrap replications.

**Result:** FAIL TO REJECT H0 (p = 0.849). No COT filter significantly improves the base strategy after correcting for multiple comparisons.

| Metric | Base (No COT) | Best COT Variant (ROC >= 20) |
|--------|---------------|------------------------------|
| OOS Sharpe | 0.505 | 0.792 |
| Sharpe Improvement | — | +0.287 |
| SPA p-value | — | 0.849 |
| Statistically Significant | — | No |

**Action:** Remove COT filters from ORB NR3 live trading.

---

## 2. COT Hypothesis Test — Trend-Following Breakout

**Files:** `tests/trend_following_test.py`, `tests/COT_TrendFollowing_Test.ipynb`

**Question:** Does any COT filter improve a diversified N-day breakout portfolio (lookbacks 5–100d, step 5)?

**Method:** Same SPA framework. 49 COT permutations, 20 lookbacks x 48 markets, 3 walk-forward windows. 578 OOS trading days, 1000 bootstrap replications.

**Result:** FAIL TO REJECT H0 (p = 0.876). No COT filter significantly helps trend-following.

| Metric | Base (No COT) | Best COT Variant (Lvl 60/40) |
|--------|---------------|------------------------------|
| OOS Sharpe | 1.501 | 1.820 |
| OOS Max Drawdown | 14.2% | 9.4% |
| Max Consec. Losers | 10 | 11 |
| Total Trades | 15,436 | — |
| Sharpe Improvement | — | +0.320 |
| SPA p-value | — | 0.876 |
| Statistically Significant | — | No |

**Action:** Remove COT filters from trend-following.

---

## 3. Lookback Robustness Analysis (no costs)

**Files:** `tests/lookback_robustness.py`, `tests/Lookback_Robustness.ipynb`

**Question:** Which breakout lookback periods are robust across time, and is the edge real or overfit?

### Why ranges, not a single "best"

A single optimised lookback (e.g., "19 days is best") is almost certainly overfit. A **contiguous range** of lookbacks that all perform well (e.g., 5–55 days) indicates a genuine structural edge — the market has a feature at that timescale that is not sensitive to the exact parameter. Isolated spikes that don't hold for neighbouring values are red flags for curve-fitting.

### Method

**Part 1 — Performance surface.** Every integer lookback from 5 to 100 (96 values) across all 48 markets on the full date range (2022-01 to 2026-04), ~4,600 backtests. For each lookback: Sharpe, CAGR, max drawdown, win rate, trade count, profit factor, plus **block-bootstrap 95% CIs** for Sharpe and CAGR (1,000 resamples; block bootstrap preserves the serial correlation present in financial returns). A lookback is **viable** if its lower 95% CI for CAGR exceeds 20%.

**Part 2 — Robust range identification.** A robust range is a contiguous stretch of at least 5 consecutive viable lookbacks. This rules out isolated spikes. Narrow viable points surrounded by non-viable neighbours are flagged as potential overfitting.

**Part 3 — Walk-forward validation.** 4 expanding training windows, each using all data up to a cutoff, tested on the unseen period that follows:

| Window | Train Period | Test Period |
|--------|-------------|-------------|
| WF-1 | 2022-01 → 2023-06 | 2023-07 → 2024-03 |
| WF-2 | 2022-01 → 2024-03 | 2024-04 → 2024-12 |
| WF-3 | 2022-01 → 2024-12 | 2025-01 → 2025-09 |
| WF-4 | 2022-01 → 2025-09 | 2025-10 → 2026-04 |

For each window, robust ranges are identified on training data only, then three portfolios are compared out-of-sample:

- **(A) Range portfolio** — equal-weight only lookbacks inside the training-identified range
- **(B) Best single lookback** — highest Sharpe from training
- **(C) All-lookback portfolio** — equal-weight all 96 lookbacks (diversified baseline)

**Interpretation key:** If A consistently beats C → range selection adds value. If B beats A → you are overfitting to a point estimate. If C beats both → safest approach is to diversify across all lookbacks.

### In-Sample (Full Period, 2022–2026, no costs)

| Metric | Robust Range (5–55d) | Best Single (5d) |
|--------|---------------------|-------------------|
| Avg Sharpe | 2.75 | 4.07 |
| Avg CAGR | 55.0% | 79.5% |
| Avg Max DD | 7.0% | 5.7% |
| Viable Lookbacks | 52 / 96 | — |
| Range Width | 51 days | — |

### Walk-Forward OOS Results (per window)

| Window | Test Period | Train Range | A: Range Sharpe | A: Range CAGR | A: Range Max DD | B: Best Sharpe | B: Best CAGR | C: All Sharpe |
|--------|-------------|-------------|-----------------|---------------|-----------------|----------------|--------------|---------------|
| WF-1 | Jul 2023 – Mar 2024 | 5–67d | 3.72 | 182.9% | 8.7% | 3.86 | 299.9% | 3.35 |
| WF-2 | Apr 2024 – Dec 2024 | 5–68d | 1.63 | 68.0% | 11.8% | 2.91 | 206.4% | 1.37 |
| WF-3 | Jan 2025 – Sep 2025 | 5–65d | 0.98 | 44.3% | 20.5% | 1.32 | 80.9% | 0.96 |
| WF-4 | Oct 2025 – Apr 2026 | 5–52d | 2.30 | 135.5% | 12.0% | 2.63 | 221.1% | 1.71 |

### Walk-Forward Aggregate (pooled OOS, no costs)

| Metric | A: Robust Range | B: Best Single | C: All Lookbacks |
|--------|----------------|----------------|------------------|
| Sharpe | 2.15 | 2.72 | 1.83 |
| Max DD | 56.6% | 11.4% | 68.8% |

### Interpretation

- **B > A > C on Sharpe in every window.** The best single lookback (mostly 5d) wins on risk-adjusted returns, and the range portfolio consistently beats the all-lookback portfolio. This means range selection adds value over blind diversification, but a single short lookback is even better — at least before costs.
- **The robust range is structurally stable** — lower bound is always 5d; upper bound narrows only slightly from 67d to 52d across 4 windows. This is strong evidence of a genuine feature, not overfitting.
- **All three approaches are profitable OOS in all 4 windows.** Nothing goes negative, even in the weakest period (WF-3, Jan–Sep 2025).
- **Aggregate max DD for multi-lookback portfolios is severe** (57–69%). This is driven by capital scaling (each lookback gets its own $30K allocation). A single lookback avoids this.
- **These results are pre-cost.** The very high CAGR numbers (44–300%) should be discounted significantly once slippage and commissions are applied, especially for the 5d lookback which generates the most trades.

---

## 4. Trend-Following Backtest With Costs

**Files:** `tf_backtest_app.py`

**Question:** What does the strategy look like with realistic transaction costs?

**Method:** Dash dashboard running the N-day breakout across all 48 markets with $30,000 capital, 1% risk per trade, $10 commission per trade, and 2-tick adverse slippage on entry (market-specific tick sizes). Default lookback 20d, full period 2022-01-01 to 2026-04-07.

| Metric | 20d Lookback (with costs) |
|--------|--------------------------|
| Total Commission | $17,160 |
| Total Slippage Cost | $8,174 |
| Combined Costs | $25,334 |
| Gross PnL (no costs) | $194,905 |
| Net PnL (after costs) | $169,571 |
| Cost as % of Gross | 13.0% |

**Finding:** Transaction costs consume ~13% of gross PnL at the 20-day lookback. Shorter lookbacks will have higher cost drag due to more trades.

---

## Comparison Table — All Strategies

All metrics are out-of-sample except where noted. The trend-following with-costs row uses full-period data.

| Strategy | Source Files | OOS Sharpe | OOS CAGR | Max DD | Trades | Win Rate | Costs Modelled | Verdict |
|----------|-------------|------------|----------|--------|--------|----------|----------------|---------|
| ORB NR3 (no COT) | `hypothesis_test.py` | 0.51 | — | — | — | — | No | Baseline |
| ORB NR3 + Best COT | `hypothesis_test.py` | 0.79 | — | — | — | — | No | Not significant (p=0.85) |
| TF Breakout (no COT) | `trend_following_test.py` | 1.50 | — | 14.2% | 15,436 | — | No | Baseline |
| TF Breakout + Best COT | `trend_following_test.py` | 1.82 | — | 9.4% | — | — | No | Not significant (p=0.88) |
| TF 5–55d Range (OOS agg) | `lookback_robustness.py` | 2.15 | — | 56.6% | — | — | No | Profitable, high DD |
| TF Best Single LB (OOS agg) | `lookback_robustness.py` | 2.72 | — | 11.4% | — | — | No | Best risk-adjusted |
| TF 20d Breakout (with costs) | `tf_backtest_app.py` | — | — | — | 1,716 | — | Yes ($10 + 2 ticks) | 13% cost drag |

---

## Key Takeaways

1. **COT does not help.** Both the ORB and trend-following strategies show no statistically significant benefit from COT filters (SPA p-values 0.85 and 0.88).

2. **Short-to-medium breakouts (5–55d) are a robust edge.** The range appears in every walk-forward training window and is consistently profitable out-of-sample.

3. **Shorter lookbacks dominate** on risk-adjusted returns, but generate more trades and are more exposed to transaction costs.

4. **Transaction costs are material but manageable.** At the 20d lookback, costs eat ~13% of gross PnL. The strategy remains profitable after costs.

5. **Max drawdown in diversified multi-lookback portfolios is severe** (57–69%). A single well-chosen lookback in the 10–30d range is a better practical approach than running all lookbacks.
