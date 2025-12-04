# Crypto Trading System - Results & Findings

This document tracks key results, decisions, and takeaways from each development phase. It serves as a quick reference for baselines, parameter choices, and lessons learned.

---

## Phase 1: Data Infrastructure (Dec 2024)

**Scope:** Database setup, exchange data ingestion, feature engineering pipeline

**Data Collected:**
- BTC/USD: 2,155 hourly candles (90 days)
- ETH/USD: 2,155 hourly candles (90 days)
- Source: Coinbase via CCXT
- Storage: TimescaleDB with hypertable partitioning

**Features Generated (22 total):**
- Price: returns (1h, 4h, 24h), log returns
- Trend: SMA (5, 10, 20, 50), EMA (12, 26)
- Volatility: 20-period volatility, ATR-14, Bollinger Bands
- Momentum: RSI-14, MACD (12, 26, 9)
- Volume: 20-period SMA, volume ratio, OBV

**Key Decisions:**
- 50-period warmup to ensure indicator stability
- Hourly timeframe balances signal quality vs data quantity
- TimescaleDB chosen for time-series optimization

**Takeaway:** Clean data pipeline established. Feature engineering produces 22 indicators with no NaN values after warmup.

---

## Phase 2: Baseline LightGBM Model (Dec 2024)

**Scope:** Baseline ML model, walk-forward validation, backtesting framework

**Data:** 90 days BTC/USD hourly (2,105 samples after preprocessing)

**Model Configuration:**
- Algorithm: LightGBM binary classifier
- Target: Next-hour price direction (1 = up, 0 = down)
- Boosting rounds: 100 (early stopping at 10)
- Walk-forward: 5 splits, 701 train / 150 test per fold, 1-period gap

**Validation Results:**
| Metric | Value |
|--------|-------|
| Overall Accuracy | 51.6% |
| Test Accuracy Range | 46.0% - 54.7% |
| Up Prediction Ratio | 62.3% |
| Train Accuracy Range | 65.3% - 82.2% |

**Backtest Results (750 hours, Nov-Dec 2024):**
| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Return | -21.94% | -15.02% |
| Sharpe Ratio | -6.81 | -3.35 |
| Max Drawdown | 24.98% | 25.72% |
| Volatility | 41.30% | 53.30% |
| Win Rate | 33.64% | - |
| Num Trades | 110 | 1 |

**Top Features by Importance:**
1. OBV (On-Balance Volume)
2. 4h Return
3. RSI-14
4. 1h Return
5. Volume Ratio

**Key Decisions:**
- Transaction costs: 0.1% commission + 0.05% slippage (Coinbase realistic)
- Long-only strategy (no shorting for baseline simplicity)
- Walk-forward validation prevents look-ahead bias
- 1-period gap between train/test to avoid data leakage

**Observations:**
- Model barely beats random (51.6% vs 50%) - expected for price-only features
- Significant train/test gap (65-82% train vs 46-55% test) suggests overfitting
- Lower volatility than buy-and-hold due to flat periods when predicting "down"
- Market was declining during test period, explaining negative Sharpe for both

**Takeaway:** Baseline established. 51.6% accuracy is the number to beat. On-chain data and sentiment in Phase 3 should provide real alpha - academic research shows 75-82% directional accuracy with these features.

---

## Phase 3: Alpha Sources (Dec 2024)

**Scope:** Reddit sentiment analysis, on-chain signal interpretation, regime classification, alpha signal combination

### Data Sources Implemented

1. **Reddit Sentiment (via public JSON endpoints)**
   - Subreddits: r/Bitcoin, r/CryptoCurrency, r/ethereum, r/CryptoMarkets, r/BitcoinMarkets
   - Public JSON API (no auth required, ~10 req/min rate limit)
   - Abstraction layer supports future PRAW OAuth integration
   - Verified: 10+ posts fetched per subreddit

2. **On-Chain Metrics (LIVE)**
   | Metric | Source | Data Availability | Status |
   |--------|--------|-------------------|--------|
   | MVRV | CoinMetrics (free API) | 3,990 days (2015-present) | ✓ Live |
   | SOPR | Dune Analytics #5130629 | 5,603 days (2010-present) | ✓ Live |
   | Exchange Netflow | Dune Analytics #1621987 | Daily | ✓ Live |
   | Stablecoin Supply | Dune Analytics #4425983 | Daily | ✓ Live |

3. **Sentiment Analysis (FinBERT)**
   - Model: ProsusAI/finbert
   - 3-class classification: positive/negative/neutral
   - Credibility weighting by Reddit post score
   - Aggregation across multiple subreddits

### Backtest Methodology: Regime Overrides

Tested on-chain signals as **regime overrides** (not ML features):

**Override Logic:**
- MVRV < 1.0 → Force bullish (cycle bottom)
- MVRV > 3.7 → Force bearish (cycle top)
- SOPR < 0.95 → Force bullish (capitulation)
- SOPR > 1.05 → Force bearish (euphoria)
- Otherwise → Trust ML prediction

**Dataset:** 2020-2024 daily BTC/USD (2,115 samples)

**Override Conditions Found in Full Dataset:**
| Condition | Days | % of Data |
|-----------|------|-----------|
| MVRV < 1.0 (bottom) | 185 | 8.5% |
| MVRV > 3.7 (top) | 8 | 0.4% |
| SOPR < 0.95 (capitulation) | 84 | 3.9% |
| SOPR > 1.05 (euphoria) | 305 | 14.0% |

### Backtest Results (70/30 Split)

**Test Period:** March 2024 - December 2025 (635 days)

| Metric | Buy & Hold | Baseline ML | ML + Overrides |
|--------|------------|-------------|----------------|
| Final Equity | $13,305 | $13,372 | $10,442 |
| Total Return | +33.0% | +33.7% | +4.4% |
| Sharpe Ratio | 2.85 | 2.98 | 1.27 |
| Max Drawdown | 32.1% | 25.7% | 25.6% |
| Win Rate | - | 51.8% | 50.0% |
| Num Trades | 0 | 56 | 70 |

**Sharpe Change: -57.5%** (target was ≥10% improvement)

### Critical Analysis

**Why Overrides Underperformed:**

1. **No MVRV extremes in test period**
   - Test period MVRV range: 1.51 - 2.78 (all neutral zone)
   - All 185 MVRV<1.0 days and 8 MVRV>3.7 days occurred in training data (2020-2022)
   - MVRV extremes are cycle events occurring years apart

2. **SOPR override accuracy was poor**
   - SOPR euphoria (>1.05): 83 days, 45.8% accuracy (worse than random)
   - SOPR capitulation (<0.95): 5 days, 40.0% accuracy
   - Overrides introduced noise, not signal

3. **More trades, more costs**
   - Overrides triggered 70 trades vs 56 baseline
   - Each unnecessary trade costs ~0.15% (commission + slippage)

**Performance by Regime Type:**
| Regime | Days | Accuracy |
|--------|------|----------|
| ML Only (no override) | 547 (86%) | 52.5% |
| SOPR Euphoria | 83 (13%) | 45.8% |
| SOPR Capitulation | 5 (1%) | 40.0% |

### Honest Takeaways

1. **Infrastructure works** - Data pipelines for MVRV, SOPR, sentiment all functional
2. **MVRV extremes untestable** - Cycle tops/bottoms are rare (8 days in 5 years above 3.7)
3. **SOPR thresholds too loose** - >1.05 fires too often with poor accuracy
4. **No alpha found** - In out-of-sample testing, on-chain signals hurt performance

### Current Market Reading (Dec 4, 2025)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MVRV | 1.66 | Fair Value (neutral zone) |
| SOPR | 0.99 | Neutral |
| Combined | - | No override triggered |

### Files Created

| File | Purpose |
|------|---------|
| `data/ingestion/reddit_sources.py` | Reddit data abstraction |
| `data/ingestion/onchain.py` | CoinMetrics + Dune on-chain data |
| `data/ingestion/sentiment.py` | FinBERT sentiment analyzer |
| `models/predictors/regime_classifier.py` | Market regime classification |
| `models/predictors/alpha_combiner.py` | Combined alpha signal generation |
| `tests/test_alpha_sources.py` | Unit tests (39 tests) |
| `scripts/phase3_regime_override_backtest.py` | Regime override backtest |

### Conclusion

The Phase 3 hypothesis that on-chain signals provide tradeable alpha was **not validated** in out-of-sample backtesting. The baseline ML model with technical features only (Sharpe 2.98) significantly outperformed the ML + regime override strategy (Sharpe 1.27).

---

## Phase 3.5: Signal Quality Deep Dive (Dec 2024)

**Scope:** Comprehensive analysis of ALL signal sources before Phase 4-6 to determine if we have alpha worth amplifying.

### Signal Correlation Analysis (2020-2025, 2,135 daily samples)

| Signal | Pearson Corr | Spearman Corr | Assessment |
|--------|--------------|---------------|------------|
| **1-Day Momentum** | **-0.0579** | -0.0580 | Best signal (mean reversion) |
| SOPR | +0.0425 | +0.0230 | Marginal |
| MVRV Z-Score | +0.0362 | +0.0294 | Marginal |
| Volatility-20 | +0.0302 | +0.0178 | Marginal |
| RSI-14 | +0.0286 | +0.0132 | Weak |
| MVRV | +0.0050 | -0.0031 | Noise |

**Interpretation:**
- Best single signal is mean reversion (-5.8% correlation)
- On-chain signals (MVRV, SOPR) are marginal at best
- Correlation < 0.05 is very weak; < 0.02 is noise

### CRITICAL FINDING: Extreme Signal Accuracy

| Signal Extreme | Days | Expected Direction | Actual Accuracy |
|----------------|------|-------------------|-----------------|
| MVRV < 1.0 (cycle bottom) | 185 (8.7%) | Bullish | **49.7%** (coin flip!) |
| MVRV > 3.0 (cycle top) | 100 (4.7%) | Bearish | 53.0% |
| RSI < 30 (oversold) | 211 (9.9%) | Bullish | **43.1%** (WORSE than random!) |
| RSI > 70 (overbought) | 415 (19.4%) | Bearish | 45.1% |
| SOPR < 0.97 (capitulation) | 155 (7.3%) | Bullish | **45.2%** |
| SOPR > 1.03 (euphoria) | 514 (24.1%) | Bearish | 51.9% |

**This is devastating for the "buy at extremes" hypothesis:**
- MVRV cycle bottoms: No better than random
- RSI oversold: Actually WRONG more often than right
- SOPR capitulation: Wrong 55% of the time

### Sentiment Analysis Limitation

**Historical backtest NOT POSSIBLE:**
- Reddit public JSON only provides recent posts (~1000)
- No Pushshift archive access
- No historical sentiment data collected

**Current sentiment sample (50 posts):** -0.053 (neutral)

### MVRV Lag Analysis

Does MVRV predict further ahead?

| Lag | Correlation with Forward Return |
|-----|--------------------------------|
| 1 day | +0.005 |
| 5 days | +0.017 |
| 10 days | +0.021 |
| 20 days | +0.019 |
| 40 days | +0.006 |
| 60 days | -0.011 |

**Finding:** MVRV shows no meaningful predictive power at any horizon.

### Honest Assessment

```
Available Alpha Sources:
- Technical (mean reversion): -0.058 correlation ← ONLY USABLE SIGNAL
- MVRV Z-Score: +0.036 correlation (marginal)
- SOPR: +0.043 correlation (marginal)
- Extreme signals: WORSE than random

Combined signal strength: VERY WEAK
```

### Path Recommendation

**Recommended: PATH B - Risk Management Focus**

Despite the automated recommendation of Path A (based on -0.058 correlation), the complete picture suggests PATH B:

**Reasons:**
1. Best signal is simple mean reversion (well-known, likely arbed away)
2. Extreme signal accuracy is WORSE than 50% - invalidates conviction-based sizing
3. On-chain signals don't work for daily trading at any threshold
4. No historical sentiment data to validate that hypothesis

**Phase 4-6 should focus on RISK MANAGEMENT, not alpha generation:**

| Phase | Revised Goal | Success Metric |
|-------|--------------|----------------|
| 4 - GNN | Regime detection, drawdown prediction | Detect crashes 5+ days ahead |
| 5 - RL | Dynamic position sizing | Max drawdown < 20% (vs BTC's 30%) |
| 6 - Production | Defensive system | Match BTC returns, halve volatility |

### Alternative: PATH C - Monthly Rebalancing

Given MVRV correlations are marginally better at 10-20 day lags, consider:

| MVRV Range | BTC Allocation | Rationale |
|------------|----------------|-----------|
| < 1.0 | 100% | Cycle bottom (rare) |
| 1.0-2.0 | 70% | Fair value |
| 2.0-3.0 | 40% | Getting expensive |
| > 3.0 | 10% | Cycle top risk |

This matches signal frequency (monthly rebalance, not daily trading).

### Key Takeaway

**We do NOT have alpha worth amplifying.** The best strategy may be:
1. Simple buy-and-hold with rebalancing
2. GNN/RL for crash detection and defensive positioning
3. Accept that beating BTC returns is unlikely; focus on risk-adjusted returns

---

## Phase 4: GNN Regime Detection (Revised Scope)

**Original Goal:** ~~Alpha amplification with graph neural networks~~

**Revised Goal:** Detect regime shifts to trigger defensive positioning BEFORE crashes develop.

### Objective

Classify market into three regimes:
- **RISK_ON**: Normal conditions → 100% exposure
- **CAUTION**: Elevated risk → 50% exposure
- **RISK_OFF**: Crisis conditions → 20% exposure

### Key Signals to Detect
| Input Pattern | Regime | Action |
|---------------|--------|--------|
| BTC-ETH correlation spike + vol expansion | CAUTION | Reduce position |
| Cross-asset correlations → 1.0 | RISK_OFF | Minimal exposure |
| Vol compression + normal correlations | RISK_ON | Full position |

### Training Data
- 2020 COVID crash (correlation spike preceded dump)
- 2021 May crash (leverage flush)
- 2022 LUNA/FTX (contagion detection)

### Success Criteria
- Detect regime shift 3-5 days before major drawdown >20%
- False positive rate <30%
- Reduces exposure before ≥60% of major crashes

*Status: Planned*

---

## Phase 5: RL Position Sizing (Revised Scope)

**Original Goal:** ~~Optimal trading decisions for maximum returns~~

**Revised Goal:** Learn optimal position sizing (0-100%) based on regime and uncertainty.

### RL Formulation
- **State:** Regime, volatility, recent drawdown, current exposure
- **Action:** Target position size (0%, 25%, 50%, 75%, 100%)
- **Reward:** Risk-adjusted returns with heavy drawdown penalties

### Reward Function
```
reward = returns
if drawdown > 10%: reward -= drawdown * 3
if drawdown > 20%: reward -= drawdown * 5
reward -= volatility * 0.5
```

### Desired Behaviors
| Situation | Action |
|-----------|--------|
| RISK_ON + low vol | 100% position |
| CAUTION regime | 50% position |
| RISK_OFF regime | 20% position |
| After 15% drawdown | Gradual re-entry (5-10 days) |

### Success Criteria
- Sharpe improvement >0.3 vs fixed position
- Drawdown reduction >30% vs buy & hold
- Rebalancing <2x per week average

*Status: Planned*

---

## Phase 6: Defensive Production System (Revised Scope)

**Original Goal:** ~~Alpha-generating ensemble~~

**Revised Goal:** Production-ready defensive system with hard risk controls.

### Architecture
```
GNN Regime Detector → Regime Signal → RL Position Sizer → Hard Limits → Execute
```

### Hard Risk Limits (Non-Negotiable)
| Limit | Value |
|-------|-------|
| Max position | 100% (no leverage) |
| Min position | 10% (always some exposure) |
| Max daily change | 25% |
| Position stop-loss | 15% |
| System halt | 25% drawdown |

### Success Declaration
| Metric | BTC Buy & Hold | Our Target |
|--------|----------------|------------|
| Returns | 100% (baseline) | >70% of BTC |
| Max Drawdown | ~70% | **<25%** |
| Sharpe | ~1.0 | >1.5 |

*Status: Planned*

---

## Strategic Pivot Summary (Dec 2024)

**What We Learned:**
- Signals (MVRV, SOPR, RSI extremes) do NOT provide tradeable daily alpha
- Extreme signal accuracy is 43-50% (worse than random)
- Best correlation is -5.8% (mean reversion) - too weak to amplify

**What We're Building Instead:**
- Risk management system, not alpha generation
- Goal: Survive crypto volatility, not predict it
- Success = 70%+ of BTC returns with <25% max drawdown

**What We're NOT Building:**
- ❌ Alpha prediction
- ❌ Daily trade signals
- ❌ Sentiment trading
- ❌ On-chain trade triggers
- ❌ Leverage

**Mantra:** "We're not trying to predict the future. We're trying to survive it."

---

## Quick Reference: Key Baselines

| Phase | Accuracy | Sharpe | Notes |
|-------|----------|--------|-------|
| 2 - LightGBM Baseline | 52.3% | 2.98 | Technical features only (2020-2025 daily) |
| 3 - Regime Overrides | 51.5% | 1.27 | **-57.5%** - On-chain overrides hurt performance |
| 3.5 - Signal Analysis | - | - | Best signal: -5.8% corr (mean reversion) |
| 4 - GNN | TBD | TBD | **REVISED**: Regime detection, not alpha |
| 5 - RL | TBD | TBD | **REVISED**: Risk management, max DD < 20% |

### Critical Finding: Extreme Signal Accuracy

| Extreme | Expected | Actual Accuracy |
|---------|----------|-----------------|
| MVRV < 1.0 | Bullish | **49.7%** (random) |
| RSI < 30 | Bullish | **43.1%** (WORSE) |
| SOPR < 0.97 | Bullish | **45.2%** (worse) |

**Conclusion:** We do NOT have alpha worth amplifying. Pivot to risk management focus.

---

## Files Created in Phase 3

| File | Purpose |
|------|---------|
| `data/ingestion/reddit_sources.py` | Reddit data abstraction (public JSON + PRAW) |
| `data/ingestion/onchain.py` | CoinMetrics + Dune Analytics on-chain data |
| `data/ingestion/sentiment.py` | FinBERT sentiment analyzer |
| `models/predictors/regime_classifier.py` | Market regime classification |
| `models/predictors/alpha_combiner.py` | Combined alpha signal generation |
| `tests/test_alpha_sources.py` | Unit tests (39 tests) |
| `scripts/phase3_backtest_comparison.py` | Initial backtest (simulated alpha) |
| `scripts/phase3_backtest_real_data.py` | Backtest with real MVRV/SOPR |
| `scripts/phase3_regime_override_backtest.py` | Regime override backtest (2020-2024) |
| `scripts/phase3_signal_quality_analysis.py` | Comprehensive signal quality analysis |

---

*Last Updated: December 2024*
