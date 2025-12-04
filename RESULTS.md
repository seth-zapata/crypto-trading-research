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

## Phase 3: Alpha Sources (Planned)

*To be completed*

**Planned components:**
- On-chain metrics (MVRV, SOPR, exchange netflows)
- Sentiment analysis (Twitter, Reddit with RVS filtering)
- Regime classification

**Success criteria:**
- Accuracy improvement to 55%+
- Positive Sharpe ratio in backtest
- Alpha sources show independent predictive power

---

## Phase 4: Advanced Models (Planned)

*To be completed*

---

## Phase 5: Reinforcement Learning (Planned)

*To be completed*

---

## Phase 6: Production Deployment (Planned)

*To be completed*

---

## Quick Reference: Key Baselines

| Phase | Accuracy | Sharpe | Notes |
|-------|----------|--------|-------|
| 2 - LightGBM Baseline | 51.6% | -6.81 | Price features only, declining market |
| 3 - With Alpha Sources | TBD | TBD | Target: 55%+, Sharpe > 0 |
| 4 - Advanced Models | TBD | TBD | LSTM, GNN ensemble |
| 5 - Hierarchical RL | TBD | TBD | Target: Sharpe > 2.0 |

---

*Last Updated: December 2024*
