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

**Data Sources Implemented:**

1. **Reddit Sentiment (via public JSON endpoints)**
   - Subreddits: r/Bitcoin, r/CryptoCurrency, r/ethereum, r/CryptoMarkets, r/BitcoinMarkets
   - Public JSON API (no auth required, ~10 req/min rate limit)
   - Abstraction layer supports future PRAW OAuth integration
   - Test run: 25 posts fetched successfully

2. **On-Chain Metrics (Dune Analytics ready)**
   - MVRV (Market Value to Realized Value)
   - SOPR (Spent Output Profit Ratio)
   - Exchange Netflows
   - SSR (Stablecoin Supply Ratio)
   - NUPL (Net Unrealized Profit/Loss)
   - Puell Multiple

3. **Sentiment Analysis (FinBERT)**
   - Model: ProsusAI/finbert
   - CARVS scoring (Credibility-Adjusted Relevance-Volume-Sentiment)
   - Relevance filtering for crypto keywords
   - Credibility weighting by post score, upvote ratio, subreddit quality

**Regime Classification:**

| Regime | Combined Signal Range | Trading Bias |
|--------|----------------------|--------------|
| Strong Bull | > 0.6 | Long, 100% size |
| Bull | 0.3 - 0.6 | Long, 80% size |
| Accumulation | 0.15 - 0.3 | Long, 60% size |
| Neutral | -0.15 - 0.15 | Flat, 30% size |
| Distribution | -0.3 - -0.15 | Short, 50% size |
| Bear | -0.6 - -0.3 | Short, 70% size |
| Strong Bear | < -0.6 | Short, 90% size |

**Alpha Combiner Configuration:**
| Signal Source | Weight |
|---------------|--------|
| On-chain | 35% |
| Sentiment | 25% |
| Technical | 25% |
| Regime | 15% |

**Signal Interpretation (On-Chain):**

| Metric | Bullish Signal | Bearish Signal |
|--------|---------------|----------------|
| MVRV | < -0.5 (extreme undervaluation) | > 7 (extreme overvaluation) |
| SOPR | < 0.95 (capitulation) | > 1.05 (profit taking) |
| Exchange Netflow | Negative (outflows) | Positive (inflows) |

**Validation Results:**
- All unit tests pass (39/39)
- Reddit fetching: 25 posts from 5 subreddits
- Sentiment signals in valid range [-1, 1]
- Regime classifier correctly identifies scenarios
- Combined signals produce actionable recommendations

**Current Market Signal (test run):**
- Combined Signal: +0.002 (neutral)
- Regime: Neutral
- Recommendation: Flat
- Note: Low confidence due to placeholder on-chain data

**Key Decisions:**
- Reddit public JSON over PRAW (no API approval needed since Nov 2025 policy)
- Dune Analytics over Glassnode (more flexible queries, has API key)
- FinBERT for financial-domain sentiment (better than general BERT)
- 7-regime classification for nuanced market states
- CARVS scoring filters low-quality sentiment signals

**Technical Notes:**
- PyTorch upgraded to 2.9.1+ (required by transformers for CVE-2025-32434 fix)
- torch-geometric, learn2learn deferred to Phase 4-5 (build dependency issues)

**Takeaway:** Alpha source infrastructure complete. Regime classifier and signal combiner ready for integration. Next phase will combine these signals with ML models for improved predictions. On-chain signals await Dune query configuration with actual query IDs.

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
| 3 - Alpha Sources | Infrastructure | Ready | Reddit, Dune, FinBERT, Regime classifier |
| 4 - Advanced Models | TBD | TBD | LSTM, GNN ensemble |
| 5 - Hierarchical RL | TBD | TBD | Target: Sharpe > 2.0 |

---

## Files Created in Phase 3

| File | Purpose |
|------|---------|
| `data/ingestion/reddit_sources.py` | Reddit data abstraction (public JSON + PRAW) |
| `data/ingestion/onchain.py` | Dune Analytics on-chain data provider |
| `data/ingestion/sentiment.py` | FinBERT sentiment analyzer with CARVS |
| `models/predictors/regime_classifier.py` | Market regime classification |
| `models/predictors/alpha_combiner.py` | Combined alpha signal generation |
| `tests/test_alpha_sources.py` | Unit tests (39 tests) |
| `notebooks/phase3/01_alpha_sources_validation.ipynb` | Validation notebook |

---

*Last Updated: December 2024*
