# Cryptocurrency Trading System - Complete Technical Specification

**Version:** 1.0  
**Last Updated:** December 2024  
**Implementation Timeline:** 6 weeks  
**Target Capital:** $10,000 - $50,000  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Foundation](#research-foundation)
3. [System Architecture](#system-architecture)
4. [Technical Stack](#technical-stack)
5. [Implementation Phases](#implementation-phases)
6. [Module Specifications](#module-specifications)
7. [Performance Targets](#performance-targets)
8. [Risk Management](#risk-management)
9. [Deployment Strategy](#deployment-strategy)

---

## Executive Summary

This specification details a production-grade cryptocurrency trading system based on validated academic research (2020-2024). The system achieves alpha generation through:

1. **On-Chain Analytics** - Blockchain metrics with 75-82% directional accuracy
2. **Graph Neural Networks** - Cross-asset relationship modeling (2.2x Sharpe improvement)
3. **Filtered Sentiment Analysis** - Social signals with volume confirmation (+291% bear market returns)
4. **Hierarchical Reinforcement Learning** - Multi-timescale decision making (2.74 Sharpe ratio)

### What This System Does NOT Do

- ❌ High-frequency trading (requires co-location)
- ❌ Arbitrage (requires millisecond execution)
- ❌ Leverage above 2x (too risky)
- ❌ Margin trading (added complexity)
- ❌ Altcoin speculation (focused on BTC/ETH/major caps)

### Key Differentiators

**Research-Backed:** Every component has peer-reviewed validation  
**Multi-Signal:** Combines on-chain, sentiment, technical, and ML signals  
**Risk-First:** Conservative position sizing with strict limits  
**Transparent:** Full logging and explainable decisions  
**Incremental:** Build baseline first, add complexity only when validated  

---

## Research Foundation

### Validated Techniques (2020-2024 Research)

#### 1. On-Chain Analytics
- **MVRV Z-Score**: Identifies cycle tops/bottoms (>3.7 = top, <1.0 = bottom)
- **SOPR**: Short-term holder profitability (#1 predictive metric per Glassnode)
- **Exchange Netflows**: 24-72 hour lead time on major moves
- **Stablecoin Supply Ratio**: Measures available buying power
- **Validation**: 75-82% accuracy (Omole & Enke, 2024), 6,654% annualized return in backtest

#### 2. Graph Neural Networks
- **Architecture**: Temporal graph attention networks (THGNN, MGAR)
- **Node Features**: Price, volume, technical indicators
- **Edge Construction**: Rolling correlation, sector membership, knowledge relations
- **Validation**: 2.2x Sharpe improvement, 164-236% returns (MGAR, 2023)

#### 3. Sentiment Analysis
- **CARVS Method**: Combined Attention Relative Volume Sentiment
- **Key Insight**: Only use sentiment when aligned with volume direction
- **Processing**: FinBERT for financial text, engagement weighting, bot filtering
- **Validation**: +291% returns in 2018 bear market vs -72.6% buy-hold (Kraaijeveld & De Smedt, 2024)

#### 4. Hierarchical Reinforcement Learning
- **Architecture**: Strategic (PPO) → Tactical (DDPG) → Execution (DQN)
- **Training**: Phased with exponential decay weighting between levels
- **Validation**: 2.74 Sharpe vs 2.27 benchmark (HRT, 2024)

#### 5. Meta-Learning for Adaptation
- **Approach**: MAML for few-shot regime adaptation
- **Performance**: 10x Sharpe improvement, 2x faster recovery from drawdowns
- **Validation**: X-Trend (2023), 5-fold Sharpe on zero-shot predictions

### Techniques Excluded (Insufficient Evidence)

- ❌ Pure transformer forecasting (linear models perform better)
- ❌ Unfiltered social sentiment (too noisy)
- ❌ Technical indicators alone (insufficient edge)
- ❌ Single-agent RL (hierarchical outperforms)
- ❌ Price prediction without risk management (dangerous)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                             │
│          (Main Loop - 1 hour frequency)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐            ┌───────▼────────┐
│  DATA LAYER    │            │  INTELLIGENCE  │
│                │            │     LAYER      │
│ • Exchanges    │            │                │
│ • On-Chain     │            │ • LightGBM     │
│ • Sentiment    │            │ • LSTM         │
│ • Order Books  │            │ • GNN          │
└───────┬────────┘            │ • Ensemble     │
        │                     │ • Meta-Learn   │
        │                     └───────┬────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │      DECISION LAYER         │
        │                             │
        │  ┌──────────────────┐      │
        │  │  Strategic Agent │      │
        │  │      (PPO)       │      │
        │  └────────┬─────────┘      │
        │           │                 │
        │  ┌────────▼─────────┐      │
        │  │  Tactical Agent  │      │
        │  │     (DDPG)       │      │
        │  └────────┬─────────┘      │
        │           │                 │
        │  ┌────────▼─────────┐      │
        │  │ Execution Agent  │      │
        │  │      (DQN)       │      │
        │  └────────┬─────────┘      │
        └───────────┼─────────────────┘
                    │
        ┌───────────▼─────────────┐
        │    RISK MANAGEMENT      │
        │                         │
        │ • Position Sizing       │
        │ • Limit Enforcement     │
        │ • Circuit Breakers      │
        └───────────┬─────────────┘
                    │
        ┌───────────▼─────────────┐
        │   EXECUTION ENGINE      │
        │                         │
        │ • Order Routing         │
        │ • Slippage Modeling     │
        │ • Trade Logging         │
        └─────────────────────────┘
```

---

## Technical Stack

### Core Technologies

**Language:** Python 3.11+  
**ML Framework:** PyTorch 2.1+  
**RL Framework:** Stable-Baselines3 2.1+, FinRL 0.3+  
**Graph ML:** PyTorch Geometric 2.4+  
**Time Series:** Pandas 2.1+, NumPy 1.24+  
**Database:** PostgreSQL 15 + TimescaleDB  
**Caching:** Redis 5.0+  
**Web:** AsyncIO, aiohttp, websockets  

### Key Libraries

```yaml
# Core ML
- pytorch==2.1
- pytorch-geometric==2.4
- scikit-learn==1.3
- lightgbm==4.1
- stable-baselines3==2.1

# Financial Data
- ccxt==4.2              # Exchange APIs
- pandas==2.1
- numpy==1.24
- ta-lib==0.4            # Technical indicators

# Specialized
- tigramite>=5.2         # Causal discovery
- learn2learn>=0.2       # Meta-learning
- transformers>=4.35     # FinBERT sentiment

# Data Providers
- glassnode>=1.0         # On-chain data
- tweepy>=4.14          # Twitter API
- praw>=7.6             # Reddit API

# Database
- psycopg2-binary==2.9
- sqlalchemy==2.0
- redis==5.0

# Monitoring
- streamlit==1.28       # Dashboard
- plotly==5.17          # Visualizations
```

### Infrastructure Requirements

**Compute:**
- CPU: 8+ cores (AMD Ryzen 7 or Intel i7)
- RAM: 32GB minimum, 64GB recommended
- GPU: Optional (RTX 3080 for LSTM/GNN acceleration)
- Storage: 250GB+ SSD

**Network:**
- Low-latency connection (< 50ms to exchange)
- Stable internet (99.9% uptime)

**Estimated Costs:**
- Development machine: $0 (local)
- Cloud VM (optional): $100-200/month
- Data providers: $0-100/month
- Exchange fees: 0.1-0.2% per trade

---

## Implementation Phases

### Phase 0: Environment Setup (Day 1)

**Deliverables:**
- Python environment configured
- PostgreSQL + TimescaleDB installed
- Redis running
- Git repository initialized
- Configuration templates created

**Validation:**
```bash
python --version  # 3.11+
psql --version   # 15+
redis-cli ping   # PONG
git status       # Clean working tree
```

### Phase 1: Data Infrastructure (Days 2-7)

**Deliverables:**
- Exchange data ingestion (OHLCV, order books)
- TimescaleDB schema created
- Historical data backfill (90+ days)
- Basic feature engineering
- Data quality monitoring

**Key Files:**
- `data/ingestion/exchanges.py`
- `data/storage/timeseries_db.py`
- `sql/schema.sql`
- `scripts/backfill_data.py`

**Validation:**
```bash
# Should populate database
python scripts/backfill_data.py --symbols BTC/USDT ETH/USDT --days 90

# Should show data
psql -d crypto_trading -c "SELECT COUNT(*) FROM ohlcv;"
# Expected: 2160+ rows (90 days * 24 hours * 1 symbol)
```

### Phase 2: Baseline Model (Days 8-14)

**Deliverables:**
- Feature engineering pipeline (100+ features)
- LightGBM training and evaluation
- Backtesting framework
- Walk-forward validation
- Performance metrics

**Key Files:**
- `data/processing/features.py`
- `models/predictors/lightgbm_baseline.py`
- `backtesting/engine.py`
- `notebooks/baseline_analysis.ipynb`

**Success Criteria:**
- Backtest Sharpe > 1.0
- Win rate > 50%
- Max drawdown < 20%
- Out-of-sample validation positive

### Phase 3: Alpha Sources (Days 15-21)

**Deliverables:**
- On-chain data integration (MVRV, SOPR, netflows, SSR)
- Sentiment analysis (Twitter + Reddit with FinBERT)
- Regime classification system
- Combined signal generation

**Key Files:**
- `data/ingestion/onchain.py`
- `data/ingestion/sentiment.py`
- `models/regime_detector.py`
- `notebooks/signal_analysis.ipynb`

**Success Criteria:**
- On-chain regime correctly identifies known bull/bear periods
- Sentiment RVS generates sensible signals
- Combined system improves Sharpe by 10%+

### Phase 4: Advanced Models (Days 22-35)

**Deliverables:**
- LSTM sequential model
- GNN cross-asset model
- Graph construction pipeline
- Model ensemble system
- Meta-learning wrapper

**Key Files:**
- `models/predictors/lstm_sequential.py`
- `models/predictors/gnn_crossasset.py`
- `data/processing/graph_construction.py`
- `models/predictors/ensemble.py`
- `models/meta_learning/maml_wrapper.py`

**Success Criteria:**
- LSTM adds value over LightGBM
- GNN captures cross-asset dynamics
- Ensemble beats best individual model by 5%+

### Phase 5: Reinforcement Learning (Days 36-42)

**Deliverables:**
- Custom trading environments (Gym)
- Strategic agent (asset allocation)
- Tactical agent (position sizing)
- Execution agent (order placement)
- Hierarchical coordination

**Key Files:**
- `agents/environments/crypto_env.py`
- `agents/hierarchical/strategic_agent.py`
- `agents/hierarchical/tactical_agent.py`
- `agents/hierarchical/execution_agent.py`
- `agents/hierarchical/hierarchical_system.py`

**Success Criteria:**
- Agents learn non-trivial policies
- Hierarchical system coordinates effectively
- RL system improves over supervised baseline

### Phase 6: Production Systems (Days 43-49)

**Deliverables:**
- Risk management enforcement
- Position sizing (Kelly criterion)
- Order execution engine
- Performance monitoring
- Alerting system
- Paper trading mode

**Key Files:**
- `risk/position_sizer.py`
- `risk/risk_manager.py`
- `execution/order_manager.py`
- `execution/execution_engine.py`
- `monitoring/performance.py`
- `orchestrator.py`

**Success Criteria:**
- Risk limits enforced correctly
- Orders execute with expected slippage
- Monitoring dashboard operational
- Paper trading runs 24/7 without crashes

---

## Module Specifications

### Data Layer

#### Exchange Data Ingestion
```python
class ExchangeDataIngester:
    """
    Multi-exchange data fetching with resilience
    
    Supported exchanges: Binance, Coinbase, Kraken
    Data types: OHLCV, order books, trades
    Features: Rate limiting, retry logic, WebSocket streaming
    """
```
**See:** Part 1, Section 1.1 for full implementation

#### On-Chain Analytics
```python
class OnChainDataProvider:
    """
    Blockchain metrics with validated predictive power
    
    Metrics: MVRV Z-Score, SOPR, exchange netflows, SSR
    Source: Glassnode API (paid) or CryptoQuant (free tier)
    Update frequency: Daily (sufficient for regime detection)
    """
```
**See:** Part 1, Section 1.2 for full implementation

#### Sentiment Analysis
```python
class SentimentAnalyzer:
    """
    CARVS method for filtered social sentiment
    
    Sources: Twitter (via tweepy), Reddit (via praw)
    Model: FinBERT for financial text classification
    Filtering: Bot detection, engagement weighting, volume confirmation
    """
```
**See:** Part 1, Section 1.3 for full implementation

### Intelligence Layer

#### Feature Engineering
```python
class FeatureEngineer:
    """
    Generate 100+ validated features
    
    Categories:
    - Price: returns, MAs, distance from MAs
    - Volume: OBV, volume ratios, VWAP
    - Volatility: realized vol, ATR, Bollinger Bands
    - Momentum: RSI, MACD, ADX, ROC
    - Microstructure: spread, Amihud illiquidity
    - Temporal: hour, day of week, session
    """
```
**See:** Part 1, Section 2.1 for full implementation

#### Graph Construction
```python
class CryptoGraphBuilder:
    """
    Build graph structures for GNN
    
    Graph types:
    - Correlation graph: dynamic daily updates
    - Transaction graph: whale wallet networks
    - Heterogeneous graph: multiple node/edge types
    - Temporal graphs: sequences for RNN
    """
```
**See:** Part 1, Section 2.2 for full implementation

#### Models
- **LightGBM Baseline**: Gradient boosting for tabular data
- **LSTM Sequential**: Temporal dependency modeling
- **GNN Cross-Asset**: Graph neural network for relationships
- **Ensemble**: Weighted combination of all models
- **Meta-Learning**: MAML for regime adaptation

**See:** Part 2, Phase 3 for all model implementations

### Decision Layer

#### Hierarchical RL System
```python
class HierarchicalTradingSystem:
    """
    Three-level decision hierarchy
    
    Strategic (weekly): Asset allocation, regime detection
    Tactical (daily): Position sizing, entry/exit timing
    Execution (minutes): Order type, placement timing
    
    Coordination: Phased training with reward propagation
    """
```
**See:** Part 2, Section 4.2 for full implementation

### Risk Layer

#### Position Sizing
```python
class PositionSizer:
    """
    Kelly criterion-based sizing
    
    Methods: Full Kelly, Half Kelly, Fixed Fractional
    Adjustments: Confidence scaling, volatility targeting
    Constraints: Max position 25%, min bet size
    """
```

#### Risk Management
```python
class RiskManager:
    """
    Portfolio-level risk enforcement
    
    Limits:
    - Max position size: 25% of capital
    - Max leverage: 2.0x
    - Max drawdown: 15% (circuit breaker)
    - Max daily loss: 5%
    - Correlation limit: 50% in correlated assets
    """
```
**See:** Part 2, Section 5.1 for full implementation

### Execution Layer

```python
class ExecutionEngine:
    """
    Smart order routing and execution
    
    Features:
    - Best venue selection
    - Limit orders with timeout
    - Market order fallback
    - Slippage modeling
    - Partial fill handling
    """
```

---

## Performance Targets

### Realistic Expectations (First 6 Months)

**Returns:**
- Monthly: 3-8% (conservative target)
- Annualized: 40-100% (before compounding)

**Risk Metrics:**
- Sharpe Ratio: > 1.5
- Max Drawdown: < 15%
- Win Rate: 45-55%
- Profit Factor: > 1.5

**Trading Metrics:**
- Trades per month: 10-30
- Average hold time: 2-5 days
- Commission cost: < 1% of returns

### Comparison to Benchmarks

| Metric | Our Target | Buy-Hold BTC | S&P 500 |
|--------|-----------|-------------|---------|
| Annual Return | 40-100% | ~50% (variable) | ~10% |
| Sharpe Ratio | > 1.5 | ~1.0 | ~0.8 |
| Max Drawdown | < 15% | -70% to -80% | -20% to -30% |
| Volatility | 30-40% | 60-80% | 15-20% |

### Degradation Scenarios

**Expected degradation over time:**
- Year 1: Full performance (new edge)
- Year 2: 70-80% of Year 1 (edge decay)
- Year 3+: 50-60% of Year 1 (market adaptation)

**Mitigation:**
- Continuous model retraining
- New feature discovery
- Regime adaptation
- Strategy diversification

---

## Risk Management

### Position Sizing Rules

```python
# Kelly Criterion (Half Kelly)
kelly_fraction = (win_rate * avg_win - (1 - win_rate)) / avg_win * 0.5

# Adjusted for confidence
position_size = kelly_fraction * confidence * account_balance / risk_per_share

# Volatility targeting
position_size *= (0.20 / asset_volatility)

# Hard caps
position_size = min(position_size, 0.25 * account_balance)
```

### Portfolio Limits

| Limit | Threshold | Action |
|-------|-----------|--------|
| Single Position | 25% | Reject trade |
| Total Leverage | 2.0x | Reject trade |
| Drawdown | 15% | Circuit breaker (stop all trading) |
| Daily Loss | 5% | Halt trading for 24h |
| Correlated Exposure | 50% | Reject correlated trade |

### Circuit Breakers

**Trigger Conditions:**
1. Flash crash detected (>10% move in 5 minutes)
2. Volatility spike (>3x normal)
3. Drawdown exceeds 20%
4. Exchange connectivity issues
5. Data quality degradation

**Actions:**
- Close all positions (market orders if necessary)
- Halt new trades
- Alert user via email/SMS
- Log event for review
- Require manual restart

---

## Deployment Strategy

### Pre-Deployment Checklist

**Backtesting:**
- [ ] 1+ year historical data
- [ ] Multiple market regimes tested
- [ ] Realistic transaction costs included
- [ ] Walk-forward validation positive
- [ ] Out-of-sample testing complete

**Paper Trading:**
- [ ] 2+ weeks live data (no real trades)
- [ ] All risk limits tested
- [ ] Execution working correctly
- [ ] No crashes or data loss
- [ ] Performance meets expectations

**System Validation:**
- [ ] Database backup strategy
- [ ] Monitoring dashboard operational
- [ ] Alerting configured
- [ ] Error logs reviewed
- [ ] Emergency stop procedure tested

### Deployment Phases

**Phase A: Paper Trading (2+ weeks)**
```bash
python orchestrator.py --mode paper --duration 14d
```
- Verify execution quality
- Validate risk management
- Ensure 24/7 operation
- Review all decisions

**Phase B: Minimal Capital ($1,000-2,000)**
```bash
python orchestrator.py --mode live --capital 1000
```
- Real trades, minimal risk
- Validate P&L calculation
- Test order execution
- Monitor for 1+ month

**Phase C: Scale Up (Gradual)**
```bash
# After 1 month profitable
python orchestrator.py --mode live --capital 5000

# After 3 months profitable
python orchestrator.py --mode live --capital 10000

# After 6 months profitable
python orchestrator.py --mode live --capital 25000
```

### Monitoring & Maintenance

**Daily:**
- Check logs for errors
- Review previous day's trades
- Verify data quality
- Monitor key metrics

**Weekly:**
- Retrain models with new data
- Review performance metrics
- Adjust parameters if needed
- Check for regime changes

**Monthly:**
- Full performance analysis
- Strategy effectiveness review
- Risk parameter adjustment
- Model retraining (full)

---

## Appendix: File Structure

```
crypto-trading-system/
├── README.md                   # Project overview
├── CLAUDE.md                   # AI implementation guide
├── PROJECT_SPEC.md            # This file
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
├── .gitignore                # Git ignore patterns
│
├── config/                    # Configuration files
│   ├── exchanges.yaml        # Exchange API configs
│   ├── models.yaml           # Model hyperparameters
│   ├── risk.yaml             # Risk management rules
│   └── strategies.yaml       # Strategy configurations
│
├── data/                      # Data handling
│   ├── ingestion/
│   │   ├── exchanges.py      # Exchange data fetching
│   │   ├── onchain.py        # On-chain data
│   │   └── sentiment.py      # Sentiment analysis
│   ├── processing/
│   │   ├── features.py       # Feature engineering
│   │   └── graph_construction.py
│   └── storage/
│       ├── timeseries_db.py  # Database interface
│       └── cache.py          # Redis caching
│
├── models/                    # ML models
│   ├── predictors/
│   │   ├── lightgbm_baseline.py
│   │   ├── lstm_sequential.py
│   │   ├── gnn_crossasset.py
│   │   └── ensemble.py
│   ├── meta_learning/
│   │   └── maml_wrapper.py
│   └── saved/                # Trained models
│
├── agents/                    # RL agents
│   ├── environments/
│   │   └── crypto_env.py
│   ├── hierarchical/
│   │   ├── strategic_agent.py
│   │   ├── tactical_agent.py
│   │   └── execution_agent.py
│   └── training/
│       └── train_*.py
│
├── risk/                      # Risk management
│   ├── position_sizer.py
│   └── risk_manager.py
│
├── execution/                 # Order execution
│   ├── order_manager.py
│   └── execution_engine.py
│
├── backtesting/              # Backtesting framework
│   ├── engine.py
│   ├── metrics.py
│   └── validation.py
│
├── monitoring/               # Monitoring & alerts
│   ├── performance.py
│   ├── alerting.py
│   └── dashboard.py
│
├── scripts/                  # Utility scripts
│   ├── backfill_data.py
│   ├── train_models.py
│   └── deploy.py
│
├── sql/                      # Database schemas
│   └── schema.sql
│
├── tests/                    # Test suite
│   ├── test_features.py
│   ├── test_models.py
│   └── integration/
│
├── notebooks/                # Analysis notebooks
│   ├── baseline_analysis.ipynb
│   └── performance_review.ipynb
│
└── orchestrator.py           # Main entry point
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial specification |

---

**End of Specification**

For implementation guidance, see **CLAUDE.md**  
For research details, see **crypto_trading_research.md**
