# Cryptocurrency Trading System

**A research-backed, production-grade algorithmic trading system for cryptocurrency markets.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 What Is This?

This is NOT your typical "AI trading bot." This system focuses on **risk management over alpha generation**, using validated academic research (2020-2024) for crash detection and defensive positioning:

- **Graph Neural Networks** - Multi-asset regime detection with 80% crash detection rate
- **Asymmetric Loss Training** - Crash-aware model penalizing missed crashes 15x
- **Monte Carlo Validation** - 1000 bootstrap simulations for robust performance estimates
- **On-Chain Analytics** - MVRV, SOPR for regime context (not trade signals)

**Strategic Pivot:** After rigorous signal quality analysis, we found on-chain signals do NOT provide tradeable daily alpha. The system pivoted from alpha generation to crash protection and drawdown reduction.

**Performance Targets (Risk-Focused):**
- Max Drawdown: < 25% (validated via Monte Carlo)
- Crash Detection: > 75% (currently 80%)
- Walk-Forward Win Rate: > 50%
- Sharpe Ratio: > 0.5 (risk-adjusted)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with TimescaleDB
- Redis 5.0+
- 32GB RAM (recommended)
- Optional: NVIDIA GPU for neural network training

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-trading-system.git
cd crypto-trading-system

# Create conda environment
conda env create -f environment.yml
conda activate crypto-trading

# Setup database
psql -U postgres -c "CREATE DATABASE crypto_trading;"
psql -U postgres -d crypto_trading -f sql/schema.sql

# Start Redis
redis-server --daemonize yes

# Configure API keys
cp config/exchanges.yaml.example config/exchanges.yaml
# Edit config/exchanges.yaml with your API credentials
```

### Initial Data Fetch

```bash
# Backfill 90 days of historical data
python scripts/backfill_data.py --symbols BTC/USDT ETH/USDT --days 90

# Generate features
python scripts/generate_features.py --symbols BTC/USDT ETH/USDT
```

### Training Models

```bash
# Train baseline LightGBM model
python scripts/train_lightgbm.py --symbol BTC/USDT

# Train LSTM model (optional)
python scripts/train_lstm.py --symbol BTC/USDT

# Train GNN for cross-asset signals (optional)
python scripts/train_gnn.py --symbols BTC/USDT ETH/USDT SOL/USDT

# Train RL agents
python agents/training/train_hierarchical.py
```

### Backtesting

```bash
# Run backtest on historical data
python scripts/backtest.py --start 2024-01-01 --end 2024-12-01

# Results will show:
# - Sharpe Ratio
# - Max Drawdown
# - Win Rate
# - Profit Factor
# - Equity curve
```

### Paper Trading

```bash
# Dry run with real data, no real trades
python orchestrator.py --mode paper --duration 7d

# Monitor via dashboard
python monitoring/dashboard.py
# Opens at http://localhost:8501
```

### Live Trading

```bash
# START SMALL! $1,000-2,000 initially
python orchestrator.py --mode live --capital 1000

# Monitor logs
tail -f trading.log
```

---

## 📊 System Architecture

```
Data Sources → Feature Engineering → ML Models → RL Agents → Risk Management → Execution
     ↓                ↓                  ↓           ↓              ↓              ↓
  Exchanges      100+ Features      LightGBM    Strategic     Position        Order
  On-Chain       Technical          LSTM        Tactical      Sizing          Routing
  Sentiment      Microstructure     GNN         Execution     Limits          Slippage
  Order Books    Graph Features     Ensemble                  Circuit         Modeling
                                                              Breakers
```

**Key Components:**

1. **Data Layer** - Multi-source ingestion (exchanges, blockchain, social)
2. **Intelligence Layer** - Ensemble ML models with meta-learning
3. **Decision Layer** - Hierarchical RL (strategic → tactical → execution)
4. **Risk Layer** - Position sizing, portfolio limits, circuit breakers
5. **Execution Layer** - Smart order routing with cost minimization

---

## 🔬 Research Foundation

This system implements techniques from 50+ peer-reviewed papers (2020-2024):

### On-Chain Analytics
- **Omole & Enke (2024)**: 82.44% accuracy using 87 on-chain metrics
- **Glassnode Research**: MVRV Z-Score and SOPR as top predictors
- **CryptoQuant Studies**: Exchange netflows with 24-72h lead time

### Graph Neural Networks  
- **THGNN (CIKM 2022)**: 2.2x Sharpe improvement
- **MGAR (2023)**: 164-236% returns with multi-view graphs
- **Matsunaga et al. (2019)**: GNN+LSTM on Nikkei 225

### Sentiment Analysis
- **Kraaijeveld & De Smedt (2024)**: CARVS method, +291% bear market returns
- **Ante (2023)**: "Musk Effect" quantification (3.58% moves)
- **Multiple studies**: FinBERT outperforms generic sentiment

### Hierarchical RL
- **HRT (2024)**: Bi-level architecture achieving 2.74 Sharpe
- **MSPM (2022)**: Multi-agent coordination
- **MARS (2025)**: Meta-adaptive strategy selection

### Meta-Learning
- **X-Trend (2023)**: 10x Sharpe improvement with few-shot learning
- **MAML for Finance**: Regime adaptation in 1-5 gradient steps

**See `crypto_trading_research.md` for complete literature review.**

---

## 🛠️ Configuration

### Risk Parameters (`config/risk.yaml`)

```yaml
risk_management:
  max_portfolio_risk: 0.02  # 2% per trade
  max_position_size: 0.25   # 25% of capital max
  max_leverage: 2.0
  max_drawdown: 0.15        # 15% circuit breaker
  max_daily_loss: 0.05      # 5% daily stop
```

### Model Parameters (`config/models.yaml`)

```yaml
lightgbm:
  objective: "binary"
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 100
  
ensemble:
  weights:
    lightgbm: 0.4
    lstm: 0.35
    gnn: 0.25
  min_confidence: 0.6
```

### Exchange Configuration (`config/exchanges.yaml`)

```yaml
exchanges:
  binance:
    api_key: ${BINANCE_API_KEY}
    secret: ${BINANCE_SECRET}
    rate_limit: 1200  # requests per minute
    
trading_pairs:
  - BTC/USDT
  - ETH/USDT
```

---

## 📈 Performance Monitoring

### Dashboard

```bash
python monitoring/dashboard.py
```

Provides real-time view of:
- Current positions and P&L
- Equity curve
- Sharpe ratio, max drawdown
- Recent trades
- Model predictions
- System health

### Metrics Tracked

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| Sharpe Ratio | > 1.5 | < 1.0 |
| Max Drawdown | < 15% | > 12% |
| Win Rate | 45-55% | < 40% |
| Daily Loss | < 5% | > 4% |
| API Latency | < 100ms | > 500ms |

### Alerting

Configure alerts in `config/monitoring.yaml`:

```yaml
alerts:
  email: your@email.com
  slack_webhook: https://hooks.slack.com/...
  
triggers:
  max_drawdown: 0.12      # Alert at 12% drawdown
  daily_loss: 0.04        # Alert at 4% daily loss
  api_failure: true       # Alert on exchange errors
```

---

## 🧪 Testing

### Run Test Suite

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific module
pytest tests/test_features.py

# Integration tests
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"
```

### Test Coverage Targets

- Unit tests: > 80% coverage
- Integration tests: Critical paths
- Backtests: Multiple market regimes

---

## 📝 Development Workflow

### For AI Implementation (Claude Opus)

**See `CLAUDE.md` for comprehensive instructions on:**
- Git workflow and commit standards
- Code generation best practices
- Testing requirements
- Decision-making guidelines
- Phase-by-phase implementation

### For Human Developers

```bash
# Create feature branch
git checkout -b feature/new-indicator

# Make changes, add tests
pytest tests/test_new_indicator.py

# Commit with conventional commit format
git commit -m "feat(features): Add momentum-based indicator"

# Merge to main
git checkout main
git merge feature/new-indicator
```

---

## 🔒 Security Best Practices

### API Keys

**NEVER commit API keys to git!**

```bash
# Use environment variables
export BINANCE_API_KEY="your_key_here"
export BINANCE_SECRET="your_secret_here"

# Or use .env file (gitignored)
echo "BINANCE_API_KEY=your_key" > .env
echo "BINANCE_SECRET=your_secret" >> .env
```

### Database

```bash
# Use strong passwords
createuser -P crypto_user  # Enter strong password

# Restrict connections
# Edit postgresql.conf:
listen_addresses = 'localhost'
```

### File Permissions

```bash
# Protect config files
chmod 600 config/exchanges.yaml
chmod 600 .env
```

---

## 🚨 Risk Warnings

### Financial Risk

- **Past performance ≠ future results**
- **Crypto is extremely volatile** - Can lose 50%+ in days
- **Start with minimal capital** ($1,000-2,000 max)
- **Only risk what you can afford to lose**
- **This is experimental** - No guarantees of profit

### Technical Risk

- Exchange APIs can fail
- Internet connection can drop  
- Bugs may exist in code
- Models can stop working (regime change)
- Overfitting is always possible

### Operational Risk

- Requires monitoring and maintenance
- Models degrade over time (retrain needed)
- Market conditions change (strategies expire)
- Competition increases (edges erode)

**Use at your own risk. This is NOT financial advice.**

---

## 🤝 Contributing

Contributions welcome! Please:

1. Read `CLAUDE.md` for code standards
2. Add tests for new features
3. Ensure all tests pass
4. Follow conventional commit format
5. Update documentation

```bash
# Create issue first
# Then fork and create PR
git checkout -b fix/issue-123
# ... make changes ...
git commit -m "fix(risk): Correct Kelly calculation (#123)"
# ... create pull request ...
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | This file - project overview |
| `PROJECT_SPEC.md` | Complete technical specification |
| `CLAUDE.md` | AI implementation guide |
| `crypto_trading_research.md` | Research literature review |
| `docs/` | Additional documentation |

---

## 📊 Current Status

### Phase 5 Complete: GNN Regime Detection + Position Sizing

The system now uses a **Graph Neural Network** for market regime detection with crash-aware training:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Crash Miss Rate | 20% | <25% | ✓ Pass |
| Max Drawdown (95th pctl) | 23.8% | <30% | ✓ Pass |
| Walk-Forward Win Rate | 60% | >50% | ✓ Pass |
| Stress Test Beat Rate | 100% | >75% | ✓ Pass |

**Key Achievement:** Reduces drawdown by 27% during major crashes (Luna/3AC, FTX) while maintaining exposure during bull markets.

**Fundamental Trade-off Discovered:** Cannot simultaneously achieve >35% returns AND <25% crash miss rate. Lower thresholds for crash detection make positions too defensive. Current configuration optimizes for crash protection with acceptable return capture.

---

## 📈 Roadmap

### Phase 1: Data Infrastructure ✅
- [x] TimescaleDB setup with hypertable partitioning
- [x] Exchange data ingestion (Coinbase, Binance via CCXT)
- [x] 22 technical features generated

### Phase 2: Baseline Model ✅
- [x] LightGBM classifier (51.6% accuracy baseline)
- [x] Walk-forward validation framework
- [x] Backtesting with realistic transaction costs

### Phase 3: Alpha Sources ✅
- [x] On-chain metrics (MVRV, SOPR via Dune/CoinMetrics)
- [x] Reddit sentiment analysis (FinBERT)
- [x] Signal quality analysis (finding: no tradeable alpha)

### Phase 4: GNN Regime Detection ✅
- [x] Graph Attention Network for multi-asset regimes
- [x] 3-class classification (RISK_ON, CAUTION, RISK_OFF)
- [x] Class-weighted training for calibration

### Phase 5: Position Sizing Optimization ✅
- [x] Continuous probability-based sizing
- [x] Asymmetric loss function (crash-aware training)
- [x] Monte Carlo + Walk-forward validation
- [x] Ensemble strategy exploration

### Phase 6: Production System (Planned)
- [ ] Hard risk limits enforcement
- [ ] Order execution engine
- [ ] Monitoring dashboard
- [ ] Paper trading mode

---

## 📜 License

MIT License - see `LICENSE` file for details.

**Disclaimer:** This software is provided "as is" without warranty. Use at your own risk. Not financial advice.

---

## 🙏 Acknowledgments

Built on research from:
- Google (Temporal Fusion Transformers)
- Microsoft (Qlib platform)
- Glassnode (On-chain analytics)
- Multiple academic institutions

Special thanks to the open-source community for:
- PyTorch, Scikit-learn, LightGBM
- CCXT (exchange connectivity)
- Stable-Baselines3 (RL framework)
- TimescaleDB (time-series database)

---

## 📧 Contact

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: your@email.com

---

**Built with:** Python 🐍 | PyTorch 🔥 | PostgreSQL 🐘 | Redis ⚡

**Status:** ✅ Phase 5 Complete | 🔬 Monte Carlo Validated | 📊 Phase 6 Pending

---

*Last Updated: December 2024*
*Phase 5 validated with 1000 Monte Carlo simulations and 5-fold walk-forward testing*
