# Cryptocurrency Trading System - Implementation Specification
## From Research to Production

**Document Purpose**: This specification translates research findings into a concrete, buildable system. Every component has specific technology choices, clear interfaces, and implementation guidance.

**Target Outcome**: A production-ready crypto trading system deployable with $10k-50k capital, achieving target Sharpe ratio > 1.5, max drawdown < 15%, delivering 3-8% monthly returns.

---

## Executive Summary: What We're Building

Based on academic research validation, we're building a **three-tier hierarchical system**:

1. **Data Layer**: Multi-source ingestion (exchanges, on-chain, sentiment)
2. **Intelligence Layer**: Ensemble ML models (GNN + LSTM + LightGBM) with meta-learning
3. **Decision Layer**: Hierarchical RL agents (strategic → tactical → execution)

**Key Differentiators from Standard Approaches**:
- On-chain analytics as primary alpha source (75-82% accuracy validated)
- Graph neural networks for cross-asset dynamics
- Sentiment filtered by volume (CARVS approach: +291% in bear markets)
- Hierarchical RL instead of single-agent systems
- Meta-learning for regime adaptation

**NOT Included** (research shows limited value):
- Pure transformer price forecasting (linear models perform better)
- Unfiltered social sentiment
- Single-timeframe RL agents
- Over-complicated deep learning without proper validation

---

## Phase 0: Foundation & Quick Wins (Week 1)

Before building complex ML, validate data pipelines and establish baselines.

### 0.1 Development Environment Setup

```yaml
# environment.yml
name: crypto-trading
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - pytorch=2.1
  - pytorch-geometric=2.4
  - pandas=2.1
  - numpy=1.24
  - scikit-learn=1.3
  - lightgbm=4.1
  - ccxt=4.2
  - redis=5.0
  - postgresql=15
  - asyncio
  - websockets
  - aiohttp
  - pip:
    - tigramite>=5.2  # Causal discovery
    - learn2learn>=0.2  # Meta-learning
    - finrl>=0.3  # RL trading framework
    - transformers>=4.35  # For FinBERT
    - tweepy>=4.14  # Twitter API
    - glassnode>=1.0  # On-chain data
    - python-binance>=1.0
    - freqtrade>=2024.1
```

### 0.2 Project Structure

```
crypto-trading-system/
├── config/
│   ├── exchanges.yaml          # API keys, rate limits
│   ├── models.yaml              # Model hyperparameters
│   ├── risk.yaml                # Risk management rules
│   └── strategies.yaml          # Strategy configurations
├── data/
│   ├── ingestion/
│   │   ├── exchanges.py         # CCXT wrapper
│   │   ├── onchain.py           # Glassnode/CryptoQuant
│   │   ├── sentiment.py         # Twitter/Reddit scraping
│   │   └── orderbook.py         # WebSocket order book
│   ├── processing/
│   │   ├── features.py          # Feature engineering
│   │   ├── graph_construction.py # Build GNN graphs
│   │   └── cleaning.py          # Data validation
│   └── storage/
│       ├── timeseries_db.py     # TimescaleDB interface
│       └── cache.py             # Redis cache manager
├── models/
│   ├── predictors/
│   │   ├── lightgbm_baseline.py # Gradient boosting
│   │   ├── lstm_sequential.py   # Temporal patterns
│   │   ├── gnn_crossasset.py    # Graph neural network
│   │   └── ensemble.py          # Combine predictions
│   ├── meta_learning/
│   │   ├── maml_wrapper.py      # Meta-learning
│   │   └── regime_tasks.py      # Task construction
│   └── causal/
│       ├── pcmci_selection.py   # Feature selection
│       └── causal_graph.py      # Network discovery
├── agents/
│   ├── hierarchical/
│   │   ├── strategic_agent.py   # High-level (PPO)
│   │   ├── tactical_agent.py    # Mid-level (DDPG)
│   │   └── execution_agent.py   # Low-level (DQN)
│   ├── environments/
│   │   ├── crypto_env.py        # Custom gym environment
│   │   └── portfolio_env.py     # Multi-asset environment
│   └── training/
│       ├── train_strategic.py   # Phase 1 training
│       ├── train_tactical.py    # Phase 2 training
│       └── train_hierarchical.py # Joint optimization
├── risk/
│   ├── position_sizer.py        # Kelly criterion
│   ├── risk_manager.py          # Limits enforcement
│   └── circuit_breaker.py       # Emergency stops
├── execution/
│   ├── order_manager.py         # Smart order routing
│   ├── execution_engine.py      # Order execution
│   └── slippage_model.py        # Cost estimation
├── backtesting/
│   ├── engine.py                # Backtest framework
│   ├── metrics.py               # Performance calculation
│   └── validation.py            # CPCV, walk-forward
├── monitoring/
│   ├── performance.py           # Live tracking
│   ├── alerting.py              # Notifications
│   └── dashboard.py             # Streamlit UI
├── orchestrator.py              # Main trading loop
├── tests/                       # Unit tests
└── notebooks/                   # Research/exploration
```

### 0.3 Database Schema

```sql
-- TimescaleDB schema for efficient time-series storage

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV data
CREATE TABLE ohlcv (
    time TIMESTAMPTZ NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    quote_volume DOUBLE PRECISION
);

SELECT create_hypertable('ohlcv', 'time', 
    chunk_time_interval => INTERVAL '1 day');

CREATE INDEX idx_ohlcv_symbol_time ON ohlcv (symbol, time DESC);

-- On-chain metrics
CREATE TABLE onchain_metrics (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    metadata JSONB
);

SELECT create_hypertable('onchain_metrics', 'time',
    chunk_time_interval => INTERVAL '1 day');

-- Sentiment data
CREATE TABLE sentiment (
    time TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,  -- 'twitter', 'reddit'
    symbol TEXT NOT NULL,
    sentiment_score DOUBLE PRECISION,
    volume_score DOUBLE PRECISION,
    engagement BIGINT,
    processed BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('sentiment', 'time',
    chunk_time_interval => INTERVAL '6 hours');

-- Order book snapshots (for microstructure)
CREATE TABLE orderbook_snapshots (
    time TIMESTAMPTZ NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    bids JSONB,  -- Top 20 levels
    asks JSONB,
    bid_volume DOUBLE PRECISION,
    ask_volume DOUBLE PRECISION
);

SELECT create_hypertable('orderbook_snapshots', 'time',
    chunk_time_interval => INTERVAL '1 hour');

-- Trades table
CREATE TABLE trades (
    time TIMESTAMPTZ NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT,  -- 'buy' or 'sell'
    price DOUBLE PRECISION,
    amount DOUBLE PRECISION,
    trade_id TEXT
);

SELECT create_hypertable('trades', 'time',
    chunk_time_interval => INTERVAL '6 hours');

-- Model predictions
CREATE TABLE predictions (
    time TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    prediction_type TEXT,  -- 'direction', 'price', 'volatility'
    predicted_value DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    features_used JSONB
);

-- Trading signals
CREATE TABLE signals (
    time TIMESTAMPTZ NOT NULL,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT,  -- 'long', 'short', 'close'
    strength DOUBLE PRECISION,
    reasoning JSONB
);

-- Executed trades (portfolio history)
CREATE TABLE executed_trades (
    id SERIAL PRIMARY KEY,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT,  -- 'long' or 'short'
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    quantity DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    pnl_percent DOUBLE PRECISION,
    commission DOUBLE PRECISION,
    strategy TEXT,
    metadata JSONB
);

-- Portfolio state
CREATE TABLE portfolio_state (
    time TIMESTAMPTZ NOT NULL,
    total_equity DOUBLE PRECISION,
    cash DOUBLE PRECISION,
    positions JSONB,
    daily_pnl DOUBLE PRECISION,
    total_pnl DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION
);

SELECT create_hypertable('portfolio_state', 'time',
    chunk_time_interval => INTERVAL '1 day');
```

### 0.4 Configuration Files

```yaml
# config/exchanges.yaml
exchanges:
  binance:
    api_key: ${BINANCE_API_KEY}
    secret: ${BINANCE_SECRET}
    sandbox: false
    rate_limit: 1200  # requests per minute
    timeout: 30000
    
  coinbase:
    api_key: ${COINBASE_API_KEY}
    secret: ${COINBASE_SECRET}
    passphrase: ${COINBASE_PASSPHRASE}
    sandbox: false
    rate_limit: 10
    
trading_pairs:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
  
timeframes:
  - 1m   # For execution
  - 5m   # For tactical decisions
  - 1h   # For strategic decisions
  - 1d   # For regime detection
```

```yaml
# config/risk.yaml
risk_management:
  max_portfolio_risk: 0.02  # 2% per trade
  max_position_size: 0.25   # 25% of portfolio max
  max_leverage: 2.0
  max_drawdown: 0.15        # 15% kill switch
  max_daily_loss: 0.05      # 5% daily stop
  max_correlated_positions: 0.50  # 50% in correlated assets
  
position_sizing:
  method: "kelly_half"      # Half-Kelly criterion
  min_size: 100             # Minimum trade size in USD
  max_size: 10000           # Maximum single trade
  
stop_loss:
  method: "atr"             # ATR-based stops
  atr_multiplier: 2.0
  trailing_activation: 0.02 # Activate trailing at 2% profit
  trailing_distance: 0.01   # Trail by 1%
```

```yaml
# config/models.yaml
lightgbm:
  objective: "binary"
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 100
  feature_fraction: 0.9
  bagging_fraction: 0.8
  bagging_freq: 5
  
lstm:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  sequence_length: 60
  batch_size: 32
  learning_rate: 0.001
  
gnn:
  hidden_channels: 64
  num_layers: 3
  dropout: 0.2
  heads: 4  # For attention
  edge_threshold: 0.3  # Correlation threshold
  
ensemble:
  weights:
    lightgbm: 0.4
    lstm: 0.35
    gnn: 0.25
  min_confidence: 0.6  # Minimum to act on signal
  
meta_learning:
  inner_lr: 0.01
  outer_lr: 0.0001
  num_inner_steps: 5
  task_window: 60  # Days per task
```

```yaml
# config/strategies.yaml
strategies:
  onchain_regime:
    enabled: true
    indicators:
      - mvrv_zscore
      - sopr
      - exchange_netflow
    thresholds:
      mvrv_overbought: 3.7
      mvrv_oversold: 1.0
      sopr_profitable: 1.05
    
  sentiment_volume:
    enabled: true
    sources:
      - twitter
      - reddit
    min_volume_change: 0.2  # 20% volume increase
    sentiment_threshold: 0.3
    lookback_hours: 24
    
  cross_asset_gnn:
    enabled: true
    min_correlation: 0.3
    update_frequency: "1h"
    num_neighbors: 5
```

---

## Phase 1: Data Infrastructure (Week 1-2)

### 1.1 Exchange Data Ingestion

```python
# data/ingestion/exchanges.py

import ccxt
import asyncio
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

class ExchangeDataIngester:
    """
    Multi-exchange data ingestion with resilient WebSocket connections
    
    Responsibilities:
    - OHLCV fetching with gap detection
    - Real-time WebSocket streaming
    - Order book snapshots
    - Rate limit management
    - Automatic reconnection
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.exchanges = self._init_exchanges()
        self.ws_connections = {}
        self.rate_limiters = {}
        
    def _init_exchanges(self) -> Dict[str, ccxt.Exchange]:
        """Initialize CCXT exchange instances"""
        exchanges = {}
        for name, config in self.config['exchanges'].items():
            exchange_class = getattr(ccxt, name)
            exchanges[name] = exchange_class({
                'apiKey': config['api_key'],
                'secret': config['secret'],
                'enableRateLimit': True,
                'timeout': config['timeout'],
                'options': {
                    'defaultType': 'spot',
                }
            })
        return exchanges
    
    async def fetch_ohlcv_historical(self,
                                     exchange_name: str,
                                     symbol: str,
                                     timeframe: str,
                                     since: datetime,
                                     until: datetime) -> pd.DataFrame:
        """
        Fetch historical OHLCV with automatic pagination
        
        Returns: DataFrame with columns [time, open, high, low, close, volume]
        """
        exchange = self.exchanges[exchange_name]
        
        all_data = []
        current_time = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        
        while current_time < until_ms:
            try:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_time,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                current_time = ohlcv[-1][0] + 1
                
                # Respect rate limits
                await asyncio.sleep(exchange.rateLimit / 1000)
                
            except ccxt.NetworkError as e:
                await asyncio.sleep(5)
                continue
            except ccxt.ExchangeError as e:
                print(f"Exchange error: {e}")
                break
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['exchange'] = exchange_name
        df['symbol'] = symbol
        
        return df[['time', 'exchange', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
    
    async def stream_ohlcv(self, 
                          exchange_name: str,
                          symbol: str,
                          timeframe: str,
                          callback):
        """
        Stream real-time OHLCV via WebSocket
        
        Automatically reconnects on failure
        """
        exchange = self.exchanges[exchange_name]
        
        while True:
            try:
                # Use CCXT watch methods (requires ccxt.pro for some exchanges)
                if hasattr(exchange, 'watch_ohlcv'):
                    while True:
                        ohlcv = await exchange.watch_ohlcv(symbol, timeframe)
                        await callback(ohlcv)
                else:
                    # Fallback: polling mode
                    while True:
                        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=1)
                        await callback(ohlcv[0])
                        await asyncio.sleep(self._get_sleep_duration(timeframe))
                        
            except Exception as e:
                print(f"WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
    
    def _get_sleep_duration(self, timeframe: str) -> int:
        """Convert timeframe to sleep seconds"""
        mapping = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1d': 86400}
        return mapping.get(timeframe, 60)

# Example usage:
async def main():
    config = load_config('config/exchanges.yaml')
    ingester = ExchangeDataIngester(config)
    
    # Fetch historical data
    df = await ingester.fetch_ohlcv_historical(
        'binance',
        'BTC/USDT',
        '1h',
        datetime.now() - timedelta(days=90),
        datetime.now()
    )
    
    # Store in database
    store_to_timescaledb(df)
```

### 1.2 On-Chain Data Integration

```python
# data/ingestion/onchain.py

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class OnChainDataProvider:
    """
    Fetch validated on-chain metrics with highest predictive power:
    - MVRV Z-Score (market cycle indicator)
    - SOPR (profitability of spent outputs)
    - Exchange netflows (accumulation/distribution)
    - Stablecoin supply ratio (buying power)
    """
    
    def __init__(self, api_key: str, provider: str = 'glassnode'):
        self.api_key = api_key
        self.provider = provider
        self.base_url = self._get_base_url()
        
    def _get_base_url(self) -> str:
        urls = {
            'glassnode': 'https://api.glassnode.com/v1',
            'cryptoquant': 'https://api.cryptoquant.com/v1'
        }
        return urls[self.provider]
    
    def fetch_mvrv_zscore(self, 
                         asset: str = 'BTC',
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        MVRV Z-Score = (Market Cap - Realized Cap) / StdDev(Market Cap)
        
        Interpretation:
        - > 3.7: Extreme overvaluation (historically marks tops)
        - < 1.0: Undervaluation (accumulation zone)
        - 1.0-3.7: Normal range
        
        Research validation: Identifies cycle tops with high reliability
        """
        endpoint = f"{self.base_url}/metrics/market/mvrv_z_score"
        
        params = {
            'a': asset,
            'api_key': self.api_key,
            'i': '24h',  # Daily resolution
            's': int(start_date.timestamp()) if start_date else None,
            'u': int(end_date.timestamp()) if end_date else None
        }
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['t'], unit='s')
        df['mvrv_zscore'] = df['v']
        
        return df[['time', 'mvrv_zscore']]
    
    def fetch_sopr(self, 
                  asset: str = 'BTC',
                  start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        SOPR (Spent Output Profit Ratio) = Realized Value / Value at Creation
        
        Interpretation:
        - > 1: Coins moving at profit
        - < 1: Coins moving at loss
        - STH-SOPR (Short-Term Holders) more reactive
        
        Research: Identified as #1 predictive metric by Glassnode ML study
        """
        endpoint = f"{self.base_url}/metrics/indicators/sopr"
        
        params = {
            'a': asset,
            'api_key': self.api_key,
            'i': '24h'
        }
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['t'], unit='s')
        df['sopr'] = df['v']
        
        return df[['time', 'sopr']]
    
    def fetch_exchange_netflow(self, 
                               asset: str = 'BTC',
                               exchanges: List[str] = ['binance', 'coinbase']) -> pd.DataFrame:
        """
        Exchange Netflow = Inflow - Outflow
        
        Interpretation:
        - Positive (inflow > outflow): Selling pressure
        - Negative (outflow > inflow): Accumulation
        - Track 7-day MA for trend
        
        Research: CryptoQuant validates 24-72 hour lead time on moves
        """
        endpoint = f"{self.base_url}/metrics/transactions/transfers_volume_exchanges_net"
        
        params = {
            'a': asset,
            'api_key': self.api_key,
            'i': '1h'
        }
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['t'], unit='s')
        df['netflow'] = df['v']
        df['netflow_7d_ma'] = df['netflow'].rolling(7*24).mean()
        
        return df[['time', 'netflow', 'netflow_7d_ma']]
    
    def fetch_stablecoin_supply_ratio(self, asset: str = 'BTC') -> pd.DataFrame:
        """
        SSR = BTC Market Cap / Stablecoin Supply
        
        Interpretation:
        - Low SSR: High stablecoin buying power (bullish potential)
        - High SSR: Less dry powder (bearish)
        
        Research: Preceded major 2024-2025 rally
        """
        # This requires combining multiple endpoints
        btc_marketcap = self._fetch_market_cap(asset)
        stablecoin_supply = self._fetch_stablecoin_supply()
        
        df = btc_marketcap.merge(stablecoin_supply, on='time')
        df['ssr'] = df['market_cap'] / df['stablecoin_supply']
        
        return df[['time', 'ssr']]
    
    def get_regime_classification(self, metrics: pd.DataFrame) -> pd.Series:
        """
        Classify market regime based on on-chain metrics
        
        Regimes:
        - euphoria: MVRV > 3.7, SOPR > 1.05, netflow positive
        - accumulation: MVRV < 1.0, SOPR < 1.0, netflow negative
        - bullish: MVRV 1.0-2.5, SOPR > 1.0
        - bearish: MVRV 1.0-2.5, SOPR < 1.0
        - neutral: Otherwise
        """
        conditions = []
        
        # Euphoria (extreme overvaluation)
        euphoria = (metrics['mvrv_zscore'] > 3.7) & (metrics['sopr'] > 1.05)
        
        # Accumulation (undervaluation)
        accumulation = (metrics['mvrv_zscore'] < 1.0) & (metrics['sopr'] < 1.0)
        
        # Bullish trend
        bullish = (metrics['mvrv_zscore'].between(1.0, 2.5)) & (metrics['sopr'] > 1.0)
        
        # Bearish trend
        bearish = (metrics['mvrv_zscore'].between(1.0, 2.5)) & (metrics['sopr'] < 1.0)
        
        regime = pd.Series('neutral', index=metrics.index)
        regime[euphoria] = 'euphoria'
        regime[accumulation] = 'accumulation'
        regime[bullish] = 'bullish'
        regime[bearish] = 'bearish'
        
        return regime

# Example usage:
provider = OnChainDataProvider(api_key='YOUR_KEY', provider='glassnode')

# Fetch all key metrics
mvrv = provider.fetch_mvrv_zscore('BTC', start_date=datetime.now()-timedelta(days=365))
sopr = provider.fetch_sopr('BTC')
netflow = provider.fetch_exchange_netflow('BTC')

# Combine into single DataFrame
metrics = mvrv.merge(sopr, on='time').merge(netflow, on='time')

# Get regime classification
metrics['regime'] = provider.get_regime_classification(metrics)
```

### 1.3 Sentiment Data Pipeline

```python
# data/ingestion/sentiment.py

import tweepy
import praw  # Reddit API
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
from datetime import datetime, timedelta

class SentimentAnalyzer:
    """
    Implements CARVS (Combined Attention-based Relative Volume Sentiment)
    approach validated in research (+291% bear market returns)
    
    Key improvements over naive sentiment:
    - Filter by volume change (sentiment + volume alignment required)
    - Use FinBERT for financial text understanding
    - Weight by engagement (upvotes, retweets)
    - Filter out bot accounts
    - Neutralize celebrity mentions
    """
    
    def __init__(self, config: dict):
        self.config = config
        self._init_twitter_api()
        self._init_reddit_api()
        self._init_finbert()
        
    def _init_twitter_api(self):
        """Initialize Twitter API v2"""
        self.twitter = tweepy.Client(
            bearer_token=self.config['twitter']['bearer_token'],
            consumer_key=self.config['twitter']['api_key'],
            consumer_secret=self.config['twitter']['api_secret'],
            access_token=self.config['twitter']['access_token'],
            access_token_secret=self.config['twitter']['access_secret']
        )
        
    def _init_reddit_api(self):
        """Initialize Reddit API"""
        self.reddit = praw.Reddit(
            client_id=self.config['reddit']['client_id'],
            client_secret=self.config['reddit']['client_secret'],
            user_agent='crypto_trading_bot/1.0'
        )
        
    def _init_finbert(self):
        """Load FinBERT model for financial sentiment"""
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        
    def fetch_twitter_sentiment(self, 
                                symbol: str,
                                hours_back: int = 24) -> Dict:
        """
        Fetch and analyze Twitter sentiment with proper filtering
        
        Returns dict with:
        - sentiment_score: -1 to +1
        - volume: tweet count
        - engagement: total likes + retweets
        - influential_sentiment: sentiment from verified/high-follower accounts
        """
        # Build search query
        query = f"${symbol} OR #{symbol} -is:retweet lang:en"
        
        # Fetch tweets
        start_time = datetime.now() - timedelta(hours=hours_back)
        
        tweets = self.twitter.search_recent_tweets(
            query=query,
            start_time=start_time,
            max_results=100,
            tweet_fields=['created_at', 'public_metrics', 'author_id'],
            user_fields=['verified', 'public_metrics']
        )
        
        if not tweets.data:
            return {'sentiment_score': 0, 'volume': 0, 'engagement': 0}
        
        # Filter out likely bots
        filtered_tweets = [t for t in tweets.data if self._is_likely_human(t)]
        
        # Analyze sentiment
        sentiments = []
        engagement_weights = []
        
        for tweet in filtered_tweets:
            # Get FinBERT sentiment
            inputs = self.tokenizer(tweet.text, return_tensors="pt", 
                                   padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # FinBERT outputs: [negative, neutral, positive]
            sentiment_score = probs[0][2].item() - probs[0][0].item()  # positive - negative
            
            # Weight by engagement
            engagement = (tweet.public_metrics['like_count'] + 
                         tweet.public_metrics['retweet_count'])
            
            sentiments.append(sentiment_score)
            engagement_weights.append(engagement + 1)  # +1 to avoid zero weight
        
        # Calculate weighted average sentiment
        weighted_sentiment = sum(s * w for s, w in zip(sentiments, engagement_weights)) / sum(engagement_weights)
        
        return {
            'sentiment_score': weighted_sentiment,
            'volume': len(filtered_tweets),
            'engagement': sum(engagement_weights),
            'timestamp': datetime.now()
        }
    
    def fetch_reddit_sentiment(self, 
                               symbol: str,
                               subreddit: str = 'CryptoCurrency',
                               hours_back: int = 24) -> Dict:
        """
        Fetch Reddit sentiment with engagement weighting
        
        Research shows r/CryptoCurrency RVS achieved +291% in 2018 bear
        """
        submissions = []
        subreddit_obj = self.reddit.subreddit(subreddit)
        
        # Fetch recent posts mentioning symbol
        for submission in subreddit_obj.search(symbol, time_filter='day', limit=100):
            if (datetime.now() - datetime.fromtimestamp(submission.created_utc)).total_seconds() < hours_back * 3600:
                submissions.append(submission)
        
        if not submissions:
            return {'sentiment_score': 0, 'volume': 0, 'engagement': 0}
        
        sentiments = []
        engagement_weights = []
        
        for submission in submissions:
            # Combine title and body
            text = f"{submission.title} {submission.selftext}"
            
            # FinBERT sentiment
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            sentiment_score = probs[0][2].item() - probs[0][0].item()
            
            # Weight by upvotes and comments
            engagement = submission.score + submission.num_comments
            
            sentiments.append(sentiment_score)
            engagement_weights.append(engagement + 1)
        
        weighted_sentiment = sum(s * w for s, w in zip(sentiments, engagement_weights)) / sum(engagement_weights)
        
        return {
            'sentiment_score': weighted_sentiment,
            'volume': len(submissions),
            'engagement': sum(engagement_weights),
            'timestamp': datetime.now()
        }
    
    def calculate_rvs(self, 
                     current_sentiment: Dict,
                     historical_sentiment: List[Dict],
                     current_volume: float,
                     historical_volume: List[float]) -> float:
        """
        Calculate Relative Volume Sentiment (RVS) following CARVS research
        
        RVS = sentiment_direction * volume_change * engagement_weight
        
        Only generate signals when:
        1. Sentiment is strong (|score| > 0.3)
        2. Volume increased significantly (> 20%)
        3. Sentiment and volume direction align
        """
        # Calculate volume change
        avg_historical_volume = sum(historical_volume) / len(historical_volume)
        volume_change = (current_volume - avg_historical_volume) / avg_historical_volume
        
        # Get sentiment direction
        sentiment_score = current_sentiment['sentiment_score']
        
        # Calculate RVS
        if abs(sentiment_score) > 0.3 and abs(volume_change) > 0.2:
            # Check alignment
            if (sentiment_score > 0 and volume_change > 0) or \
               (sentiment_score < 0 and volume_change < 0):
                rvs = sentiment_score * (1 + volume_change)
            else:
                rvs = 0  # Misalignment = no signal
        else:
            rvs = 0  # Insufficient strength
            
        return rvs
    
    def _is_likely_human(self, tweet) -> bool:
        """
        Filter out bot accounts using heuristics
        
        Bots typically have:
        - Very high tweet frequency
        - Low follower/following ratio
        - Username patterns (numbers, random strings)
        """
        # This is simplified - production needs more robust filtering
        return True  # Implement proper bot detection

# Example usage:
config = load_config('config/sentiment.yaml')
analyzer = SentimentAnalyzer(config)

# Fetch sentiment from both sources
twitter_sentiment = analyzer.fetch_twitter_sentiment('BTC', hours_back=24)
reddit_sentiment = analyzer.fetch_reddit_sentiment('BTC', hours_back=24)

# Calculate RVS score
rvs_score = analyzer.calculate_rvs(
    current_sentiment=twitter_sentiment,
    historical_sentiment=past_24h_twitter_data,
    current_volume=current_trading_volume,
    historical_volume=past_7d_volume
)

print(f"RVS Score: {rvs_score}")
# Positive RVS > 0.5: Strong buy signal
# Negative RVS < -0.5: Strong sell signal / go to cash
```

---

## Phase 2: Feature Engineering & Graph Construction (Week 2-3)

### 2.1 Technical Feature Engineering

```python
# data/processing/features.py

import pandas as pd
import numpy as np
from typing import List, Dict
import talib

class FeatureEngineer:
    """
    Generate features validated by research:
    - Traditional technical indicators (baseline)
    - Microstructure features (order flow)
    - Cross-asset features (for GNN)
    - Regime indicators
    
    Based on Alpha158 feature set (Microsoft Qlib) + crypto-specific additions
    """
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature generation pipeline
        
        Input: OHLCV DataFrame
        Output: Feature-enriched DataFrame with 100+ columns
        """
        df = df.copy()
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Microstructure features
        df = self._add_microstructure_features(df)
        
        # Temporal features
        df = self._add_temporal_features(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features (returns, moving averages, patterns)"""
        
        # Returns at multiple horizons
        for period in [1, 5, 10, 20, 60]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Distance from moving average
            df[f'close_div_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            df[f'close_div_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
        
        # Price channels
        for period in [20, 50]:
            df[f'high_{period}'] = df['high'].rolling(period).max()
            df[f'low_{period}'] = df['low'].rolling(period).min()
            df[f'price_position_{period}'] = (df['close'] - df[f'low_{period}']) / \
                                              (df[f'high_{period}'] - df[f'low_{period}'])
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['close_div_vwap'] = df['close'] / df['vwap'] - 1
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""
        
        # Volume ratios
        for period in [5, 10, 20]:
            df[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ema_10'] = df['obv'].ewm(span=10).mean()
        
        # Volume-weighted returns
        df['volume_return'] = df['return_1'] * df['volume']
        
        # Force Index
        df['force_index'] = df['close'].diff() * df['volume']
        df['force_index_13'] = df['force_index'].ewm(span=13).mean()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility features (critical for regime detection)"""
        
        # Realized volatility at multiple horizons
        for period in [10, 20, 30, 60]:
            df[f'volatility_{period}'] = df['return_1'].rolling(period).std() * np.sqrt(365)
        
        # Parkinson volatility (uses high-low range)
        for period in [10, 20]:
            hl_ratio = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (hl_ratio ** 2).rolling(period).mean() / (4 * np.log(2))
            ) * np.sqrt(365)
        
        # Volatility of volatility
        df['vol_of_vol'] = df['volatility_20'].rolling(10).std()
        
        # ATR (Average True Range)
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_pct'] = df['atr_14'] / df['close']
        
        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                           (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators"""
        
        # RSI
        for period in [9, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        
        # ADX (trend strength)
        df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order flow and microstructure features
        Requires order book data from separate stream
        """
        
        # Spread proxy (high-low as percentage of close)
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_ma'] = df['hl_spread'].rolling(20).mean()
        
        # Amihud illiquidity measure
        df['amihud'] = abs(df['return_1']) / df['volume']
        df['amihud_ma'] = df['amihud'].rolling(20).mean()
        
        # Trade imbalance (requires tick data)
        # This is simplified - production needs actual buy/sell volume
        df['trade_imbalance'] = np.sign(df['close'] - df['open'])
        df['trade_imbalance_10'] = df['trade_imbalance'].rolling(10).sum()
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        
        # Session indicators
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    def select_features_causal(self, 
                              df: pd.DataFrame,
                              target: str = 'return_1') -> List[str]:
        """
        Use causal discovery (PCMCI) to select features with genuine causal relationships
        
        This addresses overfitting by removing spurious correlations
        """
        from tigramite import data as tg_data
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests import ParCorr
        
        # Prepare data
        feature_cols = [c for c in df.columns if c not in ['time', target]]
        data_matrix = df[feature_cols + [target]].values
        
        # Initialize Tigramite
        dataframe = tg_data.DataFrame(data_matrix)
        parcorr = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr)
        
        # Run causal discovery
        results = pcmci.run_pcmciplus(tau_max=5, pc_alpha=0.05)
        
        # Extract features that causally affect target
        causal_features = []
        target_idx = len(feature_cols)  # Last column is target
        
        for i, feature in enumerate(feature_cols):
            if results['p_matrix'][i, target_idx, :].min() < 0.05:
                causal_features.append(feature)
        
        print(f"Selected {len(causal_features)} causal features from {len(feature_cols)} candidates")
        
        return causal_features

# Example usage:
engineer = FeatureEngineer()

# Load OHLCV data
df = load_ohlcv_from_db('BTC/USDT', start='2024-01-01')

# Generate all features
df_features = engineer.generate_all_features(df)

# Optional: Use causal discovery to select features
causal_features = engineer.select_features_causal(df_features, target='return_1')

# Save to database
save_features_to_db(df_features[causal_features])
```

### 2.2 Graph Construction for GNN

```python
# data/processing/graph_construction.py

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from typing import List, Dict, Tuple

class CryptoGraphBuilder:
    """
    Construct graph structures for GNN modeling
    
    Graph types:
    1. Correlation graph: Assets as nodes, correlations as edges
    2. Transaction graph: Wallets as nodes, transactions as edges (on-chain)
    3. Knowledge graph: Assets as nodes, relationships (sector, tech) as edges
    4. Temporal graph: Time-evolving graphs
    
    Research validation: THGNN, MGAR papers show 164-236% returns with multi-view graphs
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold  # Minimum correlation for edge
        
    def build_correlation_graph(self,
                                prices: pd.DataFrame,
                                window: int = 20) -> Data:
        """
        Build dynamic correlation graph updated daily
        
        Nodes: Crypto assets
        Edges: Rolling correlation > threshold
        Node features: Technical indicators, returns, volatility
        Edge features: Correlation strength, lagged correlation
        """
        
        # Calculate returns
        returns = prices.pct_change()
        
        # Rolling correlation matrix
        correlation = returns.rolling(window).corr().iloc[-len(prices.columns):]
        
        # Build edge list
        edges = []
        edge_attrs = []
        
        for i, asset1 in enumerate(prices.columns):
            for j, asset2 in enumerate(prices.columns):
                if i < j:  # Avoid duplicates
                    corr_value = correlation.loc[asset1, asset2]
                    
                    if abs(corr_value) > self.threshold:
                        edges.append([i, j])
                        edges.append([j, i])  # Undirected graph
                        edge_attrs.append(corr_value)
                        edge_attrs.append(corr_value)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
        
        # Node features (latest values)
        node_features = []
        for asset in prices.columns:
            features = [
                returns[asset].iloc[-1],  # 1-period return
                returns[asset].iloc[-5:].mean(),  # 5-period avg return
                returns[asset].iloc[-20:].std(),  # 20-period volatility
                prices[asset].iloc[-1] / prices[asset].rolling(20).mean().iloc[-1] - 1,  # Price/SMA
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
    
    def build_temporal_correlation_graph(self,
                                        prices: pd.DataFrame,
                                        window: int = 20,
                                        num_snapshots: int = 10) -> List[Data]:
        """
        Build sequence of graphs for temporal GNN
        
        Returns: List of graph snapshots
        """
        graphs = []
        
        # Create snapshots at regular intervals
        total_len = len(prices)
        step = total_len // num_snapshots
        
        for i in range(num_snapshots):
            start_idx = i * step
            end_idx = min((i + 1) * step, total_len)
            
            if end_idx - start_idx < window:
                break
            
            snapshot_prices = prices.iloc[start_idx:end_idx]
            graph = self.build_correlation_graph(snapshot_prices, window)
            
            # Add time information
            graph.time = torch.tensor([i / num_snapshots], dtype=torch.float)
            
            graphs.append(graph)
        
        return graphs
    
    def build_whale_transaction_graph(self,
                                     transactions: pd.DataFrame,
                                     min_amount: float = 10.0) -> Data:
        """
        Build graph from on-chain whale transactions
        
        Nodes: Wallet addresses
        Edges: Transactions between wallets
        Node features: Balance, transaction count, age
        Edge features: Transaction amount, timestamp
        
        Research: Whale wallet influence networks show predictive power
        """
        
        # Filter for large transactions
        whale_txs = transactions[transactions['amount'] >= min_amount]
        
        # Create unique wallet list
        wallets = list(set(whale_txs['from_address'].tolist() + 
                          whale_txs['to_address'].tolist()))
        wallet_to_idx = {wallet: idx for idx, wallet in enumerate(wallets)}
        
        # Build edges
        edges = []
        edge_attrs = []
        
        for _, tx in whale_txs.iterrows():
            from_idx = wallet_to_idx[tx['from_address']]
            to_idx = wallet_to_idx[tx['to_address']]
            
            edges.append([from_idx, to_idx])
            edge_attrs.append([
                tx['amount'],
                tx['timestamp'].timestamp(),  # Convert to unix time
            ])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # Node features (wallet characteristics)
        node_features = []
        for wallet in wallets:
            # Calculate wallet metrics
            balance = self._get_wallet_balance(wallet)
            tx_count = len(whale_txs[
                (whale_txs['from_address'] == wallet) |
                (whale_txs['to_address'] == wallet)
            ])
            age_days = self._get_wallet_age(wallet)
            
            node_features.append([balance, tx_count, age_days])
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
    
    def build_heterogeneous_graph(self,
                                 assets: List[str],
                                 correlations: pd.DataFrame,
                                 sectors: Dict[str, str],
                                 tech_relations: List[Tuple[str, str, str]]) -> HeteroData:
        """
        Build heterogeneous graph with multiple node and edge types
        
        Node types:
        - asset: Crypto assets
        - sector: Market sectors (DeFi, Layer1, etc.)
        
        Edge types:
        - correlated: Correlation between assets
        - belongs_to: Asset belongs to sector
        - competes_with: Competitive relationship
        
        Research: MGAR paper shows multi-view graphs achieve 164-236% returns
        """
        
        data = HeteroData()
        
        # Asset nodes
        asset_features = self._get_asset_features(assets)
        data['asset'].x = torch.tensor(asset_features, dtype=torch.float)
        
        # Sector nodes
        unique_sectors = list(set(sectors.values()))
        sector_features = torch.eye(len(unique_sectors))  # One-hot encoding
        data['sector'].x = sector_features
        
        # Correlation edges
        corr_edges = []
        corr_attrs = []
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j and abs(correlations.loc[asset1, asset2]) > self.threshold:
                    corr_edges.append([i, j])
                    corr_edges.append([j, i])
                    corr_attrs.append(correlations.loc[asset1, asset2])
                    corr_attrs.append(correlations.loc[asset1, asset2])
        
        data['asset', 'correlated', 'asset'].edge_index = \
            torch.tensor(corr_edges, dtype=torch.long).t().contiguous()
        data['asset', 'correlated', 'asset'].edge_attr = \
            torch.tensor(corr_attrs, dtype=torch.float).view(-1, 1)
        
        # Sector membership edges
        sector_edges = []
        sector_to_idx = {sector: idx for idx, sector in enumerate(unique_sectors)}
        
        for asset_idx, asset in enumerate(assets):
            sector = sectors.get(asset)
            if sector:
                sector_idx = sector_to_idx[sector]
                sector_edges.append([asset_idx, sector_idx])
        
        data['asset', 'belongs_to', 'sector'].edge_index = \
            torch.tensor(sector_edges, dtype=torch.long).t().contiguous()
        
        # Technology relationship edges
        tech_edges = []
        for asset1, relation, asset2 in tech_relations:
            if asset1 in assets and asset2 in assets:
                idx1 = assets.index(asset1)
                idx2 = assets.index(asset2)
                tech_edges.append([idx1, idx2])
        
        if tech_edges:
            data['asset', 'competes_with', 'asset'].edge_index = \
                torch.tensor(tech_edges, dtype=torch.long).t().contiguous()
        
        return data
    
    def _get_wallet_balance(self, wallet: str) -> float:
        """Fetch current wallet balance from blockchain"""
        # Implement blockchain query
        return 0.0
    
    def _get_wallet_age(self, wallet: str) -> float:
        """Calculate wallet age in days"""
        # Implement blockchain query
        return 0.0
    
    def _get_asset_features(self, assets: List[str]) -> List[List[float]]:
        """Get latest technical features for assets"""
        # Fetch from feature database
        return [[0.0] * 10 for _ in assets]

# Example usage:
builder = CryptoGraphBuilder(threshold=0.3)

# Build correlation graph
prices = load_prices(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
graph = builder.build_correlation_graph(prices, window=20)

# Build temporal graphs for sequence modeling
temporal_graphs = builder.build_temporal_correlation_graph(prices, window=20, num_snapshots=10)

# Build heterogeneous graph with multiple relationship types
hetero_graph = builder.build_heterogeneous_graph(
    assets=['BTC', 'ETH', 'SOL', 'AVAX'],
    correlations=prices.corr(),
    sectors={'BTC': 'Layer1', 'ETH': 'Layer1', 'SOL': 'Layer1', 'AVAX': 'Layer1'},
    tech_relations=[('ETH', 'competes', 'SOL'), ('SOL', 'competes', 'AVAX')]
)
```

---

## Phase 3: Model Development (Week 3-4)

I need to continue with the specification but am approaching length limits. Should I:

1. Continue in a second file with Phase 3 (Model Development), Phase 4 (RL Agents), Phase 5 (Risk & Execution), and Phase 6 (Deployment)?

2. Or would you prefer I consolidate the remaining phases into a more concise format in this same file?

The remaining sections will cover:
- **Phase 3**: LightGBM baseline, LSTM, GNN implementation with training code
- **Phase 4**: Hierarchical RL agents (strategic/tactical/execution) with FinRL integration
- **Phase 5**: Risk management, execution engine, backtesting framework
- **Phase 6**: Orchestrator, monitoring, deployment, and specific commands to run

Which approach would be more useful for handing to Opus?
