"""
Data Ingestion Module

Handles data collection from various sources:
- Exchange data (OHLCV via CCXT)
- Reddit sentiment (public JSON API)
- On-chain metrics (Dune Analytics)
"""

from data.ingestion.reddit_sources import (
    RedditDataSource,
    RedditPublicJSON,
    RedditPRAW,
    get_reddit_source,
)

from data.ingestion.onchain import (
    OnChainDataProvider,
    OnChainSignalGenerator,
    OnChainMetric,
    ONCHAIN_METRICS,
)

from data.ingestion.sentiment import (
    FinBERTAnalyzer,
    SentimentAggregator,
    SentimentResult,
    AggregatedSentiment,
)

__all__ = [
    # Reddit
    'RedditDataSource',
    'RedditPublicJSON',
    'RedditPRAW',
    'get_reddit_source',
    # On-chain
    'OnChainDataProvider',
    'OnChainSignalGenerator',
    'OnChainMetric',
    'ONCHAIN_METRICS',
    # Sentiment
    'FinBERTAnalyzer',
    'SentimentAggregator',
    'SentimentResult',
    'AggregatedSentiment',
]
