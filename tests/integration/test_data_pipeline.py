"""
Integration tests for Data Pipeline

Tests the complete data flow from exchange to database to features.
These tests require Docker/TimescaleDB to be running.

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.ingestion.exchanges import ExchangeDataIngester
from data.storage.timeseries_db import TimeSeriesDB
from data.processing.features import FeatureEngineer


@pytest.fixture(scope="module")
def db():
    """Create database connection for tests."""
    database = TimeSeriesDB()
    yield database
    database.close()


@pytest.fixture(scope="module")
def ingester():
    """Create exchange ingester for tests."""
    return ExchangeDataIngester(exchange_name='coinbase')


class TestDatabaseConnection:
    """Test database connectivity."""

    def test_health_check(self, db):
        """Test database health check passes."""
        health = db.health_check()
        assert health['status'] == 'healthy'
        assert health['connected'] is True

    def test_has_ohlcv_data(self, db):
        """Test that OHLCV data exists."""
        health = db.health_check()
        assert health['ohlcv_rows'] > 0, "No OHLCV data in database"

    def test_symbol_stats(self, db):
        """Test symbol statistics query."""
        stats = db.get_symbol_stats()
        assert len(stats) >= 2, "Expected at least BTC/USD and ETH/USD"
        assert 'BTC/USD' in stats['symbol'].values
        assert 'ETH/USD' in stats['symbol'].values


class TestExchangeConnection:
    """Test exchange connectivity."""

    def test_connection(self, ingester):
        """Test exchange connection works."""
        assert ingester.test_connection()

    def test_fetch_ticker(self, ingester):
        """Test fetching current ticker."""
        ticker = ingester.fetch_ticker('BTC/USD')
        assert 'last' in ticker
        assert ticker['last'] > 0

    def test_fetch_ohlcv(self, ingester):
        """Test fetching OHLCV data."""
        df = ingester.fetch_ohlcv('BTC/USD', timeframe='1h', limit=10)
        assert len(df) > 0
        assert 'time' in df.columns
        assert 'close' in df.columns


class TestDataRetrieval:
    """Test data retrieval from database."""

    def test_fetch_ohlcv_btc(self, db):
        """Test fetching BTC/USD data."""
        df = db.fetch_ohlcv('BTC/USD', days=7)
        assert len(df) > 0
        assert df['symbol'].iloc[0] == 'BTC/USD'

    def test_fetch_ohlcv_eth(self, db):
        """Test fetching ETH/USD data."""
        df = db.fetch_ohlcv('ETH/USD', days=7)
        assert len(df) > 0
        assert df['symbol'].iloc[0] == 'ETH/USD'

    def test_data_sorted_by_time(self, db):
        """Test data is sorted by time ascending."""
        df = db.fetch_ohlcv('BTC/USD', days=7)
        times = df['time'].values
        assert (times[:-1] <= times[1:]).all()

    def test_latest_timestamp(self, db):
        """Test getting latest timestamp."""
        latest = db.get_latest_timestamp('BTC/USD')
        assert latest is not None
        assert latest.tzinfo is not None  # Should be timezone-aware


class TestDataQuality:
    """Test data quality."""

    def test_no_nan_in_ohlcv(self, db):
        """Test no NaN values in OHLCV columns."""
        df = db.fetch_ohlcv('BTC/USD', days=30)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        nan_count = df[ohlcv_cols].isna().sum().sum()
        assert nan_count == 0

    def test_ohlc_relationships(self, db):
        """Test OHLC relationships are valid."""
        df = db.fetch_ohlcv('BTC/USD', days=30)

        # High should be >= Low
        assert (df['high'] >= df['low']).all()

        # High should be >= Open and Close
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()

        # Low should be <= Open and Close
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()

    def test_positive_values(self, db):
        """Test all price/volume values are positive."""
        df = db.fetch_ohlcv('BTC/USD', days=30)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        assert (df[ohlcv_cols] >= 0).all().all()


class TestEndToEndPipeline:
    """Test complete data flow."""

    def test_fetch_process_validate(self, db):
        """Test complete pipeline: fetch -> features -> validate."""
        # 1. Fetch data from database
        df = db.fetch_ohlcv('BTC/USD', days=60)
        assert len(df) > 0

        # 2. Generate features
        engineer = FeatureEngineer()
        features = engineer.generate_all_features(df)

        # 3. Validate features
        validation = engineer.validate_features(features)
        assert validation['is_valid'], f"Validation failed: {validation['issues']}"

        # 4. Remove warmup
        clean = engineer.remove_warmup(features)
        assert len(clean) > 0

        # 5. Check no NaN in clean data
        feature_names = engineer.get_feature_names()
        nan_count = clean[feature_names].isna().sum().sum()
        assert nan_count == 0

    def test_multi_symbol_pipeline(self, db):
        """Test pipeline works for multiple symbols."""
        symbols = ['BTC/USD', 'ETH/USD']
        engineer = FeatureEngineer()

        for symbol in symbols:
            df = db.fetch_ohlcv(symbol, days=30)
            features = engineer.generate_all_features(df)
            validation = engineer.validate_features(features)
            assert validation['is_valid'], f"{symbol} validation failed"
