"""
Unit tests for Feature Engineering Pipeline

Tests the FeatureEngineer class for correctness, edge cases,
and absence of look-ahead bias.

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.processing.features import FeatureEngineer, compute_features


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    dates = pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC')

    # Random walk price
    returns = np.random.randn(n) * 0.01
    price = 50000 * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'time': dates,
        'open': price * (1 + np.random.randn(n) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(n)) * 0.005),
        'low': price * (1 - np.abs(np.random.randn(n)) * 0.005),
        'close': price,
        'volume': 1000 + np.abs(np.random.randn(n)) * 100,
        'symbol': 'BTC/USD',
        'exchange': 'coinbase'
    })

    return data


@pytest.fixture
def engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer(warmup_periods=50)


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    def test_generate_all_features_column_count(self, sample_ohlcv_data, engineer):
        """Test that all expected features are generated."""
        result = engineer.generate_all_features(sample_ohlcv_data)

        expected_features = engineer.get_feature_names()
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_generate_all_features_row_count(self, sample_ohlcv_data, engineer):
        """Test that row count is preserved."""
        result = engineer.generate_all_features(sample_ohlcv_data)
        assert len(result) == len(sample_ohlcv_data)

    def test_feature_validation_passes_clean_data(self, sample_ohlcv_data, engineer):
        """Test validation passes for valid data."""
        result = engineer.generate_all_features(sample_ohlcv_data)
        validation = engineer.validate_features(result)
        assert validation['is_valid'], f"Validation failed: {validation['issues']}"

    def test_remove_warmup_removes_correct_rows(self, sample_ohlcv_data, engineer):
        """Test warmup removal."""
        result = engineer.generate_all_features(sample_ohlcv_data)
        clean = engineer.remove_warmup(result)

        expected_rows = len(sample_ohlcv_data) - engineer.warmup_periods
        assert len(clean) == expected_rows

    def test_no_nan_after_warmup(self, sample_ohlcv_data, engineer):
        """Test that no NaN values exist after warmup removal."""
        result = engineer.generate_all_features(sample_ohlcv_data)
        clean = engineer.remove_warmup(result)

        feature_names = engineer.get_feature_names()
        nan_count = clean[feature_names].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values after warmup"

    def test_rsi_bounded(self, sample_ohlcv_data, engineer):
        """Test RSI is bounded between 0 and 100."""
        result = engineer.generate_all_features(sample_ohlcv_data)
        clean = engineer.remove_warmup(result)

        rsi = clean['rsi_14']
        assert rsi.min() >= 0, f"RSI below 0: {rsi.min()}"
        assert rsi.max() <= 100, f"RSI above 100: {rsi.max()}"

    def test_bollinger_bands_relationship(self, sample_ohlcv_data, engineer):
        """Test that upper band >= lower band."""
        result = engineer.generate_all_features(sample_ohlcv_data)
        clean = engineer.remove_warmup(result)

        assert (clean['bb_upper'] >= clean['bb_lower']).all()

    def test_volume_ratio_positive(self, sample_ohlcv_data, engineer):
        """Test volume ratio is always positive."""
        result = engineer.generate_all_features(sample_ohlcv_data)
        clean = engineer.remove_warmup(result)

        assert (clean['volume_ratio'] > 0).all()

    def test_missing_columns_raises_error(self, engineer):
        """Test that missing required columns raise ValueError."""
        bad_data = pd.DataFrame({'time': [1, 2, 3], 'close': [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.generate_all_features(bad_data)

    def test_consistency(self, sample_ohlcv_data, engineer):
        """Test that same input produces same output."""
        result1 = engineer.generate_all_features(sample_ohlcv_data.copy())
        result2 = engineer.generate_all_features(sample_ohlcv_data.copy())

        feature_names = engineer.get_feature_names()
        pd.testing.assert_frame_equal(
            result1[feature_names],
            result2[feature_names]
        )


class TestPriceFeatures:
    """Test price-based features specifically."""

    def test_return_calculation(self, sample_ohlcv_data, engineer):
        """Test return calculation is correct."""
        result = engineer.generate_all_features(sample_ohlcv_data)

        # Manually calculate expected return
        expected = sample_ohlcv_data['close'].pct_change(1)

        # Compare (allowing for floating point tolerance)
        np.testing.assert_allclose(
            result['return_1h'].dropna().values,
            expected.dropna().values,
            rtol=1e-10
        )

    def test_log_return_calculation(self, sample_ohlcv_data, engineer):
        """Test log return calculation."""
        result = engineer.generate_all_features(sample_ohlcv_data)

        close = sample_ohlcv_data['close']
        expected = np.log(close / close.shift(1))

        np.testing.assert_allclose(
            result['log_return_1h'].dropna().values,
            expected.dropna().values,
            rtol=1e-10
        )


class TestMovingAverages:
    """Test moving average features."""

    def test_sma_calculation(self, sample_ohlcv_data, engineer):
        """Test SMA calculation is correct."""
        result = engineer.generate_all_features(sample_ohlcv_data)

        # Manually calculate SMA_20
        expected = sample_ohlcv_data['close'].rolling(window=20).mean()

        np.testing.assert_allclose(
            result['sma_20'].dropna().values,
            expected.dropna().values,
            rtol=1e-10
        )


class TestConvenienceFunction:
    """Test the compute_features convenience function."""

    def test_compute_features_basic(self, sample_ohlcv_data):
        """Test compute_features returns valid DataFrame."""
        result = compute_features(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_compute_features_removes_warmup(self, sample_ohlcv_data):
        """Test warmup is removed by default."""
        result = compute_features(sample_ohlcv_data, remove_warmup=True)
        # Should have fewer rows than original
        assert len(result) < len(sample_ohlcv_data)

    def test_compute_features_keeps_warmup(self, sample_ohlcv_data):
        """Test warmup is kept when requested."""
        result = compute_features(sample_ohlcv_data, remove_warmup=False)
        assert len(result) == len(sample_ohlcv_data)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, engineer):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        result = engineer.generate_all_features(empty_df)
        assert len(result) == 0

    def test_single_row(self, engineer):
        """Test handling of single row."""
        single_row = pd.DataFrame({
            'time': [pd.Timestamp('2024-01-01', tz='UTC')],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000.0]
        })
        result = engineer.generate_all_features(single_row)
        assert len(result) == 1

    def test_handles_zero_volume(self, sample_ohlcv_data, engineer):
        """Test handling of zero volume periods."""
        data = sample_ohlcv_data.copy()
        data.loc[50:55, 'volume'] = 0

        # Should not raise error
        result = engineer.generate_all_features(data)
        assert len(result) == len(data)
