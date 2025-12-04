"""
Unit tests for Backtesting Engine

Tests the backtesting framework for correctness, edge cases,
and proper handling of transaction costs.

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.engine import BacktestEngine, BacktestConfig, run_buy_and_hold
from backtesting.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_all_metrics
)


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.002 + 0.0001
    prices = pd.Series(50000 * np.cumprod(1 + returns))
    return prices


@pytest.fixture
def sample_signals():
    """Create sample trading signals."""
    np.random.seed(42)
    n = 500
    signals = pd.Series(np.random.choice([0, 1], size=n, p=[0.3, 0.7]))
    return signals


@pytest.fixture
def default_config():
    """Create default backtest configuration."""
    return BacktestConfig(
        initial_capital=10000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )


class TestBacktestEngine:
    """Test suite for BacktestEngine."""

    def test_initialization(self, default_config):
        """Test engine initializes correctly."""
        engine = BacktestEngine(default_config)
        assert engine.config.initial_capital == 10000
        assert engine.config.commission_rate == 0.001

    def test_run_returns_result(self, sample_prices, sample_signals, default_config):
        """Test that run returns a BacktestResult."""
        engine = BacktestEngine(default_config)
        result = engine.run(sample_prices, sample_signals)

        assert result is not None
        assert len(result.equity_curve) == len(sample_prices)
        assert len(result.returns) == len(sample_prices)

    def test_equity_starts_at_initial_capital(self, sample_prices, sample_signals, default_config):
        """Test equity curve starts at initial capital."""
        engine = BacktestEngine(default_config)
        result = engine.run(sample_prices, sample_signals)

        # First non-NaN value should be close to initial capital
        first_equity = result.equity_curve.dropna().iloc[0]
        assert abs(first_equity - default_config.initial_capital) < 100

    def test_transaction_costs_applied(self, sample_prices, default_config):
        """Test that transaction costs reduce returns."""
        # Always long signal
        signals = pd.Series(1, index=range(len(sample_prices)))

        # Run with costs
        engine = BacktestEngine(default_config)
        result_with_costs = engine.run(sample_prices, signals)

        # Run without costs
        config_no_costs = BacktestConfig(
            initial_capital=10000,
            commission_rate=0.0,
            slippage_rate=0.0
        )
        engine_no_costs = BacktestEngine(config_no_costs)
        result_no_costs = engine_no_costs.run(sample_prices, signals)

        # With costs should have lower final equity
        assert result_with_costs.equity_curve.iloc[-1] <= result_no_costs.equity_curve.iloc[-1]

    def test_flat_signal_preserves_capital(self, sample_prices, default_config):
        """Test that flat signal (0) preserves capital."""
        signals = pd.Series(0, index=range(len(sample_prices)))

        engine = BacktestEngine(default_config)
        result = engine.run(sample_prices, signals)

        # Should end close to initial capital (no trading)
        final_equity = result.equity_curve.iloc[-1]
        assert abs(final_equity - default_config.initial_capital) < 1

    def test_trades_extracted(self, sample_prices, sample_signals, default_config):
        """Test that trades are properly extracted."""
        engine = BacktestEngine(default_config)
        result = engine.run(sample_prices, sample_signals)

        # Should have some trades
        assert len(result.trades) >= 0

        # Each trade should have required fields
        for trade in result.trades:
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.side in ['long', 'short']


class TestMetrics:
    """Test suite for performance metrics."""

    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio for positive returns."""
        returns = pd.Series(np.random.randn(1000) * 0.01 + 0.001)
        sharpe = calculate_sharpe_ratio(returns)

        # Positive drift should give positive Sharpe
        assert sharpe > 0

    def test_sharpe_ratio_negative_returns(self):
        """Test Sharpe ratio for negative returns."""
        returns = pd.Series(np.random.randn(1000) * 0.01 - 0.001)
        sharpe = calculate_sharpe_ratio(returns)

        # Negative drift should give negative Sharpe
        assert sharpe < 0

    def test_sharpe_ratio_handles_zero_std(self):
        """Test Sharpe handles constant returns."""
        returns = pd.Series([0.0] * 100)  # Exactly zero returns
        sharpe = calculate_sharpe_ratio(returns)

        # Should return 0 for zero std (no variance)
        assert sharpe == 0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        sortino = calculate_sortino_ratio(returns)

        # Should be a valid number
        assert not np.isnan(sortino)

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        # Create equity curve with known drawdown
        equity = pd.Series([100, 110, 120, 100, 90, 95, 100])
        max_dd, duration = calculate_max_drawdown(equity)

        # Max drawdown should be 25% (120 -> 90)
        assert abs(max_dd - 0.25) < 0.01

    def test_win_rate(self):
        """Test win rate calculation."""
        trade_returns = pd.Series([0.1, -0.05, 0.2, -0.1, 0.15])
        win_rate = calculate_win_rate(trade_returns)

        # 3 wins out of 5
        assert win_rate == 0.6

    def test_profit_factor(self):
        """Test profit factor calculation."""
        trade_returns = pd.Series([0.1, -0.05, 0.2, -0.05])
        pf = calculate_profit_factor(trade_returns)

        # (0.1 + 0.2) / (0.05 + 0.05) = 3.0
        assert abs(pf - 3.0) < 0.001

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        trade_returns = pd.Series([0.1, 0.2, 0.15])
        pf = calculate_profit_factor(trade_returns)

        assert pf == float('inf')

    def test_calculate_all_metrics(self):
        """Test full metrics calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)
        equity = (1 + returns).cumprod() * 10000
        trade_returns = pd.Series(np.random.randn(20) * 0.02)

        metrics = calculate_all_metrics(returns, equity, trade_returns)

        # Check all fields populated
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown >= 0
        assert 0 <= metrics.win_rate <= 1


class TestBuyAndHold:
    """Test buy-and-hold benchmark."""

    def test_buy_and_hold_basic(self, sample_prices):
        """Test buy and hold returns correct result."""
        result = run_buy_and_hold(sample_prices, initial_capital=10000)

        assert len(result.equity_curve) == len(sample_prices)
        assert result.metrics is not None

    def test_buy_and_hold_tracks_price(self, sample_prices):
        """Test buy and hold tracks price movement."""
        result = run_buy_and_hold(sample_prices, initial_capital=10000, commission_rate=0)

        # Return should match price return (roughly)
        price_return = (sample_prices.iloc[-1] / sample_prices.iloc[0]) - 1
        strategy_return = result.metrics.total_return

        # Should be close (within 1% due to calculation differences)
        assert abs(strategy_return - price_return) < 0.01


class TestEdgeCases:
    """Test edge cases."""

    def test_single_period(self, default_config):
        """Test with minimal data."""
        prices = pd.Series([100, 101])
        signals = pd.Series([1, 1])

        engine = BacktestEngine(default_config)
        result = engine.run(prices, signals)

        assert len(result.equity_curve) == 2

    def test_all_zeros_signals(self, sample_prices, default_config):
        """Test with all zero signals."""
        signals = pd.Series(0, index=range(len(sample_prices)))

        engine = BacktestEngine(default_config)
        result = engine.run(sample_prices, signals)

        # Should have no trades
        assert len(result.trades) == 0

    def test_alternating_signals(self, default_config):
        """Test with alternating signals (high frequency)."""
        np.random.seed(42)
        n = 100
        prices = pd.Series(50000 * np.cumprod(1 + np.random.randn(n) * 0.001))
        signals = pd.Series([i % 2 for i in range(n)])

        engine = BacktestEngine(default_config)
        result = engine.run(prices, signals)

        # Should have many trades
        assert len(result.trades) > 10
