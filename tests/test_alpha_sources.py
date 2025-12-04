"""
Unit Tests for Phase 3 Alpha Sources

Tests for:
- Reddit data source abstraction
- On-chain signal interpretation
- Regime classification
- Alpha signal combination

Author: Claude Opus 4.5
Date: 2024-12-04
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predictors.regime_classifier import (
    RegimeClassifier,
    MarketRegime,
    RegimeClassification,
    RegimeHistory,
)
from models.predictors.alpha_combiner import (
    AlphaCombiner,
    AlphaSignal,
    CombinedSignal,
)
from data.ingestion.onchain import (
    OnChainSignalGenerator,
    OnChainMetric,
    ONCHAIN_METRICS,
)


class TestRegimeClassifier:
    """Test suite for RegimeClassifier."""

    @pytest.fixture
    def classifier(self):
        return RegimeClassifier()

    def test_initialization(self, classifier):
        """Test classifier initializes with correct weights."""
        assert classifier.weights is not None
        assert sum(classifier.weights.values()) == pytest.approx(1.0)

    def test_strong_bullish_classification(self, classifier):
        """Test strong bullish signals produce strong_bull regime."""
        result = classifier.classify(
            onchain_signal=0.8,
            sentiment_signal=0.7,
            technical_signal=0.6
        )

        assert result.regime == MarketRegime.STRONG_BULL
        assert result.confidence > 0.5

    def test_strong_bearish_classification(self, classifier):
        """Test strong bearish signals produce strong_bear regime."""
        result = classifier.classify(
            onchain_signal=-0.8,
            sentiment_signal=-0.7,
            technical_signal=-0.6
        )

        assert result.regime == MarketRegime.STRONG_BEAR
        assert result.confidence > 0.5

    def test_neutral_classification(self, classifier):
        """Test mixed signals produce neutral regime."""
        result = classifier.classify(
            onchain_signal=0.1,
            sentiment_signal=-0.05,
            technical_signal=0.0
        )

        assert result.regime == MarketRegime.NEUTRAL

    def test_accumulation_classification(self, classifier):
        """Test mildly positive signals produce accumulation regime."""
        result = classifier.classify(
            onchain_signal=0.2,
            sentiment_signal=0.15,
            technical_signal=0.1
        )

        # Should be either accumulation or bull
        assert result.regime in [MarketRegime.ACCUMULATION, MarketRegime.BULL]

    def test_distribution_classification(self, classifier):
        """Test mildly negative signals produce distribution regime."""
        result = classifier.classify(
            onchain_signal=-0.2,
            sentiment_signal=-0.15,
            technical_signal=-0.1
        )

        # Should be either distribution or bear
        assert result.regime in [MarketRegime.DISTRIBUTION, MarketRegime.BEAR]

    def test_technical_signal_calculation(self, classifier):
        """Test technical signal calculation from prices."""
        # Create trending price series
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.randn(100) * 0.01 + 0.002))

        tech = classifier.calculate_technical_signal(prices)

        assert 'trend' in tech
        assert 'momentum' in tech
        assert 'combined' in tech
        assert -1 <= tech['combined'] <= 1

    def test_technical_signal_short_series(self, classifier):
        """Test technical handles short series gracefully."""
        prices = pd.Series([100, 101, 102])

        tech = classifier.calculate_technical_signal(prices, lookback=20)

        assert tech['combined'] == 0.0

    def test_trading_bias(self, classifier):
        """Test trading bias recommendations."""
        for regime in MarketRegime:
            bias = classifier.get_trading_bias(regime)

            assert 'position_bias' in bias
            assert 'position_size_mult' in bias
            assert bias['position_size_mult'] >= 0
            assert bias['position_size_mult'] <= 1

    def test_signals_included_in_result(self, classifier):
        """Test that component signals are tracked."""
        result = classifier.classify(
            onchain_signal=0.5,
            sentiment_signal=0.3,
            technical_signal=0.2
        )

        assert 'onchain' in result.signals
        assert 'sentiment' in result.signals
        assert 'technical' in result.signals
        assert 'combined' in result.signals


class TestRegimeHistory:
    """Test suite for RegimeHistory."""

    @pytest.fixture
    def history(self):
        return RegimeHistory()

    def test_add_classification(self, history):
        """Test adding classifications to history."""
        classification = RegimeClassification(
            regime=MarketRegime.BULL,
            confidence=0.7,
            timestamp=datetime.now(),
            signals={'combined': 0.5},
            description="Test"
        )

        history.add(classification)
        assert len(history.history) == 1

    def test_to_dataframe(self, history):
        """Test converting history to DataFrame."""
        for regime in [MarketRegime.BULL, MarketRegime.NEUTRAL, MarketRegime.BEAR]:
            history.add(RegimeClassification(
                regime=regime,
                confidence=0.6,
                timestamp=datetime.now(),
                signals={'combined': 0.0},
                description="Test"
            ))

        df = history.to_dataframe()
        assert len(df) == 3
        assert 'regime' in df.columns
        assert 'confidence' in df.columns

    def test_regime_changes(self, history):
        """Test detecting regime changes."""
        regimes = [
            MarketRegime.BULL,
            MarketRegime.BULL,
            MarketRegime.NEUTRAL,
            MarketRegime.BEAR,
            MarketRegime.BEAR,
        ]

        for regime in regimes:
            history.add(RegimeClassification(
                regime=regime,
                confidence=0.5,
                timestamp=datetime.now(),
                signals={},
                description=""
            ))

        changes = history.get_regime_changes()
        assert len(changes) == 2  # BULL->NEUTRAL, NEUTRAL->BEAR


class TestOnChainSignals:
    """Test suite for on-chain signal interpretation."""

    @pytest.fixture
    def generator(self):
        mock_provider = Mock()
        return OnChainSignalGenerator(mock_provider)

    def test_mvrv_extreme_high(self, generator):
        """Test MVRV interpretation at extreme high."""
        result = generator.interpret_mvrv(8.0)

        assert result['signal'] == -1.0
        assert result['regime'] == 'extreme_overvaluation'

    def test_mvrv_high(self, generator):
        """Test MVRV interpretation at high."""
        result = generator.interpret_mvrv(4.0)

        assert result['signal'] == -0.5
        assert result['regime'] == 'overvaluation'

    def test_mvrv_extreme_low(self, generator):
        """Test MVRV interpretation at extreme low."""
        result = generator.interpret_mvrv(-1.0)

        assert result['signal'] == 1.0
        assert result['regime'] == 'extreme_undervaluation'

    def test_mvrv_low(self, generator):
        """Test MVRV interpretation at low."""
        result = generator.interpret_mvrv(-0.3)

        assert result['signal'] == 0.5
        assert result['regime'] == 'undervaluation'

    def test_mvrv_fair_value(self, generator):
        """Test MVRV interpretation at fair value."""
        result = generator.interpret_mvrv(1.5)

        assert result['signal'] == 0.0
        assert result['regime'] == 'fair_value'

    def test_sopr_profit_taking(self, generator):
        """Test SOPR interpretation during profit taking."""
        result = generator.interpret_sopr(1.08)

        assert result['signal'] < 0
        assert result['regime'] == 'profit_taking'

    def test_sopr_capitulation(self, generator):
        """Test SOPR interpretation during capitulation."""
        result = generator.interpret_sopr(0.92)

        assert result['signal'] > 0
        assert result['regime'] == 'capitulation'

    def test_exchange_netflow_inflow(self, generator):
        """Test netflow interpretation for inflows."""
        result = generator.interpret_exchange_netflow(15000)

        assert result['signal'] < 0
        assert result['regime'] == 'distribution'

    def test_exchange_netflow_outflow(self, generator):
        """Test netflow interpretation for outflows."""
        result = generator.interpret_exchange_netflow(-10000)

        assert result['signal'] > 0
        assert result['regime'] == 'accumulation'

    def test_combine_signals(self, generator):
        """Test combining multiple on-chain signals."""
        signals = {
            'mvrv': generator.interpret_mvrv(4.0),
            'sopr': generator.interpret_sopr(1.05),
            'exchange_netflow': generator.interpret_exchange_netflow(-5000),
        }

        combined = generator.combine_signals(signals)

        assert 'signal' in combined
        assert 'regime' in combined
        assert -1 <= combined['signal'] <= 1


class TestAlphaCombiner:
    """Test suite for AlphaCombiner."""

    @pytest.fixture
    def combiner(self):
        return AlphaCombiner()

    def test_initialization(self, combiner):
        """Test combiner initializes correctly."""
        assert combiner.weights is not None
        assert sum(combiner.weights.values()) == pytest.approx(1.0)

    def test_add_signals(self, combiner):
        """Test adding individual signals."""
        onchain = combiner.add_onchain_signal(0.5, confidence=0.7)
        sentiment = combiner.add_sentiment_signal(0.3, confidence=0.6)
        technical = combiner.add_technical_signal(0.4, confidence=0.8)

        assert onchain.signal == 0.5
        assert sentiment.confidence == 0.6
        assert technical.source == 'technical'

    def test_combine_produces_result(self, combiner):
        """Test that combine produces CombinedSignal."""
        combiner.add_onchain_signal(0.5, confidence=0.7)
        combiner.add_sentiment_signal(0.3, confidence=0.6)
        combiner.add_technical_signal(0.4, confidence=0.8)

        result = combiner.combine()

        assert isinstance(result, CombinedSignal)
        assert -1 <= result.signal <= 1
        assert result.regime is not None
        assert result.position_recommendation in ['long', 'short', 'flat']

    def test_bullish_signals_produce_long(self, combiner):
        """Test bullish signals produce long recommendation."""
        combiner.add_onchain_signal(0.6, confidence=0.8)
        combiner.add_sentiment_signal(0.5, confidence=0.7)
        combiner.add_technical_signal(0.4, confidence=0.8)

        result = combiner.combine()

        assert result.signal > 0
        assert result.position_recommendation == 'long'

    def test_bearish_signals_produce_short(self, combiner):
        """Test bearish signals produce short recommendation."""
        combiner.add_onchain_signal(-0.6, confidence=0.8)
        combiner.add_sentiment_signal(-0.5, confidence=0.7)
        combiner.add_technical_signal(-0.4, confidence=0.8)

        result = combiner.combine()

        assert result.signal < 0
        assert result.position_recommendation == 'short'

    def test_low_confidence_produces_flat(self, combiner):
        """Test low confidence produces flat recommendation."""
        combiner.add_onchain_signal(0.5, confidence=0.1)
        combiner.add_sentiment_signal(0.3, confidence=0.1)
        combiner.add_technical_signal(0.2, confidence=0.1)

        result = combiner.combine()

        assert result.position_recommendation == 'flat'

    def test_trading_decision(self, combiner):
        """Test trading decision generation."""
        combiner.add_onchain_signal(0.5, confidence=0.7)
        combiner.add_sentiment_signal(0.4, confidence=0.6)
        combiner.add_technical_signal(0.3, confidence=0.8)
        combiner.combine()

        decision = combiner.generate_trading_decision(current_position='flat')

        assert 'action' in decision
        assert 'reason' in decision
        assert 'target_position' in decision
        assert 'size' in decision

    def test_signal_summary(self, combiner):
        """Test signal summary generation."""
        combiner.add_onchain_signal(0.5, confidence=0.7)
        combiner.combine()

        summary = combiner.get_signal_summary()

        assert summary['status'] == 'active'
        assert 'combined_signal' in summary
        assert 'regime' in summary

    def test_empty_combiner_summary(self, combiner):
        """Test summary with no signals."""
        summary = combiner.get_signal_summary()
        assert summary['status'] == 'no_signal'

    def test_technical_from_prices(self, combiner):
        """Test calculating technical signal from prices."""
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.randn(50) * 0.01))

        signal = combiner.calculate_technical_from_prices(prices)

        assert signal.source == 'technical'
        assert -1 <= signal.signal <= 1


class TestOnChainMetrics:
    """Test on-chain metric definitions."""

    def test_all_metrics_defined(self):
        """Test all expected metrics are defined."""
        expected = ['mvrv', 'sopr', 'exchange_netflow', 'ssr', 'nupl', 'puell_multiple']

        for metric in expected:
            assert metric in ONCHAIN_METRICS

    def test_metric_structure(self):
        """Test metrics have required fields."""
        for name, metric in ONCHAIN_METRICS.items():
            assert isinstance(metric, OnChainMetric)
            assert metric.name is not None
            assert metric.description is not None


class TestSignalClipping:
    """Test signal value clipping."""

    @pytest.fixture
    def combiner(self):
        return AlphaCombiner()

    def test_signal_clipped_high(self, combiner):
        """Test signals above 1 are clipped."""
        signal = combiner.add_onchain_signal(2.0, confidence=0.5)
        assert signal.signal == 1.0

    def test_signal_clipped_low(self, combiner):
        """Test signals below -1 are clipped."""
        signal = combiner.add_onchain_signal(-2.0, confidence=0.5)
        assert signal.signal == -1.0

    def test_confidence_clipped_high(self, combiner):
        """Test confidence above 1 is clipped."""
        signal = combiner.add_onchain_signal(0.5, confidence=1.5)
        assert signal.confidence == 1.0

    def test_confidence_clipped_low(self, combiner):
        """Test confidence below 0 is clipped."""
        signal = combiner.add_onchain_signal(0.5, confidence=-0.5)
        assert signal.confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
