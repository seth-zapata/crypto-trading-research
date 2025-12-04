"""
Market Regime Classifier

Classifies market regimes using on-chain metrics and sentiment signals.
Regimes: Bull, Bear, Accumulation, Distribution, Neutral

Based on academic research showing regime-conditional strategies outperform.

Author: Claude Opus 4.5
Date: 2024-12-04
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_BULL = "strong_bull"      # Clear uptrend, high confidence
    BULL = "bull"                     # Uptrend
    ACCUMULATION = "accumulation"     # Sideways with buying pressure
    NEUTRAL = "neutral"               # No clear direction
    DISTRIBUTION = "distribution"     # Sideways with selling pressure
    BEAR = "bear"                     # Downtrend
    STRONG_BEAR = "strong_bear"       # Clear downtrend, high confidence


@dataclass
class RegimeClassification:
    """Result of regime classification."""
    regime: MarketRegime
    confidence: float  # 0-1
    timestamp: datetime
    signals: Dict[str, float]  # Component signals
    description: str


class RegimeClassifier:
    """
    Classifies market regimes from multiple alpha sources.

    Combines:
    - On-chain signals (MVRV, SOPR, netflows)
    - Sentiment signals (CARVS score)
    - Price/volume technical signals

    Uses a rule-based system with configurable thresholds,
    optionally enhanced with a trained classifier.
    """

    # Default thresholds for regime classification
    DEFAULT_THRESHOLDS = {
        'strong_bull': 0.6,
        'bull': 0.3,
        'accumulation': 0.15,
        'neutral_upper': 0.15,
        'neutral_lower': -0.15,
        'distribution': -0.15,
        'bear': -0.3,
        'strong_bear': -0.6,
    }

    # Signal weights for combining
    DEFAULT_WEIGHTS = {
        'onchain': 0.40,      # On-chain metrics
        'sentiment': 0.30,    # Social sentiment
        'technical': 0.30,    # Price/volume
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize regime classifier.

        Args:
            thresholds: Override default classification thresholds
            weights: Override default signal weights
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        logger.info(f"RegimeClassifier initialized with weights: {self.weights}")

    def calculate_technical_signal(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        lookback: int = 20
    ) -> Dict[str, float]:
        """
        Calculate technical signals from price/volume.

        Args:
            prices: Price series
            volumes: Optional volume series
            lookback: Lookback period

        Returns:
            Dict with signal components
        """
        if len(prices) < lookback:
            return {
                'trend': 0.0,
                'momentum': 0.0,
                'volatility_regime': 0.0,
                'combined': 0.0
            }

        # Calculate returns
        returns = prices.pct_change()

        # Trend: SMA crossover signal
        sma_short = prices.rolling(lookback // 2).mean()
        sma_long = prices.rolling(lookback).mean()
        trend = (sma_short.iloc[-1] / sma_long.iloc[-1] - 1) * 10  # Scale

        # Momentum: Recent return vs average
        recent_return = returns.iloc[-lookback:].sum()
        momentum = recent_return * 5  # Scale

        # Volatility regime: High vol often = bear
        volatility = returns.iloc[-lookback:].std() * np.sqrt(252)
        avg_vol = returns.rolling(lookback * 2).std().mean() * np.sqrt(252)

        if avg_vol > 0:
            vol_ratio = volatility / avg_vol
            volatility_regime = -0.5 if vol_ratio > 1.5 else 0.0
        else:
            volatility_regime = 0.0

        # Combine
        combined = trend * 0.4 + momentum * 0.4 + volatility_regime * 0.2
        combined = np.clip(combined, -1, 1)

        return {
            'trend': float(np.clip(trend, -1, 1)),
            'momentum': float(np.clip(momentum, -1, 1)),
            'volatility_regime': float(volatility_regime),
            'combined': float(combined)
        }

    def combine_signals(
        self,
        onchain_signal: float,
        sentiment_signal: float,
        technical_signal: float
    ) -> float:
        """
        Combine signals using configured weights.

        Args:
            onchain_signal: On-chain signal (-1 to 1)
            sentiment_signal: Sentiment signal (-1 to 1)
            technical_signal: Technical signal (-1 to 1)

        Returns:
            Combined signal (-1 to 1)
        """
        combined = (
            onchain_signal * self.weights['onchain'] +
            sentiment_signal * self.weights['sentiment'] +
            technical_signal * self.weights['technical']
        )

        return float(np.clip(combined, -1, 1))

    def classify_from_signal(
        self,
        combined_signal: float
    ) -> Tuple[MarketRegime, float]:
        """
        Classify regime from combined signal.

        Args:
            combined_signal: Combined signal value

        Returns:
            Tuple of (regime, confidence)
        """
        thresholds = self.thresholds

        if combined_signal >= thresholds['strong_bull']:
            regime = MarketRegime.STRONG_BULL
            confidence = min(1.0, combined_signal / thresholds['strong_bull'])
        elif combined_signal >= thresholds['bull']:
            regime = MarketRegime.BULL
            confidence = (combined_signal - thresholds['bull']) / (
                thresholds['strong_bull'] - thresholds['bull']
            )
        elif combined_signal >= thresholds['accumulation']:
            regime = MarketRegime.ACCUMULATION
            confidence = 0.5 + combined_signal
        elif combined_signal <= thresholds['strong_bear']:
            regime = MarketRegime.STRONG_BEAR
            confidence = min(1.0, abs(combined_signal) / abs(thresholds['strong_bear']))
        elif combined_signal <= thresholds['bear']:
            regime = MarketRegime.BEAR
            confidence = (abs(combined_signal) - abs(thresholds['bear'])) / (
                abs(thresholds['strong_bear']) - abs(thresholds['bear'])
            )
        elif combined_signal <= thresholds['distribution']:
            regime = MarketRegime.DISTRIBUTION
            confidence = 0.5 + abs(combined_signal)
        else:
            regime = MarketRegime.NEUTRAL
            confidence = 0.5

        return regime, float(np.clip(confidence, 0, 1))

    def classify(
        self,
        onchain_signal: Optional[float] = None,
        sentiment_signal: Optional[float] = None,
        technical_signal: Optional[float] = None,
        prices: Optional[pd.Series] = None,
        timestamp: Optional[datetime] = None
    ) -> RegimeClassification:
        """
        Classify current market regime.

        Args:
            onchain_signal: Pre-computed on-chain signal
            sentiment_signal: Pre-computed sentiment signal
            technical_signal: Pre-computed technical signal (or use prices)
            prices: Price series for technical calculation
            timestamp: Classification timestamp

        Returns:
            RegimeClassification result
        """
        timestamp = timestamp or datetime.now()
        signals = {}

        # Get on-chain signal
        if onchain_signal is not None:
            signals['onchain'] = onchain_signal
        else:
            signals['onchain'] = 0.0
            logger.debug("No on-chain signal provided, using 0")

        # Get sentiment signal
        if sentiment_signal is not None:
            signals['sentiment'] = sentiment_signal
        else:
            signals['sentiment'] = 0.0
            logger.debug("No sentiment signal provided, using 0")

        # Get technical signal
        if technical_signal is not None:
            signals['technical'] = technical_signal
        elif prices is not None:
            tech = self.calculate_technical_signal(prices)
            signals['technical'] = tech['combined']
        else:
            signals['technical'] = 0.0
            logger.debug("No technical signal provided, using 0")

        # Combine signals
        combined = self.combine_signals(
            signals['onchain'],
            signals['sentiment'],
            signals['technical']
        )
        signals['combined'] = combined

        # Classify
        regime, confidence = self.classify_from_signal(combined)

        # Generate description
        descriptions = {
            MarketRegime.STRONG_BULL: "Strong bullish conditions - high conviction uptrend",
            MarketRegime.BULL: "Bullish conditions - uptrend likely to continue",
            MarketRegime.ACCUMULATION: "Accumulation phase - smart money buying",
            MarketRegime.NEUTRAL: "Neutral conditions - no clear direction",
            MarketRegime.DISTRIBUTION: "Distribution phase - smart money selling",
            MarketRegime.BEAR: "Bearish conditions - downtrend likely to continue",
            MarketRegime.STRONG_BEAR: "Strong bearish conditions - high conviction downtrend",
        }

        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            timestamp=timestamp,
            signals=signals,
            description=descriptions[regime]
        )

    def get_trading_bias(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get trading bias based on regime.

        Args:
            regime: Current market regime

        Returns:
            Dict with trading recommendations
        """
        biases = {
            MarketRegime.STRONG_BULL: {
                'position_bias': 'long',
                'position_size_mult': 1.0,
                'stop_loss_mult': 0.8,  # Tighter stops in strong trend
                'take_profit_mult': 1.5,  # Let winners run
            },
            MarketRegime.BULL: {
                'position_bias': 'long',
                'position_size_mult': 0.8,
                'stop_loss_mult': 1.0,
                'take_profit_mult': 1.2,
            },
            MarketRegime.ACCUMULATION: {
                'position_bias': 'long',
                'position_size_mult': 0.6,
                'stop_loss_mult': 1.2,
                'take_profit_mult': 1.0,
            },
            MarketRegime.NEUTRAL: {
                'position_bias': 'flat',
                'position_size_mult': 0.3,
                'stop_loss_mult': 1.5,
                'take_profit_mult': 0.8,
            },
            MarketRegime.DISTRIBUTION: {
                'position_bias': 'short',
                'position_size_mult': 0.5,
                'stop_loss_mult': 1.2,
                'take_profit_mult': 1.0,
            },
            MarketRegime.BEAR: {
                'position_bias': 'short',
                'position_size_mult': 0.7,
                'stop_loss_mult': 1.0,
                'take_profit_mult': 1.2,
            },
            MarketRegime.STRONG_BEAR: {
                'position_bias': 'short',
                'position_size_mult': 0.9,
                'stop_loss_mult': 0.8,
                'take_profit_mult': 1.5,
            },
        }

        return biases.get(regime, biases[MarketRegime.NEUTRAL])


class RegimeHistory:
    """
    Tracks regime history for analysis and backtesting.
    """

    def __init__(self):
        """Initialize empty history."""
        self.history: List[RegimeClassification] = []

    def add(self, classification: RegimeClassification) -> None:
        """Add classification to history."""
        self.history.append(classification)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        if not self.history:
            return pd.DataFrame()

        records = []
        for c in self.history:
            record = {
                'timestamp': c.timestamp,
                'regime': c.regime.value,
                'confidence': c.confidence,
                **c.signals
            }
            records.append(record)

        return pd.DataFrame(records)

    def get_regime_changes(self) -> List[Tuple[datetime, MarketRegime, MarketRegime]]:
        """Get list of regime changes."""
        changes = []
        for i in range(1, len(self.history)):
            if self.history[i].regime != self.history[i-1].regime:
                changes.append((
                    self.history[i].timestamp,
                    self.history[i-1].regime,
                    self.history[i].regime
                ))
        return changes

    def get_regime_durations(self) -> Dict[MarketRegime, List[int]]:
        """Get duration of each regime period (in observations)."""
        if not self.history:
            return {}

        durations: Dict[MarketRegime, List[int]] = {r: [] for r in MarketRegime}
        current_regime = self.history[0].regime
        current_duration = 1

        for i in range(1, len(self.history)):
            if self.history[i].regime == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = self.history[i].regime
                current_duration = 1

        # Add final period
        durations[current_regime].append(current_duration)

        return durations


if __name__ == "__main__":
    # Demo
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== Market Regime Classifier Demo ===\n")

    classifier = RegimeClassifier()

    # Test various signal combinations
    test_cases = [
        {'onchain': 0.7, 'sentiment': 0.5, 'technical': 0.6, 'desc': 'All bullish'},
        {'onchain': 0.3, 'sentiment': 0.2, 'technical': 0.4, 'desc': 'Mildly bullish'},
        {'onchain': 0.0, 'sentiment': 0.1, 'technical': -0.1, 'desc': 'Neutral'},
        {'onchain': -0.3, 'sentiment': -0.4, 'technical': -0.2, 'desc': 'Mildly bearish'},
        {'onchain': -0.8, 'sentiment': -0.6, 'technical': -0.7, 'desc': 'All bearish'},
        {'onchain': 0.5, 'sentiment': -0.3, 'technical': 0.1, 'desc': 'Mixed signals'},
    ]

    print("Regime Classification Results:\n")
    for case in test_cases:
        result = classifier.classify(
            onchain_signal=case['onchain'],
            sentiment_signal=case['sentiment'],
            technical_signal=case['technical']
        )

        bias = classifier.get_trading_bias(result.regime)

        print(f"  {case['desc']:20} -> {result.regime.value:12} "
              f"(conf={result.confidence:.2f}, combined={result.signals['combined']:.2f})")
        print(f"    Trading bias: {bias['position_bias']}, "
              f"size_mult={bias['position_size_mult']:.1f}")

    print("\n✓ Regime classifier demo complete!")
