"""
Alpha Signal Combiner

Combines signals from multiple alpha sources into trading signals.
Integrates: On-chain metrics, Sentiment analysis, Technical indicators, Regime classification.

Author: Claude Opus 4.5
Date: 2024-12-04
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.predictors.regime_classifier import (
    MarketRegime,
    RegimeClassification,
    RegimeClassifier
)

logger = logging.getLogger(__name__)


@dataclass
class AlphaSignal:
    """Individual alpha signal from a source."""
    source: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CombinedSignal:
    """Combined signal from all alpha sources."""
    timestamp: datetime
    signal: float  # -1 to 1 (negative=bearish, positive=bullish)
    confidence: float  # 0 to 1
    regime: MarketRegime
    regime_confidence: float
    component_signals: Dict[str, AlphaSignal]
    position_recommendation: str  # 'long', 'short', 'flat'
    position_size_mult: float  # 0 to 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlphaCombiner:
    """
    Combines multiple alpha sources into trading signals.

    Alpha sources:
    - On-chain metrics (MVRV, SOPR, netflows)
    - Sentiment analysis (CARVS from Reddit)
    - Technical indicators (trend, momentum)
    - Regime classification

    The combiner weights these signals and produces a final trading recommendation.

    Example:
        >>> combiner = AlphaCombiner(config)
        >>> signal = await combiner.generate_signal(prices)
        >>> print(f"Signal: {signal.position_recommendation} ({signal.signal:.2f})")
    """

    # Default weights for alpha sources
    DEFAULT_WEIGHTS = {
        'onchain': 0.35,
        'sentiment': 0.25,
        'technical': 0.25,
        'regime': 0.15,
    }

    # Minimum confidence thresholds
    MIN_CONFIDENCE = {
        'signal': 0.3,        # Min confidence to act
        'position': 0.5,      # Min confidence for full position
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize alpha combiner.

        Args:
            config: Configuration dict
            weights: Override default alpha weights
        """
        self.config = config or {}
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        # Initialize components
        self.regime_classifier = RegimeClassifier()

        # Cache for recent signals
        self._signal_cache: Dict[str, AlphaSignal] = {}
        self._last_combined: Optional[CombinedSignal] = None

        logger.info(f"AlphaCombiner initialized with weights: {self.weights}")

    def add_onchain_signal(
        self,
        signal: float,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> AlphaSignal:
        """
        Add on-chain signal.

        Args:
            signal: Signal value (-1 to 1)
            confidence: Confidence (0 to 1)
            metadata: Additional metadata

        Returns:
            Created AlphaSignal
        """
        alpha = AlphaSignal(
            source='onchain',
            signal=np.clip(signal, -1, 1),
            confidence=np.clip(confidence, 0, 1),
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self._signal_cache['onchain'] = alpha
        return alpha

    def add_sentiment_signal(
        self,
        signal: float,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> AlphaSignal:
        """
        Add sentiment signal.

        Args:
            signal: CARVS or similar score (-1 to 1)
            confidence: Confidence (0 to 1)
            metadata: Additional metadata

        Returns:
            Created AlphaSignal
        """
        alpha = AlphaSignal(
            source='sentiment',
            signal=np.clip(signal, -1, 1),
            confidence=np.clip(confidence, 0, 1),
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self._signal_cache['sentiment'] = alpha
        return alpha

    def add_technical_signal(
        self,
        signal: float,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> AlphaSignal:
        """
        Add technical signal.

        Args:
            signal: Technical score (-1 to 1)
            confidence: Confidence (0 to 1)
            metadata: Additional metadata

        Returns:
            Created AlphaSignal
        """
        alpha = AlphaSignal(
            source='technical',
            signal=np.clip(signal, -1, 1),
            confidence=np.clip(confidence, 0, 1),
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self._signal_cache['technical'] = alpha
        return alpha

    def calculate_technical_from_prices(
        self,
        prices: pd.Series,
        lookback: int = 20
    ) -> AlphaSignal:
        """
        Calculate technical signal from price series.

        Args:
            prices: Price series
            lookback: Lookback period

        Returns:
            Technical AlphaSignal
        """
        tech = self.regime_classifier.calculate_technical_signal(prices, lookback=lookback)

        metadata = {
            'trend': tech['trend'],
            'momentum': tech['momentum'],
            'volatility_regime': tech['volatility_regime']
        }

        return self.add_technical_signal(
            signal=tech['combined'],
            confidence=0.6,  # Medium confidence for technical
            metadata=metadata
        )

    def combine(
        self,
        timestamp: Optional[datetime] = None
    ) -> CombinedSignal:
        """
        Combine all cached signals into final signal.

        Args:
            timestamp: Signal timestamp

        Returns:
            CombinedSignal with recommendation
        """
        timestamp = timestamp or datetime.now()

        # Get signals with defaults for missing
        onchain = self._signal_cache.get('onchain', AlphaSignal(
            source='onchain', signal=0, confidence=0, timestamp=timestamp
        ))
        sentiment = self._signal_cache.get('sentiment', AlphaSignal(
            source='sentiment', signal=0, confidence=0, timestamp=timestamp
        ))
        technical = self._signal_cache.get('technical', AlphaSignal(
            source='technical', signal=0, confidence=0, timestamp=timestamp
        ))

        # Calculate regime
        regime_class = self.regime_classifier.classify(
            onchain_signal=onchain.signal if onchain.confidence > 0 else None,
            sentiment_signal=sentiment.signal if sentiment.confidence > 0 else None,
            technical_signal=technical.signal if technical.confidence > 0 else None,
            timestamp=timestamp
        )

        # Regime signal (-1 to 1 based on regime)
        regime_signals = {
            MarketRegime.STRONG_BULL: 1.0,
            MarketRegime.BULL: 0.5,
            MarketRegime.ACCUMULATION: 0.3,
            MarketRegime.NEUTRAL: 0.0,
            MarketRegime.DISTRIBUTION: -0.3,
            MarketRegime.BEAR: -0.5,
            MarketRegime.STRONG_BEAR: -1.0,
        }
        regime_signal = regime_signals.get(regime_class.regime, 0.0)

        # Weighted combination
        weighted_signals = []
        total_confidence_weight = 0.0

        for source, alpha in [
            ('onchain', onchain),
            ('sentiment', sentiment),
            ('technical', technical),
        ]:
            if alpha.confidence > 0:
                weight = self.weights[source] * alpha.confidence
                weighted_signals.append(alpha.signal * weight)
                total_confidence_weight += weight

        # Add regime contribution
        regime_weight = self.weights['regime'] * regime_class.confidence
        weighted_signals.append(regime_signal * regime_weight)
        total_confidence_weight += regime_weight

        # Calculate combined signal
        if total_confidence_weight > 0:
            combined_signal = sum(weighted_signals) / total_confidence_weight
        else:
            combined_signal = 0.0

        combined_signal = float(np.clip(combined_signal, -1, 1))

        # Calculate combined confidence
        confidences = [
            onchain.confidence * self.weights['onchain'],
            sentiment.confidence * self.weights['sentiment'],
            technical.confidence * self.weights['technical'],
            regime_class.confidence * self.weights['regime'],
        ]
        combined_confidence = sum(confidences)

        # Position recommendation
        if combined_confidence < self.MIN_CONFIDENCE['signal']:
            position = 'flat'
            size_mult = 0.0
        elif combined_signal > 0.2:
            position = 'long'
            size_mult = min(1.0, combined_signal) * (combined_confidence / 1.0)
        elif combined_signal < -0.2:
            position = 'short'
            size_mult = min(1.0, abs(combined_signal)) * (combined_confidence / 1.0)
        else:
            position = 'flat'
            size_mult = 0.0

        # Get regime bias adjustments
        bias = self.regime_classifier.get_trading_bias(regime_class.regime)

        # Adjust size mult based on regime
        size_mult *= bias['position_size_mult']

        result = CombinedSignal(
            timestamp=timestamp,
            signal=combined_signal,
            confidence=combined_confidence,
            regime=regime_class.regime,
            regime_confidence=regime_class.confidence,
            component_signals={
                'onchain': onchain,
                'sentiment': sentiment,
                'technical': technical,
            },
            position_recommendation=position,
            position_size_mult=size_mult,
            metadata={
                'regime_bias': bias,
                'weights': self.weights,
            }
        )

        self._last_combined = result
        return result

    def generate_trading_decision(
        self,
        current_position: str = 'flat',
        max_position_size: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate actionable trading decision.

        Args:
            current_position: Current position ('long', 'short', 'flat')
            max_position_size: Maximum position size (0 to 1)

        Returns:
            Dict with trading decision
        """
        if self._last_combined is None:
            return {
                'action': 'hold',
                'reason': 'No signal generated yet',
                'target_position': current_position,
                'size': 0.0
            }

        signal = self._last_combined
        recommended = signal.position_recommendation

        # Determine action
        if recommended == current_position:
            action = 'hold'
            reason = f'Already {current_position}'
        elif recommended == 'flat':
            action = 'close' if current_position != 'flat' else 'hold'
            reason = 'Signal suggests flat position'
        elif recommended == 'long':
            if current_position == 'short':
                action = 'reverse'
                reason = 'Reversing from short to long'
            else:
                action = 'open_long'
                reason = f'Bullish signal ({signal.signal:.2f})'
        else:  # short
            if current_position == 'long':
                action = 'reverse'
                reason = 'Reversing from long to short'
            else:
                action = 'open_short'
                reason = f'Bearish signal ({signal.signal:.2f})'

        target_size = signal.position_size_mult * max_position_size

        return {
            'action': action,
            'reason': reason,
            'target_position': recommended,
            'size': target_size,
            'signal': signal.signal,
            'confidence': signal.confidence,
            'regime': signal.regime.value,
            'timestamp': signal.timestamp.isoformat()
        }

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current signals."""
        if self._last_combined is None:
            return {'status': 'no_signal'}

        signal = self._last_combined
        return {
            'status': 'active',
            'timestamp': signal.timestamp.isoformat(),
            'combined_signal': signal.signal,
            'confidence': signal.confidence,
            'regime': signal.regime.value,
            'recommendation': signal.position_recommendation,
            'size_mult': signal.position_size_mult,
            'components': {
                name: {
                    'signal': alpha.signal,
                    'confidence': alpha.confidence
                }
                for name, alpha in signal.component_signals.items()
            }
        }


async def create_alpha_pipeline(
    config: Dict[str, Any],
    prices: pd.Series
) -> AlphaCombiner:
    """
    Create and initialize alpha pipeline with all data sources.

    This is a helper function that sets up the full pipeline:
    1. Fetches Reddit data
    2. Runs sentiment analysis
    3. Calculates on-chain signals (if configured)
    4. Calculates technical signals
    5. Returns ready-to-use combiner

    Args:
        config: Configuration with API keys, etc.
        prices: Price series for technical analysis

    Returns:
        Initialized AlphaCombiner with signals
    """
    from data.ingestion.reddit_sources import get_reddit_source
    from data.ingestion.sentiment import FinBERTAnalyzer, SentimentAggregator

    combiner = AlphaCombiner(config)

    # 1. Fetch Reddit data
    logger.info("Fetching Reddit data...")
    reddit = get_reddit_source(config.get('reddit', {}))

    try:
        posts = await reddit.fetch_multiple_subreddits(limit_per_sub=25)
        logger.info(f"Fetched {len(posts)} Reddit posts")
    except Exception as e:
        logger.warning(f"Failed to fetch Reddit: {e}")
        posts = []

    # 2. Run sentiment analysis
    if posts:
        logger.info("Running sentiment analysis...")
        try:
            analyzer = FinBERTAnalyzer()
            aggregator = SentimentAggregator(analyzer)
            aggregated = aggregator.aggregate_sentiment(posts)
            sentiment_signal = aggregator.generate_signal(aggregated)

            combiner.add_sentiment_signal(
                signal=sentiment_signal['signal'],
                confidence=sentiment_signal['confidence'],
                metadata={
                    'carvs_score': sentiment_signal.get('carvs_score', 0),
                    'num_posts': sentiment_signal['num_posts'],
                    'bullish_ratio': sentiment_signal.get('bullish_ratio', 0),
                }
            )
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")

    # 3. Calculate technical signals
    logger.info("Calculating technical signals...")
    combiner.calculate_technical_from_prices(prices)

    # 4. On-chain signals would be added here if Dune queries are configured
    # This requires query_ids to be set up by the user
    if config.get('dune_api_key') and config.get('dune_query_ids'):
        logger.info("On-chain signals not yet configured")
        # Would add on-chain signals here

    # 5. Combine all signals
    logger.info("Combining signals...")
    combiner.combine()

    return combiner


if __name__ == "__main__":
    # Demo
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== Alpha Combiner Demo ===\n")

    # Create combiner
    combiner = AlphaCombiner()

    # Add mock signals
    combiner.add_onchain_signal(0.3, confidence=0.7, metadata={'mvrv': 2.5})
    combiner.add_sentiment_signal(0.4, confidence=0.6, metadata={'carvs': 0.35})
    combiner.add_technical_signal(0.2, confidence=0.8, metadata={'trend': 0.3})

    # Combine
    result = combiner.combine()

    print("Combined Signal Result:")
    print(f"  Signal: {result.signal:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Regime: {result.regime.value}")
    print(f"  Position: {result.position_recommendation}")
    print(f"  Size Mult: {result.position_size_mult:.2f}")

    print("\nComponent Signals:")
    for name, alpha in result.component_signals.items():
        print(f"  {name:12}: signal={alpha.signal:+.2f}, conf={alpha.confidence:.2f}")

    # Trading decision
    print("\nTrading Decision:")
    decision = combiner.generate_trading_decision(current_position='flat')
    print(f"  Action: {decision['action']}")
    print(f"  Reason: {decision['reason']}")
    print(f"  Target: {decision['target_position']} @ {decision['size']:.2f}")

    print("\n✓ Alpha combiner demo complete!")
