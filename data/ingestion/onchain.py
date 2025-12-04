"""
On-Chain Data Provider

Fetches on-chain metrics from Dune Analytics API for crypto market analysis.
Key metrics: MVRV, SOPR, Exchange Netflows, Stablecoin Supply Ratio.

Author: Claude Opus 4.5
Date: 2024-12-04
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetric:
    """Represents an on-chain metric configuration."""
    name: str
    description: str
    query_id: Optional[int] = None  # Dune query ID
    interpretation: str = ""  # How to interpret the metric


# Pre-defined on-chain metrics used for crypto analysis
# Query IDs should be populated with actual Dune query IDs
ONCHAIN_METRICS = {
    'mvrv': OnChainMetric(
        name='MVRV Z-Score',
        description='Market Value to Realized Value ratio (z-scored)',
        query_id=None,  # User needs to create/find this query on Dune
        interpretation='High (>7): Market top likely. Low (<0): Market bottom likely.'
    ),
    'sopr': OnChainMetric(
        name='SOPR',
        description='Spent Output Profit Ratio - measures profit/loss of moved coins',
        query_id=None,
        interpretation='>1: Coins moving at profit. <1: Coins moving at loss.'
    ),
    'exchange_netflow': OnChainMetric(
        name='Exchange Netflow',
        description='Net BTC flowing into/out of exchanges',
        query_id=None,
        interpretation='Positive: Selling pressure. Negative: Accumulation.'
    ),
    'ssr': OnChainMetric(
        name='Stablecoin Supply Ratio',
        description='BTC market cap / Stablecoin supply',
        query_id=None,
        interpretation='Low: High buying power available. High: Limited dry powder.'
    ),
    'nupl': OnChainMetric(
        name='Net Unrealized Profit/Loss',
        description='Aggregate profit/loss of all coins',
        query_id=None,
        interpretation='>0.75: Euphoria (sell). <0: Capitulation (buy).'
    ),
    'puell_multiple': OnChainMetric(
        name='Puell Multiple',
        description='Daily coin issuance value / 365-day MA',
        query_id=None,
        interpretation='>4: Miners selling (top). <0.5: Miners capitulating (bottom).'
    ),
}


class OnChainDataProvider:
    """
    Fetches on-chain metrics from Dune Analytics.

    Dune Analytics provides SQL-queryable blockchain data. Users create queries
    on the Dune platform and this provider executes them via API.

    Example:
        >>> provider = OnChainDataProvider(api_key='your_key')
        >>> mvrv_data = await provider.fetch_metric('mvrv', query_id=12345)
    """

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Dune Analytics provider.

        Args:
            api_key: Dune Analytics API key
            config: Optional configuration with query_ids mapping
        """
        try:
            from dune_client.client import DuneClient
        except ImportError:
            raise ImportError(
                "dune-client not installed. Install with: pip install dune-client"
            )

        self.dune = DuneClient(api_key)
        self.config = config or {}

        # Allow config to override default query IDs
        self.query_ids = self.config.get('query_ids', {})

        logger.info("OnChainDataProvider initialized")

    def get_metric_info(self, metric_name: str) -> Optional[OnChainMetric]:
        """Get information about a metric."""
        return ONCHAIN_METRICS.get(metric_name)

    def list_available_metrics(self) -> List[str]:
        """List all defined metrics."""
        return list(ONCHAIN_METRICS.keys())

    async def fetch_metric(
        self,
        metric_name: str,
        query_id: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fetch on-chain metric data from Dune.

        Args:
            metric_name: Name of the metric (for logging)
            query_id: Dune query ID to execute
            params: Optional query parameters

        Returns:
            DataFrame with metric data

        Raises:
            ValueError: If no query_id provided and not configured
        """
        from dune_client.query import QueryBase

        # Get query ID from config if not provided
        if query_id is None:
            query_id = self.query_ids.get(metric_name)
            if query_id is None:
                metric = ONCHAIN_METRICS.get(metric_name)
                if metric and metric.query_id:
                    query_id = metric.query_id

        if query_id is None:
            raise ValueError(
                f"No query_id provided for metric '{metric_name}'. "
                f"Either pass query_id parameter or configure in query_ids dict."
            )

        logger.info(f"Fetching {metric_name} (query_id={query_id})")

        try:
            query = QueryBase(query_id=query_id)
            results = self.dune.run_query(query)

            if results.result and results.result.rows:
                df = pd.DataFrame(results.result.rows)
                logger.info(f"Fetched {len(df)} rows for {metric_name}")
                return df
            else:
                logger.warning(f"No data returned for {metric_name}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching {metric_name}: {e}")
            raise

    async def fetch_latest_results(
        self,
        query_id: int,
        metric_name: str = "unknown"
    ) -> pd.DataFrame:
        """
        Fetch latest cached results from a Dune query without re-executing.

        Faster than fetch_metric() but may return stale data.

        Args:
            query_id: Dune query ID
            metric_name: Name for logging

        Returns:
            DataFrame with cached results
        """
        logger.info(f"Fetching latest cached results for {metric_name}")

        try:
            results = self.dune.get_latest_result(query_id)

            if results.result and results.result.rows:
                df = pd.DataFrame(results.result.rows)
                logger.info(f"Got {len(df)} cached rows for {metric_name}")
                return df
            else:
                logger.warning(f"No cached data for {metric_name}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching cached {metric_name}: {e}")
            raise

    async def fetch_multiple_metrics(
        self,
        metrics: Dict[str, int]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple metrics.

        Args:
            metrics: Dict mapping metric_name -> query_id

        Returns:
            Dict mapping metric_name -> DataFrame
        """
        results = {}

        for metric_name, query_id in metrics.items():
            try:
                df = await self.fetch_metric(metric_name, query_id=query_id)
                results[metric_name] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {metric_name}: {e}")
                results[metric_name] = pd.DataFrame()

        return results


class OnChainSignalGenerator:
    """
    Generates trading signals from on-chain metrics.

    Uses thresholds from academic research to classify market conditions.
    """

    # Thresholds for signal generation (from research)
    THRESHOLDS = {
        'mvrv': {
            'extreme_high': 7.0,   # Strong sell signal
            'high': 3.0,          # Caution - overvalued
            'low': 0.0,           # Caution - undervalued
            'extreme_low': -0.5,  # Strong buy signal
        },
        'sopr': {
            'profit_taking': 1.05,  # Coins moving at profit
            'capitulation': 0.95,   # Coins moving at loss
        },
        'nupl': {
            'euphoria': 0.75,     # Extreme greed
            'belief': 0.5,        # Confidence
            'optimism': 0.25,     # Early bull
            'hope': 0.0,          # Neutral
            'capitulation': -0.25, # Extreme fear
        },
    }

    def __init__(self, provider: OnChainDataProvider):
        """
        Initialize signal generator.

        Args:
            provider: OnChainDataProvider instance
        """
        self.provider = provider

    def interpret_mvrv(self, mvrv_zscore: float) -> Dict[str, Any]:
        """
        Interpret MVRV Z-Score.

        Args:
            mvrv_zscore: Current MVRV Z-Score value

        Returns:
            Dict with signal (-1 to 1), regime, and description
        """
        thresholds = self.THRESHOLDS['mvrv']

        if mvrv_zscore >= thresholds['extreme_high']:
            return {
                'signal': -1.0,
                'regime': 'extreme_overvaluation',
                'description': 'Historical top territory - strong sell signal',
                'value': mvrv_zscore
            }
        elif mvrv_zscore >= thresholds['high']:
            return {
                'signal': -0.5,
                'regime': 'overvaluation',
                'description': 'Market overvalued - reduce exposure',
                'value': mvrv_zscore
            }
        elif mvrv_zscore <= thresholds['extreme_low']:
            return {
                'signal': 1.0,
                'regime': 'extreme_undervaluation',
                'description': 'Historical bottom territory - strong buy signal',
                'value': mvrv_zscore
            }
        elif mvrv_zscore <= thresholds['low']:
            return {
                'signal': 0.5,
                'regime': 'undervaluation',
                'description': 'Market undervalued - accumulation zone',
                'value': mvrv_zscore
            }
        else:
            return {
                'signal': 0.0,
                'regime': 'fair_value',
                'description': 'Market fairly valued - neutral',
                'value': mvrv_zscore
            }

    def interpret_sopr(self, sopr: float) -> Dict[str, Any]:
        """
        Interpret SOPR value.

        Args:
            sopr: Current SOPR value

        Returns:
            Dict with signal, regime, and description
        """
        thresholds = self.THRESHOLDS['sopr']

        if sopr >= thresholds['profit_taking']:
            return {
                'signal': -0.3,
                'regime': 'profit_taking',
                'description': 'Coins moving at profit - potential distribution',
                'value': sopr
            }
        elif sopr <= thresholds['capitulation']:
            return {
                'signal': 0.5,
                'regime': 'capitulation',
                'description': 'Coins moving at loss - potential accumulation',
                'value': sopr
            }
        else:
            return {
                'signal': 0.0,
                'regime': 'neutral',
                'description': 'SOPR near break-even',
                'value': sopr
            }

    def interpret_exchange_netflow(
        self,
        netflow: float,
        avg_netflow: float = 0.0
    ) -> Dict[str, Any]:
        """
        Interpret exchange netflow.

        Args:
            netflow: Current netflow (positive = into exchanges)
            avg_netflow: Average netflow for comparison

        Returns:
            Dict with signal, regime, and description
        """
        # Normalize relative to average
        relative_flow = netflow - avg_netflow

        if relative_flow > 0:
            intensity = min(1.0, relative_flow / 10000)  # Normalize
            return {
                'signal': -intensity * 0.5,
                'regime': 'distribution',
                'description': 'Net inflow to exchanges - selling pressure',
                'value': netflow
            }
        else:
            intensity = min(1.0, abs(relative_flow) / 10000)
            return {
                'signal': intensity * 0.5,
                'regime': 'accumulation',
                'description': 'Net outflow from exchanges - accumulation',
                'value': netflow
            }

    def combine_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Combine multiple on-chain signals into aggregate signal.

        Args:
            signals: Dict of metric_name -> signal_dict
            weights: Optional weights for each metric

        Returns:
            Combined signal dict
        """
        default_weights = {
            'mvrv': 0.35,
            'sopr': 0.25,
            'exchange_netflow': 0.25,
            'nupl': 0.15,
        }
        weights = weights or default_weights

        weighted_signal = 0.0
        total_weight = 0.0
        regimes = []

        for metric, signal_dict in signals.items():
            if metric in weights:
                weight = weights[metric]
                weighted_signal += signal_dict['signal'] * weight
                total_weight += weight
                regimes.append(f"{metric}:{signal_dict['regime']}")

        if total_weight > 0:
            combined_signal = weighted_signal / total_weight
        else:
            combined_signal = 0.0

        # Classify overall regime
        if combined_signal >= 0.5:
            overall_regime = 'strong_bullish'
        elif combined_signal >= 0.2:
            overall_regime = 'bullish'
        elif combined_signal <= -0.5:
            overall_regime = 'strong_bearish'
        elif combined_signal <= -0.2:
            overall_regime = 'bearish'
        else:
            overall_regime = 'neutral'

        return {
            'signal': combined_signal,
            'regime': overall_regime,
            'component_regimes': regimes,
            'description': f"Combined on-chain signal: {overall_regime}",
        }


if __name__ == "__main__":
    # Demo of signal interpretation (no API call)
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== On-Chain Signal Interpretation Demo ===\n")

    # Create mock provider for demo
    class MockProvider:
        pass

    generator = OnChainSignalGenerator(MockProvider())

    # Demo MVRV interpretations
    print("MVRV Z-Score Interpretations:")
    for mvrv in [8.0, 4.0, 1.5, -0.3, -1.0]:
        result = generator.interpret_mvrv(mvrv)
        print(f"  MVRV={mvrv:>5.1f}: {result['regime']:>25} | signal={result['signal']:>5.2f}")

    print("\nSOPR Interpretations:")
    for sopr in [1.1, 1.02, 0.98, 0.92]:
        result = generator.interpret_sopr(sopr)
        print(f"  SOPR={sopr:>5.2f}: {result['regime']:>15} | signal={result['signal']:>5.2f}")

    print("\nExchange Netflow Interpretations:")
    for flow in [15000, 5000, -3000, -12000]:
        result = generator.interpret_exchange_netflow(flow)
        print(f"  Netflow={flow:>7}: {result['regime']:>15} | signal={result['signal']:>5.2f}")

    # Combined signal
    print("\nCombined Signal Example:")
    signals = {
        'mvrv': generator.interpret_mvrv(3.5),
        'sopr': generator.interpret_sopr(1.03),
        'exchange_netflow': generator.interpret_exchange_netflow(-5000),
    }
    combined = generator.combine_signals(signals)
    print(f"  Overall: {combined['regime']} | signal={combined['signal']:.2f}")
    print(f"  Components: {', '.join(combined['component_regimes'])}")

    print("\n✓ On-chain signal interpretation demo complete!")
