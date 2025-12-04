"""
On-Chain Data Provider

Fetches on-chain metrics from multiple sources:
- CoinMetrics Community API (free, no key needed): MVRV, Realized Cap
- Dune Analytics: SOPR, Exchange Netflows, Stablecoin Supply

Author: Claude Opus 4.5
Date: 2024-12-04
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetric:
    """Represents an on-chain metric configuration."""
    name: str
    description: str
    source: str  # 'coinmetrics' or 'dune'
    query_id: Optional[int] = None  # Dune query ID
    interpretation: str = ""  # How to interpret the metric


# Pre-defined on-chain metrics with public query IDs
ONCHAIN_METRICS = {
    'mvrv': OnChainMetric(
        name='MVRV',
        description='Market Value to Realized Value ratio',
        source='coinmetrics',
        query_id=None,  # CoinMetrics - no query ID needed
        interpretation='< 1.0: Undervalued. 1.0-2.5: Fair. > 3.7: Overvalued.'
    ),
    'mvrv_zscore': OnChainMetric(
        name='MVRV Z-Score',
        description='MVRV normalized using rolling mean/std',
        source='coinmetrics',
        query_id=None,
        interpretation='< -1.0: Strong buy. -1.0 to 1.0: Neutral. > 2.0: Strong sell.'
    ),
    'sopr': OnChainMetric(
        name='SOPR',
        description='Spent Output Profit Ratio - measures profit/loss of moved coins',
        source='dune',
        query_id=5130629,  # Public query by @sagarfieldelevate
        interpretation='< 0.97: Capitulation (buy). 0.97-1.03: Neutral. > 1.03: Profit taking (sell).'
    ),
    'exchange_netflow': OnChainMetric(
        name='Exchange Netflow',
        description='Net BTC flowing into/out of exchanges',
        source='dune',
        query_id=1621987,  # Public exchange netflow query
        interpretation='Positive: Selling pressure. Negative: Accumulation.'
    ),
    'stablecoin_supply': OnChainMetric(
        name='Stablecoin Supply',
        description='Total stablecoin market cap (buying power indicator)',
        source='dune',
        query_id=4425983,  # Public stablecoin supply query
        interpretation='High supply = dry powder for buying.'
    ),
    'nupl': OnChainMetric(
        name='Net Unrealized Profit/Loss',
        description='Aggregate profit/loss of all coins',
        source='coinmetrics',
        query_id=None,
        interpretation='>0.75: Euphoria (sell). <0: Capitulation (buy).'
    ),
}


class CoinMetricsProvider:
    """
    Fetches on-chain metrics from CoinMetrics Community API.

    Free API - no key required. Rate limit: 10 requests per 6 seconds.

    Example:
        >>> provider = CoinMetricsProvider()
        >>> mvrv_df = await provider.fetch_mvrv('btc', days_back=365)
    """

    BASE_URL = "https://community-api.coinmetrics.io/v4"

    def __init__(self):
        """Initialize CoinMetrics provider."""
        logger.info("CoinMetricsProvider initialized (no API key needed)")

    async def fetch_mvrv(
        self,
        asset: str = "btc",
        days_back: int = 365,
        start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch MVRV and related metrics from CoinMetrics.

        Args:
            asset: Asset symbol (btc, eth)
            days_back: Number of days of history (ignored if start_date provided)
            start_date: Optional start date string (YYYY-MM-DD) for full history

        Returns:
            DataFrame with time, mvrv, market_cap columns
            Note: CapRealUSD not available in free tier
        """
        import aiohttp

        # Use start_date if provided, otherwise calculate from days_back
        if start_date:
            start_time = start_date
        else:
            start_time = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Only request free-tier metrics (CapRealUSD is paid-only)
        params = {
            "assets": asset,
            "metrics": "CapMVRVCur,CapMrktCurUSD",
            "frequency": "1d",
            "start_time": start_time,
            "page_size": 10000,  # Get maximum data per request
        }

        url = f"{self.BASE_URL}/timeseries/asset-metrics"
        all_rows = []

        try:
            async with aiohttp.ClientSession() as session:
                # Paginate through all results
                while True:
                    async with session.get(url, params=params, timeout=60) as resp:
                        resp.raise_for_status()
                        data = await resp.json()

                    for item in data.get('data', []):
                        all_rows.append({
                            'time': pd.to_datetime(item['time']),
                            'mvrv': float(item.get('CapMVRVCur', 0)) if item.get('CapMVRVCur') else None,
                            'market_cap': float(item.get('CapMrktCurUSD', 0)) if item.get('CapMrktCurUSD') else None,
                        })

                    # Check for next page
                    next_page = data.get('next_page_url')
                    if not next_page:
                        break
                    url = next_page
                    params = {}  # URL already has params

            df = pd.DataFrame(all_rows)

            if len(df) > 0:
                df = df.sort_values('time').reset_index(drop=True)
                # Calculate MVRV Z-Score locally using rolling window
                df['mvrv_zscore'] = (
                    (df['mvrv'] - df['mvrv'].rolling(365, min_periods=30).mean()) /
                    df['mvrv'].rolling(365, min_periods=30).std()
                )
                logger.info(f"Fetched {len(df)} MVRV records from CoinMetrics ({df['time'].min()} to {df['time'].max()})")

            return df

        except Exception as e:
            logger.error(f"Error fetching MVRV from CoinMetrics: {e}")
            raise

    async def fetch_metrics(
        self,
        asset: str = "btc",
        metrics: List[str] = None,
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Fetch multiple metrics from CoinMetrics.

        Args:
            asset: Asset symbol
            metrics: List of CoinMetrics metric names
            days_back: Number of days of history

        Returns:
            DataFrame with time and requested metrics
        """
        import aiohttp

        if metrics is None:
            # Only free-tier metrics (CapRealUSD and NUPLAll are paid-only)
            metrics = ["CapMVRVCur", "CapMrktCurUSD"]

        params = {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "start_time": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
        }

        url = f"{self.BASE_URL}/timeseries/asset-metrics"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            rows = []
            for item in data.get('data', []):
                row = {'time': pd.to_datetime(item['time'])}
                for metric in metrics:
                    val = item.get(metric)
                    row[metric] = float(val) if val else None
                rows.append(row)

            df = pd.DataFrame(rows)
            logger.info(f"Fetched {len(df)} records for {len(metrics)} metrics from CoinMetrics")
            return df

        except Exception as e:
            logger.error(f"Error fetching metrics from CoinMetrics: {e}")
            raise


class OnChainDataProvider:
    """
    Fetches on-chain metrics from Dune Analytics.

    Dune Analytics provides SQL-queryable blockchain data. Uses public
    community queries for SOPR, Exchange Netflows, and Stablecoin Supply.

    Example:
        >>> provider = OnChainDataProvider(api_key='your_key')
        >>> sopr_data = await provider.fetch_sopr()
    """

    # Default public query IDs
    DEFAULT_QUERY_IDS = {
        'sopr': 5130629,
        'exchange_netflow': 1621987,
        'stablecoin_supply': 4425983,
    }

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

        # Merge config query IDs with defaults
        self.query_ids = {**self.DEFAULT_QUERY_IDS, **self.config.get('query_ids', {})}

        # Also initialize CoinMetrics provider
        self.coinmetrics = CoinMetricsProvider()

        logger.info(f"OnChainDataProvider initialized with query_ids: {self.query_ids}")

    def get_metric_info(self, metric_name: str) -> Optional[OnChainMetric]:
        """Get information about a metric."""
        return ONCHAIN_METRICS.get(metric_name)

    def list_available_metrics(self) -> List[str]:
        """List all defined metrics."""
        return list(ONCHAIN_METRICS.keys())

    async def fetch_sopr(self) -> pd.DataFrame:
        """
        Fetch SOPR data from Dune (query 5130629).

        Returns:
            DataFrame with SOPR time series
        """
        query_id = self.query_ids.get('sopr', 5130629)
        return await self.fetch_latest_results(query_id, 'sopr')

    async def fetch_exchange_netflow(self) -> pd.DataFrame:
        """
        Fetch exchange netflow data from Dune (query 1621987).

        Returns:
            DataFrame with exchange netflow time series
        """
        query_id = self.query_ids.get('exchange_netflow', 1621987)
        return await self.fetch_latest_results(query_id, 'exchange_netflow')

    async def fetch_stablecoin_supply(self) -> pd.DataFrame:
        """
        Fetch stablecoin supply data from Dune (query 4425983).

        Returns:
            DataFrame with stablecoin supply time series
        """
        query_id = self.query_ids.get('stablecoin_supply', 4425983)
        return await self.fetch_latest_results(query_id, 'stablecoin_supply')

    async def fetch_mvrv(self, asset: str = "btc", days_back: int = 365) -> pd.DataFrame:
        """
        Fetch MVRV data from CoinMetrics (free API).

        Args:
            asset: Asset symbol (btc, eth)
            days_back: Number of days of history

        Returns:
            DataFrame with MVRV and Z-Score
        """
        return await self.coinmetrics.fetch_mvrv(asset, days_back)

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
        logger.info(f"Fetching latest cached results for {metric_name} (query_id={query_id})")

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

    async def fetch_all_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available on-chain metrics.

        Returns:
            Dict mapping metric_name -> DataFrame
        """
        results = {}

        # Fetch from CoinMetrics
        try:
            mvrv_df = await self.fetch_mvrv()
            results['mvrv'] = mvrv_df
        except Exception as e:
            logger.warning(f"Failed to fetch MVRV: {e}")
            results['mvrv'] = pd.DataFrame()

        # Fetch from Dune
        for metric_name in ['sopr', 'exchange_netflow', 'stablecoin_supply']:
            try:
                query_id = self.query_ids.get(metric_name)
                if query_id:
                    df = await self.fetch_latest_results(query_id, metric_name)
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

    # Updated thresholds based on research
    THRESHOLDS = {
        'mvrv': {
            'strong_sell': 3.7,   # Overvalued - strong sell
            'sell': 2.5,          # Caution - reduce exposure
            'buy': 1.0,           # Undervalued - accumulate
            'strong_buy': 0.8,    # Strong buy signal
        },
        'mvrv_zscore': {
            'strong_sell': 2.0,   # Strong sell signal
            'sell': 1.0,          # Caution
            'buy': -1.0,          # Accumulation zone
            'strong_buy': -1.5,   # Strong buy signal
        },
        'sopr': {
            'profit_taking': 1.03,  # Coins moving at profit
            'neutral_high': 1.03,
            'neutral_low': 0.97,
            'capitulation': 0.97,   # Coins moving at loss
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

    def interpret_mvrv(self, mvrv: float, use_zscore: bool = False) -> Dict[str, Any]:
        """
        Interpret MVRV or MVRV Z-Score.

        Args:
            mvrv: Current MVRV or MVRV Z-Score value
            use_zscore: Whether the value is a Z-Score

        Returns:
            Dict with signal (-1 to 1), regime, and description
        """
        thresholds = self.THRESHOLDS['mvrv_zscore'] if use_zscore else self.THRESHOLDS['mvrv']

        if mvrv >= thresholds['strong_sell']:
            return {
                'signal': -1.0,
                'regime': 'extreme_overvaluation',
                'description': 'Historical top territory - strong sell signal',
                'value': mvrv
            }
        elif mvrv >= thresholds['sell']:
            return {
                'signal': -0.5,
                'regime': 'overvaluation',
                'description': 'Market overvalued - reduce exposure',
                'value': mvrv
            }
        elif mvrv <= thresholds['strong_buy']:
            return {
                'signal': 1.0,
                'regime': 'extreme_undervaluation',
                'description': 'Historical bottom territory - strong buy signal',
                'value': mvrv
            }
        elif mvrv <= thresholds['buy']:
            return {
                'signal': 0.5,
                'regime': 'undervaluation',
                'description': 'Market undervalued - accumulation zone',
                'value': mvrv
            }
        else:
            return {
                'signal': 0.0,
                'regime': 'fair_value',
                'description': 'Market fairly valued - neutral',
                'value': mvrv
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

    async def get_current_signals(self) -> Dict[str, Any]:
        """
        Fetch latest data and generate current signals.

        Returns:
            Dict with all signals and combined result
        """
        signals = {}

        # Fetch MVRV from CoinMetrics
        try:
            mvrv_df = await self.provider.fetch_mvrv(days_back=30)
            if len(mvrv_df) > 0:
                latest_mvrv = mvrv_df.iloc[-1]['mvrv']
                signals['mvrv'] = self.interpret_mvrv(latest_mvrv)
        except Exception as e:
            logger.warning(f"Could not fetch MVRV: {e}")

        # Fetch SOPR from Dune
        try:
            sopr_df = await self.provider.fetch_sopr()
            if len(sopr_df) > 0:
                # Find the SOPR column (may vary by query)
                sopr_col = next((c for c in sopr_df.columns if 'sopr' in c.lower()), None)
                if sopr_col:
                    latest_sopr = float(sopr_df.iloc[-1][sopr_col])
                    signals['sopr'] = self.interpret_sopr(latest_sopr)
        except Exception as e:
            logger.warning(f"Could not fetch SOPR: {e}")

        # Fetch Exchange Netflow from Dune
        try:
            netflow_df = await self.provider.fetch_exchange_netflow()
            if len(netflow_df) > 0:
                # Find the netflow column
                netflow_col = next((c for c in netflow_df.columns if 'netflow' in c.lower() or 'net' in c.lower()), None)
                if netflow_col:
                    latest_netflow = float(netflow_df.iloc[-1][netflow_col])
                    signals['exchange_netflow'] = self.interpret_exchange_netflow(latest_netflow)
        except Exception as e:
            logger.warning(f"Could not fetch exchange netflow: {e}")

        # Combine signals
        if signals:
            combined = self.combine_signals(signals)
        else:
            combined = {
                'signal': 0.0,
                'regime': 'unknown',
                'description': 'No on-chain data available'
            }

        return {
            'signals': signals,
            'combined': combined,
            'timestamp': datetime.now().isoformat()
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

    # Demo MVRV interpretations (updated thresholds)
    print("MVRV Interpretations (raw value):")
    for mvrv in [4.0, 3.0, 2.0, 1.5, 1.0, 0.7]:
        result = generator.interpret_mvrv(mvrv, use_zscore=False)
        print(f"  MVRV={mvrv:>5.1f}: {result['regime']:>25} | signal={result['signal']:>5.2f}")

    print("\nMVRV Z-Score Interpretations:")
    for zscore in [2.5, 1.5, 0.5, -0.5, -1.2, -2.0]:
        result = generator.interpret_mvrv(zscore, use_zscore=True)
        print(f"  Z-Score={zscore:>5.1f}: {result['regime']:>25} | signal={result['signal']:>5.2f}")

    print("\nSOPR Interpretations:")
    for sopr in [1.1, 1.02, 0.99, 0.95]:
        result = generator.interpret_sopr(sopr)
        print(f"  SOPR={sopr:>5.2f}: {result['regime']:>15} | signal={result['signal']:>5.2f}")

    print("\nExchange Netflow Interpretations:")
    for flow in [15000, 5000, -3000, -12000]:
        result = generator.interpret_exchange_netflow(flow)
        print(f"  Netflow={flow:>7}: {result['regime']:>15} | signal={result['signal']:>5.2f}")

    # Combined signal
    print("\nCombined Signal Example:")
    signals = {
        'mvrv': generator.interpret_mvrv(1.8),
        'sopr': generator.interpret_sopr(1.01),
        'exchange_netflow': generator.interpret_exchange_netflow(-5000),
    }
    combined = generator.combine_signals(signals)
    print(f"  Overall: {combined['regime']} | signal={combined['signal']:.2f}")
    print(f"  Components: {', '.join(combined['component_regimes'])}")

    print("\n✓ On-chain signal interpretation demo complete!")
