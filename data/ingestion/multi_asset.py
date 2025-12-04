"""
Multi-Asset Data Fetcher for Regime Detection
==============================================

Fetches price data for multiple crypto assets to detect cross-asset
correlation patterns that precede market crashes.

Key assets:
- BTC: Market leader
- ETH: DeFi proxy
- SOL: Alt-coin proxy
- Stablecoins (USDT/USDC market cap): Flight to safety indicator
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MultiAssetFetcher:
    """
    Fetches and aligns multi-asset crypto data.

    Used for:
    - Cross-asset correlation analysis
    - Regime detection features
    - Flight-to-safety indicators
    """

    # Core assets for regime detection
    ASSETS = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'SOL': 'SOL-USD',
        'BNB': 'BNB-USD',
        'XRP': 'XRP-USD',
        'ADA': 'ADA-USD',
        'AVAX': 'AVAX-USD',
        'DOGE': 'DOGE-USD',
    }

    def __init__(self, assets: Optional[List[str]] = None):
        """
        Initialize fetcher.

        Args:
            assets: List of asset symbols to fetch. Defaults to all.
        """
        if assets:
            self.assets = {k: v for k, v in self.ASSETS.items() if k in assets}
        else:
            self.assets = self.ASSETS

    def fetch_all(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch price data for all assets.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (default: today)
            interval: Data interval ('1d', '1h', etc.)

        Returns:
            DataFrame with columns: {asset}_close, {asset}_volume, etc.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        all_data = {}

        for symbol, ticker in self.assets.items():
            logger.info(f"Fetching {symbol}...")
            try:
                data = yf.Ticker(ticker).history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )

                if len(data) > 0:
                    # Rename columns with asset prefix
                    data = data.rename(columns={
                        'Open': f'{symbol}_open',
                        'High': f'{symbol}_high',
                        'Low': f'{symbol}_low',
                        'Close': f'{symbol}_close',
                        'Volume': f'{symbol}_volume',
                    })

                    # Keep only relevant columns
                    cols = [c for c in data.columns if symbol in c]
                    all_data[symbol] = data[cols]

                    logger.info(f"  {symbol}: {len(data)} rows")
                else:
                    logger.warning(f"  {symbol}: No data returned")

            except Exception as e:
                logger.error(f"  {symbol}: Error fetching - {e}")

        # Merge all dataframes on index
        if not all_data:
            raise ValueError("No data fetched for any asset")

        result = pd.concat(all_data.values(), axis=1)
        result.index = result.index.tz_localize(None)

        # Forward fill missing values (some assets have gaps)
        result = result.ffill()

        logger.info(f"Combined dataset: {len(result)} rows, {len(result.columns)} columns")

        return result

    def calculate_returns(
        self,
        df: pd.DataFrame,
        periods: List[int] = [1, 5, 20]
    ) -> pd.DataFrame:
        """
        Calculate returns for all assets at multiple horizons.

        Args:
            df: DataFrame with {asset}_close columns
            periods: List of return periods

        Returns:
            DataFrame with {asset}_return_{period} columns added
        """
        result = df.copy()

        # Find all close columns
        close_cols = [c for c in df.columns if c.endswith('_close')]

        for col in close_cols:
            asset = col.replace('_close', '')

            for period in periods:
                result[f'{asset}_return_{period}d'] = df[col].pct_change(period)

        return result

    def calculate_correlations(
        self,
        df: pd.DataFrame,
        window: int = 20,
        base_asset: str = 'BTC'
    ) -> pd.DataFrame:
        """
        Calculate rolling correlations between assets.

        Args:
            df: DataFrame with return columns
            window: Rolling window size
            base_asset: Asset to calculate correlations against

        Returns:
            DataFrame with {asset}_corr_to_{base} columns
        """
        result = df.copy()

        base_col = f'{base_asset}_return_1d'
        if base_col not in df.columns:
            raise ValueError(f"Base asset return column {base_col} not found")

        # Find all return columns
        return_cols = [c for c in df.columns if c.endswith('_return_1d')]

        for col in return_cols:
            asset = col.replace('_return_1d', '')
            if asset != base_asset:
                result[f'{asset}_corr_to_{base_asset}'] = (
                    df[col].rolling(window).corr(df[base_col])
                )

        return result

    def calculate_volatility(
        self,
        df: pd.DataFrame,
        windows: List[int] = [10, 20, 60]
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility for all assets.

        Args:
            df: DataFrame with return columns
            windows: List of volatility windows

        Returns:
            DataFrame with {asset}_vol_{window} columns
        """
        result = df.copy()

        return_cols = [c for c in df.columns if c.endswith('_return_1d')]

        for col in return_cols:
            asset = col.replace('_return_1d', '')

            for window in windows:
                result[f'{asset}_vol_{window}d'] = (
                    df[col].rolling(window).std() * np.sqrt(252)  # Annualized
                )

        return result


class RegimeLabeler:
    """
    Labels historical periods as RISK_ON, CAUTION, or RISK_OFF.

    Based on realized drawdowns and volatility expansion.
    """

    def __init__(
        self,
        caution_drawdown: float = 0.10,
        risk_off_drawdown: float = 0.20,
        lookforward: int = 20
    ):
        """
        Initialize labeler.

        Args:
            caution_drawdown: Drawdown threshold for CAUTION (10%)
            risk_off_drawdown: Drawdown threshold for RISK_OFF (20%)
            lookforward: Days to look ahead for labeling
        """
        self.caution_dd = caution_drawdown
        self.risk_off_dd = risk_off_drawdown
        self.lookforward = lookforward

    def label_regimes(
        self,
        prices: pd.Series,
        method: str = 'forward_drawdown'
    ) -> pd.Series:
        """
        Label each day with its regime.

        Args:
            prices: Price series (typically BTC close)
            method: Labeling method
                - 'forward_drawdown': Based on max drawdown in next N days
                - 'realized_vol': Based on realized volatility

        Returns:
            Series with regime labels: 'RISK_ON', 'CAUTION', 'RISK_OFF'
        """
        if method == 'forward_drawdown':
            return self._label_by_forward_drawdown(prices)
        elif method == 'realized_vol':
            return self._label_by_volatility(prices)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _label_by_forward_drawdown(self, prices: pd.Series) -> pd.Series:
        """Label based on maximum drawdown in the next N days."""

        labels = pd.Series(index=prices.index, dtype=str)

        for i in range(len(prices) - self.lookforward):
            current_price = prices.iloc[i]
            future_prices = prices.iloc[i+1:i+1+self.lookforward]

            # Calculate max drawdown from current price
            min_future = future_prices.min()
            max_drawdown = (current_price - min_future) / current_price

            # Assign regime
            if max_drawdown >= self.risk_off_dd:
                labels.iloc[i] = 'RISK_OFF'
            elif max_drawdown >= self.caution_dd:
                labels.iloc[i] = 'CAUTION'
            else:
                labels.iloc[i] = 'RISK_ON'

        # Fill last N days with forward fill
        labels = labels.ffill()

        return labels

    def _label_by_volatility(self, prices: pd.Series) -> pd.Series:
        """Label based on realized volatility regime."""

        returns = prices.pct_change()
        vol = returns.rolling(20).std() * np.sqrt(252)

        # Calculate volatility percentiles
        vol_percentile = vol.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        labels = pd.Series(index=prices.index, dtype=str)
        labels[vol_percentile <= 0.5] = 'RISK_ON'
        labels[(vol_percentile > 0.5) & (vol_percentile <= 0.8)] = 'CAUTION'
        labels[vol_percentile > 0.8] = 'RISK_OFF'

        return labels

    def get_regime_statistics(
        self,
        prices: pd.Series,
        labels: pd.Series
    ) -> Dict:
        """
        Calculate statistics for each regime.

        Returns dict with counts, avg returns, volatility per regime.
        """
        returns = prices.pct_change()

        stats = {}
        for regime in ['RISK_ON', 'CAUTION', 'RISK_OFF']:
            mask = labels == regime
            regime_returns = returns[mask]

            stats[regime] = {
                'count': mask.sum(),
                'pct_of_total': mask.mean(),
                'avg_daily_return': regime_returns.mean(),
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252))
                         if regime_returns.std() > 0 else 0
            }

        return stats


def build_regime_detection_features(
    start_date: str = "2020-01-01",
    assets: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build complete feature set for regime detection.

    Returns:
        Tuple of (features DataFrame, regime labels Series)
    """
    # Fetch data
    fetcher = MultiAssetFetcher(assets)
    df = fetcher.fetch_all(start_date=start_date)

    # Calculate features
    df = fetcher.calculate_returns(df)
    df = fetcher.calculate_correlations(df)
    df = fetcher.calculate_volatility(df)

    # Add cross-asset correlation mean (risk indicator)
    corr_cols = [c for c in df.columns if '_corr_to_BTC' in c]
    if corr_cols:
        df['avg_corr_to_BTC'] = df[corr_cols].mean(axis=1)

    # Add volatility regime indicator
    df['BTC_vol_regime'] = df['BTC_vol_20d'] / df['BTC_vol_60d']

    # Label regimes
    labeler = RegimeLabeler()
    labels = labeler.label_regimes(df['BTC_close'])

    # Drop NaN rows
    valid_idx = df.dropna().index.intersection(labels.dropna().index)
    df = df.loc[valid_idx]
    labels = labels.loc[valid_idx]

    return df, labels


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print("Fetching multi-asset data...")
    df, labels = build_regime_detection_features(
        start_date="2020-01-01",
        assets=['BTC', 'ETH', 'SOL']
    )

    print(f"\nDataset shape: {df.shape}")
    print(f"\nRegime distribution:")
    print(labels.value_counts())

    # Get statistics
    labeler = RegimeLabeler()
    stats = labeler.get_regime_statistics(df['BTC_close'], labels)

    print("\nRegime Statistics:")
    for regime, s in stats.items():
        print(f"\n{regime}:")
        print(f"  Days: {s['count']} ({s['pct_of_total']:.1%})")
        print(f"  Avg Daily Return: {s['avg_daily_return']:.3%}")
        print(f"  Annualized Vol: {s['volatility']:.1%}")
        print(f"  Sharpe: {s['sharpe']:.2f}")
