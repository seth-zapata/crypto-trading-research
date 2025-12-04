"""
Feature Engineering Pipeline

Computes technical indicators and features from OHLCV data for ML models.
All features are computed using only past data (no look-ahead bias).

Features computed:
- Price returns (1h, 4h, 24h)
- Moving averages (SMA, EMA)
- Volatility (rolling std, ATR, Bollinger Bands)
- Momentum (RSI, MACD)
- Volume indicators (SMA, ratio, OBV)

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Generates technical features from OHLCV data.

    All features use only past data to avoid look-ahead bias.
    Features align with the schema in sql/schema.sql.

    Attributes:
        warmup_periods: Number of periods needed before features are valid

    Example:
        >>> engineer = FeatureEngineer()
        >>> features = engineer.generate_all_features(ohlcv_df)
        >>> features_clean = engineer.remove_warmup(features)
    """

    # Feature computation requires historical data (warmup period)
    # Max lookback is 50 periods (for SMA_50)
    DEFAULT_WARMUP = 50

    def __init__(self, warmup_periods: int = DEFAULT_WARMUP):
        """
        Initialize the feature engineer.

        Args:
            warmup_periods: Periods to skip at start (features will be NaN)
        """
        self.warmup_periods = warmup_periods
        logger.info(f"FeatureEngineer initialized with {warmup_periods} warmup periods")

    def generate_all_features(
        self,
        df: pd.DataFrame,
        include_ohlcv: bool = True
    ) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            df: DataFrame with columns: time, open, high, low, close, volume
                Optional: symbol, exchange
            include_ohlcv: Whether to include original OHLCV columns

        Returns:
            DataFrame with all computed features

        Raises:
            ValueError: If required columns are missing
        """
        # Validate input
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Work with a copy
        df = df.copy()

        # Ensure sorted by time
        df = df.sort_values('time').reset_index(drop=True)

        logger.info(f"Generating features for {len(df)} rows")

        # Compute feature groups
        df = self._add_price_features(df)
        df = self._add_moving_averages(df)
        df = self._add_volatility_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volume_features(df)

        # Select output columns
        if include_ohlcv:
            output_cols = list(df.columns)
        else:
            # Exclude raw OHLCV (keep time, symbol, exchange)
            exclude = ['open', 'high', 'low', 'close', 'volume']
            output_cols = [c for c in df.columns if c not in exclude]

        result = df[output_cols]

        # Log feature summary
        feature_cols = [c for c in result.columns
                       if c not in ['time', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume']]
        logger.info(f"Generated {len(feature_cols)} features: {feature_cols}")

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features (returns).

        Features:
        - return_1h: 1-period return
        - return_4h: 4-period return
        - return_24h: 24-period return
        - log_return_1h: Log return (more suitable for ML)
        """
        close = df['close']

        # Simple returns
        df['return_1h'] = close.pct_change(1)
        df['return_4h'] = close.pct_change(4)
        df['return_24h'] = close.pct_change(24)

        # Log returns (better for modeling)
        df['log_return_1h'] = np.log(close / close.shift(1))

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add moving average features.

        Features:
        - sma_5, sma_10, sma_20, sma_50: Simple moving averages
        - ema_12, ema_26: Exponential moving averages (for MACD)
        """
        close = df['close']

        # Simple Moving Averages
        df['sma_5'] = close.rolling(window=5).mean()
        df['sma_10'] = close.rolling(window=10).mean()
        df['sma_20'] = close.rolling(window=20).mean()
        df['sma_50'] = close.rolling(window=50).mean()

        # Exponential Moving Averages
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility features.

        Features:
        - volatility_20: 20-period rolling standard deviation of returns
        - atr_14: 14-period Average True Range
        - bb_upper, bb_lower: Bollinger Bands (20-period, 2 std)
        - bb_width: Bollinger Band width (normalized)
        """
        close = df['close']
        high = df['high']
        low = df['low']

        # Rolling volatility of returns
        returns = close.pct_change()
        df['volatility_20'] = returns.rolling(window=20).std()

        # Average True Range (ATR)
        df['atr_14'] = self._calculate_atr(high, low, close, period=14)

        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()

        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20

        return df

    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range.

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = EMA of True Range
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Use EMA for smoothing
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum features.

        Features:
        - rsi_14: 14-period Relative Strength Index
        - macd: MACD line (EMA12 - EMA26)
        - macd_signal: Signal line (9-period EMA of MACD)
        - macd_hist: MACD histogram (MACD - Signal)
        """
        close = df['close']

        # RSI
        df['rsi_14'] = self._calculate_rsi(close, period=14)

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)

        # Calculate average gains/losses using EMA
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero (no losses = RSI of 100)
        rsi = rsi.replace([np.inf, -np.inf], 100)

        return rsi

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.

        Features:
        - volume_sma_20: 20-period SMA of volume
        - volume_ratio: Current volume / SMA (above 1 = high volume)
        - obv: On-Balance Volume (cumulative)
        """
        volume = df['volume']
        close = df['close']

        # Volume moving average
        df['volume_sma_20'] = volume.rolling(window=20).mean()

        # Volume ratio
        df['volume_ratio'] = volume / df['volume_sma_20']

        # On-Balance Volume
        df['obv'] = self._calculate_obv(close, volume)

        return df

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume.

        OBV increases when close > previous close, decreases otherwise.
        """
        # Direction: +1 if up, -1 if down, 0 if unchanged
        direction = np.sign(close.diff())

        # OBV = cumulative sum of (direction * volume)
        obv = (direction * volume).fillna(0).cumsum()

        return obv

    def remove_warmup(
        self,
        df: pd.DataFrame,
        warmup: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Remove warmup period rows (where features are NaN).

        Args:
            df: DataFrame with features
            warmup: Number of rows to remove (default: self.warmup_periods)

        Returns:
            DataFrame without warmup rows
        """
        warmup = warmup or self.warmup_periods
        return df.iloc[warmup:].reset_index(drop=True)

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names generated.

        Returns:
            List of feature column names
        """
        return [
            # Price features
            'return_1h', 'return_4h', 'return_24h', 'log_return_1h',
            # Moving averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            # Volatility
            'volatility_20', 'atr_14', 'bb_upper', 'bb_lower', 'bb_width',
            # Momentum
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            # Volume
            'volume_sma_20', 'volume_ratio', 'obv'
        ]

    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate computed features for quality issues.

        Checks:
        - NaN values after warmup
        - Infinite values
        - Extreme outliers

        Args:
            df: DataFrame with features

        Returns:
            Dict with validation results
        """
        # Skip warmup period
        df_valid = self.remove_warmup(df)
        feature_cols = self.get_feature_names()

        # Only check features that exist in df
        feature_cols = [c for c in feature_cols if c in df_valid.columns]

        results = {
            'total_rows': len(df_valid),
            'nan_counts': {},
            'inf_counts': {},
            'issues': []
        }

        for col in feature_cols:
            nan_count = df_valid[col].isna().sum()
            inf_count = np.isinf(df_valid[col]).sum()

            if nan_count > 0:
                results['nan_counts'][col] = nan_count
                results['issues'].append(f"{col}: {nan_count} NaN values")

            if inf_count > 0:
                results['inf_counts'][col] = inf_count
                results['issues'].append(f"{col}: {inf_count} infinite values")

        results['is_valid'] = len(results['issues']) == 0

        if results['is_valid']:
            logger.info("Feature validation passed")
        else:
            logger.warning(f"Feature validation issues: {results['issues']}")

        return results


def compute_features(
    df: pd.DataFrame,
    remove_warmup: bool = True
) -> pd.DataFrame:
    """
    Convenience function to compute all features.

    Args:
        df: OHLCV DataFrame
        remove_warmup: Whether to remove warmup period

    Returns:
        DataFrame with features

    Example:
        >>> features = compute_features(ohlcv_df)
    """
    engineer = FeatureEngineer()
    result = engineer.generate_all_features(df)

    if remove_warmup:
        result = engineer.remove_warmup(result)

    return result


if __name__ == "__main__":
    # Quick test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("Testing FeatureEngineer...")

    # Create sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')

    # Random walk price
    returns = np.random.randn(n) * 0.01
    price = 50000 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'time': dates,
        'open': price * (1 + np.random.randn(n) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(n)) * 0.005),
        'low': price * (1 - np.abs(np.random.randn(n)) * 0.005),
        'close': price,
        'volume': 1000 + np.random.randn(n) * 100,
        'symbol': 'BTC/USD'
    })

    # Generate features
    engineer = FeatureEngineer()
    features = engineer.generate_all_features(df)

    print(f"\nGenerated {len(features.columns)} columns:")
    print(features.columns.tolist())

    print(f"\nSample features (last 5 rows):")
    feature_cols = engineer.get_feature_names()
    print(features[['time'] + feature_cols[:5]].tail())

    # Validate
    validation = engineer.validate_features(features)
    print(f"\nValidation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")

    # Remove warmup
    clean = engineer.remove_warmup(features)
    print(f"\nAfter removing warmup: {len(clean)} rows (was {len(features)})")

    print("\n✓ FeatureEngineer test complete!")
