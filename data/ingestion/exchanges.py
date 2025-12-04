"""
Exchange Data Ingestion Module

Handles fetching OHLCV and other market data from cryptocurrency exchanges
using the ccxt library. Primary exchange is Coinbase (Advanced Trade API).

Features:
- Rate-limited API calls with automatic retry
- Historical data backfill
- Real-time data streaming (future)
- Graceful error handling with fallbacks

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import ccxt
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)


def fix_pem_newlines(pem_string: str) -> str:
    """
    Fix PEM-encoded private key newlines.

    .env files store \\n as literal two-character sequences,
    but PEM format requires actual newline characters.

    Args:
        pem_string: PEM string potentially with literal \\n

    Returns:
        PEM string with proper newline characters
    """
    if pem_string and '\\n' in pem_string:
        return pem_string.replace('\\n', '\n')
    return pem_string


class ExchangeDataIngester:
    """
    Fetches market data from cryptocurrency exchanges.

    Supports multiple exchanges via ccxt, with Coinbase as the primary.
    Handles rate limiting, retries, and data validation.

    Attributes:
        exchange: The ccxt exchange instance
        exchange_name: Name of the exchange (e.g., 'coinbase')
        rate_limit_delay: Seconds to wait between API calls

    Example:
        >>> ingester = ExchangeDataIngester()
        >>> df = ingester.fetch_ohlcv_historical('BTC/USD', days=7)
        >>> print(f"Fetched {len(df)} candles")
    """

    # Supported exchanges and their ccxt class names
    EXCHANGE_CLASSES = {
        'coinbase': 'coinbaseadvanced',
        'coinbaseadvanced': 'coinbaseadvanced',
        'binance': 'binance',
    }

    def __init__(
        self,
        exchange_name: str = 'coinbase',
        config: Optional[Dict[str, Any]] = None,
        load_env: bool = True
    ):
        """
        Initialize the exchange connection.

        Args:
            exchange_name: Name of exchange to connect to
            config: Optional config dict with api_key, secret, etc.
            load_env: Whether to load credentials from .env file
        """
        self.exchange_name = exchange_name.lower()
        self.config = config or {}

        # Load environment variables
        if load_env:
            env_path = Path(__file__).parent.parent.parent / '.env'
            load_dotenv(dotenv_path=env_path)

        # Initialize exchange
        self.exchange = self._create_exchange()
        self.rate_limit_delay = 0.2  # 200ms between calls (conservative)

        logger.info(f"Initialized {self.exchange_name} exchange connection")

    def _create_exchange(self) -> ccxt.Exchange:
        """Create and configure the ccxt exchange instance."""
        # Get the correct ccxt class
        ccxt_class_name = self.EXCHANGE_CLASSES.get(
            self.exchange_name,
            self.exchange_name
        )

        if not hasattr(ccxt, ccxt_class_name):
            raise ValueError(f"Unknown exchange: {self.exchange_name}")

        ccxt_class = getattr(ccxt, ccxt_class_name)

        # Get credentials
        api_key = self.config.get('api_key') or self._get_env_credential('API_KEY')
        api_secret = self.config.get('api_secret') or self._get_env_credential('API_SECRET')

        # Fix PEM newlines for Coinbase
        if api_secret and self.exchange_name in ['coinbase', 'coinbaseadvanced']:
            api_secret = fix_pem_newlines(api_secret)

        # Create exchange instance
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        }

        # Add password/passphrase if provided (for exchanges that need it)
        password = self.config.get('password') or self._get_env_credential('PASSPHRASE')
        if password:
            exchange_config['password'] = password

        return ccxt_class(exchange_config)

    def _get_env_credential(self, credential_type: str) -> Optional[str]:
        """Get credential from environment variable."""
        # Try exchange-specific first, then generic
        prefix = self.exchange_name.upper()

        # Try: COINBASE_API_KEY, then EXCHANGE_API_KEY
        value = os.getenv(f'{prefix}_{credential_type}')
        if not value:
            value = os.getenv(f'EXCHANGE_{credential_type}')

        return value

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USD')
            timeframe: Candle interval ('1m', '5m', '1h', '1d', etc.)
            since: Start timestamp in milliseconds (optional)
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with columns: time, open, high, low, close, volume

        Raises:
            ccxt.NetworkError: On network issues (will retry)
            ccxt.ExchangeError: On exchange-specific errors
        """
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )

                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )

                # Convert timestamp to datetime
                df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df = df.drop('timestamp', axis=1)

                # Reorder columns
                df = df[['time', 'open', 'high', 'low', 'close', 'volume']]

                # Add metadata
                df['symbol'] = symbol
                df['exchange'] = self.exchange_name

                logger.debug(f"Fetched {len(df)} candles for {symbol}")
                return df

            except ccxt.NetworkError as e:
                logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit hit, waiting 60s: {e}")
                time.sleep(60)
                # Don't count this as a retry

            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise

    def fetch_ohlcv_historical(
        self,
        symbol: str,
        timeframe: str = '1h',
        days: int = 90,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data by paginating through time.

        Args:
            symbol: Trading pair (e.g., 'BTC/USD')
            timeframe: Candle interval
            days: Number of days of history (ignored if start_date provided)
            start_date: Optional start datetime (UTC)
            end_date: Optional end datetime (UTC), defaults to now
            progress_callback: Optional function(fetched, total) for progress

        Returns:
            DataFrame with all historical candles

        Example:
            >>> df = ingester.fetch_ohlcv_historical('BTC/USD', days=90)
            >>> print(f"Fetched {len(df)} hourly candles")
        """
        # Calculate time range
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Ensure UTC timezone
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")

        # Calculate expected candles for progress tracking
        timeframe_ms = self._timeframe_to_ms(timeframe)
        total_ms = int((end_date - start_date).total_seconds() * 1000)
        expected_candles = total_ms // timeframe_ms

        # Fetch in batches
        all_data = []
        current_time = int(start_date.timestamp() * 1000)
        end_time = int(end_date.timestamp() * 1000)
        batch_size = 300  # Coinbase allows up to 300 per request

        fetched_count = 0

        while current_time < end_time:
            # Fetch batch
            df = self.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_time,
                limit=batch_size
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol} at {current_time}")
                break

            all_data.append(df)
            fetched_count += len(df)

            # Progress callback
            if progress_callback:
                progress_callback(fetched_count, expected_candles)

            # Move to next batch
            last_time = df['time'].max()
            current_time = int(last_time.timestamp() * 1000) + timeframe_ms

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            # Log progress every 500 candles
            if fetched_count % 500 == 0:
                logger.info(f"Fetched {fetched_count} candles for {symbol}...")

        if not all_data:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()

        # Combine all batches
        result = pd.concat(all_data, ignore_index=True)

        # Remove duplicates (can happen at batch boundaries)
        result = result.drop_duplicates(subset=['time', 'symbol'], keep='last')

        # Sort by time
        result = result.sort_values('time').reset_index(drop=True)

        # Filter to exact time range
        result = result[
            (result['time'] >= start_date) &
            (result['time'] <= end_date)
        ]

        logger.info(f"Fetched {len(result)} total candles for {symbol}")
        return result

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
        }

        unit = timeframe[-1]
        value = int(timeframe[:-1])

        if unit not in multipliers:
            raise ValueError(f"Unknown timeframe unit: {unit}")

        return value * multipliers[unit]

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USD')

        Returns:
            Dict with last, bid, ask, volume, etc.
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.debug(f"Ticker {symbol}: {ticker['last']}")
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs."""
        try:
            self.exchange.load_markets()
            return list(self.exchange.markets.keys())
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test the exchange connection.

        Returns:
            True if connection works, False otherwise
        """
        try:
            # Test public endpoint
            ticker = self.fetch_ticker('BTC/USD')
            logger.info(f"Connection test passed. BTC/USD = ${ticker['last']:,.2f}")

            # Test authenticated endpoint
            balance = self.exchange.fetch_balance()
            logger.info("Authenticated connection test passed.")

            return True

        except ccxt.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Convenience function for quick data fetching
def fetch_ohlcv(
    symbol: str,
    exchange: str = 'coinbase',
    timeframe: str = '1h',
    days: int = 7
) -> pd.DataFrame:
    """
    Quick function to fetch OHLCV data.

    Args:
        symbol: Trading pair (e.g., 'BTC/USD')
        exchange: Exchange name
        timeframe: Candle interval
        days: Number of days of history

    Returns:
        DataFrame with OHLCV data

    Example:
        >>> df = fetch_ohlcv('BTC/USD', days=7)
    """
    ingester = ExchangeDataIngester(exchange_name=exchange)
    return ingester.fetch_ohlcv_historical(symbol, timeframe=timeframe, days=days)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing ExchangeDataIngester...")
    ingester = ExchangeDataIngester()

    if ingester.test_connection():
        print("\n✓ Connection successful!")

        # Fetch a small sample
        print("\nFetching 24h of BTC/USD...")
        df = ingester.fetch_ohlcv_historical('BTC/USD', days=1)
        print(f"Fetched {len(df)} candles")
        print(df.head())
    else:
        print("\n✗ Connection failed!")
