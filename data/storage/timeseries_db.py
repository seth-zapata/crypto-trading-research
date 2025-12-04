"""
TimescaleDB Storage Interface

Handles storing and retrieving time-series data (OHLCV, features, signals)
from TimescaleDB. Provides efficient bulk inserts and time-range queries.

Features:
- Connection pooling for performance
- Upsert support (insert or update on conflict)
- Batch operations for bulk inserts
- Graceful error handling

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

import pandas as pd
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)


class TimeSeriesDB:
    """
    Interface for TimescaleDB time-series storage.

    Manages connection pooling and provides methods for storing
    and retrieving OHLCV data, features, and signals.

    Attributes:
        connection_pool: PostgreSQL connection pool
        min_connections: Minimum pool connections
        max_connections: Maximum pool connections

    Example:
        >>> db = TimeSeriesDB()
        >>> db.store_ohlcv(df)
        >>> data = db.fetch_ohlcv('BTC/USD', days=7)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10
    ):
        """
        Initialize database connection pool.

        Args:
            host: Database host (default: from env or localhost)
            port: Database port (default: from env or 5432)
            database: Database name (default: from env or crypto_trading)
            user: Database user (default: from env or crypto_user)
            password: Database password (default: from env)
            min_connections: Minimum pool size
            max_connections: Maximum pool size
        """
        # Load environment
        env_path = Path(__file__).parent.parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)

        # Get connection parameters
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', '5432'))
        self.database = database or os.getenv('DB_NAME', 'crypto_trading')
        self.user = user or os.getenv('DB_USER', 'crypto_user')
        self.password = password or os.getenv('DB_PASSWORD', 'borger123!@#')

        self.min_connections = min_connections
        self.max_connections = max_connections

        # Create connection pool
        self._create_pool()

        logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}/{self.database}")

    def _create_pool(self) -> None:
        """Create the connection pool."""
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
        except psycopg2.Error as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.

        Usage:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT ...")
        """
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def close(self) -> None:
        """Close all connections in the pool."""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")

    # =========================================================================
    # OHLCV Operations
    # =========================================================================

    def store_ohlcv(
        self,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> int:
        """
        Store OHLCV data with upsert (insert or update on conflict).

        Args:
            df: DataFrame with columns: time, symbol, exchange, open, high, low, close, volume
            batch_size: Number of rows per batch insert

        Returns:
            Number of rows inserted/updated

        Raises:
            ValueError: If required columns are missing
            psycopg2.Error: On database errors
        """
        required_cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Add exchange if not present
        if 'exchange' not in df.columns:
            df = df.copy()
            df['exchange'] = 'coinbase'

        # Ensure time is timezone-aware UTC
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')

        # Prepare data for insert
        columns = ['time', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume']
        data = df[columns].values.tolist()

        # Upsert query
        query = """
            INSERT INTO ohlcv (time, symbol, exchange, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (time, symbol, exchange)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """

        total_rows = 0
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Insert in batches
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    execute_values(cur, query, batch)
                    total_rows += len(batch)

            conn.commit()

        logger.info(f"Stored {total_rows} OHLCV rows")
        return total_rows

    def fetch_ohlcv(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None,
        exchange: str = 'coinbase'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USD')
            start: Start datetime (UTC)
            end: End datetime (UTC), defaults to now
            days: Alternative to start - fetch last N days
            exchange: Exchange name filter

        Returns:
            DataFrame with OHLCV data, sorted by time ascending

        Example:
            >>> df = db.fetch_ohlcv('BTC/USD', days=7)
            >>> df = db.fetch_ohlcv('ETH/USD', start=start_dt, end=end_dt)
        """
        # Calculate time range
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            if days:
                start = end - pd.Timedelta(days=days)
            else:
                start = datetime(2020, 1, 1, tzinfo=timezone.utc)

        query = """
            SELECT time, symbol, exchange, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s
              AND exchange = %s
              AND time >= %s
              AND time <= %s
            ORDER BY time ASC
        """

        with self.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol, exchange, start, end)
            )

        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], utc=True)

        logger.debug(f"Fetched {len(df)} OHLCV rows for {symbol}")
        return df

    def get_latest_timestamp(
        self,
        symbol: str,
        exchange: str = 'coinbase'
    ) -> Optional[datetime]:
        """
        Get the latest timestamp for a symbol.

        Useful for incremental data fetching.

        Args:
            symbol: Trading pair
            exchange: Exchange name

        Returns:
            Latest timestamp or None if no data
        """
        query = """
            SELECT MAX(time) as latest
            FROM ohlcv
            WHERE symbol = %s AND exchange = %s
        """

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (symbol, exchange))
                result = cur.fetchone()

        if result and result[0]:
            return result[0].replace(tzinfo=timezone.utc)
        return None

    def get_symbol_stats(self, exchange: str = 'coinbase') -> pd.DataFrame:
        """
        Get statistics for all symbols in the database.

        Returns:
            DataFrame with symbol, row_count, min_time, max_time
        """
        query = """
            SELECT
                symbol,
                COUNT(*) as row_count,
                MIN(time) as first_time,
                MAX(time) as last_time
            FROM ohlcv
            WHERE exchange = %s
            GROUP BY symbol
            ORDER BY symbol
        """

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(exchange,))

        return df

    # =========================================================================
    # Features Operations
    # =========================================================================

    def store_features(
        self,
        df: pd.DataFrame,
        feature_version: str = 'v1',
        batch_size: int = 1000
    ) -> int:
        """
        Store computed features.

        Args:
            df: DataFrame with time, symbol, and feature columns
            feature_version: Version tag for these features
            batch_size: Rows per batch

        Returns:
            Number of rows stored
        """
        if df.empty:
            return 0

        # Get feature columns (exclude time, symbol, exchange)
        meta_cols = ['time', 'symbol', 'exchange', 'feature_version']
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # Prepare for insert
        df = df.copy()
        df['feature_version'] = feature_version

        # Build dynamic insert query
        all_cols = ['time', 'symbol', 'feature_version'] + feature_cols
        col_names = ', '.join(all_cols)
        placeholders = ', '.join(['%s'] * len(all_cols))

        query = f"""
            INSERT INTO features ({col_names})
            VALUES ({placeholders})
            ON CONFLICT (time, symbol, feature_version)
            DO UPDATE SET {', '.join(f'{c} = EXCLUDED.{c}' for c in feature_cols)}
        """

        # Only include columns that exist in the dataframe
        available_cols = [c for c in all_cols if c in df.columns]
        data = df[available_cols].values.tolist()

        total_rows = 0
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    cur.executemany(query, batch)
                    total_rows += len(batch)
            conn.commit()

        logger.info(f"Stored {total_rows} feature rows")
        return total_rows

    def fetch_features(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        days: Optional[int] = None,
        feature_version: str = 'v1'
    ) -> pd.DataFrame:
        """
        Fetch features for a symbol.

        Args:
            symbol: Trading pair
            start: Start datetime
            end: End datetime
            days: Alternative - fetch last N days
            feature_version: Version to fetch

        Returns:
            DataFrame with features
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None and days:
            start = end - pd.Timedelta(days=days)

        query = """
            SELECT *
            FROM features
            WHERE symbol = %s
              AND feature_version = %s
              AND time >= %s
              AND time <= %s
            ORDER BY time ASC
        """

        with self.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol, feature_version, start, end)
            )

        return df

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute a raw SQL query and return results."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:  # SELECT query
                    return cur.fetchall()
                conn.commit()
                return []

    def get_table_sizes(self) -> pd.DataFrame:
        """Get sizes of all tables."""
        query = """
            SELECT
                tablename as table_name,
                pg_size_pretty(pg_total_relation_size(quote_ident(tablename))) as size
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(quote_ident(tablename)) DESC
        """

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.

        Returns:
            Dict with connection status, table counts, etc.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check connection
                    cur.execute("SELECT 1")

                    # Get row counts
                    cur.execute("SELECT COUNT(*) FROM ohlcv")
                    ohlcv_count = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(*) FROM features")
                    features_count = cur.fetchone()[0]

            return {
                'status': 'healthy',
                'connected': True,
                'ohlcv_rows': ohlcv_count,
                'features_rows': features_count,
                'host': self.host,
                'database': self.database
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e)
            }


# Convenience function
def get_db() -> TimeSeriesDB:
    """Get a TimeSeriesDB instance with default configuration."""
    return TimeSeriesDB()


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing TimeSeriesDB...")
    db = TimeSeriesDB()

    # Health check
    health = db.health_check()
    print(f"\nHealth check: {health}")

    # Test store and fetch
    if health['status'] == 'healthy':
        print("\n✓ Database connection successful!")

        # Show table stats
        stats = db.get_symbol_stats()
        if not stats.empty:
            print("\nSymbol stats:")
            print(stats)
        else:
            print("\nNo data in database yet.")

    db.close()
