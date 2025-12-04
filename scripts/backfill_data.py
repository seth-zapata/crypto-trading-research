#!/usr/bin/env python3
"""
Historical Data Backfill Script

Fetches historical OHLCV data from exchanges and stores in TimescaleDB.
Supports incremental updates - only fetches data newer than what's already stored.

Usage:
    python scripts/backfill_data.py --symbols BTC/USD ETH/USD --days 90
    python scripts/backfill_data.py --symbols BTC/USD --days 365 --timeframe 1h
    python scripts/backfill_data.py --incremental  # Update existing data

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.exchanges import ExchangeDataIngester
from data.storage.timeseries_db import TimeSeriesDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_symbol(
    ingester: ExchangeDataIngester,
    db: TimeSeriesDB,
    symbol: str,
    days: int = 90,
    timeframe: str = '1h',
    incremental: bool = True
) -> int:
    """
    Backfill historical data for a single symbol.

    Args:
        ingester: Exchange data ingester
        db: Database connection
        symbol: Trading pair (e.g., 'BTC/USD')
        days: Number of days to backfill
        timeframe: Candle interval
        incremental: If True, only fetch data newer than existing

    Returns:
        Number of rows stored
    """
    logger.info(f"Starting backfill for {symbol}")

    # Determine time range
    end_date = datetime.now(timezone.utc)

    if incremental:
        # Check for existing data
        latest = db.get_latest_timestamp(symbol)
        if latest:
            # Start from latest + 1 interval
            start_date = latest + timedelta(hours=1)
            logger.info(f"Incremental mode: fetching from {start_date}")
        else:
            start_date = end_date - timedelta(days=days)
            logger.info(f"No existing data, fetching {days} days")
    else:
        start_date = end_date - timedelta(days=days)

    # Check if we need to fetch anything
    if start_date >= end_date:
        logger.info(f"Data is up to date for {symbol}")
        return 0

    # Progress callback
    def progress(fetched: int, total: int) -> None:
        pct = (fetched / total * 100) if total > 0 else 0
        print(f"\r  Progress: {fetched}/{total} candles ({pct:.1f}%)", end='', flush=True)

    # Fetch data
    try:
        df = ingester.fetch_ohlcv_historical(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress
        )
        print()  # Newline after progress

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return 0

    if df.empty:
        logger.warning(f"No data fetched for {symbol}")
        return 0

    # Store in database
    try:
        rows = db.store_ohlcv(df)
        logger.info(f"Stored {rows} rows for {symbol}")
        return rows

    except Exception as e:
        logger.error(f"Error storing {symbol}: {e}")
        return 0


def backfill_all(
    symbols: List[str],
    days: int = 90,
    timeframe: str = '1h',
    exchange: str = 'coinbase',
    incremental: bool = True
) -> dict:
    """
    Backfill data for multiple symbols.

    Args:
        symbols: List of trading pairs
        days: Number of days to backfill
        timeframe: Candle interval
        exchange: Exchange name
        incremental: Use incremental mode

    Returns:
        Dict with results per symbol
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting backfill: {len(symbols)} symbols, {days} days")
    logger.info(f"Exchange: {exchange}, Timeframe: {timeframe}")
    logger.info(f"Mode: {'incremental' if incremental else 'full'}")
    logger.info(f"=" * 60)

    # Initialize connections
    ingester = ExchangeDataIngester(exchange_name=exchange)
    db = TimeSeriesDB()

    # Test connections
    if not ingester.test_connection():
        logger.error("Exchange connection failed!")
        return {}

    health = db.health_check()
    if health['status'] != 'healthy':
        logger.error("Database connection failed!")
        return {}

    results = {}
    total_rows = 0

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}")

        rows = backfill_symbol(
            ingester=ingester,
            db=db,
            symbol=symbol,
            days=days,
            timeframe=timeframe,
            incremental=incremental
        )

        results[symbol] = rows
        total_rows += rows

    # Summary
    logger.info(f"\n" + "=" * 60)
    logger.info(f"Backfill complete!")
    logger.info(f"Total rows stored: {total_rows:,}")
    logger.info(f"Results by symbol:")
    for symbol, rows in results.items():
        logger.info(f"  {symbol}: {rows:,} rows")

    # Show database stats
    stats = db.get_symbol_stats()
    if not stats.empty:
        logger.info(f"\nDatabase status:")
        for _, row in stats.iterrows():
            logger.info(f"  {row['symbol']}: {row['row_count']:,} rows "
                       f"({row['first_time']} to {row['last_time']})")

    db.close()
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Backfill historical OHLCV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill 90 days for BTC and ETH
  python scripts/backfill_data.py --symbols BTC/USD ETH/USD --days 90

  # Backfill 1 year of data
  python scripts/backfill_data.py --symbols BTC/USD --days 365

  # Update existing data (incremental)
  python scripts/backfill_data.py --incremental

  # Full refresh (ignore existing data)
  python scripts/backfill_data.py --symbols BTC/USD --days 90 --full
        """
    )

    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['BTC/USD', 'ETH/USD'],
        help='Trading pairs to backfill (default: BTC/USD ETH/USD)'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=90,
        help='Number of days to backfill (default: 90)'
    )

    parser.add_argument(
        '--timeframe', '-t',
        default='1h',
        help='Candle interval (default: 1h)'
    )

    parser.add_argument(
        '--exchange', '-e',
        default='coinbase',
        help='Exchange to fetch from (default: coinbase)'
    )

    parser.add_argument(
        '--incremental', '-i',
        action='store_true',
        help='Only fetch data newer than existing (default behavior)'
    )

    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Full refresh - ignore existing data'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine incremental mode
    incremental = not args.full

    # Run backfill
    try:
        results = backfill_all(
            symbols=args.symbols,
            days=args.days,
            timeframe=args.timeframe,
            exchange=args.exchange,
            incremental=incremental
        )

        # Exit with error if no data was fetched
        if not any(results.values()):
            logger.warning("No new data was fetched")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nBackfill interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
