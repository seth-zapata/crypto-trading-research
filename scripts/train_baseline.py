#!/usr/bin/env python3
"""
Baseline Model Training Script

Trains LightGBM model on historical OHLCV data with walk-forward validation.
Runs backtest to evaluate trading performance.

Usage:
    python scripts/train_baseline.py --symbol BTC/USD --days 90
    python scripts/train_baseline.py --symbol ETH/USD --days 60 --n-splits 10

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.storage.timeseries_db import TimeSeriesDB
from data.processing.features import FeatureEngineer
from models.predictors.lightgbm_model import LightGBMPredictor, create_target_variable
from models.predictors.walk_forward import WalkForwardValidator
from backtesting.engine import BacktestEngine, BacktestConfig, run_buy_and_hold
from backtesting.metrics import compare_to_benchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    symbol: str,
    days: int = 90
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load OHLCV data and prepare features/target.

    Args:
        symbol: Trading pair
        days: Days of historical data

    Returns:
        Tuple of (features_df, X, y)
    """
    logger.info(f"Loading {days} days of {symbol} data...")

    # Load from database
    db = TimeSeriesDB()
    ohlcv = db.fetch_ohlcv(symbol, days=days)
    db.close()

    if ohlcv.empty:
        raise ValueError(f"No data found for {symbol}")

    logger.info(f"Loaded {len(ohlcv)} rows")

    # Generate features
    engineer = FeatureEngineer(warmup_periods=50)
    features_df = engineer.generate_all_features(ohlcv)

    # Create target (1 if next period return > 0, else 0)
    features_df['target'] = create_target_variable(features_df, 'close', horizon=1)

    # Remove warmup and NaN
    features_df = engineer.remove_warmup(features_df)
    features_df = features_df.dropna()

    logger.info(f"After preprocessing: {len(features_df)} rows")

    # Feature columns (exclude metadata and target)
    exclude_cols = ['time', 'symbol', 'exchange', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]

    X = features_df[feature_cols]
    y = features_df['target']

    logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return features_df, X, y


def train_and_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    train_size: int = 1000,
    test_size: int = 200
) -> Dict:
    """
    Train model with walk-forward validation.

    Args:
        X: Features
        y: Target
        n_splits: Number of CV splits
        train_size: Training window size
        test_size: Test window size

    Returns:
        Dict with validation results
    """
    logger.info(f"Running walk-forward validation with {n_splits} splits...")

    # Adjust sizes based on data
    n_samples = len(X)
    if train_size + (n_splits * test_size) > n_samples:
        # Reduce sizes to fit
        test_size = max(50, (n_samples - train_size) // (n_splits + 1))
        train_size = min(train_size, n_samples - n_splits * test_size - 50)
        logger.warning(f"Adjusted sizes: train={train_size}, test={test_size}")

    validator = WalkForwardValidator(
        n_splits=n_splits,
        train_size=train_size,
        test_size=test_size,
        expanding=False,
        gap=1  # 1 period gap to avoid look-ahead
    )

    # Model parameters
    model_params = {
        'num_boost_round': 100,
        'early_stopping_rounds': 10,
        'params': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'min_child_samples': 20
        }
    }

    # Run validation
    results = validator.validate(
        LightGBMPredictor,
        X, y,
        model_params=model_params
    )

    # Aggregate results
    summary = validator.aggregate_results(results)

    return summary


def run_backtest_on_predictions(
    features_df: pd.DataFrame,
    predictions: np.ndarray,
    actuals: np.ndarray,
    test_indices: np.ndarray,
    initial_capital: float = 10000.0
) -> Dict:
    """
    Run backtest using model predictions as signals.

    Args:
        features_df: Original features DataFrame
        predictions: Model predictions (0/1)
        actuals: Actual target values
        test_indices: Indices corresponding to predictions
        initial_capital: Starting capital

    Returns:
        Dict with backtest results
    """
    logger.info("Running backtest on predictions...")

    # Get prices for test period
    test_data = features_df.iloc[test_indices].copy()
    prices = test_data['close']

    # Convert predictions to signals (1 = long, 0 = flat)
    signals = pd.Series(predictions, index=prices.index)

    # Configure backtest
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=0.001,  # 0.1% Coinbase maker fee
        slippage_rate=0.0005,   # 0.05% slippage
        position_size=1.0,
        allow_short=False
    )

    # Run strategy backtest
    engine = BacktestEngine(config)
    strategy_result = engine.run(prices, signals)

    # Run buy-and-hold benchmark
    bh_result = run_buy_and_hold(prices, initial_capital)

    # Compare to benchmark
    comparison = compare_to_benchmark(
        strategy_result.returns,
        bh_result.returns
    )

    return {
        'strategy': strategy_result,
        'benchmark': bh_result,
        'comparison': comparison
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train baseline LightGBM model with walk-forward validation'
    )

    parser.add_argument(
        '--symbol', '-s',
        default='BTC/USD',
        help='Trading pair (default: BTC/USD)'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=90,
        help='Days of historical data (default: 90)'
    )

    parser.add_argument(
        '--n-splits', '-n',
        type=int,
        default=5,
        help='Number of walk-forward splits (default: 5)'
    )

    parser.add_argument(
        '--train-size',
        type=int,
        default=1000,
        help='Training window size (default: 1000)'
    )

    parser.add_argument(
        '--test-size',
        type=int,
        default=200,
        help='Test window size (default: 200)'
    )

    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained model to models/saved/'
    )

    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Data: {args.days} days")
    print(f"Walk-forward: {args.n_splits} splits")
    print("=" * 60)

    try:
        # Load data
        features_df, X, y = load_and_prepare_data(args.symbol, args.days)

        # Train and validate
        validation_results = train_and_validate(
            X, y,
            n_splits=args.n_splits,
            train_size=args.train_size,
            test_size=args.test_size
        )

        # Print validation results
        print("\n" + "=" * 60)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("=" * 60)
        print(f"Number of folds: {validation_results['n_folds']}")
        print(f"\nOverall accuracy: {validation_results['overall_accuracy']:.4f}")
        print("\nTest metrics (mean ± std):")
        for key, value in validation_results['test_metrics'].items():
            if key.endswith('_mean'):
                metric_name = key.replace('_mean', '')
                std_key = f'{metric_name}_std'
                std_value = validation_results['test_metrics'].get(std_key, 0)
                print(f"  {metric_name}: {value:.4f} ± {std_value:.4f}")

        # Feature importance
        if validation_results['feature_importance'] is not None:
            print("\nTop 10 features:")
            print(validation_results['feature_importance'].head(10).to_string(index=False))

        # Get test indices for backtest
        # Use the last test_size * n_splits samples
        n_test_samples = len(validation_results['all_predictions'])
        test_start_idx = len(features_df) - n_test_samples
        test_indices = np.arange(test_start_idx, len(features_df))

        # Run backtest
        backtest_results = run_backtest_on_predictions(
            features_df,
            validation_results['all_predictions'],
            validation_results['all_actuals'],
            test_indices
        )

        # Print backtest results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        strategy = backtest_results['strategy']
        benchmark = backtest_results['benchmark']

        print("\nStrategy Performance:")
        print(strategy.metrics)

        print("\nBenchmark (Buy & Hold):")
        print(f"  Total Return: {benchmark.metrics.total_return:.2%}")
        print(f"  Sharpe Ratio: {benchmark.metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {benchmark.metrics.max_drawdown:.2%}")

        print("\nStrategy vs Benchmark:")
        for key, value in backtest_results['comparison'].items():
            print(f"  {key}: {value:.4f}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        sharpe = strategy.metrics.sharpe_ratio
        win_rate = validation_results['overall_accuracy']

        print(f"Model Accuracy: {win_rate:.2%}")
        print(f"Strategy Sharpe: {sharpe:.2f}")
        print(f"Strategy Return: {strategy.metrics.total_return:.2%}")
        print(f"Max Drawdown: {strategy.metrics.max_drawdown:.2%}")
        print(f"Number of Trades: {len(strategy.trades)}")

        # Pass/fail criteria
        print("\n" + "=" * 60)
        print("VALIDATION CRITERIA")
        print("=" * 60)

        criteria = {
            'Win rate > 50%': win_rate > 0.5,
            'Sharpe ratio > 0': sharpe > 0,
            'Not always-up bias': validation_results['all_predictions'].mean() < 0.95
        }

        all_passed = True
        for criterion, passed in criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {criterion}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n✓ All criteria passed!")
        else:
            print("\n⚠ Some criteria failed - model may need tuning")

        # Save model if requested
        if args.save_model:
            # Train final model on all data
            final_model = LightGBMPredictor(num_boost_round=100)
            final_model.fit(X, y)

            model_path = Path('models/saved') / f'lightgbm_{args.symbol.replace("/", "_")}_{datetime.now().strftime("%Y%m%d")}.pkl'
            final_model.save(model_path)
            print(f"\nModel saved to: {model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
