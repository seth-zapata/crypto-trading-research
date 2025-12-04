"""
Walk-Forward Validation

Implements time-series cross-validation that respects temporal ordering.
Critical for avoiding look-ahead bias in trading models.

Walk-forward process:
1. Train on historical window
2. Predict on next period(s)
3. Slide window forward
4. Repeat

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """Single train/test split for walk-forward validation."""
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""
    fold: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    probabilities: Optional[np.ndarray] = None
    feature_importance: Optional[pd.DataFrame] = None


class WalkForwardValidator:
    """
    Walk-forward (rolling window) cross-validation.

    Unlike standard k-fold CV, walk-forward validation:
    - Respects temporal ordering (no future data leakage)
    - Uses expanding or sliding training window
    - Tests on out-of-sample future data

    Example:
        >>> validator = WalkForwardValidator(
        ...     n_splits=5,
        ...     train_size=500,
        ...     test_size=100
        ... )
        >>> results = validator.validate(model, X, y, feature_names)
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: int = 100,
        expanding: bool = False,
        gap: int = 0
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of train/test splits
            train_size: Training window size (None = use all available)
            test_size: Test window size
            expanding: If True, training window grows; if False, slides
            gap: Gap between train and test (to avoid look-ahead)
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.expanding = expanding
        self.gap = gap

        logger.info(
            f"WalkForwardValidator: {n_splits} splits, "
            f"train_size={train_size}, test_size={test_size}, "
            f"expanding={expanding}"
        )

    def get_splits(self, n_samples: int) -> List[WalkForwardSplit]:
        """
        Generate train/test split indices.

        Args:
            n_samples: Total number of samples

        Returns:
            List of WalkForwardSplit objects
        """
        splits = []

        # Calculate minimum required data
        min_train = self.train_size or (n_samples // (self.n_splits + 1))
        total_test = self.n_splits * self.test_size

        if min_train + total_test + self.gap > n_samples:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits. "
                f"Need {min_train + total_test + self.gap}, have {n_samples}"
            )

        # Generate splits
        for i in range(self.n_splits):
            if self.expanding:
                # Expanding window: train from start
                train_start = 0
                train_end = min_train + i * self.test_size
            else:
                # Sliding window: fixed train size
                train_start = i * self.test_size
                train_end = train_start + (self.train_size or min_train)

            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            # Ensure we don't exceed data
            if test_end > n_samples:
                test_end = n_samples
                if test_end - test_start < 1:
                    break

            splits.append(WalkForwardSplit(
                fold=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            ))

        return splits

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices (sklearn-compatible interface).

        Args:
            X: Features (used for length only)
            y: Labels (unused, for API compatibility)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        splits = self.get_splits(n_samples)

        for split in splits:
            train_idx = np.arange(split.train_start, split.train_end)
            test_idx = np.arange(split.test_start, split.test_end)
            yield train_idx, test_idx

    def validate(
        self,
        model_class: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model_params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None
    ) -> List[WalkForwardResult]:
        """
        Run full walk-forward validation.

        Args:
            model_class: Model class with fit/predict/predict_proba methods
            X: Features
            y: Labels
            model_params: Parameters for model initialization
            feature_names: Feature names for importance tracking

        Returns:
            List of WalkForwardResult for each fold
        """
        model_params = model_params or {}
        results = []

        # Convert to numpy if needed
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        if feature_names is None and isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)

        splits = self.get_splits(len(X_arr))

        for split in splits:
            logger.info(
                f"Fold {split.fold + 1}/{len(splits)}: "
                f"train[{split.train_start}:{split.train_end}] "
                f"test[{split.test_start}:{split.test_end}]"
            )

            # Get train/test data
            X_train = X_arr[split.train_start:split.train_end]
            y_train = y_arr[split.train_start:split.train_end]
            X_test = X_arr[split.test_start:split.test_end]
            y_test = y_arr[split.test_start:split.test_end]

            # Initialize and train model
            model = model_class(**model_params)

            # Try to use validation set for early stopping
            if hasattr(model, 'fit') and 'X_val' in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
            else:
                model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Get probabilities if available
            probas = None
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_test)

            # Evaluate
            train_metrics = model.evaluate(X_train, y_train) if hasattr(model, 'evaluate') else {}
            test_metrics = model.evaluate(X_test, y_test) if hasattr(model, 'evaluate') else {}

            # Feature importance
            importance = None
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()

            results.append(WalkForwardResult(
                fold=split.fold,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                predictions=y_pred,
                actuals=y_test,
                probabilities=probas,
                feature_importance=importance
            ))

            logger.info(f"  Train acc: {train_metrics.get('accuracy', 0):.4f}, "
                       f"Test acc: {test_metrics.get('accuracy', 0):.4f}")

        return results

    def aggregate_results(
        self,
        results: List[WalkForwardResult]
    ) -> Dict[str, Any]:
        """
        Aggregate results across all folds.

        Args:
            results: List of fold results

        Returns:
            Dict with mean/std metrics and combined predictions
        """
        # Collect metrics
        test_metrics = {}
        for key in results[0].test_metrics.keys():
            values = [r.test_metrics[key] for r in results]
            test_metrics[f'{key}_mean'] = np.mean(values)
            test_metrics[f'{key}_std'] = np.std(values)

        # Combine predictions
        all_predictions = np.concatenate([r.predictions for r in results])
        all_actuals = np.concatenate([r.actuals for r in results])

        # Combined accuracy
        overall_accuracy = (all_predictions == all_actuals).mean()

        # Aggregate feature importance
        importance_dfs = [r.feature_importance for r in results if r.feature_importance is not None]
        if importance_dfs:
            combined_importance = pd.concat(importance_dfs)
            avg_importance = combined_importance.groupby('feature')['importance'].mean()
            avg_importance = avg_importance.sort_values(ascending=False).reset_index()
            avg_importance.columns = ['feature', 'importance']
        else:
            avg_importance = None

        return {
            'n_folds': len(results),
            'test_metrics': test_metrics,
            'overall_accuracy': overall_accuracy,
            'all_predictions': all_predictions,
            'all_actuals': all_actuals,
            'feature_importance': avg_importance
        }


def time_series_train_test_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple train/test split respecting time order.

    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y

    return (
        X_arr[:split_idx],
        X_arr[split_idx:],
        y_arr[:split_idx],
        y_arr[split_idx:]
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    import sys
    sys.path.insert(0, '.')

    from models.predictors.lightgbm_model import LightGBMPredictor

    np.random.seed(42)

    # Generate sample data
    n = 1000
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    # Add some signal
    y = ((X['feature_0'] + X['feature_1'] + np.random.randn(n) * 0.5) > 0).astype(int)

    # Walk-forward validation
    validator = WalkForwardValidator(
        n_splits=5,
        train_size=500,
        test_size=100,
        expanding=False
    )

    print("Running walk-forward validation...")
    results = validator.validate(
        LightGBMPredictor,
        X, y,
        model_params={'num_boost_round': 50}
    )

    # Aggregate
    summary = validator.aggregate_results(results)

    print(f"\n=== Walk-Forward Results ({summary['n_folds']} folds) ===")
    print(f"Overall accuracy: {summary['overall_accuracy']:.4f}")
    print(f"\nTest metrics:")
    for k, v in summary['test_metrics'].items():
        print(f"  {k}: {v:.4f}")

    if summary['feature_importance'] is not None:
        print(f"\nTop 5 features:")
        print(summary['feature_importance'].head())

    print("\n✓ Walk-forward validation test complete!")
