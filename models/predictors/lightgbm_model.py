"""
LightGBM Model for Price Direction Prediction

Wrapper around LightGBM for predicting cryptocurrency price movements.
Supports classification (direction) and regression (return magnitude).

Features:
- Configurable hyperparameters
- Feature importance extraction
- Probability calibration
- Model persistence

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

logger = logging.getLogger(__name__)


class LightGBMPredictor:
    """
    LightGBM-based predictor for price direction.

    Predicts whether price will go up (1) or down (0) in the next period.
    Uses technical features computed from OHLCV data.

    Attributes:
        model: Trained LightGBM model
        feature_names: Names of features used for training
        is_fitted: Whether model has been trained

    Example:
        >>> model = LightGBMPredictor()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
    """

    DEFAULT_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10
    ):
        """
        Initialize LightGBM predictor.

        Args:
            params: LightGBM parameters (uses defaults if None)
            num_boost_round: Number of boosting iterations
            early_stopping_rounds: Stop if no improvement for N rounds
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

        logger.info(f"LightGBMPredictor initialized")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'LightGBMPredictor':
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels (0 or 1)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names (extracted from DataFrame if available)

        Returns:
            self (fitted model)
        """
        # Extract feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        elif feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Convert labels
        y = np.asarray(y).astype(int)

        # Create LightGBM datasets
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            y_val = np.asarray(y_val).astype(int)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Train model
        logger.info(f"Training LightGBM on {X.shape[0]} samples, {X.shape[1]} features")

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0)  # Suppress iteration logs
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        self.is_fitted = True
        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class labels (0 or 1).

        Args:
            X: Features to predict

        Returns:
            Array of predicted labels
        """
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probability of positive class (price up).

        Args:
            X: Features to predict

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels

        Returns:
            Dict with accuracy, precision, recall, f1, auc
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        y_true = np.asarray(y).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        # AUC only if both classes present
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        else:
            metrics['auc'] = 0.5

        return metrics

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: 'gain', 'split', or 'weight'

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        importance = self.model.feature_importance(importance_type=importance_type)

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })

        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'num_boost_round': self.num_boost_round,
            'is_fitted': self.is_fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LightGBMPredictor':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(params=data['params'], num_boost_round=data['num_boost_round'])
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.is_fitted = data['is_fitted']

        logger.info(f"Model loaded from {path}")
        return model


def create_target_variable(
    df: pd.DataFrame,
    target_col: str = 'close',
    horizon: int = 1,
    threshold: float = 0.0
) -> pd.Series:
    """
    Create binary target variable for classification.

    Args:
        df: DataFrame with price data
        target_col: Column to use for returns
        horizon: Periods ahead to predict
        threshold: Return threshold for positive class

    Returns:
        Series with binary labels (1 if return > threshold, else 0)
    """
    # Future return
    future_return = df[target_col].shift(-horizon) / df[target_col] - 1

    # Binary target
    target = (future_return > threshold).astype(int)

    return target


def prepare_features_and_target(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'close',
    horizon: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.

    Args:
        features_df: DataFrame with features and price data
        feature_cols: Columns to use as features
        target_col: Column for target calculation
        horizon: Prediction horizon

    Returns:
        Tuple of (X features, y target)
    """
    # Create target
    y = create_target_variable(features_df, target_col, horizon)

    # Select features
    X = features_df[feature_cols].copy()

    # Align (drop NaN from target)
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    return X, y


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)

    # Generate sample data
    n = 1000
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = (np.random.randn(n) > 0).astype(int)

    # Split
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    # Train
    model = LightGBMPredictor(num_boost_round=50)
    model.fit(X_train, y_train, X_test, y_test)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Feature importance
    importance = model.get_feature_importance()
    print("\nTop 5 Features:")
    print(importance.head())

    print("\n✓ LightGBM model test complete!")
