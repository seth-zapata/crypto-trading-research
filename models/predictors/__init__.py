"""
Predictors Module

Machine learning models for price prediction and signal generation.
"""

from models.predictors.lightgbm_model import (
    LightGBMPredictor,
    create_target_variable,
    prepare_features_and_target
)

from models.predictors.walk_forward import (
    WalkForwardValidator,
    WalkForwardResult,
    WalkForwardSplit,
    time_series_train_test_split
)

from models.predictors.regime_classifier import (
    RegimeClassifier,
    RegimeClassification,
    MarketRegime,
    RegimeHistory,
)

from models.predictors.alpha_combiner import (
    AlphaCombiner,
    AlphaSignal,
    CombinedSignal,
    create_alpha_pipeline,
)

__all__ = [
    # Phase 2 - LightGBM
    'LightGBMPredictor',
    'create_target_variable',
    'prepare_features_and_target',
    'WalkForwardValidator',
    'WalkForwardResult',
    'WalkForwardSplit',
    'time_series_train_test_split',
    # Phase 3 - Regime & Alpha
    'RegimeClassifier',
    'RegimeClassification',
    'MarketRegime',
    'RegimeHistory',
    'AlphaCombiner',
    'AlphaSignal',
    'CombinedSignal',
    'create_alpha_pipeline',
]
