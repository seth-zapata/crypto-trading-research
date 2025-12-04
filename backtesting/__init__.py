"""
Backtesting Module

Provides backtesting engine and performance metrics for evaluating
trading strategies with realistic transaction costs.
"""

from backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
    run_buy_and_hold
)

from backtesting.metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    compare_to_benchmark
)

__all__ = [
    'BacktestConfig',
    'BacktestEngine',
    'BacktestResult',
    'Trade',
    'run_buy_and_hold',
    'PerformanceMetrics',
    'calculate_all_metrics',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'compare_to_benchmark'
]
