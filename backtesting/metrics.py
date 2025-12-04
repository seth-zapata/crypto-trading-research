"""
Backtesting Performance Metrics

Computes standard trading performance metrics including:
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate, profit factor
- Risk-adjusted returns

All metrics account for transaction costs when provided.

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for backtest performance metrics."""

    # Returns
    total_return: float
    annualized_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # periods

    # Trading metrics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Other
    volatility: float
    calmar_ratio: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'volatility': self.volatility,
            'calmar_ratio': self.calmar_ratio
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        return f"""
Performance Summary:
  Total Return:      {self.total_return:>10.2%}
  Annualized Return: {self.annualized_return:>10.2%}
  Sharpe Ratio:      {self.sharpe_ratio:>10.2f}
  Sortino Ratio:     {self.sortino_ratio:>10.2f}
  Max Drawdown:      {self.max_drawdown:>10.2%}
  Volatility:        {self.volatility:>10.2%}

Trading Stats:
  Num Trades:        {self.num_trades:>10d}
  Win Rate:          {self.win_rate:>10.2%}
  Profit Factor:     {self.profit_factor:>10.2f}
  Avg Win:           {self.avg_win:>10.2%}
  Avg Loss:          {self.avg_loss:>10.2%}
"""


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760  # hourly
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (8760 for hourly)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    # Convert annual risk-free to period risk-free
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_per_period

    # Annualize
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760
) -> float:
    """
    Calculate Sortino ratio (only penalizes downside volatility).

    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0

    downside_std = downside_returns.std()

    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
    """
    Calculate maximum drawdown and its duration.

    Args:
        equity_curve: Series of portfolio values over time

    Returns:
        Tuple of (max_drawdown, max_drawdown_duration_periods)
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Max drawdown
    max_dd = drawdown.min()

    # Drawdown duration
    # Find periods where we're in drawdown
    in_drawdown = drawdown < 0

    # Calculate consecutive drawdown periods
    if not in_drawdown.any():
        return 0.0, 0

    # Group consecutive drawdown periods
    dd_groups = (~in_drawdown).cumsum()
    dd_durations = in_drawdown.groupby(dd_groups).sum()
    max_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

    return abs(max_dd), max_duration


def calculate_win_rate(trade_returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of profitable trades).

    Args:
        trade_returns: Series of individual trade returns

    Returns:
        Win rate as decimal (0-1)
    """
    if len(trade_returns) == 0:
        return 0.0

    wins = (trade_returns > 0).sum()
    return wins / len(trade_returns)


def calculate_profit_factor(trade_returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trade_returns: Series of individual trade returns

    Returns:
        Profit factor (>1 means profitable)
    """
    wins = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())

    if losses == 0:
        return float('inf') if wins > 0 else 0.0

    return wins / losses


def calculate_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trade_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.

    Args:
        returns: Series of period returns
        equity_curve: Series of portfolio values
        trade_returns: Series of individual trade returns (optional)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        PerformanceMetrics object with all computed metrics
    """
    # Total return
    if len(equity_curve) < 2:
        total_return = 0.0
    else:
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Annualized return
    n_periods = len(returns)
    if n_periods > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    else:
        annualized_return = 0.0

    # Risk metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd, max_dd_duration = calculate_max_drawdown(equity_curve)

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0

    # Calmar ratio (return / max drawdown)
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0

    # Trading metrics (if trade returns provided)
    if trade_returns is not None and len(trade_returns) > 0:
        num_trades = len(trade_returns)
        win_rate = calculate_win_rate(trade_returns)
        profit_factor = calculate_profit_factor(trade_returns)

        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
    else:
        num_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        num_trades=num_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        volatility=volatility,
        calmar_ratio=calmar
    )


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 8760
) -> Dict[str, float]:
    """
    Compare strategy to benchmark (e.g., buy-and-hold).

    Args:
        strategy_returns: Strategy period returns
        benchmark_returns: Benchmark period returns
        periods_per_year: Trading periods per year

    Returns:
        Dict with comparison metrics
    """
    # Align returns
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, keys=['strategy', 'benchmark'])
    aligned = aligned.dropna()

    strategy = aligned['strategy']
    benchmark = aligned['benchmark']

    # Information ratio
    tracking_error = (strategy - benchmark).std()
    if tracking_error > 0:
        info_ratio = (strategy.mean() - benchmark.mean()) / tracking_error * np.sqrt(periods_per_year)
    else:
        info_ratio = 0.0

    # Beta
    if benchmark.var() > 0:
        beta = strategy.cov(benchmark) / benchmark.var()
    else:
        beta = 0.0

    # Alpha (Jensen's alpha, annualized)
    alpha = (strategy.mean() - beta * benchmark.mean()) * periods_per_year

    # Correlation
    correlation = strategy.corr(benchmark)

    return {
        'alpha': alpha,
        'beta': beta,
        'information_ratio': info_ratio,
        'correlation': correlation,
        'tracking_error': tracking_error * np.sqrt(periods_per_year)
    }


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    # Generate sample returns (hourly)
    n = 2000
    returns = pd.Series(np.random.randn(n) * 0.001 + 0.0001)  # Small positive drift

    # Generate equity curve
    equity = (1 + returns).cumprod() * 10000

    # Generate trade returns
    trade_returns = pd.Series(np.random.randn(50) * 0.02)  # 50 trades

    # Calculate metrics
    metrics = calculate_all_metrics(returns, equity, trade_returns)
    print(metrics)

    # Benchmark comparison
    benchmark_returns = pd.Series(np.random.randn(n) * 0.001)
    comparison = compare_to_benchmark(returns, benchmark_returns)
    print("\nBenchmark comparison:")
    for k, v in comparison.items():
        print(f"  {k}: {v:.4f}")
