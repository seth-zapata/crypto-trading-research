"""
Backtesting Engine

Event-driven backtesting engine that simulates trading strategies
with realistic transaction costs, slippage, and position management.

Features:
- Vectorized backtesting for speed
- Realistic transaction costs (spread + commission)
- Position sizing support
- Long-only or long/short strategies
- Detailed trade logging

Author: Claude Opus 4.5
Date: 2024-12-03
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.metrics import PerformanceMetrics, calculate_all_metrics

logger = logging.getLogger(__name__)


class Position(Enum):
    """Trading position states."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1% per trade (Coinbase maker fee)
    slippage_rate: float = 0.0005  # 0.05% slippage estimate
    position_size: float = 1.0  # Fraction of capital to use (1.0 = all-in)
    allow_short: bool = False  # Only long positions by default
    risk_free_rate: float = 0.0  # For Sharpe calculation


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: BacktestConfig
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: List[Trade]
    metrics: PerformanceMetrics
    signals: pd.Series  # Original signals
    prices: pd.Series  # Price series used


class BacktestEngine:
    """
    Vectorized backtesting engine.

    Simulates trading based on signals with realistic costs.

    Example:
        >>> engine = BacktestEngine(config)
        >>> result = engine.run(prices, signals)
        >>> print(result.metrics)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()
        logger.info(f"BacktestEngine initialized with capital=${self.config.initial_capital:,.2f}")

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        timestamps: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run backtest on price series with signals.

        Args:
            prices: Price series (e.g., close prices)
            signals: Signal series where:
                     1 = go long / stay long
                     0 = go flat / exit
                    -1 = go short (if allowed)
            timestamps: Optional datetime index

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        # Align data
        if timestamps is not None:
            prices = prices.copy()
            prices.index = timestamps
            signals = signals.copy()
            signals.index = timestamps

        # Ensure alignment
        df = pd.DataFrame({'price': prices, 'signal': signals})
        df = df.dropna()

        if len(df) < 2:
            raise ValueError("Need at least 2 data points for backtest")

        logger.info(f"Running backtest on {len(df)} periods")

        # Calculate positions (signals shifted by 1 to avoid look-ahead)
        # We see signal at time t, can only act at t+1
        df['position'] = df['signal'].shift(1).fillna(0)

        # Handle short positions if not allowed
        if not self.config.allow_short:
            df['position'] = df['position'].clip(lower=0)

        # Calculate returns
        df['price_return'] = df['price'].pct_change()

        # Strategy returns (position * price return)
        df['strategy_return'] = df['position'] * df['price_return']

        # Apply transaction costs when position changes
        df['position_change'] = df['position'].diff().abs()
        df['position_change'] = df['position_change'].fillna(0)

        # Transaction cost = (commission + slippage) * position_change
        total_cost_rate = self.config.commission_rate + self.config.slippage_rate
        df['transaction_cost'] = df['position_change'] * total_cost_rate

        # Net returns after costs
        df['net_return'] = df['strategy_return'] - df['transaction_cost']

        # Equity curve
        df['equity'] = self.config.initial_capital * (1 + df['net_return']).cumprod()

        # Extract trades
        trades = self._extract_trades(df)

        # Calculate trade returns for metrics
        trade_returns = pd.Series([t.pnl_pct for t in trades]) if trades else pd.Series(dtype=float)

        # Calculate metrics
        metrics = calculate_all_metrics(
            returns=df['net_return'].dropna(),
            equity_curve=df['equity'].dropna(),
            trade_returns=trade_returns,
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=8760  # hourly
        )

        return BacktestResult(
            config=self.config,
            equity_curve=df['equity'],
            returns=df['net_return'],
            positions=df['position'],
            trades=trades,
            metrics=metrics,
            signals=df['signal'],
            prices=df['price']
        )

    def _extract_trades(self, df: pd.DataFrame) -> List[Trade]:
        """Extract individual trades from position changes."""
        trades = []
        position = Position.FLAT
        entry_time = None
        entry_price = None
        entry_idx = None

        for idx, row in df.iterrows():
            current_pos = row['position']
            price = row['price']

            # Position change detection
            if position == Position.FLAT and current_pos > 0:
                # Enter long
                position = Position.LONG
                entry_time = idx
                entry_price = price * (1 + self.config.slippage_rate)  # Buy at slightly higher

            elif position == Position.FLAT and current_pos < 0:
                # Enter short
                position = Position.SHORT
                entry_time = idx
                entry_price = price * (1 - self.config.slippage_rate)  # Sell at slightly lower

            elif position == Position.LONG and current_pos <= 0:
                # Exit long
                exit_price = price * (1 - self.config.slippage_rate)  # Sell at slightly lower
                pnl_pct = (exit_price / entry_price) - 1
                commission = (self.config.commission_rate * 2)  # Entry + exit

                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=idx,
                    symbol='BTC/USD',  # TODO: make dynamic
                    side='long',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=1.0,
                    pnl=pnl_pct * self.config.initial_capital * self.config.position_size,
                    pnl_pct=pnl_pct - commission,
                    commission=commission
                ))

                position = Position.FLAT
                if current_pos < 0:
                    # Flip to short
                    position = Position.SHORT
                    entry_time = idx
                    entry_price = price * (1 - self.config.slippage_rate)

            elif position == Position.SHORT and current_pos >= 0:
                # Exit short
                exit_price = price * (1 + self.config.slippage_rate)  # Buy to cover at higher
                pnl_pct = (entry_price / exit_price) - 1
                commission = (self.config.commission_rate * 2)

                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=idx,
                    symbol='BTC/USD',
                    side='short',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=1.0,
                    pnl=pnl_pct * self.config.initial_capital * self.config.position_size,
                    pnl_pct=pnl_pct - commission,
                    commission=commission
                ))

                position = Position.FLAT
                if current_pos > 0:
                    # Flip to long
                    position = Position.LONG
                    entry_time = idx
                    entry_price = price * (1 + self.config.slippage_rate)

        return trades

    def run_with_sizing(
        self,
        prices: pd.Series,
        signals: pd.Series,
        position_sizes: pd.Series,
        timestamps: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run backtest with variable position sizing.

        Args:
            prices: Price series
            signals: Signal series (direction)
            position_sizes: Position size at each period (0-1)
            timestamps: Optional datetime index

        Returns:
            BacktestResult
        """
        # Combine signal direction with sizing
        sized_signals = signals * position_sizes.clip(0, 1)

        # Use standard run with sized signals
        return self.run(prices, sized_signals, timestamps)


def run_buy_and_hold(
    prices: pd.Series,
    initial_capital: float = 10000.0,
    commission_rate: float = 0.001
) -> BacktestResult:
    """
    Run buy-and-hold benchmark.

    Args:
        prices: Price series
        initial_capital: Starting capital
        commission_rate: One-time entry commission

    Returns:
        BacktestResult for buy-and-hold strategy
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=commission_rate
    )

    # Signals: always long
    signals = pd.Series(1, index=prices.index)

    engine = BacktestEngine(config)
    return engine.run(prices, signals)


def generate_random_signals(n: int, seed: int = 42) -> pd.Series:
    """Generate random trading signals for testing."""
    np.random.seed(seed)
    # Random walk between -1, 0, 1
    signals = np.random.choice([-1, 0, 1], size=n, p=[0.2, 0.3, 0.5])
    return pd.Series(signals)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)

    # Generate sample price data (random walk)
    n = 1000
    returns = np.random.randn(n) * 0.002 + 0.0001
    prices = pd.Series(50000 * np.cumprod(1 + returns))

    # Generate simple signals (momentum-like)
    signals = pd.Series(np.where(returns > 0, 1, 0))  # Long when up

    # Run backtest
    config = BacktestConfig(
        initial_capital=10000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )

    engine = BacktestEngine(config)
    result = engine.run(prices, signals)

    print("Backtest Results:")
    print(result.metrics)

    print(f"\nNumber of trades: {len(result.trades)}")
    print(f"Final equity: ${result.equity_curve.iloc[-1]:,.2f}")

    # Compare to buy-and-hold
    bh_result = run_buy_and_hold(prices)
    print(f"\nBuy-and-hold final: ${bh_result.equity_curve.iloc[-1]:,.2f}")
    print(f"Buy-and-hold Sharpe: {bh_result.metrics.sharpe_ratio:.2f}")
