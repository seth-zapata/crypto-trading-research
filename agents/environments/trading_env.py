"""
Trading Environment for RL Position Sizing

Gymnasium-compatible environment where the agent learns optimal
position sizing based on GNN regime probabilities and market conditions.

Author: Claude Opus 4.5
Date: December 2024
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any


class PositionSizingEnv(gym.Env):
    """
    RL environment for learning position sizing.

    State:
        - GNN regime probabilities (3)
        - Volatility features (2)
        - Recent returns (1)
        - Current position (1)
        - Current drawdown (1)
        - Days since rebalance (1)

    Action:
        - Position size: continuous [0.10, 1.00]

    Reward:
        - Risk-adjusted returns with heavy drawdown penalties
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        gnn_probs: np.ndarray,
        volatility_20d: np.ndarray,
        volatility_5d: np.ndarray,
        returns_5d: np.ndarray,
        transaction_cost: float = 0.002,
        min_position: float = 0.10,
        max_position: float = 1.00,
    ):
        """
        Initialize the trading environment.

        Args:
            prices: Array of daily prices
            gnn_probs: Array of GNN probabilities (N x 3) for each day
            volatility_20d: 20-day rolling volatility
            volatility_5d: 5-day rolling volatility
            returns_5d: 5-day rolling returns
            transaction_cost: Cost per unit of position change (default 0.2%)
            min_position: Minimum allowed position (default 10%)
            max_position: Maximum allowed position (default 100%)
        """
        super().__init__()

        self.prices = prices
        self.gnn_probs = gnn_probs
        self.volatility_20d = volatility_20d
        self.volatility_5d = volatility_5d
        self.returns_5d = returns_5d
        self.transaction_cost = transaction_cost
        self.min_position = min_position
        self.max_position = max_position

        # Calculate daily returns
        self.daily_returns = np.diff(prices) / prices[:-1]
        self.n_steps = len(self.daily_returns)

        # State space: 10 dimensions
        # [risk_on_prob, caution_prob, risk_off_prob, vol_20d, vol_5d,
        #  returns_5d, current_position, current_drawdown, days_since_rebalance, normalized_step]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 2, 2, 1, 1, 1, 30, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: continuous position size [0.10, 1.00]
        self.action_space = spaces.Box(
            low=np.array([min_position], dtype=np.float32),
            high=np.array([max_position], dtype=np.float32),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.current_position = 1.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.days_since_rebalance = 0
        self.episode_returns = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.current_position = 1.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.days_since_rebalance = 0
        self.episode_returns = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Build current observation/state vector."""
        idx = self.current_step

        # GNN probabilities
        risk_on_prob = self.gnn_probs[idx, 0]
        caution_prob = self.gnn_probs[idx, 1]
        risk_off_prob = self.gnn_probs[idx, 2]

        # Market conditions (clipped to reasonable ranges)
        vol_20d = np.clip(self.volatility_20d[idx], 0, 2)
        vol_5d = np.clip(self.volatility_5d[idx], 0, 2)
        ret_5d = np.clip(self.returns_5d[idx], -1, 1)

        # Portfolio state
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        days_since_rebal = min(self.days_since_rebalance, 30) / 30.0
        normalized_step = self.current_step / self.n_steps

        obs = np.array([
            risk_on_prob,
            caution_prob,
            risk_off_prob,
            vol_20d,
            vol_5d,
            ret_5d,
            self.current_position,
            current_drawdown,
            days_since_rebal,
            normalized_step
        ], dtype=np.float32)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Position size [0.10, 1.00]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Extract and clip position
        new_position = np.clip(action[0], self.min_position, self.max_position)
        position_change = abs(new_position - self.current_position)

        # Track rebalancing
        if position_change > 0.01:  # Threshold for counting as rebalance
            self.days_since_rebalance = 0
        else:
            self.days_since_rebalance += 1

        # Get daily return
        daily_return = self.daily_returns[self.current_step]

        # Calculate portfolio return (position * market return - transaction costs)
        portfolio_return = (daily_return * new_position) - (position_change * self.transaction_cost)

        # Update equity
        self.equity *= (1 + portfolio_return)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.episode_returns.append(portfolio_return)

        # Calculate drawdown
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity

        # Calculate reward
        reward = self._calculate_reward(
            daily_return=daily_return,
            portfolio_return=portfolio_return,
            position=new_position,
            position_change=position_change,
            drawdown=current_drawdown
        )

        # Update state
        self.current_position = new_position
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.n_steps
        truncated = False

        # Build info dict
        info = {
            'equity': self.equity,
            'drawdown': current_drawdown,
            'position': new_position,
            'daily_return': daily_return,
            'portfolio_return': portfolio_return
        }

        if terminated:
            episode_return = (self.equity - 1) * 100
            max_dd = self._calculate_max_drawdown()
            sharpe = self._calculate_sharpe()

            info['episode_return'] = episode_return
            info['max_drawdown'] = max_dd
            info['sharpe'] = sharpe

            # TERMINAL BONUS: Reward good total return, penalize poor return
            # This encourages the agent to care about cumulative performance
            if episode_return > 20:
                reward += episode_return * 0.5  # Bonus for good returns
            elif episode_return < 0:
                reward -= abs(episode_return) * 0.3  # Penalty for losses

            # Sharpe bonus
            if sharpe > 0.5:
                reward += sharpe * 2.0

        obs = self._get_observation() if not terminated else np.zeros(10, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def _calculate_reward(
        self,
        daily_return: float,
        portfolio_return: float,
        position: float,
        position_change: float,
        drawdown: float
    ) -> float:
        """
        Calculate reward using Sharpe-like risk-adjusted return.

        Key: Use actual risk-adjusted metric, not heuristics.
        """
        # Base: Daily portfolio return
        reward = portfolio_return * 100

        # Risk-adjusted component: penalize high position during volatility
        vol_5d = self.volatility_5d[self.current_step] if hasattr(self, '_current_vol') else 0.3
        vol_penalty = position * vol_5d * 0.5  # Higher position + higher vol = more penalty
        reward -= vol_penalty

        # Regime alignment bonus (use GNN signal)
        risk_off_prob = self.gnn_probs[self.current_step, 2]
        risk_on_prob = self.gnn_probs[self.current_step, 0]

        # Reward alignment with GNN predictions
        # Low position when RISK_OFF is high
        if risk_off_prob > 0.5:
            alignment = (1.0 - position) * risk_off_prob
            reward += alignment * 0.5
        # High position when RISK_ON is high
        if risk_on_prob > 0.5:
            alignment = position * risk_on_prob
            reward += alignment * 0.3

        # Drawdown penalty (only above 20%)
        if drawdown > 0.20:
            reward -= (drawdown - 0.20) * 5.0
        if drawdown > 0.25:
            reward -= (drawdown - 0.25) * 20.0

        # Position baseline: slight reward for being invested
        # This counters the tendency to stay at minimum
        reward += position * 0.2

        # Transaction cost
        reward -= position_change * 0.2

        return reward

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown of the episode."""
        equity_curve = np.cumprod(1 + np.array(self.episode_returns))
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return drawdown.max() * 100

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio of the episode."""
        returns = np.array(self.episode_returns)
        if returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    def render(self, mode: str = "human"):
        """Render current state."""
        dd = (self.peak_equity - self.equity) / self.peak_equity * 100
        print(f"Step {self.current_step}: Equity={self.equity:.4f}, "
              f"Position={self.current_position:.1%}, DD={dd:.1f}%")


class DiscretePositionSizingEnv(PositionSizingEnv):
    """
    Discrete action variant for easier training.

    Actions: 7 position sizes [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
    """

    POSITION_LEVELS = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override action space to discrete
        self.action_space = spaces.Discrete(len(self.POSITION_LEVELS))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Convert discrete action to position and step."""
        position = self.POSITION_LEVELS[action]
        return super().step(np.array([position]))
