"""
Phase 5: RL Position Sizing - V2
================================

Fixed version that:
1. Uses discrete action space (easier to learn)
2. Trains for longer with better exploration
3. Uses regime-guided reward shaping
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

logging.basicConfig(level=logging.WARNING)

print("=" * 70)
print("PHASE 5: RL POSITION SIZING (V2 - DISCRETE ACTIONS)")
print("=" * 70)


# ============================================================
# CUSTOM DISCRETE ENVIRONMENT
# ============================================================

class DiscretePositionEnv(gym.Env):
    """
    Simplified discrete action environment.

    Actions: 5 levels [0.2, 0.4, 0.6, 0.8, 1.0]
    State: GNN probs + volatility + drawdown
    Reward: Risk-adjusted return with regime alignment
    """

    POSITIONS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    def __init__(
        self,
        prices: np.ndarray,
        gnn_probs: np.ndarray,
        volatility: np.ndarray,
        transaction_cost: float = 0.001
    ):
        super().__init__()

        self.prices = prices
        self.gnn_probs = gnn_probs
        self.volatility = volatility
        self.transaction_cost = transaction_cost

        self.daily_returns = np.diff(prices) / prices[:-1]
        self.n_steps = len(self.daily_returns)

        # State: [risk_on, caution, risk_off, vol, position, drawdown]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 2, 1, 1], dtype=np.float32)
        )

        # 5 discrete actions
        self.action_space = spaces.Discrete(5)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.position = 1.0  # Start fully invested
        self.equity = 1.0
        self.peak_equity = 1.0
        self.returns_list = []
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.gnn_probs[self.step_idx, 0],  # RISK_ON
            self.gnn_probs[self.step_idx, 1],  # CAUTION
            self.gnn_probs[self.step_idx, 2],  # RISK_OFF
            min(self.volatility[self.step_idx], 2.0),
            self.position,
            (self.peak_equity - self.equity) / self.peak_equity
        ], dtype=np.float32)

    def step(self, action):
        new_position = self.POSITIONS[action]
        pos_change = abs(new_position - self.position)

        # Get return
        daily_ret = self.daily_returns[self.step_idx]

        # Portfolio return
        port_ret = daily_ret * new_position - pos_change * self.transaction_cost

        # Update equity
        self.equity *= (1 + port_ret)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.returns_list.append(port_ret)

        drawdown = (self.peak_equity - self.equity) / self.peak_equity

        # Calculate reward
        reward = self._calc_reward(daily_ret, port_ret, new_position, drawdown)

        self.position = new_position
        self.step_idx += 1

        done = self.step_idx >= self.n_steps

        if done:
            # Terminal bonus
            total_ret = (self.equity - 1) * 100
            max_dd = self._max_drawdown()

            # Strong bonus for achieving target
            if max_dd < 25 and total_ret > 30:
                reward += 50  # Big bonus for meeting both targets
            elif max_dd < 25:
                reward += 20  # Bonus for meeting DD target
            elif total_ret > 30:
                reward += 10  # Small bonus for return

        obs = self._get_obs() if not done else np.zeros(6, dtype=np.float32)
        return obs, reward, done, False, {'equity': self.equity}

    def _calc_reward(self, daily_ret, port_ret, position, drawdown):
        """Regime-aligned reward."""
        # Base return
        reward = port_ret * 100

        # Regime alignment
        risk_off_prob = self.gnn_probs[self.step_idx, 2]
        risk_on_prob = self.gnn_probs[self.step_idx, 0]

        # Position should match regime
        if risk_off_prob > 0.5:
            # Should be defensive
            if position <= 0.4:
                reward += 0.5  # Good alignment
            elif position >= 0.8:
                reward -= 0.5  # Bad - too aggressive

        if risk_on_prob > 0.6:
            # Should be aggressive
            if position >= 0.8:
                reward += 0.3  # Good
            elif position <= 0.4:
                reward -= 0.3  # Missing upside

        # Drawdown penalty
        if drawdown > 0.20:
            reward -= drawdown * 5
        if drawdown > 0.25:
            reward -= (drawdown - 0.25) * 20

        return reward

    def _max_drawdown(self):
        eq = np.cumprod(1 + np.array(self.returns_list))
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        return dd.max() * 100


# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")

df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

# Train GNN with calibrated weights
np.random.seed(42)  # For reproducibility
detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector.prepare_dataset(df, labels)

history = detector.train(
    graphs, targets,
    graphs, targets,
    epochs=100, batch_size=32,
    class_weights=[1.0, 2.0, 8.0]
)

preds, probs = detector.predict(graphs)
print(f"GNN trained. Dataset: {len(df)} samples")


# ============================================================
# 2. PREPARE ENVIRONMENT DATA
# ============================================================
print("\n[2/5] Preparing environment...")

graph_start = len(df) - len(graphs)
aligned_df = df.iloc[graph_start:].reset_index(drop=True)

prices = aligned_df['BTC_close'].values
volatility = aligned_df['BTC_vol_20d'].values

# Split 80/20
split = int(len(prices) * 0.8)

train_prices = prices[:split]
train_probs = probs[:split]
train_vol = volatility[:split]

test_prices = prices[split:]
test_probs = probs[split:]
test_vol = volatility[split:]

print(f"Train: {len(train_prices)} days, Test: {len(test_prices)} days")


# ============================================================
# 3. TRAIN PPO
# ============================================================
print("\n[3/5] Training PPO with discrete actions...")

def make_env():
    return DiscretePositionEnv(train_prices, train_probs, train_vol)

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.1,  # Higher entropy for exploration
    verbose=0,
    seed=42,
    device='cpu'
)

total_timesteps = 200_000
print(f"Training for {total_timesteps:,} timesteps...")
model.learn(total_timesteps=total_timesteps, progress_bar=True)

model_path = Path(__file__).parent.parent / 'models' / 'saved' / 'position_sizer_discrete.zip'
model.save(str(model_path))
print(f"Model saved to: {model_path}")


# ============================================================
# 4. BACKTEST
# ============================================================
print("\n[4/5] Backtesting...")

def backtest(prices, positions, cost=0.001):
    """Run backtest."""
    equity = [1.0]
    prev_pos = 1.0
    returns = np.diff(prices) / prices[:-1]

    for i, ret in enumerate(returns):
        pos = positions[i] if i < len(positions) else 1.0
        port_ret = ret * pos - abs(pos - prev_pos) * cost
        equity.append(equity[-1] * (1 + port_ret))
        prev_pos = pos

    eq = np.array(equity)
    daily_rets = np.diff(eq) / eq[:-1]

    total_ret = (eq[-1] - 1) * 100
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

    peak = np.maximum.accumulate(eq)
    max_dd = ((peak - eq) / peak).max() * 100

    return total_ret, sharpe, max_dd


# Buy & Hold
bh_ret, bh_sharpe, bh_dd = backtest(test_prices, np.ones(len(test_prices)-1))

# GNN + Fixed Rules
RISK_ON, CAUTION, RISK_OFF = 0, 1, 2
pos_map = {RISK_ON: 1.0, CAUTION: 0.5, RISK_OFF: 0.2}
test_preds = np.argmax(test_probs, axis=1)
fixed_pos = np.array([pos_map[p] for p in test_preds])
fixed_ret, fixed_sharpe, fixed_dd = backtest(test_prices, fixed_pos)

# GNN + RL
test_env = DiscretePositionEnv(test_prices, test_probs, test_vol)
rl_positions = []
obs, _ = test_env.reset()
for _ in range(len(test_prices) - 1):
    action, _ = model.predict(obs, deterministic=True)
    rl_positions.append(DiscretePositionEnv.POSITIONS[action])
    obs, _, done, _, _ = test_env.step(action)
    if done:
        break

rl_positions = np.array(rl_positions)
rl_ret, rl_sharpe, rl_dd = backtest(test_prices, rl_positions)

# Simple vol rule baseline
vol_pos = np.where(test_vol > 0.8, 0.2, np.where(test_vol > 0.5, 0.5, 1.0))
vol_ret, vol_sharpe, vol_dd = backtest(test_prices, vol_pos)


# ============================================================
# 5. RESULTS
# ============================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\n{'Strategy':<22} {'Return':>10} {'Sharpe':>8} {'Max DD':>10}")
print("-" * 55)
print(f"{'Buy & Hold':<22} {bh_ret:>+9.1f}% {bh_sharpe:>8.2f} {bh_dd:>9.1f}%")
print(f"{'Simple Vol Rule':<22} {vol_ret:>+9.1f}% {vol_sharpe:>8.2f} {vol_dd:>9.1f}%")
print(f"{'GNN + Fixed Rules':<22} {fixed_ret:>+9.1f}% {fixed_sharpe:>8.2f} {fixed_dd:>9.1f}%")
print(f"{'GNN + RL (discrete)':<22} {rl_ret:>+9.1f}% {rl_sharpe:>8.2f} {rl_dd:>9.1f}%")

print("\n" + "=" * 70)
print("POSITION ANALYSIS")
print("=" * 70)

print(f"\nRL positions: Avg={rl_positions.mean():.1%}, Min={rl_positions.min():.1%}, Max={rl_positions.max():.1%}")
print(f"Position distribution:")
for level in DiscretePositionEnv.POSITIONS:
    pct = (rl_positions == level).mean() * 100
    print(f"  {level:.0%}: {pct:.1f}%")

# Regime-specific positions
for regime_idx, regime_name in enumerate(['RISK_ON', 'CAUTION', 'RISK_OFF']):
    mask = test_preds[:len(rl_positions)] == regime_idx
    if mask.sum() > 0:
        avg = rl_positions[mask].mean()
        print(f"Avg position during {regime_name}: {avg:.1%}")

print("\n" + "=" * 70)
print("SUCCESS CHECK")
print("=" * 70)

print(f"\n1. Max DD < 25%: {rl_dd:.1f}% - {'PASS' if rl_dd < 25 else 'FAIL'}")
print(f"2. Return > 0%: {rl_ret:.1f}% - {'PASS' if rl_ret > 0 else 'FAIL'}")
print(f"3. Sharpe > Fixed: {rl_sharpe:.2f} vs {fixed_sharpe:.2f} - {'PASS' if rl_sharpe > fixed_sharpe else 'FAIL'}")
print(f"4. Beat Vol Rule: {rl_sharpe:.2f} vs {vol_sharpe:.2f} - {'PASS' if rl_sharpe > vol_sharpe else 'FAIL'}")

print("\n" + "=" * 70)
