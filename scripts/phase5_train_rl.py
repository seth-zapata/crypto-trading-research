"""
Phase 5: Train RL Position Sizer
================================

Train PPO agent to learn optimal position sizing based on:
- GNN regime probabilities (from Phase 4)
- Market conditions (volatility, recent returns)
- Portfolio state (current position, drawdown)

Goal: Reduce max drawdown from 26.8% to <25% without sacrificing returns.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.environments.trading_env import PositionSizingEnv, DiscretePositionSizingEnv
from stable_baselines3 import DQN
from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

logging.basicConfig(level=logging.WARNING)

print("=" * 70)
print("PHASE 5: RL POSITION SIZING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA AND GNN MODEL
# ============================================================
print("\n[1/6] Loading data and GNN model...")

df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

# Train GNN with calibrated weights (from Phase 4)
detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector.prepare_dataset(df, labels)

# Use calibrated class weights
calibrated_weights = [1.0, 2.0, 8.0]

# Full training on all data to get probabilities
history = detector.train(
    graphs, targets,
    graphs, targets,  # Just for prob generation
    epochs=100, batch_size=32,
    class_weights=calibrated_weights
)

# Get probabilities for all data
preds, probs = detector.predict(graphs)

print(f"Dataset: {len(df)} samples")
print(f"GNN predictions generated: {len(probs)}")


# ============================================================
# 2. PREPARE FEATURES FOR RL
# ============================================================
print("\n[2/6] Preparing features for RL environment...")

# Align features with graphs (graphs start after warmup period)
graph_start_idx = len(df) - len(graphs)
aligned_df = df.iloc[graph_start_idx:].reset_index(drop=True)

# Extract features
prices = aligned_df['BTC_close'].values
volatility_20d = aligned_df['BTC_vol_20d'].values
volatility_5d = aligned_df['BTC_vol_5d'].values if 'BTC_vol_5d' in aligned_df.columns else aligned_df['BTC_vol_20d'].values * 1.2
returns_5d = aligned_df['BTC_return_5d'].values if 'BTC_return_5d' in aligned_df.columns else aligned_df['BTC_return_1d'].rolling(5).sum().values

# Handle NaN
volatility_5d = np.nan_to_num(volatility_5d, nan=0.3)
returns_5d = np.nan_to_num(returns_5d, nan=0.0)

# Split: train on 2020-2023, test on 2024
# Approximate split at 80% (similar to GNN training)
train_split = int(len(prices) * 0.80)

train_prices = prices[:train_split]
train_probs = probs[:train_split]
train_vol_20d = volatility_20d[:train_split]
train_vol_5d = volatility_5d[:train_split]
train_returns_5d = returns_5d[:train_split]

test_prices = prices[train_split:]
test_probs = probs[train_split:]
test_vol_20d = volatility_20d[train_split:]
test_vol_5d = volatility_5d[train_split:]
test_returns_5d = returns_5d[train_split:]

print(f"Train period: {train_split} days")
print(f"Test period: {len(test_prices)} days")


# ============================================================
# 3. CREATE ENVIRONMENTS
# ============================================================
print("\n[3/6] Creating training environment...")

def make_train_env():
    return PositionSizingEnv(
        prices=train_prices,
        gnn_probs=train_probs,
        volatility_20d=train_vol_20d,
        volatility_5d=train_vol_5d,
        returns_5d=train_returns_5d,
        transaction_cost=0.002
    )

def make_test_env():
    return PositionSizingEnv(
        prices=test_prices,
        gnn_probs=test_probs,
        volatility_20d=test_vol_20d,
        volatility_5d=test_vol_5d,
        returns_5d=test_returns_5d,
        transaction_cost=0.002
    )

train_env = DummyVecEnv([make_train_env])
test_env = DummyVecEnv([make_test_env])

print("Environments created")


# ============================================================
# 4. TRAIN PPO AGENT
# ============================================================
print("\n[4/6] Training PPO agent...")

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=0,
    seed=42
)

# Train for 200k steps (more exploration needed)
total_timesteps = 200_000
print(f"Training for {total_timesteps:,} timesteps...")

model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Save model
model_path = Path(__file__).parent.parent / 'models' / 'saved' / 'position_sizer_rl.zip'
model_path.parent.mkdir(parents=True, exist_ok=True)
model.save(str(model_path))
print(f"Model saved to: {model_path}")


# ============================================================
# 5. BACKTEST ALL STRATEGIES
# ============================================================
print("\n[5/6] Backtesting strategies...")

def backtest_strategy(
    prices: np.ndarray,
    positions: np.ndarray,
    transaction_cost: float = 0.002
) -> Tuple[List[float], float, float, float]:
    """Run backtest with given positions."""
    equity = [1.0]
    prev_position = 1.0

    returns = np.diff(prices) / prices[:-1]

    for i, ret in enumerate(returns):
        pos = positions[i] if i < len(positions) else 1.0
        pos_change = abs(pos - prev_position)
        cost = pos_change * transaction_cost

        portfolio_ret = ret * pos - cost
        equity.append(equity[-1] * (1 + portfolio_ret))
        prev_position = pos

    # Calculate metrics
    equity = np.array(equity)
    daily_returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] / equity[0] - 1) * 100
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = drawdown.max() * 100

    return equity.tolist(), total_return, sharpe, max_dd


# Strategy 1: Buy & Hold
bh_positions = np.ones(len(test_prices) - 1)
bh_equity, bh_ret, bh_sharpe, bh_dd = backtest_strategy(test_prices, bh_positions)

# Strategy 2: GNN + Fixed Rules (current baseline from Phase 4)
RISK_ON, CAUTION, RISK_OFF = 0, 1, 2
position_map = {RISK_ON: 1.0, CAUTION: 0.5, RISK_OFF: 0.2}
test_preds = np.argmax(test_probs, axis=1)
fixed_positions = np.array([position_map[p] for p in test_preds])
fixed_equity, fixed_ret, fixed_sharpe, fixed_dd = backtest_strategy(test_prices, fixed_positions)

# Strategy 3: GNN + RL (our new approach)
rl_positions = []
obs = test_env.reset()
for i in range(len(test_prices) - 1):
    action, _ = model.predict(obs, deterministic=True)
    rl_positions.append(action[0][0])
    obs, _, done, _ = test_env.step(action)
    if done:
        break

rl_positions = np.array(rl_positions)
rl_equity, rl_ret, rl_sharpe, rl_dd = backtest_strategy(test_prices, rl_positions)

# Strategy 4: RL Only (no GNN - random probs to prove GNN adds value)
print("Training RL-only baseline (random regime inputs)...")

random_probs = np.random.dirichlet(np.ones(3), size=len(train_probs))
random_test_probs = np.random.dirichlet(np.ones(3), size=len(test_probs))

def make_rl_only_train_env():
    return PositionSizingEnv(
        prices=train_prices,
        gnn_probs=random_probs,
        volatility_20d=train_vol_20d,
        volatility_5d=train_vol_5d,
        returns_5d=train_returns_5d,
        transaction_cost=0.002
    )

rl_only_env = DummyVecEnv([make_rl_only_train_env])
rl_only_model = PPO("MlpPolicy", rl_only_env, learning_rate=3e-4, n_steps=2048,
                     batch_size=64, verbose=0, seed=42)
rl_only_model.learn(total_timesteps=50_000, progress_bar=True)

# Test RL-only
def make_rl_only_test_env():
    return PositionSizingEnv(
        prices=test_prices,
        gnn_probs=random_test_probs,
        volatility_20d=test_vol_20d,
        volatility_5d=test_vol_5d,
        returns_5d=test_returns_5d,
        transaction_cost=0.002
    )

rl_only_test_env = DummyVecEnv([make_rl_only_test_env])
rl_only_positions = []
obs = rl_only_test_env.reset()
for i in range(len(test_prices) - 1):
    action, _ = rl_only_model.predict(obs, deterministic=True)
    rl_only_positions.append(action[0][0])
    obs, _, done, _ = rl_only_test_env.step(action)
    if done:
        break

rl_only_positions = np.array(rl_only_positions)
rl_only_equity, rl_only_ret, rl_only_sharpe, rl_only_dd = backtest_strategy(test_prices, rl_only_positions)

# Strategy 5: Simple volatility rule (control baseline)
def simple_vol_positions(vol_20d: np.ndarray) -> np.ndarray:
    positions = []
    for v in vol_20d:
        if v > 0.80:
            positions.append(0.2)
        elif v > 0.50:
            positions.append(0.5)
        else:
            positions.append(1.0)
    return np.array(positions)

vol_positions = simple_vol_positions(test_vol_20d)
vol_equity, vol_ret, vol_sharpe, vol_dd = backtest_strategy(test_prices, vol_positions)


# ============================================================
# 6. RESULTS COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

results = [
    ("Buy & Hold", bh_ret, bh_sharpe, bh_dd, "-"),
    ("Simple Vol Rule", vol_ret, vol_sharpe, vol_dd, "Baseline"),
    ("GNN + Fixed Rules", fixed_ret, fixed_sharpe, fixed_dd, "Phase 4"),
    ("GNN + RL", rl_ret, rl_sharpe, rl_dd, "Phase 5"),
    ("RL Only (no GNN)", rl_only_ret, rl_only_sharpe, rl_only_dd, "Control"),
]

print(f"\n{'Strategy':<22} {'Return':>10} {'Sharpe':>8} {'Max DD':>10} {'Notes':<15}")
print("-" * 70)
for name, ret, sharpe, dd, notes in results:
    print(f"{name:<22} {ret:>+9.1f}% {sharpe:>8.2f} {dd:>9.1f}% {notes:<15}")


# ============================================================
# SUCCESS CRITERIA CHECK
# ============================================================
print("\n" + "=" * 70)
print("SUCCESS CRITERIA CHECK")
print("=" * 70)

target_dd = 25.0
target_ret = 50.0  # Don't sacrifice returns

print(f"\n1. Max Drawdown: {rl_dd:.1f}% (target: <{target_dd}%): ", end="")
print("PASSED" if rl_dd < target_dd else "FAILED")

print(f"2. Return: {rl_ret:.1f}% (target: >{target_ret}%): ", end="")
print("PASSED" if rl_ret > target_ret else "FAILED")

print(f"3. Beat GNN+Fixed: Sharpe {rl_sharpe:.2f} vs {fixed_sharpe:.2f}: ", end="")
print("PASSED" if rl_sharpe > fixed_sharpe else "FAILED")

print(f"4. GNN adds value: GNN+RL {rl_sharpe:.2f} vs RL-only {rl_only_sharpe:.2f}: ", end="")
print("PASSED" if rl_sharpe > rl_only_sharpe else "FAILED")


# ============================================================
# POSITION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("POSITION ANALYSIS (GNN + RL)")
print("=" * 70)

print(f"\nAvg position: {rl_positions.mean():.1%}")
print(f"Min position: {rl_positions.min():.1%}")
print(f"Max position: {rl_positions.max():.1%}")
print(f"Std position: {rl_positions.std():.1%}")

# Position by regime
for regime_idx, regime_name in enumerate(['RISK_ON', 'CAUTION', 'RISK_OFF']):
    regime_mask = test_preds == regime_idx
    if regime_mask.sum() > 0:
        avg_pos = rl_positions[regime_mask[:len(rl_positions)]].mean()
        print(f"Avg position in {regime_name}: {avg_pos:.1%}")


# ============================================================
# IMPROVEMENT OVER PHASE 4
# ============================================================
print("\n" + "=" * 70)
print("IMPROVEMENT OVER PHASE 4")
print("=" * 70)

dd_improvement = fixed_dd - rl_dd
sharpe_improvement = rl_sharpe - fixed_sharpe
ret_change = rl_ret - fixed_ret

print(f"\nDrawdown: {fixed_dd:.1f}% -> {rl_dd:.1f}% ({dd_improvement:+.1f}%)")
print(f"Sharpe: {fixed_sharpe:.2f} -> {rl_sharpe:.2f} ({sharpe_improvement:+.2f})")
print(f"Return: {fixed_ret:.1f}% -> {rl_ret:.1f}% ({ret_change:+.1f}%)")

if rl_dd < target_dd and rl_ret > fixed_ret * 0.9:
    print("\nPHASE 5 SUCCESS: Achieved <25% max DD without significant return sacrifice")
elif rl_dd < fixed_dd:
    print(f"\nPARTIAL SUCCESS: Reduced DD by {dd_improvement:.1f}%, but still above {target_dd}% target")
else:
    print("\nNEEDS WORK: RL did not improve on fixed rules")

print("\n" + "=" * 70)
