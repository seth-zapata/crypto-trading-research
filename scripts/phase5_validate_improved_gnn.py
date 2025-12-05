"""
Phase 5: Monte Carlo Validation of Improved GNN
================================================

Validates the improved GNN (asymmetric loss + low threshold) using:
1. Block Bootstrap Monte Carlo (1000 simulations)
2. Walk-Forward Validation (5 splits)
3. Stress Testing on known crisis periods

Success Criteria:
- Miss rate < 25%
- Monte Carlo 95th percentile Max DD < 40%
- Walk-forward win rate > 45%
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

logging.basicConfig(level=logging.WARNING)

print("=" * 70)
print("PHASE 5: MONTE CARLO VALIDATION OF IMPROVED GNN")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'miss_penalty': 15.0,  # Higher penalty for more consistent crash detection
    'risk_off_threshold': 0.20,  # Balance between crash detection and false alarms
    'position_weights': {'risk_on': 0.85, 'caution': 0.65, 'risk_off': 0.20},
    'n_bootstrap': 1000,
    'block_size': 20,
    'n_walk_forward_splits': 5
}

print(f"\nConfiguration:")
print(f"  Miss penalty: {CONFIG['miss_penalty']}")
print(f"  RISK_OFF threshold: {CONFIG['risk_off_threshold']}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(equity: np.ndarray) -> Dict:
    """Calculate return, Sharpe, max drawdown."""
    if len(equity) < 2:
        return {'return': 0, 'sharpe': 0, 'max_dd': 0}

    returns = np.diff(equity) / equity[:-1]
    total_return = (equity[-1] / equity[0] - 1) * 100

    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown) * 100

    return {
        'return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd
    }


def positions_from_probs(probs: np.ndarray, threshold: float, weights: Dict) -> np.ndarray:
    """Convert probabilities to positions using threshold and weights."""
    positions = (
        probs[:, 0] * weights['risk_on'] +
        probs[:, 1] * weights['caution'] +
        probs[:, 2] * weights['risk_off']
    )
    # Override: go more defensive if p(RISK_OFF) > threshold
    defensive_mask = probs[:, 2] > threshold
    positions[defensive_mask] = weights['risk_off']

    return np.clip(positions, 0.1, 1.0)


def backtest(returns: np.ndarray, positions: np.ndarray) -> Dict:
    """Run backtest and return metrics."""
    equity = [1.0]
    for i, ret in enumerate(returns):
        pos = positions[i] if i < len(positions) else 1.0
        equity.append(equity[-1] * (1 + ret * pos))
    return calculate_metrics(np.array(equity))


def calculate_miss_rate(predictions: np.ndarray, returns: np.ndarray, threshold: float = -0.05) -> float:
    """Calculate crash miss rate."""
    is_crash = returns < threshold
    n_crashes = is_crash.sum()
    if n_crashes == 0:
        return 0

    pred_risk_on = predictions == 0
    misses = (pred_risk_on & is_crash).sum()
    return misses / n_crashes


def block_bootstrap(returns: np.ndarray, probs: np.ndarray, n_simulations: int = 1000, block_size: int = 20):
    """Generate bootstrap samples using block resampling."""
    n = len(returns)
    n_blocks = n // block_size + 1
    samples = []

    for _ in range(n_simulations):
        block_starts = np.random.randint(0, n - block_size, size=n_blocks)
        ret_sample, prob_sample = [], []

        for start in block_starts:
            ret_sample.extend(returns[start:start + block_size])
            prob_sample.extend(probs[start:start + block_size])

        samples.append((np.array(ret_sample[:n]), np.array(prob_sample[:n])))

    return samples


# ============================================================
# 1. LOAD DATA AND TRAIN IMPROVED MODEL
# ============================================================
print("\n[1/4] Loading data and training improved GNN...")

df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector.prepare_dataset(df, labels)

split = int(len(graphs) * 0.8)
train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]

btc_returns = df['BTC_return_1d'].values
train_returns_arr = btc_returns[:split]
val_returns = btc_returns[split:split + len(val_graphs)]

# Train with asymmetric loss
history = detector.train(
    train_graphs, train_labels,
    val_graphs, val_labels,
    epochs=100, batch_size=32,
    train_returns=train_returns_arr,
    use_asymmetric_loss=True,
    miss_penalty=CONFIG['miss_penalty']
)

# Get predictions
preds, probs = detector.predict(val_graphs)

# Apply low threshold for predictions
preds_adj = preds.copy()
preds_adj[probs[:, 2] > CONFIG['risk_off_threshold']] = 2

# Calculate single-path metrics
positions = positions_from_probs(probs, CONFIG['risk_off_threshold'], CONFIG['position_weights'])
single_path_metrics = backtest(val_returns, positions)
miss_rate = calculate_miss_rate(preds_adj, val_returns)

print(f"\nSingle-Path Results (Validation Set):")
print(f"  Miss Rate: {miss_rate:.1%}")
print(f"  Return: {single_path_metrics['return']:.1f}%")
print(f"  Max DD: {single_path_metrics['max_dd']:.1f}%")
print(f"  Sharpe: {single_path_metrics['sharpe']:.2f}")


# ============================================================
# 2. BLOCK BOOTSTRAP MONTE CARLO
# ============================================================
print(f"\n[2/4] Running Block Bootstrap ({CONFIG['n_bootstrap']} simulations)...")

bootstrap_results = []
samples = block_bootstrap(val_returns, probs, CONFIG['n_bootstrap'], CONFIG['block_size'])

for ret_sample, prob_sample in tqdm(samples, desc="Bootstrap"):
    positions_sample = positions_from_probs(prob_sample, CONFIG['risk_off_threshold'], CONFIG['position_weights'])
    metrics = backtest(ret_sample, positions_sample)
    bootstrap_results.append(metrics)

bootstrap_df = pd.DataFrame(bootstrap_results)

# Calculate percentiles
percentiles = {
    'return_5th': bootstrap_df['return'].quantile(0.05),
    'return_50th': bootstrap_df['return'].quantile(0.50),
    'return_95th': bootstrap_df['return'].quantile(0.95),
    'max_dd_5th': bootstrap_df['max_dd'].quantile(0.05),
    'max_dd_50th': bootstrap_df['max_dd'].quantile(0.50),
    'max_dd_95th': bootstrap_df['max_dd'].quantile(0.95),
    'sharpe_5th': bootstrap_df['sharpe'].quantile(0.05),
    'sharpe_50th': bootstrap_df['sharpe'].quantile(0.50),
}

print("\nBootstrap Results:")
print(f"  Return: {percentiles['return_5th']:.1f}% / {percentiles['return_50th']:.1f}% / {percentiles['return_95th']:.1f}% (5th/50th/95th)")
print(f"  Max DD: {percentiles['max_dd_5th']:.1f}% / {percentiles['max_dd_50th']:.1f}% / {percentiles['max_dd_95th']:.1f}% (5th/50th/95th)")
print(f"  Sharpe: {percentiles['sharpe_5th']:.2f} / {percentiles['sharpe_50th']:.2f} (5th/50th)")


# ============================================================
# 3. WALK-FORWARD VALIDATION
# ============================================================
print(f"\n[3/4] Running Walk-Forward Validation ({CONFIG['n_walk_forward_splits']} splits)...")

n_splits = CONFIG['n_walk_forward_splits']
total_samples = len(graphs)
train_size = int(total_samples * 0.5)
step = (total_samples - train_size) // n_splits

walk_forward_results = []

for i in range(n_splits):
    train_end = train_size + i * step
    val_start = train_end
    val_end = min(train_end + step, total_samples)

    if val_end <= val_start:
        continue

    # Split data
    wf_train_graphs = graphs[:train_end]
    wf_train_labels = targets[:train_end]
    wf_val_graphs = graphs[val_start:val_end]
    wf_val_labels = targets[val_start:val_end]
    wf_train_returns = btc_returns[:train_end]
    wf_val_returns = btc_returns[val_start:val_end]

    if len(wf_val_graphs) == 0:
        continue

    # Train new detector for this fold
    wf_detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
    wf_detector.num_node_features = detector.num_node_features

    wf_detector.train(
        wf_train_graphs, wf_train_labels,
        None, None,
        epochs=50, batch_size=32,
        train_returns=wf_train_returns,
        use_asymmetric_loss=True,
        miss_penalty=CONFIG['miss_penalty']
    )

    # Predict
    wf_preds, wf_probs = wf_detector.predict(wf_val_graphs)

    # Apply threshold
    wf_preds_adj = wf_preds.copy()
    wf_preds_adj[wf_probs[:, 2] > CONFIG['risk_off_threshold']] = 2

    # Calculate metrics
    wf_positions = positions_from_probs(wf_probs, CONFIG['risk_off_threshold'], CONFIG['position_weights'])
    strategy_metrics = backtest(wf_val_returns, wf_positions)

    # Buy & Hold baseline
    bh_metrics = backtest(wf_val_returns, np.ones(len(wf_val_returns)))

    # Miss rate for this fold
    fold_miss_rate = calculate_miss_rate(wf_preds_adj, wf_val_returns)

    walk_forward_results.append({
        'fold': i + 1,
        'train_end': train_end,
        'val_samples': len(wf_val_graphs),
        'strategy_return': strategy_metrics['return'],
        'strategy_dd': strategy_metrics['max_dd'],
        'strategy_sharpe': strategy_metrics['sharpe'],
        'bh_return': bh_metrics['return'],
        'bh_dd': bh_metrics['max_dd'],
        'beats_bh': strategy_metrics['sharpe'] > bh_metrics['sharpe'],
        'miss_rate': fold_miss_rate
    })

    print(f"  Fold {i+1}: Strategy Sharpe={strategy_metrics['sharpe']:.2f} vs B&H Sharpe={bh_metrics['sharpe']:.2f}, "
          f"Miss Rate={fold_miss_rate:.1%}")

wf_df = pd.DataFrame(walk_forward_results)
wf_win_rate = wf_df['beats_bh'].mean() * 100
wf_avg_miss_rate = wf_df['miss_rate'].mean()

print(f"\nWalk-Forward Summary:")
print(f"  Win Rate (vs B&H): {wf_win_rate:.0f}%")
print(f"  Avg Miss Rate: {wf_avg_miss_rate:.1%}")


# ============================================================
# 4. STRESS TESTING
# ============================================================
print("\n[4/4] Running Stress Tests on Crisis Periods...")

# For stress testing, we need to get predictions on the FULL dataset
# Use 30% train to ensure crisis periods are in test set
print("  Training model for stress test (30% train, 70% test)...")

stress_split = int(len(graphs) * 0.3)  # Earlier split to capture more crises
stress_detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
stress_detector.num_node_features = detector.num_node_features

stress_detector.train(
    graphs[:stress_split], targets[:stress_split],
    None, None,
    epochs=75, batch_size=32,
    train_returns=btc_returns[:stress_split],
    use_asymmetric_loss=True,
    miss_penalty=CONFIG['miss_penalty']
)

# Get predictions on remaining 50%
stress_test_graphs = graphs[stress_split:]
stress_preds, stress_probs = stress_detector.predict(stress_test_graphs)
stress_returns = btc_returns[stress_split:stress_split + len(stress_test_graphs)]
stress_dates = df.index[stress_split:stress_split + len(stress_test_graphs)]

# Define crisis periods
crisis_periods = {
    'COVID Crash': ('2020-02-15', '2020-03-31'),
    'May 2021 Crash': ('2021-05-10', '2021-05-25'),
    'Luna/3AC Crisis': ('2022-05-01', '2022-06-30'),
    'FTX Collapse': ('2022-11-01', '2022-11-30'),
}

stress_results = []

for name, (start, end) in crisis_periods.items():
    # Find indices for this period in stress test data
    mask = (stress_dates >= start) & (stress_dates <= end)

    if mask.sum() == 0:
        print(f"  Skipping {name} - not in test period")
        continue

    local_idx = np.where(mask)[0]
    period_returns = stress_returns[local_idx]
    period_probs = stress_probs[local_idx]

    # Calculate metrics
    period_positions = positions_from_probs(period_probs, CONFIG['risk_off_threshold'], CONFIG['position_weights'])
    strategy_metrics = backtest(period_returns, period_positions)
    bh_metrics = backtest(period_returns, np.ones(len(period_returns)))

    stress_results.append({
        'crisis': name,
        'days': len(local_idx),
        'strategy_return': strategy_metrics['return'],
        'strategy_dd': strategy_metrics['max_dd'],
        'bh_return': bh_metrics['return'],
        'bh_dd': bh_metrics['max_dd'],
        'beats_bh': strategy_metrics['max_dd'] < bh_metrics['max_dd']
    })

print("\nStress Test Results:")
print(f"{'Crisis':<20} {'Days':>6} {'Strategy':>10} {'B&H':>10} {'Protected?':>12}")
print("-" * 65)
for r in stress_results:
    protected = "YES" if r['beats_bh'] else "NO"
    print(f"{r['crisis']:<20} {r['days']:>6} {r['strategy_dd']:>9.1f}% {r['bh_dd']:>9.1f}% {protected:>12}")

stress_beat_rate = sum(r['beats_bh'] for r in stress_results) / len(stress_results) if stress_results else 0


# ============================================================
# FINAL ASSESSMENT
# ============================================================
print("\n" + "=" * 70)
print("FINAL ASSESSMENT")
print("=" * 70)

print("\nSuccess Criteria Check:")
print(f"  1. Miss Rate < 25%:                    {miss_rate:.1%} {'PASS' if miss_rate < 0.25 else 'FAIL'}")
print(f"  2. Monte Carlo 95th pct DD < 40%:      {percentiles['max_dd_95th']:.1f}% {'PASS' if percentiles['max_dd_95th'] < 40 else 'FAIL'}")
print(f"  3. Walk-Forward Win Rate > 45%:        {wf_win_rate:.0f}% {'PASS' if wf_win_rate > 45 else 'FAIL'}")
print(f"  4. Stress Test Beat Rate >= 75%:       {stress_beat_rate*100:.0f}% {'PASS' if stress_beat_rate >= 0.75 else 'FAIL'}")

all_pass = (
    miss_rate < 0.25 and
    percentiles['max_dd_95th'] < 40 and
    wf_win_rate > 45 and
    stress_beat_rate >= 0.75
)

if all_pass:
    print("\n>>> ALL CRITERIA PASSED - GNN improvements validated! <<<")
else:
    print("\n>>> SOME CRITERIA FAILED - Further improvements needed <<<")
    if miss_rate >= 0.25:
        print("  - Consider higher miss_penalty or SMOTE oversampling")
    if percentiles['max_dd_95th'] >= 40:
        print("  - Consider adding funding rates feature")
    if wf_win_rate <= 45:
        print("  - Consider GNN ensemble for stability")


# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'config': CONFIG,
    'single_path': {
        'miss_rate': float(miss_rate),
        'return': float(single_path_metrics['return']),
        'max_dd': float(single_path_metrics['max_dd']),
        'sharpe': float(single_path_metrics['sharpe'])
    },
    'bootstrap': {
        'return_ci': [float(percentiles['return_5th']), float(percentiles['return_95th'])],
        'max_dd_ci': [float(percentiles['max_dd_5th']), float(percentiles['max_dd_95th'])],
        'sharpe_median': float(percentiles['sharpe_50th'])
    },
    'walk_forward': {
        'win_rate': float(wf_win_rate),
        'avg_miss_rate': float(wf_avg_miss_rate),
        'folds': walk_forward_results
    },
    'stress_test': {
        'beat_rate': float(stress_beat_rate),
        'results': stress_results
    },
    'all_criteria_passed': all_pass
}

output_path = Path(__file__).parent.parent / 'config' / 'improved_gnn_validation.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")
print("=" * 70)
