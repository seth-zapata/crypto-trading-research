"""
Full sweep: miss_penalty AND threshold to find optimal configuration.
Target: 45-55% avg position, <30% DD 95th pctl.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

print("=" * 70)
print("FULL SWEEP: Penalty + Threshold Optimization")
print("=" * 70)

# Load data
df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

btc_returns = df['BTC_return_1d'].values
split = int(len(df) * 0.8)
val_returns = btc_returns[split:]

# Prepare graphs once
detector_base = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector_base.prepare_dataset(df, labels)
train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]
train_returns = btc_returns[:split]

def positions_from_probs(probs, threshold):
    """Position sizing with variable threshold."""
    positions = probs[:, 0] * 0.85 + probs[:, 1] * 0.65 + probs[:, 2] * 0.20
    positions[probs[:, 2] > threshold] = 0.20
    return np.clip(positions, 0.1, 1.0)

def backtest(returns, positions):
    equity = [1.0]
    for i, ret in enumerate(returns):
        equity.append(equity[-1] * (1 + ret * positions[i]))
    equity = np.array(equity)
    total_return = (equity[-1] / equity[0] - 1) * 100
    peak = np.maximum.accumulate(equity)
    max_dd = ((peak - equity) / peak).max() * 100
    return total_return, max_dd

def block_bootstrap(returns, positions, n_sims=500, block_size=20):
    n = len(returns)
    n_blocks = n // block_size + 1
    results = []
    for _ in range(n_sims):
        block_starts = np.random.randint(0, n - block_size, size=n_blocks)
        ret_sample, pos_sample = [], []
        for start in block_starts:
            ret_sample.extend(returns[start:start + block_size])
            pos_sample.extend(positions[start:start + block_size])
        ret_sample = np.array(ret_sample[:n])
        pos_sample = np.array(pos_sample[:n])
        total_ret, max_dd = backtest(ret_sample, pos_sample)
        results.append({'return': total_ret, 'max_dd': max_dd})
    return pd.DataFrame(results)

# Test combinations
penalties = [3, 5, 8, 10]
thresholds = [0.30, 0.40, 0.50, 0.60]

results = []

# First, train models for each penalty (reuse across thresholds)
print("\nTraining models...")
trained_models = {}

for penalty in penalties:
    print(f"  Training penalty={penalty}...")
    detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
    detector.num_node_features = detector_base.num_node_features

    detector.train(
        train_graphs, train_labels,
        val_graphs, val_labels,
        epochs=100, batch_size=32,
        train_returns=train_returns,
        use_asymmetric_loss=True,
        miss_penalty=penalty
    )

    _, probs = detector.predict(val_graphs)
    trained_models[penalty] = probs

print("\nEvaluating combinations...")

for penalty in penalties:
    probs = trained_models[penalty]

    for threshold in thresholds:
        positions = positions_from_probs(probs, threshold)

        # Calculate miss rate
        crash_mask = val_returns[:len(probs)] < -0.05
        preds = probs.argmax(axis=1)
        preds[probs[:, 2] > threshold] = 2
        n_crashes = crash_mask.sum()
        misses = ((preds == 0) & crash_mask).sum()
        miss_rate = misses / n_crashes if n_crashes > 0 else 0

        # Bootstrap (reduced for speed)
        bootstrap = block_bootstrap(val_returns[:len(positions)], positions)

        result = {
            'penalty': penalty,
            'threshold': threshold,
            'avg_position': positions.mean() * 100,
            'return_median': bootstrap['return'].quantile(0.50),
            'dd_95th': bootstrap['max_dd'].quantile(0.95),
            'miss_rate': miss_rate * 100
        }
        results.append(result)

# Summary table
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Penalty':>8} {'Thresh':>8} {'Avg Pos':>10} {'Ret Med':>10} {'DD 95th':>10} {'Miss %':>10} {'Target?':>10}")
print("-" * 75)

for r in sorted(results, key=lambda x: (x['penalty'], x['threshold'])):
    in_target = "YES" if 45 <= r['avg_position'] <= 55 and r['dd_95th'] < 30 else ""
    close = "*" if 40 <= r['avg_position'] <= 60 and r['dd_95th'] < 30 else ""
    print(f"{r['penalty']:>8} {r['threshold']:>8.2f} {r['avg_position']:>9.1f}% {r['return_median']:>9.1f}% "
          f"{r['dd_95th']:>9.1f}% {r['miss_rate']:>9.1f}% {in_target or close:>10}")

# Find best configuration
valid = [r for r in results if 45 <= r['avg_position'] <= 55 and r['dd_95th'] < 30]
if valid:
    best = max(valid, key=lambda x: x['return_median'])
    print(f"\n>>> OPTIMAL: penalty={best['penalty']}, threshold={best['threshold']:.2f} <<<")
else:
    # Relax to 40-60%
    valid = [r for r in results if 40 <= r['avg_position'] <= 60 and r['dd_95th'] < 30]
    if valid:
        best = max(valid, key=lambda x: x['return_median'])
        print(f"\n>>> BEST (40-60% range): penalty={best['penalty']}, threshold={best['threshold']:.2f} <<<")
    else:
        print("\n>>> No configuration meets criteria <<<")
        best = min(results, key=lambda x: abs(x['avg_position'] - 50) + (0 if r['dd_95th'] < 30 else 100))
        print(f"Closest: penalty={best['penalty']}, threshold={best['threshold']:.2f}")

print(f"\n    Avg Position: {best['avg_position']:.1f}%")
print(f"    Return Median: {best['return_median']:.1f}%")
print(f"    DD 95th pctl: {best['dd_95th']:.1f}%")
print(f"    Miss Rate: {best['miss_rate']:.1f}%")
