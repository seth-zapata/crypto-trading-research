"""
Tune Conservative Gate thresholds to achieve <25% miss rate
while maintaining >35% return and <25% DD.
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
print("TUNING CONSERVATIVE GATE THRESHOLDS")
print("=" * 70)

# Load and train (reuse from previous)
df, labels = build_regime_detection_features(start_date='2020-01-01', assets=['BTC', 'ETH', 'SOL'])
btc_returns = df['BTC_return_1d'].values
split = int(len(df) * 0.8)
val_returns = btc_returns[split:]

detector_base = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector_base.prepare_dataset(df, labels)
train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]
train_returns = btc_returns[:split]

print("\nTraining models...")
detector_cons = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
detector_cons.num_node_features = detector_base.num_node_features
detector_cons.train(train_graphs, train_labels, val_graphs, val_labels,
                   epochs=100, batch_size=32, train_returns=train_returns,
                   use_asymmetric_loss=True, miss_penalty=5.0)
_, probs_cons = detector_cons.predict(val_graphs)

detector_aggr = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
detector_aggr.num_node_features = detector_base.num_node_features
detector_aggr.train(train_graphs, train_labels, val_graphs, val_labels,
                   epochs=100, batch_size=32, train_returns=train_returns,
                   use_asymmetric_loss=True, miss_penalty=10.0)
_, probs_aggr = detector_aggr.predict(val_graphs)

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

def calc_miss_rate(positions, returns):
    crash_mask = returns < -0.05
    n_crashes = crash_mask.sum()
    # Miss = high position during crash
    misses = ((positions > 0.30) & crash_mask[:len(positions)]).sum()
    return misses / n_crashes if n_crashes > 0 else 0

# Sweep thresholds
print("\nSweeping thresholds...")

results = []
for ro_thresh in [0.20, 0.25, 0.30]:  # RISK_OFF threshold
    for caution_thresh in [0.30, 0.35, 0.40]:  # CAUTION threshold
        for agree_thresh in [0.40, 0.50, 0.60]:  # Both-agree threshold

            pos = np.ones(len(probs_cons))
            # Conservative exit triggers
            pos[probs_cons[:, 2] > ro_thresh] = 0.20
            pos[probs_cons[:, 1] > caution_thresh] = np.minimum(pos[probs_cons[:, 1] > caution_thresh], 0.50)
            # Only aggressive if both agree
            both_risk_on = (probs_cons[:, 0] > agree_thresh) & (probs_aggr[:, 0] > agree_thresh)
            pos[both_risk_on & (pos > 0.50)] = 0.85

            bootstrap = block_bootstrap(val_returns[:len(pos)], pos)
            miss_rate = calc_miss_rate(pos, val_returns[:len(pos)])

            results.append({
                'ro_thresh': ro_thresh,
                'caution_thresh': caution_thresh,
                'agree_thresh': agree_thresh,
                'avg_pos': pos.mean() * 100,
                'return_med': bootstrap['return'].median(),
                'dd_95': bootstrap['max_dd'].quantile(0.95),
                'miss_rate': miss_rate * 100
            })

# Find configurations meeting all targets
print("\n" + "=" * 80)
print("CONFIGURATIONS MEETING ALL TARGETS (Return>35%, DD<25%, Miss<25%)")
print("=" * 80)

valid = [r for r in results if r['return_med'] > 35 and r['dd_95'] < 25 and r['miss_rate'] < 25]

if valid:
    print(f"\n{'RO':>6} {'Caut':>6} {'Agree':>6} {'Pos':>8} {'Return':>10} {'DD95':>8} {'Miss%':>8}")
    print("-" * 60)
    for r in sorted(valid, key=lambda x: -x['return_med']):
        print(f"{r['ro_thresh']:>6.2f} {r['caution_thresh']:>6.2f} {r['agree_thresh']:>6.2f} "
              f"{r['avg_pos']:>7.1f}% {r['return_med']:>9.1f}% {r['dd_95']:>7.1f}% {r['miss_rate']:>7.1f}%")

    best = max(valid, key=lambda x: x['return_med'])
    print(f"\n>>> OPTIMAL CONFIG <<<")
    print(f"  RISK_OFF threshold: {best['ro_thresh']}")
    print(f"  CAUTION threshold: {best['caution_thresh']}")
    print(f"  Agree threshold: {best['agree_thresh']}")
    print(f"  Avg Position: {best['avg_pos']:.1f}%")
    print(f"  Return Median: {best['return_med']:.1f}%")
    print(f"  DD 95th: {best['dd_95']:.1f}%")
    print(f"  Miss Rate: {best['miss_rate']:.1f}%")
else:
    print("\nNo configuration meets all targets.")

    # Show closest
    print("\nClosest configurations (miss rate sorted):")
    close = [r for r in results if r['dd_95'] < 25]
    close = sorted(close, key=lambda x: x['miss_rate'])[:5]
    print(f"\n{'RO':>6} {'Caut':>6} {'Agree':>6} {'Pos':>8} {'Return':>10} {'DD95':>8} {'Miss%':>8}")
    print("-" * 60)
    for r in close:
        print(f"{r['ro_thresh']:>6.2f} {r['caution_thresh']:>6.2f} {r['agree_thresh']:>6.2f} "
              f"{r['avg_pos']:>7.1f}% {r['return_med']:>9.1f}% {r['dd_95']:>7.1f}% {r['miss_rate']:>7.1f}%")
