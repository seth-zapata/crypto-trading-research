"""
Revised Ensemble: Conservative voting approach
- Use penalty=5 (conservative) as PRIMARY for crash detection
- Use penalty=10 (aggressive) only to CONFIRM bullish signals
- Exit-first logic: any defensive signal reduces position immediately

Target: >35% return, <25% miss rate, <25% DD (95th pctl)
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
print("CONSERVATIVE ENSEMBLE: Exit-First Logic")
print("=" * 70)

# Load data
df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

btc_returns = df['BTC_return_1d'].values
split = int(len(df) * 0.8)
val_returns = btc_returns[split:]

# Prepare graphs
detector_base = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector_base.prepare_dataset(df, labels)
train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]
train_returns = btc_returns[:split]

# Train both models
print("\n[1/3] Training models...")

detector_conservative = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
detector_conservative.num_node_features = detector_base.num_node_features
detector_conservative.train(
    train_graphs, train_labels, val_graphs, val_labels,
    epochs=100, batch_size=32, train_returns=train_returns,
    use_asymmetric_loss=True, miss_penalty=5.0
)
_, probs_cons = detector_conservative.predict(val_graphs)

detector_aggressive = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
detector_aggressive.num_node_features = detector_base.num_node_features
detector_aggressive.train(
    train_graphs, train_labels, val_graphs, val_labels,
    epochs=100, batch_size=32, train_returns=train_returns,
    use_asymmetric_loss=True, miss_penalty=10.0
)
_, probs_aggr = detector_aggressive.predict(val_graphs)

# ============================================================
# Conservative Ensemble: Multiple strategies
# ============================================================
print("\n[2/3] Testing ensemble strategies...")

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

def calc_miss_rate(positions, returns, threshold=0.30):
    crash_mask = returns < -0.05
    n_crashes = crash_mask.sum()
    # Consider it a "miss" if position > threshold during crash
    misses = ((positions > threshold) & crash_mask[:len(positions)]).sum()
    return misses / n_crashes if n_crashes > 0 else 0

strategies = {}

# Strategy 1: Average probabilities
print("  Testing: Average probabilities...")
avg_probs = (probs_cons + probs_aggr) / 2
pos_avg = avg_probs[:, 0] * 0.85 + avg_probs[:, 1] * 0.65 + avg_probs[:, 2] * 0.20
pos_avg[avg_probs[:, 2] > 0.40] = 0.20
pos_avg = np.clip(pos_avg, 0.1, 1.0)
strategies['Avg Probs'] = pos_avg

# Strategy 2: Max RISK_OFF (conservative)
print("  Testing: Max RISK_OFF...")
max_risk_off = np.maximum(probs_cons[:, 2], probs_aggr[:, 2])
pos_max_ro = 1.0 - max_risk_off * 0.80  # Scale down by max RISK_OFF prob
pos_max_ro = np.clip(pos_max_ro, 0.20, 0.85)
strategies['Max RISK_OFF'] = pos_max_ro

# Strategy 3: Conservative gate (use p=5 for exit signals)
print("  Testing: Conservative gate...")
pos_gate = np.ones(len(probs_cons))
# Conservative model triggers exit
pos_gate[probs_cons[:, 2] > 0.30] = 0.20  # Lower threshold for conservative
pos_gate[probs_cons[:, 1] > 0.40] = 0.50  # CAUTION
# Only go aggressive if both agree
both_risk_on = (probs_cons[:, 0] > 0.50) & (probs_aggr[:, 0] > 0.50)
pos_gate[both_risk_on & (pos_gate > 0.50)] = 0.85
strategies['Conservative Gate'] = pos_gate

# Strategy 4: Weighted average (more weight on conservative)
print("  Testing: Weighted average (70/30)...")
weighted_probs = 0.7 * probs_cons + 0.3 * probs_aggr
pos_weighted = weighted_probs[:, 0] * 0.85 + weighted_probs[:, 1] * 0.60 + weighted_probs[:, 2] * 0.20
pos_weighted[weighted_probs[:, 2] > 0.35] = 0.20
pos_weighted = np.clip(pos_weighted, 0.15, 0.85)
strategies['Weighted 70/30'] = pos_weighted

# Strategy 5: Min position (most conservative at each point)
print("  Testing: Min position...")
pos_cons_solo = probs_cons[:, 0] * 0.85 + probs_cons[:, 1] * 0.65 + probs_cons[:, 2] * 0.20
pos_cons_solo[probs_cons[:, 2] > 0.40] = 0.20
pos_aggr_solo = probs_aggr[:, 0] * 0.85 + probs_aggr[:, 1] * 0.65 + probs_aggr[:, 2] * 0.20
pos_aggr_solo[probs_aggr[:, 2] > 0.40] = 0.20
pos_min = np.minimum(pos_cons_solo, pos_aggr_solo)
pos_min = np.clip(pos_min, 0.15, 0.85)
strategies['Min Position'] = pos_min

# ============================================================
# Evaluate all strategies
# ============================================================
print("\n[3/3] Evaluating strategies...")

results = []
for name, positions in strategies.items():
    bootstrap = block_bootstrap(val_returns[:len(positions)], positions)
    miss_rate = calc_miss_rate(positions, val_returns[:len(positions)])

    result = {
        'strategy': name,
        'avg_pos': positions.mean() * 100,
        'return_med': bootstrap['return'].median(),
        'dd_95': bootstrap['max_dd'].quantile(0.95),
        'miss_rate': miss_rate * 100
    }
    results.append(result)

# Also add single models for comparison
for name, probs in [('Single p=5', probs_cons), ('Single p=10', probs_aggr)]:
    positions = probs[:, 0] * 0.85 + probs[:, 1] * 0.65 + probs[:, 2] * 0.20
    positions[probs[:, 2] > 0.40] = 0.20
    positions = np.clip(positions, 0.1, 1.0)
    bootstrap = block_bootstrap(val_returns[:len(positions)], positions)
    miss_rate = calc_miss_rate(positions, val_returns[:len(positions)])
    results.append({
        'strategy': name,
        'avg_pos': positions.mean() * 100,
        'return_med': bootstrap['return'].median(),
        'dd_95': bootstrap['max_dd'].quantile(0.95),
        'miss_rate': miss_rate * 100
    })

# ============================================================
# Results
# ============================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Strategy':<20} {'Avg Pos':>10} {'Ret Med':>12} {'DD 95th':>10} {'Miss %':>10} {'Targets?':>10}")
print("-" * 75)

for r in sorted(results, key=lambda x: -x['return_med']):
    meets = "YES" if r['return_med'] > 35 and r['dd_95'] < 25 and r['miss_rate'] < 25 else ""
    close = "*" if r['dd_95'] < 25 and r['miss_rate'] < 25 else ""
    print(f"{r['strategy']:<20} {r['avg_pos']:>9.1f}% {r['return_med']:>11.1f}% "
          f"{r['dd_95']:>9.1f}% {r['miss_rate']:>9.1f}% {meets or close:>10}")

# Find best
valid = [r for r in results if r['return_med'] > 35 and r['dd_95'] < 25 and r['miss_rate'] < 25]
if valid:
    best = max(valid, key=lambda x: x['return_med'])
    print(f"\n>>> OPTIMAL: {best['strategy']} <<<")
else:
    # Relax targets
    valid2 = [r for r in results if r['dd_95'] < 25 and r['miss_rate'] < 30]
    if valid2:
        best = max(valid2, key=lambda x: x['return_med'])
        print(f"\n>>> BEST (relaxed targets): {best['strategy']} <<<")
    else:
        best = max(results, key=lambda x: x['return_med'])
        print(f"\n>>> BEST by return: {best['strategy']} <<<")

print(f"\n    Avg Position: {best['avg_pos']:.1f}%")
print(f"    Return Median: {best['return_med']:.1f}%")
print(f"    DD 95th pctl: {best['dd_95']:.1f}%")
print(f"    Miss Rate: {best['miss_rate']:.1f}%")
