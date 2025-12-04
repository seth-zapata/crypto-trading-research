"""
Phase 5 Addendum: Position Weight Optimization
===============================================

Can we capture more upside without sacrificing crash protection?

Analysis:
1. GNN miss rate during crashes
2. Expanded grid search
3. Pareto frontier identification
4. Bull/bear sensitivity analysis
"""

import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

logging.basicConfig(level=logging.WARNING)
np.random.seed(42)

print("=" * 70)
print("PHASE 5 ADDENDUM: WEIGHT OPTIMIZATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA AND TRAIN GNN
# ============================================================
print("\n[1/6] Loading data and training GNN...")

df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector.prepare_dataset(df, labels)

history = detector.train(
    graphs, targets, graphs, targets,
    epochs=100, batch_size=32,
    class_weights=[1.0, 2.0, 8.0]
)

preds, probs = detector.predict(graphs)

# Align data
graph_start = len(df) - len(graphs)
aligned_df = df.iloc[graph_start:].reset_index(drop=True)

# Use full dataset for analysis (not just validation)
prices = aligned_df['BTC_close'].values
returns = np.diff(prices) / prices[:-1]

RISK_ON, CAUTION, RISK_OFF = 0, 1, 2

print(f"Full dataset: {len(aligned_df)} days")


# ============================================================
# 2. GNN MISS ANALYSIS
# ============================================================
print("\n[2/6] Analyzing GNN miss rate during crashes...")

def find_crashes(prices, threshold=-0.10, lookforward=5):
    """Find periods where price dropped >threshold in next N days."""
    crashes = []
    for i in range(len(prices) - lookforward):
        future_return = (prices[i + lookforward] - prices[i]) / prices[i]
        if future_return < threshold:
            crashes.append({
                'idx': i,
                'drop': future_return * 100
            })
    return crashes

crashes = find_crashes(prices, threshold=-0.10, lookforward=5)
print(f"\nTotal crashes (>10% drop in 5 days): {len(crashes)}")

# Analyze GNN state during crashes
gnn_states_during_crashes = []
for crash in crashes:
    idx = crash['idx']
    if idx < len(preds):
        pred = preds[idx]
        prob = probs[idx]
        gnn_states_during_crashes.append({
            'idx': idx,
            'drop': crash['drop'],
            'prediction': detector.REGIME_LABELS[pred],
            'risk_on_prob': prob[RISK_ON],
            'caution_prob': prob[CAUTION],
            'risk_off_prob': prob[RISK_OFF]
        })

crash_df = pd.DataFrame(gnn_states_during_crashes)

# Count by prediction
print("\nGNN state at crash start:")
state_counts = crash_df['prediction'].value_counts()
for state, count in state_counts.items():
    pct = count / len(crash_df) * 100
    print(f"  {state}: {count} ({pct:.1f}%)")

# Miss rate = was in RISK_ON during crash
misses = crash_df[crash_df['prediction'] == 'RISK_ON']
miss_rate = len(misses) / len(crash_df) * 100 if len(crash_df) > 0 else 0

print(f"\nGNN MISS RATE: {miss_rate:.1f}%")
print(f"  Detected (CAUTION/RISK_OFF): {len(crash_df) - len(misses)}")
print(f"  Missed (RISK_ON): {len(misses)}")

if miss_rate < 20:
    print("\n→ Low miss rate: More aggressive weights are SAFER")
elif miss_rate < 30:
    print("\n→ Moderate miss rate: Current conservative weights are reasonable")
else:
    print("\n→ High miss rate: Conservative weights are JUSTIFIED")


# ============================================================
# 3. EXPANDED GRID SEARCH
# ============================================================
print("\n[3/6] Running expanded grid search...")

def backtest(prices, positions, cost=0.001):
    """Run backtest."""
    equity = [1.0]
    prev_pos = positions[0]
    rets = np.diff(prices) / prices[:-1]

    for i, ret in enumerate(rets):
        pos = positions[i] if i < len(positions) else prev_pos
        port_ret = ret * pos - abs(pos - prev_pos) * cost
        equity.append(equity[-1] * (1 + port_ret))
        prev_pos = pos

    eq = np.array(equity)
    daily_rets = np.diff(eq) / eq[:-1]
    total_ret = (eq[-1] - 1) * 100
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    peak = np.maximum.accumulate(eq)
    max_dd = ((peak - eq) / peak).max() * 100

    return {'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd, 'equity': eq}


def continuous_position(probs, risk_on, caution, risk_off):
    """Calculate continuous position."""
    return probs[:, 0] * risk_on + probs[:, 1] * caution + probs[:, 2] * risk_off


# Expanded grid
risk_on_levels = [0.80, 0.85, 0.90, 0.95, 1.00]
caution_levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
risk_off_levels = [0.15, 0.20, 0.25, 0.30, 0.35]

results = []
for ro, ca, rf in product(risk_on_levels, caution_levels, risk_off_levels):
    positions = continuous_position(probs[:len(prices)-1], ro, ca, rf)
    metrics = backtest(prices, positions)
    results.append({
        'risk_on': ro, 'caution': ca, 'risk_off': rf,
        'avg_pos': positions.mean(),
        **metrics
    })

results_df = pd.DataFrame(results)
print(f"Total configurations tested: {len(results_df)}")

# Filter valid (DD < 25%)
valid = results_df[results_df['max_dd'] < 25].copy()
print(f"Valid configurations (DD < 25%): {len(valid)}")


# ============================================================
# 4. PARETO FRONTIER
# ============================================================
print("\n[4/6] Finding Pareto frontier...")

def is_pareto_optimal(row, df):
    """Check if config is Pareto optimal (not dominated)."""
    # A config is dominated if another has better return AND better DD
    dominated = df[
        (df['return'] >= row['return']) &
        (df['max_dd'] <= row['max_dd']) &
        ((df['return'] > row['return']) | (df['max_dd'] < row['max_dd']))
    ]
    return len(dominated) == 0

# If no valid configs with DD < 25%, relax threshold
if len(valid) == 0:
    print("No configs with DD < 25%. Relaxing to < 30%...")
    valid = results_df[results_df['max_dd'] < 30].copy()
    print(f"Valid configurations (DD < 30%): {len(valid)}")

if len(valid) == 0:
    print("Still no valid configs. Using top 20 by DD...")
    valid = results_df.nsmallest(20, 'max_dd').copy()

valid['pareto'] = valid.apply(lambda r: is_pareto_optimal(r, valid), axis=1)
pareto = valid[valid['pareto']].sort_values('return', ascending=False)

print(f"\nPareto optimal configurations: {len(pareto)}")
print("\nPareto Frontier (sorted by return):")
print(f"{'RISK_ON':>8} {'CAUTION':>8} {'RISK_OFF':>9} {'Return':>10} {'Sharpe':>8} {'Max DD':>10} {'Avg Pos':>10}")
print("-" * 75)
for _, row in pareto.iterrows():
    print(f"{row['risk_on']:>7.0%} {row['caution']:>8.0%} {row['risk_off']:>9.0%} "
          f"{row['return']:>+9.1f}% {row['sharpe']:>8.2f} {row['max_dd']:>9.1f}% {row['avg_pos']:>9.1%}")


# ============================================================
# 5. BULL/BEAR SENSITIVITY ANALYSIS
# ============================================================
print("\n[5/6] Bull/Bear sensitivity analysis...")

# Define periods
# Bull: 2023 (roughly index 1095 to 1460 in our dataset)
# Bear: 2022 (roughly index 730 to 1095)

# Find approximate indices
dates = aligned_df.index
total_days = len(dates)

# Rough splits based on 2020-2024 data
# 2020: 0-365, 2021: 365-730, 2022: 730-1095, 2023: 1095-1460, 2024: 1460+
bear_start, bear_end = int(total_days * 0.36), int(total_days * 0.55)  # 2022-ish
bull_start, bull_end = int(total_days * 0.55), int(total_days * 0.73)  # 2023-ish

bear_prices = prices[bear_start:bear_end]
bear_probs = probs[bear_start:bear_end-1]
bull_prices = prices[bull_start:bull_end]
bull_probs = probs[bull_start:bull_end-1]

print(f"\nBear period: {bear_end - bear_start} days")
print(f"Bull period: {bull_end - bull_start} days")

# Test top configurations in both periods
test_configs = [
    (0.85, 0.65, 0.30, "Current"),
    (1.00, 0.75, 0.25, "Aggressive"),
    (0.90, 0.70, 0.25, "Balanced"),
    (0.80, 0.60, 0.35, "Defensive"),
]

if len(pareto) > 0:
    best_pareto = pareto.iloc[0]
    test_configs.append((best_pareto['risk_on'], best_pareto['caution'],
                         best_pareto['risk_off'], "Best Pareto"))

print("\nBull Market Performance:")
print(f"{'Config':<15} {'Return':>10} {'Max DD':>10}")
print("-" * 40)
for ro, ca, rf, name in test_configs:
    if len(bull_probs) > 0:
        pos = continuous_position(bull_probs, ro, ca, rf)
        metrics = backtest(bull_prices, pos)
        print(f"{name:<15} {metrics['return']:>+9.1f}% {metrics['max_dd']:>9.1f}%")

print("\nBear Market Performance:")
print(f"{'Config':<15} {'Return':>10} {'Max DD':>10}")
print("-" * 40)
for ro, ca, rf, name in test_configs:
    if len(bear_probs) > 0:
        pos = continuous_position(bear_probs, ro, ca, rf)
        metrics = backtest(bear_prices, pos)
        print(f"{name:<15} {metrics['return']:>+9.1f}% {metrics['max_dd']:>9.1f}%")


# ============================================================
# 6. RECOMMENDATION
# ============================================================
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Find best config: highest return with lowest DD
if len(valid) > 0:
    best = valid.loc[valid['return'].idxmax()]
    best_sharpe = valid.loc[valid['sharpe'].idxmax()]
else:
    best = results_df.loc[results_df['max_dd'].idxmin()]
    best_sharpe = best

print(f"\nCurrent config: (0.85, 0.65, 0.30)")
print(f"  Return: {results_df[(results_df['risk_on']==0.85) & (results_df['caution']==0.65) & (results_df['risk_off']==0.30)]['return'].values[0]:+.1f}%")
print(f"  Max DD: {results_df[(results_df['risk_on']==0.85) & (results_df['caution']==0.65) & (results_df['risk_off']==0.30)]['max_dd'].values[0]:.1f}%")

print(f"\nBest by Return (DD < 25%): ({best['risk_on']:.2f}, {best['caution']:.2f}, {best['risk_off']:.2f})")
print(f"  Return: {best['return']:+.1f}%")
print(f"  Max DD: {best['max_dd']:.1f}%")
print(f"  Avg Pos: {best['avg_pos']:.1%}")

print(f"\nBest by Sharpe (DD < 25%): ({best_sharpe['risk_on']:.2f}, {best_sharpe['caution']:.2f}, {best_sharpe['risk_off']:.2f})")
print(f"  Return: {best_sharpe['return']:+.1f}%")
print(f"  Sharpe: {best_sharpe['sharpe']:.2f}")
print(f"  Max DD: {best_sharpe['max_dd']:.1f}%")

# Decision based on miss rate
print("\n" + "-" * 70)
if miss_rate < 25:
    print("RECOMMENDATION: Use more aggressive weights")
    rec = best
    print(f"\nRecommended: ({rec['risk_on']:.2f}, {rec['caution']:.2f}, {rec['risk_off']:.2f})")
    print(f"\nRationale:")
    print(f"  - GNN miss rate is {miss_rate:.1f}% (low)")
    print(f"  - GNN detects {100-miss_rate:.1f}% of crashes")
    print(f"  - Safe to increase exposure during RISK_ON periods")
else:
    print("RECOMMENDATION: Keep conservative weights")
    rec = best_sharpe
    print(f"\nRecommended: ({rec['risk_on']:.2f}, {rec['caution']:.2f}, {rec['risk_off']:.2f})")
    print(f"\nRationale:")
    print(f"  - GNN miss rate is {miss_rate:.1f}% (high)")
    print(f"  - Conservative weights protect against GNN failures")

print("\n" + "=" * 70)

# Save recommendation
import json
config = {
    'risk_on_weight': float(rec['risk_on']),
    'caution_weight': float(rec['caution']),
    'risk_off_weight': float(rec['risk_off']),
    'expected_return': float(rec['return']),
    'expected_sharpe': float(rec['sharpe']),
    'expected_max_dd': float(rec['max_dd']),
    'gnn_miss_rate': float(miss_rate)
}

config_path = Path(__file__).parent.parent / 'config' / 'position_sizing.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"\nConfiguration saved to: {config_path}")


# ============================================================
# 7. VISUALIZATION
# ============================================================
print("\n[6/6] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Return vs Drawdown scatter (Pareto frontier)
ax = axes[0, 0]
scatter = ax.scatter(
    valid['return'], valid['max_dd'],
    c=valid['avg_pos'], cmap='RdYlGn',
    alpha=0.6, s=30
)
ax.scatter(
    pareto['return'], pareto['max_dd'],
    color='red', s=100, marker='*',
    label='Pareto Optimal', zorder=5
)
# Mark current and recommended
current = results_df[(results_df['risk_on']==0.85) &
                      (results_df['caution']==0.65) &
                      (results_df['risk_off']==0.30)].iloc[0]
ax.scatter([current['return']], [current['max_dd']],
           color='blue', s=150, marker='s', label='Current', zorder=6)
ax.scatter([rec['return']], [rec['max_dd']],
           color='green', s=150, marker='^', label='Recommended', zorder=6)
ax.axhline(25, color='red', linestyle='--', alpha=0.5, label='DD Limit')
ax.set_xlabel('Return (%)')
ax.set_ylabel('Max Drawdown (%)')
ax.set_title('Return vs Drawdown Tradeoff')
ax.legend(loc='upper left')
plt.colorbar(scatter, ax=ax, label='Avg Position')

# 2. GNN miss analysis
ax = axes[0, 1]
if len(crash_df) > 0:
    labels_pie = crash_df['prediction'].value_counts().index.tolist()
    sizes = crash_df['prediction'].value_counts().values.tolist()
    colors = {'RISK_ON': 'red', 'CAUTION': 'orange', 'RISK_OFF': 'green'}
    pie_colors = [colors.get(l, 'gray') for l in labels_pie]
    ax.pie(sizes, labels=labels_pie, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'GNN State During {len(crash_df)} Crashes\n(Miss Rate: {miss_rate:.1f}%)')
else:
    ax.text(0.5, 0.5, 'No crashes in period', ha='center', va='center')
    ax.set_title('GNN State During Crashes')

# 3. Weight sensitivity heatmap (RISK_ON vs CAUTION, fixed RISK_OFF)
ax = axes[1, 0]
fixed_rf = 0.25
subset = results_df[results_df['risk_off'] == fixed_rf].copy()
pivot = subset.pivot_table(values='return', index='caution', columns='risk_on')
im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels([f'{x:.0%}' for x in pivot.columns])
ax.set_yticklabels([f'{x:.0%}' for x in pivot.index])
ax.set_xlabel('RISK_ON Weight')
ax.set_ylabel('CAUTION Weight')
ax.set_title(f'Return (%) by Weights\n(RISK_OFF = {fixed_rf:.0%})')
plt.colorbar(im, ax=ax)

# 4. Position distribution comparison
ax = axes[1, 1]
current_pos = continuous_position(probs[:len(prices)-1], 0.85, 0.65, 0.30)
rec_pos = continuous_position(probs[:len(prices)-1], rec['risk_on'], rec['caution'], rec['risk_off'])
ax.hist(current_pos, bins=20, alpha=0.5, label=f'Current (avg: {current_pos.mean():.1%})')
ax.hist(rec_pos, bins=20, alpha=0.5, label=f'Recommended (avg: {rec_pos.mean():.1%})')
ax.set_xlabel('Position Size')
ax.set_ylabel('Frequency')
ax.set_title('Position Distribution Comparison')
ax.legend()

plt.tight_layout()
output_path = Path(__file__).parent.parent / 'notebooks' / 'phase5' / 'weight_optimization.png'
plt.savefig(output_path, dpi=150)
print(f"Visualization saved to: {output_path}")

plt.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
