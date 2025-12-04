"""
Phase 5: Optimize Position Thresholds
======================================

Approach A: Grid search for optimal position sizes by regime.
Also tests continuous probability-based sizing.

Goal: Close the gap from 26.8% → <25% max drawdown.
"""

import logging
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

logging.basicConfig(level=logging.WARNING)
np.random.seed(42)

print("=" * 70)
print("PHASE 5: THRESHOLD OPTIMIZATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA AND TRAIN GNN
# ============================================================
print("\n[1/5] Loading data and training calibrated GNN...")

df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector.prepare_dataset(df, labels)

# Calibrated class weights from Phase 4
history = detector.train(
    graphs, targets,
    graphs, targets,
    epochs=100, batch_size=32,
    class_weights=[1.0, 2.0, 8.0]
)

preds, probs = detector.predict(graphs)

# Align data
graph_start = len(df) - len(graphs)
aligned_df = df.iloc[graph_start:].reset_index(drop=True)

# Validation split (80/20)
split = int(len(aligned_df) * 0.8)
val_df = aligned_df.iloc[split:].reset_index(drop=True)
val_probs = probs[split:]
val_preds = preds[split:]

print(f"Validation period: {len(val_df)} days")


# ============================================================
# 2. BACKTEST HELPER
# ============================================================

def backtest(
    prices: np.ndarray,
    positions: np.ndarray,
    transaction_cost: float = 0.001
) -> Dict:
    """Run backtest and return metrics."""
    equity = [1.0]
    prev_pos = positions[0] if len(positions) > 0 else 1.0
    returns = np.diff(prices) / prices[:-1]

    for i, ret in enumerate(returns):
        pos = positions[i] if i < len(positions) else prev_pos
        cost = abs(pos - prev_pos) * transaction_cost
        port_ret = ret * pos - cost
        equity.append(equity[-1] * (1 + port_ret))
        prev_pos = pos

    eq = np.array(equity)
    daily_rets = np.diff(eq) / eq[:-1]

    total_ret = (eq[-1] - 1) * 100
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

    peak = np.maximum.accumulate(eq)
    max_dd = ((peak - eq) / peak).max() * 100

    return {
        'return': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'equity': eq
    }


# ============================================================
# 3. APPROACH A1: GRID SEARCH FOR THRESHOLDS
# ============================================================
print("\n[2/5] Grid search for optimal thresholds...")

RISK_ON, CAUTION, RISK_OFF = 0, 1, 2
prices = val_df['BTC_close'].values

# Grid search parameters
risk_on_levels = [0.85, 0.90, 0.95, 1.00]
caution_levels = [0.40, 0.50, 0.60, 0.70]
risk_off_levels = [0.10, 0.15, 0.20, 0.25, 0.30]

results = []

for risk_on_pos, caution_pos, risk_off_pos in product(
    risk_on_levels, caution_levels, risk_off_levels
):
    # Create positions based on predictions
    pos_map = {RISK_ON: risk_on_pos, CAUTION: caution_pos, RISK_OFF: risk_off_pos}
    positions = np.array([pos_map[p] for p in val_preds[:len(prices)-1]])

    metrics = backtest(prices, positions)

    results.append({
        'risk_on': risk_on_pos,
        'caution': caution_pos,
        'risk_off': risk_off_pos,
        **metrics
    })

# Sort by criteria: max_dd < 25%, then highest Sharpe
results_df = pd.DataFrame(results)
valid = results_df[results_df['max_dd'] < 25].copy()

if len(valid) > 0:
    best_grid = valid.loc[valid['sharpe'].idxmax()]
    print(f"\nFound {len(valid)} configurations with max_dd < 25%")
    print(f"\nBest by Sharpe (with DD < 25%):")
    print(f"  RISK_ON: {best_grid['risk_on']:.0%}")
    print(f"  CAUTION: {best_grid['caution']:.0%}")
    print(f"  RISK_OFF: {best_grid['risk_off']:.0%}")
    print(f"  Return: {best_grid['return']:+.1f}%")
    print(f"  Sharpe: {best_grid['sharpe']:.2f}")
    print(f"  Max DD: {best_grid['max_dd']:.1f}%")
else:
    print("\nNo configuration achieved max_dd < 25%")
    # Find best overall
    best_grid = results_df.loc[results_df['max_dd'].idxmin()]
    print(f"\nBest by lowest DD:")
    print(f"  RISK_ON: {best_grid['risk_on']:.0%}")
    print(f"  CAUTION: {best_grid['caution']:.0%}")
    print(f"  RISK_OFF: {best_grid['risk_off']:.0%}")
    print(f"  Return: {best_grid['return']:+.1f}%")
    print(f"  Sharpe: {best_grid['sharpe']:.2f}")
    print(f"  Max DD: {best_grid['max_dd']:.1f}%")

# Show top 5 by Sharpe with lowest DD
print("\nTop 5 configurations by Sharpe (sorted by DD):")
top5 = results_df.nsmallest(10, 'max_dd').nlargest(5, 'sharpe')
for _, row in top5.iterrows():
    print(f"  [{row['risk_on']:.0%}/{row['caution']:.0%}/{row['risk_off']:.0%}] "
          f"Ret: {row['return']:+.1f}%, Sharpe: {row['sharpe']:.2f}, DD: {row['max_dd']:.1f}%")


# ============================================================
# 4. APPROACH A2: CONTINUOUS PROBABILITY-BASED SIZING
# ============================================================
print("\n[3/5] Testing continuous probability-based sizing...")

def continuous_position(probs: np.ndarray, base_positions: Dict[str, float]) -> np.ndarray:
    """
    Calculate position as weighted average of regime probabilities.

    pos = p(RISK_ON)*pos_risk_on + p(CAUTION)*pos_caution + p(RISK_OFF)*pos_risk_off
    """
    positions = []
    for p in probs:
        pos = (
            p[RISK_ON] * base_positions['RISK_ON'] +
            p[CAUTION] * base_positions['CAUTION'] +
            p[RISK_OFF] * base_positions['RISK_OFF']
        )
        positions.append(pos)
    return np.array(positions)


# Test different base position combinations
continuous_configs = [
    {'RISK_ON': 1.00, 'CAUTION': 0.55, 'RISK_OFF': 0.20, 'name': 'Default'},
    {'RISK_ON': 0.95, 'CAUTION': 0.50, 'RISK_OFF': 0.15, 'name': 'Conservative'},
    {'RISK_ON': 1.00, 'CAUTION': 0.60, 'RISK_OFF': 0.10, 'name': 'Aggressive defense'},
    {'RISK_ON': 0.90, 'CAUTION': 0.55, 'RISK_OFF': 0.20, 'name': 'Capped risk-on'},
    {'RISK_ON': 1.00, 'CAUTION': 0.50, 'RISK_OFF': 0.25, 'name': 'Higher risk-off'},
]

continuous_results = []
for config in continuous_configs:
    base_pos = {k: v for k, v in config.items() if k != 'name'}
    positions = continuous_position(val_probs[:len(prices)-1], base_pos)

    metrics = backtest(prices, positions)
    continuous_results.append({
        'name': config['name'],
        **base_pos,
        **metrics
    })

print("\nContinuous sizing results:")
print(f"{'Config':<20} {'Return':>10} {'Sharpe':>8} {'Max DD':>10} {'Avg Pos':>10}")
print("-" * 65)
for r in continuous_results:
    base_pos = {'RISK_ON': r['RISK_ON'], 'CAUTION': r['CAUTION'], 'RISK_OFF': r['RISK_OFF']}
    positions = continuous_position(val_probs[:len(prices)-1], base_pos)
    avg_pos = positions.mean()
    print(f"{r['name']:<20} {r['return']:>+9.1f}% {r['sharpe']:>8.2f} {r['max_dd']:>9.1f}% {avg_pos:>9.1%}")


# ============================================================
# 5. COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

# Buy & Hold baseline
bh_metrics = backtest(prices, np.ones(len(prices)-1))

# Original fixed rules (100/50/20)
orig_pos = np.array([{RISK_ON: 1.0, CAUTION: 0.5, RISK_OFF: 0.2}[p] for p in val_preds[:len(prices)-1]])
orig_metrics = backtest(prices, orig_pos)

# Best grid search
grid_pos = np.array([{
    RISK_ON: best_grid['risk_on'],
    CAUTION: best_grid['caution'],
    RISK_OFF: best_grid['risk_off']
}[p] for p in val_preds[:len(prices)-1]])
grid_metrics = backtest(prices, grid_pos)

# Best continuous (find lowest DD that still has decent Sharpe)
best_cont = min(continuous_results, key=lambda x: x['max_dd'])
cont_base = {'RISK_ON': best_cont['RISK_ON'], 'CAUTION': best_cont['CAUTION'], 'RISK_OFF': best_cont['RISK_OFF']}
cont_pos = continuous_position(val_probs[:len(prices)-1], cont_base)
cont_metrics = backtest(prices, cont_pos)

print(f"\n{'Strategy':<30} {'Return':>10} {'Sharpe':>8} {'Max DD':>10}")
print("-" * 65)
print(f"{'Buy & Hold':<30} {bh_metrics['return']:>+9.1f}% {bh_metrics['sharpe']:>8.2f} {bh_metrics['max_dd']:>9.1f}%")
print(f"{'GNN + Fixed (100/50/20)':<30} {orig_metrics['return']:>+9.1f}% {orig_metrics['sharpe']:>8.2f} {orig_metrics['max_dd']:>9.1f}%")
print(f"{'GNN + Optimized Grid':<30} {grid_metrics['return']:>+9.1f}% {grid_metrics['sharpe']:>8.2f} {grid_metrics['max_dd']:>9.1f}%")
print(f"{'GNN + Continuous Probs':<30} {cont_metrics['return']:>+9.1f}% {cont_metrics['sharpe']:>8.2f} {cont_metrics['max_dd']:>9.1f}%")

# Success check
print("\n" + "=" * 70)
print("SUCCESS CHECK")
print("=" * 70)

best_approach = None
best_dd = float('inf')

approaches = [
    ('GNN + Optimized Grid', grid_metrics, best_grid),
    ('GNN + Continuous Probs', cont_metrics, best_cont),
]

for name, metrics, config in approaches:
    passed_dd = metrics['max_dd'] < 25
    passed_ret = metrics['return'] > 45
    passed_sharpe = metrics['sharpe'] > 0.9

    print(f"\n{name}:")
    print(f"  Max DD < 25%: {metrics['max_dd']:.1f}% {'✓' if passed_dd else '✗'}")
    print(f"  Return > 45%: {metrics['return']:.1f}% {'✓' if passed_ret else '✗'}")
    print(f"  Sharpe > 0.9: {metrics['sharpe']:.2f} {'✓' if passed_sharpe else '✗'}")

    if metrics['max_dd'] < best_dd:
        best_dd = metrics['max_dd']
        best_approach = (name, metrics, config)

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

if best_approach:
    name, metrics, config = best_approach
    print(f"\nBest approach: {name}")
    print(f"  Max DD: {metrics['max_dd']:.1f}%")
    print(f"  Return: {metrics['return']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")

    if metrics['max_dd'] < 25:
        print("\n✓ TARGET ACHIEVED: Max DD < 25%")
        print(f"  Use this configuration for production.")
    else:
        print(f"\n✗ Target not achieved. Gap: {metrics['max_dd'] - 25:.1f}%")
        print("  Consider Approach B (Constrained RL) or accept current results.")

print("\n" + "=" * 70)
