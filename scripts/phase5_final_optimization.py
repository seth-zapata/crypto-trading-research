"""
Phase 5: Final Continuous Sizing Optimization
==============================================

Grid search for optimal continuous probability-based position sizing.
"""

import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

logging.basicConfig(level=logging.WARNING)
np.random.seed(42)

print("=" * 70)
print("PHASE 5: FINAL CONTINUOUS SIZING OPTIMIZATION")
print("=" * 70)


# Load and train
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

graph_start = len(df) - len(graphs)
aligned_df = df.iloc[graph_start:].reset_index(drop=True)

split = int(len(aligned_df) * 0.8)
val_df = aligned_df.iloc[split:].reset_index(drop=True)
val_probs = probs[split:]

prices = val_df['BTC_close'].values
RISK_ON, CAUTION, RISK_OFF = 0, 1, 2


def backtest(prices, positions, cost=0.001):
    equity = [1.0]
    prev_pos = positions[0]
    returns = np.diff(prices) / prices[:-1]

    for i, ret in enumerate(returns):
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

    return total_ret, sharpe, max_dd


def continuous_position(probs, risk_on, caution, risk_off):
    positions = []
    for p in probs:
        pos = p[0] * risk_on + p[1] * caution + p[2] * risk_off
        positions.append(pos)
    return np.array(positions)


print("\nGrid searching continuous sizing parameters...")

# Fine-grained grid search
risk_on_levels = [0.85, 0.90, 0.95, 1.00]
caution_levels = [0.45, 0.50, 0.55, 0.60, 0.65]
risk_off_levels = [0.15, 0.20, 0.25, 0.30]

results = []
for ro, ca, rf in product(risk_on_levels, caution_levels, risk_off_levels):
    positions = continuous_position(val_probs[:len(prices)-1], ro, ca, rf)
    ret, sharpe, max_dd = backtest(prices, positions)

    results.append({
        'risk_on': ro, 'caution': ca, 'risk_off': rf,
        'return': ret, 'sharpe': sharpe, 'max_dd': max_dd
    })

results_df = pd.DataFrame(results)

# Filter: DD < 25%
valid = results_df[results_df['max_dd'] < 25].copy()

print(f"\nFound {len(valid)} configurations with max_dd < 25%")

if len(valid) > 0:
    # Best by Sharpe
    best_sharpe = valid.loc[valid['sharpe'].idxmax()]
    # Best by return
    best_return = valid.loc[valid['return'].idxmax()]
    # Lowest DD
    lowest_dd = valid.loc[valid['max_dd'].idxmin()]

    print("\n--- BEST BY SHARPE (DD < 25%) ---")
    print(f"  RISK_ON: {best_sharpe['risk_on']:.0%}")
    print(f"  CAUTION: {best_sharpe['caution']:.0%}")
    print(f"  RISK_OFF: {best_sharpe['risk_off']:.0%}")
    print(f"  Return: {best_sharpe['return']:+.1f}%")
    print(f"  Sharpe: {best_sharpe['sharpe']:.2f}")
    print(f"  Max DD: {best_sharpe['max_dd']:.1f}%")

    print("\n--- BEST BY RETURN (DD < 25%) ---")
    print(f"  RISK_ON: {best_return['risk_on']:.0%}")
    print(f"  CAUTION: {best_return['caution']:.0%}")
    print(f"  RISK_OFF: {best_return['risk_off']:.0%}")
    print(f"  Return: {best_return['return']:+.1f}%")
    print(f"  Sharpe: {best_return['sharpe']:.2f}")
    print(f"  Max DD: {best_return['max_dd']:.1f}%")

    print("\n--- LOWEST DD (DD < 25%) ---")
    print(f"  RISK_ON: {lowest_dd['risk_on']:.0%}")
    print(f"  CAUTION: {lowest_dd['caution']:.0%}")
    print(f"  RISK_OFF: {lowest_dd['risk_off']:.0%}")
    print(f"  Return: {lowest_dd['return']:+.1f}%")
    print(f"  Sharpe: {lowest_dd['sharpe']:.2f}")
    print(f"  Max DD: {lowest_dd['max_dd']:.1f}%")

    # Top 10 by Sharpe
    print("\n--- TOP 10 CONFIGURATIONS BY SHARPE ---")
    top10 = valid.nlargest(10, 'sharpe')
    print(f"{'RISK_ON':>8} {'CAUTION':>8} {'RISK_OFF':>9} {'Return':>10} {'Sharpe':>8} {'Max DD':>10}")
    print("-" * 60)
    for _, row in top10.iterrows():
        print(f"{row['risk_on']:>7.0%} {row['caution']:>8.0%} {row['risk_off']:>9.0%} "
              f"{row['return']:>+9.1f}% {row['sharpe']:>8.2f} {row['max_dd']:>9.1f}%")


# Compare with baselines
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

bh_ret, bh_sharpe, bh_dd = backtest(prices, np.ones(len(prices)-1))

# Best configuration
if len(valid) > 0:
    best = valid.loc[valid['sharpe'].idxmax()]
    best_pos = continuous_position(
        val_probs[:len(prices)-1],
        best['risk_on'], best['caution'], best['risk_off']
    )
    best_ret, best_sharpe_val, best_dd = backtest(prices, best_pos)

    print(f"\n{'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'Max DD':>10}")
    print("-" * 70)
    print(f"{'Buy & Hold':<35} {bh_ret:>+9.1f}% {bh_sharpe:>8.2f} {bh_dd:>9.1f}%")
    print(f"{'GNN + Continuous (optimized)':<35} {best_ret:>+9.1f}% {best_sharpe_val:>8.2f} {best_dd:>9.1f}%")

    print("\n" + "=" * 70)
    print("FINAL CONFIGURATION")
    print("=" * 70)
    print(f"\nOptimal continuous sizing parameters:")
    print(f"  RISK_ON weight:  {best['risk_on']:.0%}")
    print(f"  CAUTION weight:  {best['caution']:.0%}")
    print(f"  RISK_OFF weight: {best['risk_off']:.0%}")
    print(f"\nPosition formula:")
    print(f"  position = p(RISK_ON)*{best['risk_on']:.2f} + p(CAUTION)*{best['caution']:.2f} + p(RISK_OFF)*{best['risk_off']:.2f}")

    # Average position
    avg_pos = best_pos.mean()
    print(f"\nAverage position: {avg_pos:.1%}")

    # Save configuration
    config = {
        'risk_on_weight': float(best['risk_on']),
        'caution_weight': float(best['caution']),
        'risk_off_weight': float(best['risk_off']),
        'validation_return': float(best_ret),
        'validation_sharpe': float(best_sharpe_val),
        'validation_max_dd': float(best_dd)
    }

    import json
    config_path = Path(__file__).parent.parent / 'config' / 'position_sizing.json'
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")

print("\n" + "=" * 70)
