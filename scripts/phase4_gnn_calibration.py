"""
Phase 4: GNN Calibration - Fixing Over-Conservatism
====================================================

The GNN predicts RISK_OFF 15.2x more often than actual.
This script implements and compares fixes:

1. Threshold-based classification (require high confidence)
2. Continuous position sizing (scale by probability)
3. Retrained model with better class weights
4. Simple volatility rule (baseline control)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector, RegimeGNN
from torch_geometric.data import Batch

logging.basicConfig(level=logging.WARNING)

print("=" * 70)
print("PHASE 4: GNN CALIBRATION - FIXING OVER-CONSERVATISM")
print("=" * 70)


# ============================================================
# 1. LOAD DATA AND TRAIN BASE MODEL
# ============================================================
print("\n[1/6] Loading data and training base GNN...")

df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

detector = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector.prepare_dataset(df, labels)

split = int(len(graphs) * 0.8)
train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]

# Original class weights
train_counts = np.bincount(train_labels, minlength=3)
class_weights = len(train_labels) / (3 * train_counts + 1)
class_weights = class_weights / class_weights.sum() * 3

# Train base model
history = detector.train(
    train_graphs, train_labels,
    val_graphs, val_labels,
    epochs=100, batch_size=32,
    class_weights=class_weights.tolist()
)

# Get predictions and probabilities
preds, probs = detector.predict(val_graphs)

print(f"Base model trained. Val accuracy: {(preds == val_labels).mean():.1%}")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calc_metrics(equity: List[float]) -> Tuple[float, float, float]:
    """Calculate return, Sharpe, max drawdown."""
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] / equity[0] - 1) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = drawdown.max() * 100

    return total_return, sharpe, max_dd


def backtest_strategy(
    btc_returns: np.ndarray,
    positions: np.ndarray
) -> Tuple[List[float], float, float, float]:
    """Run backtest with given positions."""
    equity = [1.0]
    for i, ret in enumerate(btc_returns):
        pos = positions[i] if i < len(positions) else 1.0
        equity.append(equity[-1] * (1 + ret * pos))

    ret, sharpe, dd = calc_metrics(equity)
    return equity, ret, sharpe, dd


# Get BTC returns for validation period
btc_returns = df['BTC_return_1d'].iloc[split:].values[:len(preds)]


# ============================================================
# 3. FIX 1: THRESHOLD-BASED CLASSIFICATION
# ============================================================
print("\n[2/6] Testing threshold-based classification...")

RISK_ON, CAUTION, RISK_OFF = 0, 1, 2

def classify_with_threshold(
    probs: np.ndarray,
    risk_off_threshold: float = 0.75,
    caution_threshold: float = 0.50
) -> np.ndarray:
    """Classify with high confidence thresholds."""
    classifications = []
    for p in probs:
        if p[RISK_OFF] > risk_off_threshold:
            classifications.append(RISK_OFF)
        elif p[CAUTION] > caution_threshold:
            classifications.append(CAUTION)
        else:
            classifications.append(RISK_ON)
    return np.array(classifications)


def positions_from_regime(regimes: np.ndarray) -> np.ndarray:
    """Convert regimes to positions."""
    position_map = {RISK_ON: 1.0, CAUTION: 0.5, RISK_OFF: 0.2}
    return np.array([position_map[r] for r in regimes])


threshold_results = []
for risk_off_thresh in [0.50, 0.60, 0.70, 0.75, 0.80]:
    regimes = classify_with_threshold(probs, risk_off_threshold=risk_off_thresh)
    positions = positions_from_regime(regimes)

    _, ret, sharpe, dd = backtest_strategy(btc_returns, positions)
    risk_off_pct = (regimes == RISK_OFF).mean() * 100

    threshold_results.append({
        'threshold': risk_off_thresh,
        'return': ret,
        'sharpe': sharpe,
        'max_dd': dd,
        'risk_off_pct': risk_off_pct
    })

print("Threshold results:")
for r in threshold_results:
    print(f"  Thresh {r['threshold']:.2f}: Return {r['return']:+.1f}%, "
          f"Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1f}%, "
          f"RISK_OFF {r['risk_off_pct']:.1f}%")


# ============================================================
# 4. FIX 2: CONTINUOUS POSITION SIZING
# ============================================================
print("\n[3/6] Testing continuous position sizing...")

def continuous_positions(probs: np.ndarray) -> np.ndarray:
    """Scale position continuously based on crash probability."""
    positions = []
    for p in probs:
        # Base position is 100%
        position = 1.0

        # Reduce based on RISK_OFF probability (max 80% reduction)
        position -= p[RISK_OFF] * 0.8

        # Reduce based on CAUTION probability (max 30% reduction)
        position -= p[CAUTION] * 0.3

        # Floor at 10%
        positions.append(max(position, 0.10))

    return np.array(positions)


continuous_pos = continuous_positions(probs)
_, cont_ret, cont_sharpe, cont_dd = backtest_strategy(btc_returns, continuous_pos)

print(f"Continuous sizing: Return {cont_ret:+.1f}%, Sharpe {cont_sharpe:.2f}, DD {cont_dd:.1f}%")
print(f"Avg position: {continuous_pos.mean():.1%}, Min: {continuous_pos.min():.1%}")


# ============================================================
# 5. FIX 3: RETRAIN WITH BETTER CLASS WEIGHTS
# ============================================================
print("\n[4/6] Retraining with conservative class weights...")

# New class weights: penalize false RISK_OFF (missed gains)
# High weight on RISK_OFF means model must be very sure to predict it
new_class_weights = [1.0, 2.0, 8.0]  # RISK_ON, CAUTION, RISK_OFF

detector_retrained = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs_r, targets_r = detector_retrained.prepare_dataset(df, labels)
train_g_r, val_g_r = graphs_r[:split], graphs_r[split:]
train_l_r, val_l_r = targets_r[:split], targets_r[split:]

history_r = detector_retrained.train(
    train_g_r, train_l_r,
    val_g_r, val_l_r,
    epochs=100, batch_size=32,
    class_weights=new_class_weights
)

preds_r, probs_r = detector_retrained.predict(val_g_r)
positions_r = positions_from_regime(preds_r)
_, ret_r, sharpe_r, dd_r = backtest_strategy(btc_returns, positions_r)

risk_off_pct_r = (preds_r == RISK_OFF).mean() * 100
print(f"Retrained: Return {ret_r:+.1f}%, Sharpe {sharpe_r:.2f}, DD {dd_r:.1f}%, "
      f"RISK_OFF {risk_off_pct_r:.1f}%")


# ============================================================
# 6. SIMPLE VOLATILITY RULE (CONTROL BASELINE)
# ============================================================
print("\n[5/6] Testing simple volatility rule baseline...")

def simple_vol_strategy(vol_20d: np.ndarray) -> np.ndarray:
    """Baseline: reduce exposure when vol is high."""
    positions = []
    for v in vol_20d:
        if v > 0.80:  # Annualized vol > 80%
            positions.append(0.2)
        elif v > 0.50:  # Annualized vol > 50%
            positions.append(0.5)
        else:
            positions.append(1.0)
    return np.array(positions)


vol_20d = df['BTC_vol_20d'].iloc[split:].values[:len(preds)]
vol_positions = simple_vol_strategy(vol_20d)
_, vol_ret, vol_sharpe, vol_dd = backtest_strategy(btc_returns, vol_positions)

vol_defensive_pct = (vol_positions < 1.0).mean() * 100
print(f"Simple vol rule: Return {vol_ret:+.1f}%, Sharpe {vol_sharpe:.2f}, DD {vol_dd:.1f}%, "
      f"Defensive {vol_defensive_pct:.1f}%")


# ============================================================
# 7. BUY AND HOLD BASELINE
# ============================================================
bh_positions = np.ones(len(btc_returns))
_, bh_ret, bh_sharpe, bh_dd = backtest_strategy(btc_returns, bh_positions)


# ============================================================
# RESULTS COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

# Current GNN (from original)
orig_positions = positions_from_regime(preds)
_, orig_ret, orig_sharpe, orig_dd = backtest_strategy(btc_returns, orig_positions)
orig_risk_off_pct = (preds == RISK_OFF).mean() * 100

# Best threshold result
best_thresh = max(threshold_results, key=lambda x: x['sharpe'])

results = [
    ("Buy & Hold", bh_ret, bh_sharpe, bh_dd, 0.0, "-"),
    ("Current GNN (broken)", orig_ret, orig_sharpe, orig_dd, orig_risk_off_pct, "15.2x over-predict"),
    (f"Threshold {best_thresh['threshold']:.2f}", best_thresh['return'], best_thresh['sharpe'],
     best_thresh['max_dd'], best_thresh['risk_off_pct'], "Calibrated"),
    ("Continuous sizing", cont_ret, cont_sharpe, cont_dd,
     (continuous_pos < 0.5).mean() * 100, "Smooth"),
    ("Retrained (weighted)", ret_r, sharpe_r, dd_r, risk_off_pct_r, "Better weights"),
    ("Simple vol rule", vol_ret, vol_sharpe, vol_dd, vol_defensive_pct, "Baseline"),
]

print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Max DD':>8} {'Def %':>8} {'Notes':<20}")
print("-" * 85)
for name, ret, sharpe, dd, def_pct, notes in results:
    print(f"{name:<25} {ret:>+9.1f}% {sharpe:>8.2f} {dd:>7.1f}% {def_pct:>7.1f}% {notes:<20}")


# ============================================================
# RECOMMENDATION
# ============================================================
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Find best approach that beats simple vol rule
gnn_approaches = [r for r in results if "GNN" in r[0] or "Threshold" in r[0] or
                  "Continuous" in r[0] or "Retrained" in r[0]]

best_gnn = max(gnn_approaches, key=lambda x: x[2])  # Best Sharpe
vol_baseline = [r for r in results if "vol rule" in r[0]][0]

print(f"\nBest GNN approach: {best_gnn[0]}")
print(f"  Sharpe: {best_gnn[2]:.2f} vs Vol rule: {vol_baseline[2]:.2f}")
print(f"  Max DD: {best_gnn[3]:.1f}% vs Vol rule: {vol_baseline[3]:.1f}%")

if best_gnn[2] > vol_baseline[2]:
    print(f"\n✓ GNN adds value over simple volatility rule")
    print(f"  Sharpe improvement: {(best_gnn[2] - vol_baseline[2]):.2f}")
else:
    print(f"\n✗ GNN does NOT beat simple volatility rule")
    print(f"  Consider using vol rule instead (simpler, similar results)")

# Check if we meet success criteria
target_risk_off = 8.0  # Target: 3-8%
target_return = 40.0
target_dd = 25.0

print(f"\n--- Success Criteria Check ---")
print(f"RISK_OFF prediction rate: {best_gnn[4]:.1f}% (target: 3-8%): "
      f"{'✓' if 3 <= best_gnn[4] <= 8 else '✗'}")
print(f"Return: {best_gnn[1]:.1f}% (target: >40%): "
      f"{'✓' if best_gnn[1] > target_return else '✗'}")
print(f"Max DD: {best_gnn[3]:.1f}% (target: <25%): "
      f"{'✓' if best_gnn[3] < target_dd else '✗'}")
print(f"Beat vol rule: {'✓' if best_gnn[2] > vol_baseline[2] else '✗'}")

print("\n" + "=" * 70)
