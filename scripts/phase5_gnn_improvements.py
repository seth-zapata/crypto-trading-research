"""
Phase 5 GNN Improvements - Reduce Crash Miss Rate
==================================================

The robust validation revealed a 44% crash miss rate (GNN in RISK_ON during crashes).
This script implements and evaluates improvements to reduce miss rate to <25%.

Priorities (from user spec):
1. Asymmetric Loss Function - penalize misses 10x more than false alarms
2. Lower RISK_OFF Confidence Threshold - trigger on lower probability
3. Add funding rates feature (future work)
4. Crash oversampling with SMOTE (future work)

Success Criteria:
- Miss rate < 25%
- Monte Carlo 95th pct Max DD < 40%
- Walk-forward win rate > 45%
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector, AsymmetricCrashLoss

logging.basicConfig(level=logging.WARNING)

print("=" * 70)
print("PHASE 5: GNN IMPROVEMENTS - REDUCING CRASH MISS RATE")
print("=" * 70)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_miss_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    crash_threshold: float = -0.05
) -> Dict:
    """
    Calculate crash detection metrics.

    Miss = Predicted RISK_ON when crash occurred
    False Alarm = Predicted RISK_OFF when no crash
    """
    RISK_ON, RISK_OFF = 0, 2

    # Identify actual crash days (using returns)
    is_crash = returns < crash_threshold
    n_crashes = is_crash.sum()

    # Identify predictions
    pred_risk_on = predictions == RISK_ON
    pred_risk_off = predictions == RISK_OFF

    # Calculate metrics
    misses = (pred_risk_on & is_crash).sum()
    miss_rate = misses / n_crashes if n_crashes > 0 else 0

    false_alarms = (pred_risk_off & ~is_crash).sum()
    false_alarm_rate = false_alarms / (~is_crash).sum() if (~is_crash).sum() > 0 else 0

    # True positives: predicted RISK_OFF and was crash
    true_positives = (pred_risk_off & is_crash).sum()
    detection_rate = true_positives / n_crashes if n_crashes > 0 else 0

    return {
        'n_crashes': int(n_crashes),
        'misses': int(misses),
        'miss_rate': float(miss_rate),
        'true_positives': int(true_positives),
        'detection_rate': float(detection_rate),
        'false_alarms': int(false_alarms),
        'false_alarm_rate': float(false_alarm_rate)
    }


def calculate_lead_time(
    predictions: np.ndarray,
    returns: np.ndarray,
    crash_threshold: float = -0.05,
    lookback: int = 10
) -> Dict:
    """
    Calculate average lead time: how many days before crash did model go defensive?
    """
    RISK_OFF = 2
    crash_days = np.where(returns < crash_threshold)[0]

    lead_times = []
    for crash_day in crash_days:
        # Look back up to 10 days before crash
        start = max(0, crash_day - lookback)
        window = predictions[start:crash_day]

        # Find first RISK_OFF in window
        risk_off_days = np.where(window == RISK_OFF)[0]
        if len(risk_off_days) > 0:
            lead_time = crash_day - start - risk_off_days[0]
            lead_times.append(lead_time)

    return {
        'avg_lead_time': np.mean(lead_times) if lead_times else 0,
        'pct_with_warning': len(lead_times) / len(crash_days) if len(crash_days) > 0 else 0
    }


def backtest_strategy(
    btc_returns: np.ndarray,
    positions: np.ndarray
) -> Tuple[List[float], float, float, float]:
    """Run backtest with given positions."""
    equity = [1.0]
    for i, ret in enumerate(btc_returns):
        pos = positions[i] if i < len(positions) else 1.0
        equity.append(equity[-1] * (1 + ret * pos))

    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] / equity[0] - 1) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = drawdown.max() * 100

    return equity.tolist(), total_return, sharpe, max_dd


def positions_from_probs(
    probs: np.ndarray,
    risk_off_threshold: float = 0.3,
    weights: Dict = None
) -> np.ndarray:
    """
    Convert probabilities to positions.

    Two modes:
    1. Threshold-based: go defensive if p(RISK_OFF) > threshold
    2. Continuous: weighted sum of probabilities
    """
    if weights is not None:
        # Continuous mode
        positions = (
            probs[:, 0] * weights.get('risk_on', 0.85) +
            probs[:, 1] * weights.get('caution', 0.65) +
            probs[:, 2] * weights.get('risk_off', 0.30)
        )
        return np.clip(positions, 0.1, 1.0)
    else:
        # Threshold mode
        positions = np.ones(len(probs))
        positions[probs[:, 2] > risk_off_threshold] = 0.2
        positions[(probs[:, 1] > 0.4) & (positions == 1.0)] = 0.5
        return positions


# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")

df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

print(f"Dataset: {len(df)} samples")
print(f"Regime distribution: {labels.value_counts().to_dict()}")


# ============================================================
# 2. TRAIN BASELINE MODEL (Standard Loss)
# ============================================================
print("\n[2/5] Training baseline GNN (standard CrossEntropyLoss)...")

detector_baseline = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector_baseline.prepare_dataset(df, labels)

split = int(len(graphs) * 0.8)
train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]

# Get returns for crash detection
btc_returns = df['BTC_return_1d'].values
train_returns = btc_returns[:split]
val_returns = btc_returns[split:split + len(val_graphs)]

# Train baseline
history_baseline = detector_baseline.train(
    train_graphs, train_labels,
    val_graphs, val_labels,
    epochs=100, batch_size=32,
    use_asymmetric_loss=False
)

preds_baseline, probs_baseline = detector_baseline.predict(val_graphs)

# Calculate baseline metrics
baseline_miss = calculate_miss_metrics(preds_baseline, val_labels, val_returns)
baseline_lead = calculate_lead_time(preds_baseline, val_returns)

print(f"Baseline Miss Rate: {baseline_miss['miss_rate']:.1%}")
print(f"Baseline Detection Rate: {baseline_miss['detection_rate']:.1%}")
print(f"Baseline False Alarm Rate: {baseline_miss['false_alarm_rate']:.1%}")


# ============================================================
# 3. TRAIN WITH ASYMMETRIC LOSS
# ============================================================
print("\n[3/5] Training GNN with Asymmetric Loss (miss_penalty=10x)...")

# Test different miss penalties
miss_penalties = [5.0, 10.0, 15.0, 20.0]
asymmetric_results = []

for penalty in miss_penalties:
    print(f"\n  Testing miss_penalty={penalty}...")

    detector_asym = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
    graphs_a, targets_a = detector_asym.prepare_dataset(df, labels)
    train_g_a, val_g_a = graphs_a[:split], graphs_a[split:]
    train_l_a, val_l_a = targets_a[:split], targets_a[split:]

    # Train with asymmetric loss
    history_asym = detector_asym.train(
        train_g_a, train_l_a,
        val_g_a, val_l_a,
        epochs=100, batch_size=32,
        train_returns=np.array(train_returns),
        use_asymmetric_loss=True,
        miss_penalty=penalty
    )

    preds_asym, probs_asym = detector_asym.predict(val_g_a)

    # Calculate metrics
    miss_metrics = calculate_miss_metrics(preds_asym, val_l_a, val_returns)
    lead_metrics = calculate_lead_time(preds_asym, val_returns)

    # Backtest
    positions_asym = positions_from_probs(probs_asym, risk_off_threshold=0.3)
    _, ret, sharpe, dd = backtest_strategy(val_returns, positions_asym)

    result = {
        'penalty': penalty,
        'miss_rate': miss_metrics['miss_rate'],
        'detection_rate': miss_metrics['detection_rate'],
        'false_alarm_rate': miss_metrics['false_alarm_rate'],
        'avg_lead_time': lead_metrics['avg_lead_time'],
        'return': ret,
        'sharpe': sharpe,
        'max_dd': dd
    }
    asymmetric_results.append(result)

    print(f"    Miss Rate: {miss_metrics['miss_rate']:.1%} (target <25%)")
    print(f"    Detection Rate: {miss_metrics['detection_rate']:.1%}")
    print(f"    Max DD: {dd:.1f}%")


# ============================================================
# 4. TEST LOWER RISK_OFF THRESHOLDS
# ============================================================
print("\n[4/5] Testing lower RISK_OFF confidence thresholds...")

# Use baseline model but with lower thresholds
threshold_results = []

for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    # Convert to regime based on threshold
    preds_thresh = np.argmax(probs_baseline, axis=1)
    # Override: if p(RISK_OFF) > threshold, predict RISK_OFF
    preds_thresh[probs_baseline[:, 2] > thresh] = 2

    miss_metrics = calculate_miss_metrics(preds_thresh, val_labels, val_returns)
    lead_metrics = calculate_lead_time(preds_thresh, val_returns)

    # Backtest with threshold-based positions
    positions_thresh = positions_from_probs(probs_baseline, risk_off_threshold=thresh)
    _, ret, sharpe, dd = backtest_strategy(val_returns, positions_thresh)

    result = {
        'threshold': thresh,
        'miss_rate': miss_metrics['miss_rate'],
        'detection_rate': miss_metrics['detection_rate'],
        'false_alarm_rate': miss_metrics['false_alarm_rate'],
        'avg_lead_time': lead_metrics['avg_lead_time'],
        'return': ret,
        'sharpe': sharpe,
        'max_dd': dd,
        'pct_defensive': (positions_thresh < 1.0).mean() * 100
    }
    threshold_results.append(result)


# ============================================================
# 5. COMBINED: ASYMMETRIC LOSS + LOW THRESHOLD
# ============================================================
print("\n[5/5] Testing combined approach (Asymmetric + Low Threshold)...")

# Use best asymmetric model with low threshold
best_penalty = min(asymmetric_results, key=lambda x: x['miss_rate'])['penalty']

detector_combined = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs_c, targets_c = detector_combined.prepare_dataset(df, labels)
train_g_c, val_g_c = graphs_c[:split], graphs_c[split:]
train_l_c, val_l_c = targets_c[:split], targets_c[split:]

history_combined = detector_combined.train(
    train_g_c, train_l_c,
    val_g_c, val_l_c,
    epochs=100, batch_size=32,
    train_returns=np.array(train_returns),
    use_asymmetric_loss=True,
    miss_penalty=best_penalty
)

preds_combined, probs_combined = detector_combined.predict(val_g_c)

# Test with different thresholds on the asymmetric model
combined_results = []
for thresh in [0.15, 0.20, 0.25]:
    preds_c = np.argmax(probs_combined, axis=1)
    preds_c[probs_combined[:, 2] > thresh] = 2

    miss_metrics = calculate_miss_metrics(preds_c, val_l_c, val_returns)
    positions_c = positions_from_probs(probs_combined, risk_off_threshold=thresh)
    _, ret, sharpe, dd = backtest_strategy(val_returns, positions_c)

    combined_results.append({
        'threshold': thresh,
        'miss_rate': miss_metrics['miss_rate'],
        'detection_rate': miss_metrics['detection_rate'],
        'return': ret,
        'sharpe': sharpe,
        'max_dd': dd
    })


# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n1. BASELINE (Standard CrossEntropyLoss)")
print(f"   Miss Rate: {baseline_miss['miss_rate']:.1%}")
print(f"   Detection Rate: {baseline_miss['detection_rate']:.1%}")

print("\n2. ASYMMETRIC LOSS RESULTS")
print(f"   {'Penalty':<10} {'Miss Rate':>12} {'Detection':>12} {'Max DD':>10} {'Sharpe':>10}")
print("   " + "-" * 55)
for r in asymmetric_results:
    print(f"   {r['penalty']:<10.1f} {r['miss_rate']:>11.1%} {r['detection_rate']:>11.1%} "
          f"{r['max_dd']:>9.1f}% {r['sharpe']:>10.2f}")

print("\n3. THRESHOLD TUNING (on baseline model)")
print(f"   {'Threshold':>10} {'Miss Rate':>12} {'Detection':>12} {'Max DD':>10} {'Defensive%':>12}")
print("   " + "-" * 60)
for r in threshold_results:
    print(f"   {r['threshold']:>10.2f} {r['miss_rate']:>11.1%} {r['detection_rate']:>11.1%} "
          f"{r['max_dd']:>9.1f}% {r['pct_defensive']:>11.1f}%")

print("\n4. COMBINED (Asymmetric + Low Threshold)")
print(f"   {'Threshold':>10} {'Miss Rate':>12} {'Detection':>12} {'Max DD':>10} {'Sharpe':>10}")
print("   " + "-" * 55)
for r in combined_results:
    print(f"   {r['threshold']:>10.2f} {r['miss_rate']:>11.1%} {r['detection_rate']:>11.1%} "
          f"{r['max_dd']:>9.1f}% {r['sharpe']:>10.2f}")


# ============================================================
# FIND BEST CONFIGURATION
# ============================================================
print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

# Find configuration that meets miss rate target
all_results = []

# Add asymmetric results
for r in asymmetric_results:
    all_results.append({
        'config': f"Asym(penalty={r['penalty']})",
        'miss_rate': r['miss_rate'],
        'detection_rate': r['detection_rate'],
        'max_dd': r['max_dd'],
        'sharpe': r['sharpe']
    })

# Add combined results
for r in combined_results:
    all_results.append({
        'config': f"Combined(thresh={r['threshold']})",
        'miss_rate': r['miss_rate'],
        'detection_rate': r['detection_rate'],
        'max_dd': r['max_dd'],
        'sharpe': r['sharpe']
    })

# Filter by miss rate target
target_miss_rate = 0.25
valid_configs = [r for r in all_results if r['miss_rate'] < target_miss_rate]

if valid_configs:
    # Pick best by Sharpe among valid
    best = max(valid_configs, key=lambda x: x['sharpe'])
    print(f"\nBest configuration meeting <25% miss rate:")
    print(f"  Config: {best['config']}")
    print(f"  Miss Rate: {best['miss_rate']:.1%}")
    print(f"  Detection Rate: {best['detection_rate']:.1%}")
    print(f"  Max DD: {best['max_dd']:.1f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
else:
    print("\n WARNING: No configuration meets <25% miss rate target.")
    print("Consider:")
    print("  1. Higher miss_penalty (try 25.0, 30.0)")
    print("  2. Even lower threshold (try 0.10)")
    print("  3. Add funding rates feature")
    print("  4. Apply SMOTE oversampling")

    # Show best anyway
    best = min(all_results, key=lambda x: x['miss_rate'])
    print(f"\nLowest miss rate achieved:")
    print(f"  Config: {best['config']}")
    print(f"  Miss Rate: {best['miss_rate']:.1%}")


# ============================================================
# SAVE BEST MODEL
# ============================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save best asymmetric model
best_asym_penalty = min(asymmetric_results, key=lambda x: x['miss_rate'])['penalty']

# Retrain with best penalty and save
print(f"\nRetraining with best penalty ({best_asym_penalty}) and saving...")

detector_best = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs_best, targets_best = detector_best.prepare_dataset(df, labels)
train_g_best = graphs_best[:split]
train_l_best = targets_best[:split]

detector_best.train(
    train_g_best, train_l_best,
    None, None,  # No validation during final training
    epochs=100, batch_size=32,
    train_returns=np.array(train_returns),
    use_asymmetric_loss=True,
    miss_penalty=best_asym_penalty
)

# Save model
model_path = Path(__file__).parent.parent / 'models' / 'saved' / 'gnn_asymmetric_best.pth'
model_path.parent.mkdir(parents=True, exist_ok=True)
detector_best.save(str(model_path))
print(f"Model saved to: {model_path}")

# Save results to JSON
import json
results_path = Path(__file__).parent.parent / 'config' / 'gnn_improvement_results.json'
results_data = {
    'baseline': {
        'miss_rate': baseline_miss['miss_rate'],
        'detection_rate': baseline_miss['detection_rate'],
        'false_alarm_rate': baseline_miss['false_alarm_rate']
    },
    'asymmetric_results': asymmetric_results,
    'threshold_results': threshold_results,
    'combined_results': combined_results,
    'best_config': best['config'] if 'best' in dir() else None,
    'best_miss_penalty': best_asym_penalty
}

with open(results_path, 'w') as f:
    json.dump(results_data, f, indent=2)
print(f"Results saved to: {results_path}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
if valid_configs:
    print("1. Run Monte Carlo validation on best config")
    print("2. If still >40% worst-case DD, add funding rates feature")
else:
    print("1. Try higher miss_penalty (25.0-50.0)")
    print("2. Add funding rates feature")
    print("3. Apply SMOTE oversampling for crash periods")
    print("4. Consider GNN ensemble")
