"""
Phase 4: Train and Evaluate Regime GNN
======================================

Training script with:
- Proper class weighting for imbalanced data
- Lead time analysis (how early do we detect crashes?)
- Backtesting: does detecting regime shifts reduce drawdown?
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import (
    build_regime_detection_features,
    MultiAssetFetcher,
    RegimeLabeler
)
from models.predictors.regime_gnn import RegimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 70)
print("PHASE 4: GNN REGIME DETECTION")
print("=" * 70)


# ============================================================
# 1. PREPARE DATA
# ============================================================
print("\n[1/5] Preparing data...")

df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

print(f"Dataset: {len(df)} samples")
print(f"Features: {len(df.columns)} columns")
print(f"\nRegime distribution:")
for regime, count in labels.value_counts().items():
    pct = count / len(labels) * 100
    print(f"  {regime}: {count} ({pct:.1f}%)")


# ============================================================
# 2. PREPARE GRAPHS
# ============================================================
print("\n[2/5] Preparing graph dataset...")

detector = RegimeDetector(
    assets=['BTC', 'ETH', 'SOL'],
    hidden_dim=64
)

graphs, targets = detector.prepare_dataset(df, labels)
print(f"Prepared {len(graphs)} graphs")

# Time series split (80/20)
split = int(len(graphs) * 0.8)
train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]

print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

# Class weights (inverse frequency)
train_counts = np.bincount(train_labels, minlength=3)
class_weights = len(train_labels) / (3 * train_counts + 1)
class_weights = class_weights / class_weights.sum() * 3
print(f"\nClass weights: {class_weights}")


# ============================================================
# 3. TRAIN MODEL
# ============================================================
print("\n[3/5] Training GNN...")

history = detector.train(
    train_graphs, train_labels,
    val_graphs, val_labels,
    epochs=100,
    lr=0.001,
    batch_size=32,
    class_weights=class_weights.tolist()
)

print(f"\nFinal train acc: {history['train_acc'][-1]:.3f}")
print(f"Final val acc: {history['val_acc'][-1]:.3f}")


# ============================================================
# 4. EVALUATE MODEL
# ============================================================
print("\n[4/5] Evaluating model...")

preds, probs = detector.predict(val_graphs)

print("\n--- Classification Report ---")
print(classification_report(
    val_labels, preds,
    target_names=detector.REGIME_LABELS
))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(val_labels, preds)
print(f"             Pred: RISK_ON  CAUTION  RISK_OFF")
for i, regime in enumerate(detector.REGIME_LABELS):
    print(f"Actual {regime:8s}: {cm[i]}")


# ============================================================
# 5. LEAD TIME ANALYSIS
# ============================================================
print("\n[5/5] Analyzing lead time for regime shifts...")

# Find regime transitions in validation set
val_df = df.iloc[split:].copy()
val_labels_series = labels.iloc[split:].copy()
val_preds_series = pd.Series(
    [detector.REGIME_LABELS[p] for p in preds],
    index=val_df.index[:len(preds)]
)

# Find actual RISK_OFF periods
risk_off_starts = []
in_risk_off = False

for i, label in enumerate(val_labels_series):
    if label == 'RISK_OFF' and not in_risk_off:
        risk_off_starts.append(i)
        in_risk_off = True
    elif label != 'RISK_OFF':
        in_risk_off = False

print(f"\nFound {len(risk_off_starts)} RISK_OFF periods in validation")

# For each RISK_OFF period, find when we detected it
lead_times = []
detected_count = 0

for start_idx in risk_off_starts:
    # Look back from the actual start
    detected_at = None

    # Check if we predicted CAUTION or RISK_OFF before actual RISK_OFF
    for lookback in range(1, min(21, start_idx)):
        check_idx = start_idx - lookback
        if check_idx < len(preds):
            pred = detector.REGIME_LABELS[preds[check_idx]]
            if pred in ['CAUTION', 'RISK_OFF']:
                detected_at = lookback
                break

    if detected_at is not None:
        lead_times.append(detected_at)
        detected_count += 1
        print(f"  RISK_OFF period {len(lead_times)}: Detected {detected_at} days before")
    else:
        print(f"  RISK_OFF period {len(lead_times)+1}: NOT detected ahead of time")

if lead_times:
    print(f"\n--- Lead Time Statistics ---")
    print(f"Detected: {detected_count}/{len(risk_off_starts)} ({detected_count/len(risk_off_starts)*100:.1f}%)")
    print(f"Avg lead time: {np.mean(lead_times):.1f} days")
    print(f"Min lead time: {np.min(lead_times)} days")
    print(f"Max lead time: {np.max(lead_times)} days")
else:
    print("\nNo RISK_OFF periods detected ahead of time")


# ============================================================
# 6. BACKTEST: Does this reduce drawdown?
# ============================================================
print("\n" + "=" * 70)
print("BACKTEST: Position Sizing Based on Regime")
print("=" * 70)

# Simple strategy:
# - RISK_ON: 100% exposure
# - CAUTION: 50% exposure
# - RISK_OFF: 20% exposure

position_map = {
    'RISK_ON': 1.0,
    'CAUTION': 0.5,
    'RISK_OFF': 0.2
}

# Get BTC returns for validation period
btc_returns = df['BTC_return_1d'].iloc[split:].values[:len(preds)]

# Buy and hold
bh_equity = [1.0]
for ret in btc_returns:
    bh_equity.append(bh_equity[-1] * (1 + ret))

# Strategy based on TRUE regimes (oracle - upper bound)
oracle_equity = [1.0]
for i, ret in enumerate(btc_returns):
    true_regime = detector.REGIME_LABELS[val_labels[i]] if i < len(val_labels) else 'RISK_ON'
    position = position_map[true_regime]
    oracle_equity.append(oracle_equity[-1] * (1 + ret * position))

# Strategy based on PREDICTED regimes
strat_equity = [1.0]
for i, ret in enumerate(btc_returns):
    pred_regime = detector.REGIME_LABELS[preds[i]] if i < len(preds) else 'RISK_ON'
    position = position_map[pred_regime]
    strat_equity.append(strat_equity[-1] * (1 + ret * position))

# Calculate metrics
def calc_metrics(equity):
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] / equity[0] - 1) * 100
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = drawdown.max() * 100

    return total_return, sharpe, max_dd

bh_ret, bh_sharpe, bh_dd = calc_metrics(bh_equity)
oracle_ret, oracle_sharpe, oracle_dd = calc_metrics(oracle_equity)
strat_ret, strat_sharpe, strat_dd = calc_metrics(strat_equity)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ Strategy              │ Return  │ Sharpe │ Max DD │ DD Reduction│
├─────────────────────────────────────────────────────────────────┤
│ Buy & Hold            │ {bh_ret:>6.1f}% │ {bh_sharpe:>6.2f} │ {bh_dd:>5.1f}% │ (baseline)  │
│ Oracle (true regimes) │ {oracle_ret:>6.1f}% │ {oracle_sharpe:>6.2f} │ {oracle_dd:>5.1f}% │ {(bh_dd-oracle_dd)/bh_dd*100:>5.1f}%      │
│ GNN Strategy          │ {strat_ret:>6.1f}% │ {strat_sharpe:>6.2f} │ {strat_dd:>5.1f}% │ {(bh_dd-strat_dd)/bh_dd*100:>5.1f}%      │
└─────────────────────────────────────────────────────────────────┘
""")

# Success criteria
target_dd_reduction = 30  # Target: 30% drawdown reduction
actual_dd_reduction = (bh_dd - strat_dd) / bh_dd * 100

print("=" * 70)
if actual_dd_reduction >= target_dd_reduction:
    print(f"✓ SUCCESS: Drawdown reduced by {actual_dd_reduction:.1f}% (target: ≥{target_dd_reduction}%)")
else:
    print(f"✗ Target not met: {actual_dd_reduction:.1f}% DD reduction (target: ≥{target_dd_reduction}%)")
print("=" * 70)

# Save model
model_path = Path(__file__).parent.parent / 'models' / 'saved' / 'regime_gnn.pth'
model_path.parent.mkdir(parents=True, exist_ok=True)
detector.save(str(model_path))
print(f"\nModel saved to: {model_path}")
