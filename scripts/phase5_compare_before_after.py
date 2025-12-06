"""
Direct comparison: Run BOTH configurations on the same data/methodology
to properly compare before/after Monte Carlo distributions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

print("=" * 70)
print("FAIR COMPARISON: BEFORE vs AFTER (Same Methodology)")
print("=" * 70)

# ============================================================
# Load data and prepare
# ============================================================
df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

btc_returns = df['BTC_return_1d'].values
split = int(len(df) * 0.8)
val_returns = btc_returns[split:]

print(f"Validation period: {len(val_returns)} days")

# ============================================================
# Train BEFORE model (standard loss, no threshold adjustment)
# ============================================================
print("\n[1/4] Training BEFORE model (standard CrossEntropyLoss)...")

detector_before = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector_before.prepare_dataset(df, labels)

train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]

detector_before.train(
    train_graphs, train_labels,
    val_graphs, val_labels,
    epochs=100, batch_size=32,
    use_asymmetric_loss=False
)

_, probs_before = detector_before.predict(val_graphs)

# ============================================================
# Train AFTER model (asymmetric loss)
# ============================================================
print("\n[2/4] Training AFTER model (AsymmetricCrashLoss, penalty=15)...")

detector_after = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs_a, targets_a = detector_after.prepare_dataset(df, labels)

detector_after.train(
    graphs_a[:split], targets_a[:split],
    graphs_a[split:], targets_a[split:],
    epochs=100, batch_size=32,
    train_returns=btc_returns[:split],
    use_asymmetric_loss=True,
    miss_penalty=15.0
)

_, probs_after = detector_after.predict(graphs_a[split:])

# ============================================================
# Define position sizing functions
# ============================================================
def positions_before(probs, threshold=0.50):
    """Original: high threshold, same weights"""
    positions = probs[:, 0] * 0.85 + probs[:, 1] * 0.65 + probs[:, 2] * 0.30
    return np.clip(positions, 0.1, 1.0)

def positions_after(probs, threshold=0.20):
    """Improved: low threshold, acts more defensively"""
    positions = probs[:, 0] * 0.85 + probs[:, 1] * 0.65 + probs[:, 2] * 0.20
    # Lower threshold triggers more defensive behavior
    positions[probs[:, 2] > threshold] = 0.20
    return np.clip(positions, 0.1, 1.0)

pos_before = positions_before(probs_before)
pos_after = positions_after(probs_after)

print(f"\nAverage position BEFORE: {pos_before.mean():.1%}")
print(f"Average position AFTER: {pos_after.mean():.1%}")

# ============================================================
# Bootstrap comparison (same methodology for both)
# ============================================================
print("\n[3/4] Running Monte Carlo bootstrap (1000 simulations each)...")

def backtest(returns, positions):
    equity = [1.0]
    for i, ret in enumerate(returns):
        equity.append(equity[-1] * (1 + ret * positions[i]))
    equity = np.array(equity)

    total_return = (equity[-1] / equity[0] - 1) * 100
    peak = np.maximum.accumulate(equity)
    max_dd = ((peak - equity) / peak).max() * 100

    return total_return, max_dd

def block_bootstrap(returns, positions, n_sims=1000, block_size=20):
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

# Run bootstrap for both
print("  Running BEFORE bootstrap...")
bootstrap_before = block_bootstrap(val_returns, pos_before)

print("  Running AFTER bootstrap...")
bootstrap_after = block_bootstrap(val_returns, pos_after)

# ============================================================
# Results comparison
# ============================================================
print("\n[4/4] Comparing results...")

print("\n" + "=" * 70)
print("MONTE CARLO COMPARISON (Same methodology, same data)")
print("=" * 70)

print("\n                        BEFORE              AFTER")
print("-" * 55)
print(f"Return 5th pctl:     {bootstrap_before['return'].quantile(0.05):>8.1f}%          {bootstrap_after['return'].quantile(0.05):>8.1f}%")
print(f"Return median:       {bootstrap_before['return'].quantile(0.50):>8.1f}%          {bootstrap_after['return'].quantile(0.50):>8.1f}%")
print(f"Return 95th pctl:    {bootstrap_before['return'].quantile(0.95):>8.1f}%          {bootstrap_after['return'].quantile(0.95):>8.1f}%")
print()
print(f"Max DD 5th pctl:     {bootstrap_before['max_dd'].quantile(0.05):>8.1f}%          {bootstrap_after['max_dd'].quantile(0.05):>8.1f}%")
print(f"Max DD median:       {bootstrap_before['max_dd'].quantile(0.50):>8.1f}%          {bootstrap_after['max_dd'].quantile(0.50):>8.1f}%")
print(f"Max DD 95th pctl:    {bootstrap_before['max_dd'].quantile(0.95):>8.1f}%          {bootstrap_after['max_dd'].quantile(0.95):>8.1f}%")

# Calculate miss rates
crash_mask = val_returns < -0.05
n_crashes = crash_mask.sum()

preds_before = probs_before.argmax(axis=1)
preds_after = probs_after.argmax(axis=1)
preds_after[probs_after[:, 2] > 0.20] = 2  # Apply low threshold

miss_before = ((preds_before == 0) & crash_mask[:len(preds_before)]).sum()
miss_after = ((preds_after == 0) & crash_mask[:len(preds_after)]).sum()

print(f"\nCrashes in validation: {n_crashes}")
print(f"Misses BEFORE: {miss_before} ({miss_before/n_crashes*100:.1f}%)")
print(f"Misses AFTER:  {miss_after} ({miss_after/n_crashes*100:.1f}%)")

# ============================================================
# Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(bootstrap_before['max_dd'], bins=30, alpha=0.6, label='BEFORE', color='coral')
ax.hist(bootstrap_after['max_dd'], bins=30, alpha=0.6, label='AFTER', color='steelblue')
ax.axvline(bootstrap_before['max_dd'].quantile(0.95), color='coral', linestyle='--',
           label=f'Before 95th: {bootstrap_before["max_dd"].quantile(0.95):.1f}%')
ax.axvline(bootstrap_after['max_dd'].quantile(0.95), color='steelblue', linestyle='--',
           label=f'After 95th: {bootstrap_after["max_dd"].quantile(0.95):.1f}%')
ax.axvline(40, color='red', linestyle=':', linewidth=2, label='Target (<40%)')
ax.set_xlabel('Max Drawdown (%)')
ax.set_ylabel('Frequency')
ax.set_title('Monte Carlo Max DD Distribution')
ax.legend()

ax = axes[1]
ax.hist(bootstrap_before['return'], bins=30, alpha=0.6, label='BEFORE', color='coral')
ax.hist(bootstrap_after['return'], bins=30, alpha=0.6, label='AFTER', color='steelblue')
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Return (%)')
ax.set_ylabel('Frequency')
ax.set_title('Monte Carlo Return Distribution')
ax.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'notebooks/phase5/before_after_comparison.png',
            dpi=100, bbox_inches='tight')
plt.close()

print(f"\nVisualization saved to: notebooks/phase5/before_after_comparison.png")
