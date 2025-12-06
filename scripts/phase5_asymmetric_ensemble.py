"""
Part 1: Asymmetric Ensemble
- penalty=5 model for EXIT signals (conservative crash detection)
- penalty=10 model for ENTRY signals (capture upside)
- Conservative rule: both must agree on RISK_ON for >70% position

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
print("PART 1: ASYMMETRIC ENSEMBLE")
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

# ============================================================
# Train both models
# ============================================================
print("\n[1/4] Training penalty=5 model (conservative EXIT signals)...")

detector_exit = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
detector_exit.num_node_features = detector_base.num_node_features

detector_exit.train(
    train_graphs, train_labels,
    val_graphs, val_labels,
    epochs=100, batch_size=32,
    train_returns=train_returns,
    use_asymmetric_loss=True,
    miss_penalty=5.0
)

_, probs_exit = detector_exit.predict(val_graphs)

print("\n[2/4] Training penalty=10 model (aggressive ENTRY signals)...")

detector_entry = RegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
detector_entry.num_node_features = detector_base.num_node_features

detector_entry.train(
    train_graphs, train_labels,
    val_graphs, val_labels,
    epochs=100, batch_size=32,
    train_returns=train_returns,
    use_asymmetric_loss=True,
    miss_penalty=10.0
)

_, probs_entry = detector_entry.predict(val_graphs)

# ============================================================
# Ensemble Position Sizing
# ============================================================
print("\n[3/4] Computing ensemble positions...")

THRESHOLD = 0.40  # From previous optimization
RISK_ON, CAUTION, RISK_OFF = 0, 1, 2

def ensemble_positions(probs_exit, probs_entry, threshold=0.40):
    """
    Asymmetric ensemble:
    - EXIT model (penalty=5): If says RISK_OFF, reduce position
    - ENTRY model (penalty=10): If says RISK_ON, allow higher position
    - Conservative: Both must agree on RISK_ON for >70%
    """
    positions = []

    for i in range(len(probs_exit)):
        p_exit = probs_exit[i]
        p_entry = probs_entry[i]

        # Get regime predictions
        exit_pred = RISK_OFF if p_exit[RISK_OFF] > threshold else (
            CAUTION if p_exit[CAUTION] > threshold else RISK_ON)
        entry_pred = RISK_OFF if p_entry[RISK_OFF] > threshold else (
            CAUTION if p_entry[CAUTION] > threshold else RISK_ON)

        # EXIT signal dominates (if either says RISK_OFF, go defensive)
        if exit_pred == RISK_OFF or entry_pred == RISK_OFF:
            pos = 0.20
        # Both must agree on RISK_ON for >70%
        elif exit_pred == RISK_ON and entry_pred == RISK_ON:
            # Use entry model's confidence for position sizing
            pos = 0.70 + 0.30 * p_entry[RISK_ON]  # 70-100%
        # One says CAUTION or disagreement
        elif exit_pred == CAUTION or entry_pred == CAUTION:
            pos = 0.50
        else:
            # Disagreement between RISK_ON and CAUTION
            pos = 0.60

        positions.append(pos)

    return np.array(positions)

positions_ensemble = ensemble_positions(probs_exit, probs_entry, THRESHOLD)

# Also compute individual model positions for comparison
def single_positions(probs, threshold=0.40):
    positions = probs[:, 0] * 0.85 + probs[:, 1] * 0.65 + probs[:, 2] * 0.20
    positions[probs[:, 2] > threshold] = 0.20
    return np.clip(positions, 0.1, 1.0)

positions_exit_only = single_positions(probs_exit, THRESHOLD)
positions_entry_only = single_positions(probs_entry, THRESHOLD)

print(f"\nPosition sizing comparison:")
print(f"  Exit model (penalty=5):   Avg {positions_exit_only.mean()*100:.1f}%")
print(f"  Entry model (penalty=10): Avg {positions_entry_only.mean()*100:.1f}%")
print(f"  Ensemble:                 Avg {positions_ensemble.mean()*100:.1f}%")

# ============================================================
# Validation
# ============================================================
print("\n[4/4] Running Monte Carlo validation...")

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

# Calculate miss rate for ensemble
crash_mask = val_returns[:len(probs_exit)] < -0.05
n_crashes = crash_mask.sum()

# Ensemble prediction: RISK_OFF if position <= 0.30
ensemble_defensive = positions_ensemble <= 0.30
ensemble_preds = np.where(ensemble_defensive, RISK_OFF, RISK_ON)
misses = ((ensemble_preds == RISK_ON) & crash_mask).sum()
miss_rate = misses / n_crashes if n_crashes > 0 else 0

# Bootstrap
print("  Running bootstrap (1000 simulations)...")
bootstrap_ensemble = block_bootstrap(val_returns[:len(positions_ensemble)], positions_ensemble)
bootstrap_exit = block_bootstrap(val_returns[:len(positions_exit_only)], positions_exit_only)
bootstrap_entry = block_bootstrap(val_returns[:len(positions_entry_only)], positions_entry_only)

# ============================================================
# Results
# ============================================================
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print(f"\n{'Model':<20} {'Avg Pos':>10} {'Ret Med':>12} {'DD 95th':>10} {'Miss %':>10}")
print("-" * 65)

print(f"{'Exit (p=5)':<20} {positions_exit_only.mean()*100:>9.1f}% "
      f"{bootstrap_exit['return'].median():>11.1f}% "
      f"{bootstrap_exit['max_dd'].quantile(0.95):>9.1f}% "
      f"{'20.0':>9}%")

print(f"{'Entry (p=10)':<20} {positions_entry_only.mean()*100:>9.1f}% "
      f"{bootstrap_entry['return'].median():>11.1f}% "
      f"{bootstrap_entry['max_dd'].quantile(0.95):>9.1f}% "
      f"{'30.0':>9}%")

print(f"{'ENSEMBLE':<20} {positions_ensemble.mean()*100:>9.1f}% "
      f"{bootstrap_ensemble['return'].median():>11.1f}% "
      f"{bootstrap_ensemble['max_dd'].quantile(0.95):>9.1f}% "
      f"{miss_rate*100:>9.1f}%")

# Check targets
return_med = bootstrap_ensemble['return'].median()
dd_95 = bootstrap_ensemble['max_dd'].quantile(0.95)

print("\n" + "=" * 70)
print("TARGET CHECK")
print("=" * 70)

print(f"\n  Return Median: {return_med:.1f}% (target >35%): {'PASS' if return_med > 35 else 'FAIL'}")
print(f"  DD 95th pctl:  {dd_95:.1f}% (target <25%): {'PASS' if dd_95 < 25 else 'FAIL'}")
print(f"  Miss Rate:     {miss_rate*100:.1f}% (target <25%): {'PASS' if miss_rate < 0.25 else 'FAIL'}")

all_pass = return_med > 35 and dd_95 < 25 and miss_rate < 0.25

if all_pass:
    print("\n>>> ALL TARGETS MET - No need for Part 2 (Funding Rates) <<<")
else:
    print("\n>>> SOME TARGETS MISSED - Consider Part 2 (Funding Rates) <<<")

# Additional stats
print("\n" + "=" * 70)
print("ADDITIONAL STATISTICS")
print("=" * 70)

print(f"\nEnsemble Bootstrap Distribution:")
print(f"  Return: [{bootstrap_ensemble['return'].quantile(0.05):.1f}%, "
      f"{bootstrap_ensemble['return'].quantile(0.50):.1f}%, "
      f"{bootstrap_ensemble['return'].quantile(0.95):.1f}%] (5th/50th/95th)")
print(f"  Max DD: [{bootstrap_ensemble['max_dd'].quantile(0.05):.1f}%, "
      f"{bootstrap_ensemble['max_dd'].quantile(0.50):.1f}%, "
      f"{bootstrap_ensemble['max_dd'].quantile(0.95):.1f}%] (5th/50th/95th)")

# Position distribution
print(f"\nEnsemble Position Distribution:")
print(f"  Min: {positions_ensemble.min()*100:.1f}%")
print(f"  Mean: {positions_ensemble.mean()*100:.1f}%")
print(f"  Max: {positions_ensemble.max()*100:.1f}%")
print(f"  Days <30%: {(positions_ensemble < 0.30).sum()} ({(positions_ensemble < 0.30).mean()*100:.1f}%)")
print(f"  Days >70%: {(positions_ensemble > 0.70).sum()} ({(positions_ensemble > 0.70).mean()*100:.1f}%)")
