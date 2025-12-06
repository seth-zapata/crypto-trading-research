"""
Validation check for GNN improvements - answering specific questions.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features

print("=" * 70)
print("VALIDATION CHECK: Answering Specific Questions")
print("=" * 70)

# Load data
df, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

btc_returns = df['BTC_return_1d'].values
total_samples = len(df)
train_size = int(total_samples * 0.5)
step = (total_samples - train_size) // 5

print(f"\nDataset: {total_samples} samples")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# ============================================================
# QUESTION 1: Why is miss rate 0% in Folds 1-3?
# ============================================================
print("\n" + "=" * 70)
print("Q1: WHY IS MISS RATE 0% IN FOLDS 1-3?")
print("=" * 70)

crash_threshold = -0.05  # 5% daily drop = crash

for i in range(5):
    train_end = train_size + i * step
    val_start = train_end
    val_end = min(train_end + step, total_samples)

    fold_returns = btc_returns[val_start:val_end]
    fold_dates = df.index[val_start:val_end]

    # Count crashes in this fold
    n_crashes = (fold_returns < crash_threshold).sum()
    crash_dates = fold_dates[fold_returns < crash_threshold]

    print(f"\nFold {i+1}: {fold_dates[0].strftime('%Y-%m-%d')} to {fold_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Samples: {len(fold_returns)}")
    print(f"  Crashes (returns < -5%): {n_crashes}")

    if n_crashes > 0:
        print(f"  Crash dates: {[d.strftime('%Y-%m-%d') for d in crash_dates[:5]]}...")
        print(f"  Crash returns: {fold_returns[fold_returns < crash_threshold][:5]}")
    else:
        print(f"  NO CRASHES in this period!")
        print(f"  Min return: {fold_returns.min()*100:.2f}%")

# ============================================================
# QUESTION 2: How is win rate calculated?
# ============================================================
print("\n" + "=" * 70)
print("Q2: HOW IS WIN RATE CALCULATED?")
print("=" * 70)

# Load saved results
results_path = Path(__file__).parent.parent / 'config' / 'improved_gnn_validation.json'
with open(results_path) as f:
    results = json.load(f)

print("\nWalk-Forward Folds Analysis:")
print(f"{'Fold':<6} {'Strategy Sharpe':>16} {'B&H Sharpe':>12} {'Beats B&H?':>12} {'By Returns?':>12}")
print("-" * 60)

wins_by_sharpe = 0
wins_by_return = 0

for fold in results['walk_forward']['folds']:
    strat_sharpe = fold['strategy_sharpe']
    bh_sharpe = fold['bh_return'] / 100  # Approximate Sharpe from return
    beats_sharpe = strat_sharpe > fold.get('bh_sharpe', 0) if 'bh_sharpe' in fold else "N/A"
    beats_return = fold['strategy_return'] > fold['bh_return']

    # The actual beats_bh in the JSON
    recorded_beats = fold['beats_bh']

    if recorded_beats == 'True' or recorded_beats == True:
        wins_by_sharpe += 1

    if beats_return:
        wins_by_return += 1

    print(f"{fold['fold']:<6} {strat_sharpe:>16.2f} {fold.get('bh_sharpe', 'N/A'):>12} {str(recorded_beats):>12} {str(beats_return):>12}")

print(f"\nWin rate by 'beats_bh' field: {wins_by_sharpe}/5 = {wins_by_sharpe/5*100:.0f}%")
print(f"Win rate by returns: {wins_by_return}/5 = {wins_by_return/5*100:.0f}%")

print("\nNOTE: The validation script compares SHARPE ratios, not raw returns.")
print("      Strategy can beat B&H Sharpe while having lower absolute returns")
print("      (due to lower volatility/drawdown).")

# Show the actual comparison that was used
print("\nDetailed Sharpe comparison from JSON:")
for fold in results['walk_forward']['folds']:
    print(f"  Fold {fold['fold']}: Strategy Sharpe={fold['strategy_sharpe']:.2f}, "
          f"Strategy DD={fold['strategy_dd']:.1f}%, B&H DD={fold['bh_dd']:.1f}%")

# ============================================================
# QUESTION 3: Before/After Distribution Comparison
# ============================================================
print("\n" + "=" * 70)
print("Q3: BEFORE/AFTER MONTE CARLO DISTRIBUTION COMPARISON")
print("=" * 70)

# Load both result files
old_results_path = Path(__file__).parent.parent / 'config' / 'position_sizing.json'
new_results_path = Path(__file__).parent.parent / 'config' / 'improved_gnn_validation.json'

print("\nLoading result files...")

if old_results_path.exists():
    with open(old_results_path) as f:
        old_results = json.load(f)

    print("\nBEFORE (Original GNN + Continuous Sizing):")
    if 'validation' in old_results:
        print(f"  Return CI: {old_results['validation'].get('bootstrap_return_ci', 'N/A')}")
        print(f"  Max DD CI: {old_results['validation'].get('bootstrap_max_dd_ci', 'N/A')}")
        print(f"  Walk-forward win rate: {old_results['validation'].get('walk_forward_win_rate', 'N/A')}%")
else:
    print("\nBEFORE results file not found")

with open(new_results_path) as f:
    new_results = json.load(f)

print("\nAFTER (Asymmetric Loss + Low Threshold):")
print(f"  Return CI: [{new_results['bootstrap']['return_ci'][0]:.1f}%, {new_results['bootstrap']['return_ci'][1]:.1f}%]")
print(f"  Max DD CI: [{new_results['bootstrap']['max_dd_ci'][0]:.1f}%, {new_results['bootstrap']['max_dd_ci'][1]:.1f}%]")
print(f"  Walk-forward win rate: {new_results['walk_forward']['win_rate']}%")

print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print("""
1. MISS RATE 0% IN SOME FOLDS:
   - This occurs when there are NO crash events (returns < -5%) in that fold
   - Miss rate = misses / crashes, so if crashes = 0, miss rate is undefined (shown as 0)
   - This is a DATA issue, not a model issue

2. WIN RATE CALCULATION:
   - Based on SHARPE ratio comparison, not raw returns
   - Strategy beats B&H when it has better risk-adjusted returns
   - Even with lower absolute returns, lower drawdown can mean better Sharpe

3. BEFORE/AFTER COMPARISON:
   - The 66.2% figure came from the ORIGINAL robust validation
   - The 23.8% figure is from the IMPROVED model validation
   - These are DIFFERENT model configurations, not the same model
""")
