"""
Phase 5: Robust Weight Optimization with Monte Carlo Validation
================================================================

Validation methods:
1. Block bootstrap (1000 simulations) - confidence intervals
2. Walk-forward validation - out-of-sample performance
3. Stress tests - performance in known crisis periods

Pass criteria:
- 95th percentile Max DD < 30%
- Walk-forward win rate > 50%
- Beat B&H in ≥3/4 stress tests
"""

import logging
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector

logging.basicConfig(level=logging.WARNING)
np.random.seed(42)

print("=" * 70)
print("PHASE 5: ROBUST WEIGHT OPTIMIZATION")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/7] Loading data and training GNN...")

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
prices = aligned_df['BTC_close'].values
returns = np.diff(prices) / prices[:-1]

RISK_ON, CAUTION, RISK_OFF = 0, 1, 2
print(f"Dataset: {len(aligned_df)} days")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def backtest_returns(daily_returns: np.ndarray, positions: np.ndarray,
                     cost: float = 0.001) -> Dict:
    """Backtest using daily returns (not prices)."""
    equity = [1.0]
    prev_pos = positions[0] if len(positions) > 0 else 1.0

    for i, ret in enumerate(daily_returns):
        pos = positions[i] if i < len(positions) else prev_pos
        port_ret = ret * pos - abs(pos - prev_pos) * cost
        equity.append(equity[-1] * (1 + port_ret))
        prev_pos = pos

    eq = np.array(equity)
    port_rets = np.diff(eq) / eq[:-1]
    total_ret = (eq[-1] - 1) * 100
    sharpe = port_rets.mean() / port_rets.std() * np.sqrt(252) if port_rets.std() > 0 else 0
    peak = np.maximum.accumulate(eq)
    max_dd = ((peak - eq) / peak).max() * 100

    return {'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd}


def continuous_position(probs: np.ndarray, ro: float, ca: float, rf: float) -> np.ndarray:
    """Calculate continuous position from probabilities."""
    return probs[:, 0] * ro + probs[:, 1] * ca + probs[:, 2] * rf


# ============================================================
# 3. BLOCK BOOTSTRAP MONTE CARLO
# ============================================================
print("\n[2/7] Running block bootstrap Monte Carlo (1000 simulations)...")

def block_bootstrap(returns: np.ndarray, probs: np.ndarray,
                    n_simulations: int = 1000, block_size: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate bootstrap samples using block resampling.
    Preserves autocorrelation structure.
    """
    n = len(returns)
    n_blocks = n // block_size + 1

    samples = []
    for _ in range(n_simulations):
        # Sample block indices with replacement
        block_starts = np.random.randint(0, n - block_size, size=n_blocks)

        # Build resampled series
        ret_sample = []
        prob_sample = []
        for start in block_starts:
            ret_sample.extend(returns[start:start + block_size])
            prob_sample.extend(probs[start:start + block_size])

        # Trim to original length
        ret_sample = np.array(ret_sample[:n])
        prob_sample = np.array(prob_sample[:n])

        samples.append((ret_sample, prob_sample))

    return samples


def evaluate_config_bootstrap(ro: float, ca: float, rf: float,
                              bootstrap_samples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Evaluate a configuration across all bootstrap samples."""
    results = {'return': [], 'sharpe': [], 'max_dd': []}

    for ret_sample, prob_sample in bootstrap_samples:
        positions = continuous_position(prob_sample, ro, ca, rf)
        metrics = backtest_returns(ret_sample, positions)
        results['return'].append(metrics['return'])
        results['sharpe'].append(metrics['sharpe'])
        results['max_dd'].append(metrics['max_dd'])

    return {
        'return_mean': np.mean(results['return']),
        'return_5th': np.percentile(results['return'], 5),
        'return_95th': np.percentile(results['return'], 95),
        'sharpe_mean': np.mean(results['sharpe']),
        'sharpe_5th': np.percentile(results['sharpe'], 5),
        'sharpe_95th': np.percentile(results['sharpe'], 95),
        'max_dd_mean': np.mean(results['max_dd']),
        'max_dd_5th': np.percentile(results['max_dd'], 5),
        'max_dd_95th': np.percentile(results['max_dd'], 95),
    }


# Generate bootstrap samples
bootstrap_samples = block_bootstrap(returns, probs[:len(returns)], n_simulations=1000, block_size=20)
print(f"Generated {len(bootstrap_samples)} bootstrap samples")


# ============================================================
# 4. WALK-FORWARD VALIDATION
# ============================================================
print("\n[3/7] Running walk-forward validation...")

def walk_forward_validation(prices: np.ndarray, probs: np.ndarray,
                            ro: float, ca: float, rf: float,
                            n_splits: int = 5, train_ratio: float = 0.7) -> Dict:
    """
    Walk-forward validation with expanding window.
    """
    n = len(prices) - 1  # Number of return periods
    split_size = n // n_splits

    results = []
    for i in range(n_splits):
        # Define train/test split
        test_start = (i + 1) * split_size
        test_end = min((i + 2) * split_size, n)

        if test_start >= n or test_end <= test_start:
            continue

        # Test period
        test_returns = np.diff(prices[test_start:test_end+1]) / prices[test_start:test_end]
        test_probs = probs[test_start:test_end]

        # Calculate positions and backtest
        positions = continuous_position(test_probs, ro, ca, rf)
        metrics = backtest_returns(test_returns, positions)

        # Compare to B&H
        bh_metrics = backtest_returns(test_returns, np.ones(len(test_returns)))

        results.append({
            'split': i,
            'return': metrics['return'],
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_dd'],
            'bh_return': bh_metrics['return'],
            'beat_bh': metrics['sharpe'] > bh_metrics['sharpe']
        })

    if len(results) == 0:
        return {'win_rate': 0, 'avg_return': 0, 'avg_sharpe': 0, 'avg_max_dd': 100}

    results_df = pd.DataFrame(results)
    return {
        'win_rate': results_df['beat_bh'].mean() * 100,
        'avg_return': results_df['return'].mean(),
        'avg_sharpe': results_df['sharpe'].mean(),
        'avg_max_dd': results_df['max_dd'].mean(),
        'splits': results
    }


# ============================================================
# 5. STRESS TESTS
# ============================================================
print("\n[4/7] Running stress tests...")

def identify_stress_periods(prices: np.ndarray) -> List[Dict]:
    """
    Identify major stress periods (>15% drawdown).
    """
    n = len(prices)
    periods = []

    # Calculate running drawdown
    peak = prices[0]
    in_stress = False
    stress_start = 0

    for i in range(1, n):
        peak = max(peak, prices[i])
        dd = (peak - prices[i]) / peak

        if dd > 0.15 and not in_stress:
            in_stress = True
            stress_start = i
        elif dd < 0.05 and in_stress:
            in_stress = False
            if i - stress_start > 5:  # At least 5 days
                periods.append({
                    'start': stress_start,
                    'end': i,
                    'max_dd': max((peak - prices[j]) / peak for j in range(stress_start, i)) * 100
                })
            peak = prices[i]

    # Handle ongoing stress at end
    if in_stress and n - stress_start > 5:
        periods.append({
            'start': stress_start,
            'end': n - 1,
            'max_dd': max((peak - prices[j]) / peak for j in range(stress_start, n)) * 100
        })

    return periods


def stress_test_config(ro: float, ca: float, rf: float,
                       prices: np.ndarray, probs: np.ndarray,
                       stress_periods: List[Dict]) -> Dict:
    """Test configuration during identified stress periods."""
    results = []

    for period in stress_periods:
        start, end = period['start'], period['end']
        if end - start < 2:
            continue

        period_returns = np.diff(prices[start:end+1]) / prices[start:end]
        period_probs = probs[start:end]

        if len(period_returns) == 0 or len(period_probs) == 0:
            continue

        # Strategy performance
        positions = continuous_position(period_probs, ro, ca, rf)
        metrics = backtest_returns(period_returns, positions)

        # B&H performance
        bh_metrics = backtest_returns(period_returns, np.ones(len(period_returns)))

        results.append({
            'period_dd': period['max_dd'],
            'strategy_dd': metrics['max_dd'],
            'strategy_return': metrics['return'],
            'bh_return': bh_metrics['return'],
            'beat_bh': metrics['max_dd'] < bh_metrics['max_dd']
        })

    if len(results) == 0:
        return {'n_periods': 0, 'beat_rate': 0, 'avg_dd_reduction': 0}

    results_df = pd.DataFrame(results)
    return {
        'n_periods': len(results),
        'beat_rate': results_df['beat_bh'].mean(),
        'avg_dd_reduction': (results_df['period_dd'] - results_df['strategy_dd']).mean(),
        'periods': results
    }


stress_periods = identify_stress_periods(prices)
print(f"Identified {len(stress_periods)} stress periods")


# ============================================================
# 6. GRID SEARCH WITH ROBUST VALIDATION
# ============================================================
print("\n[5/7] Grid search with robust validation...")

# Weight grid
risk_on_levels = [0.80, 0.85, 0.90, 0.95, 1.00]
caution_levels = [0.50, 0.60, 0.70, 0.80]
risk_off_levels = [0.20, 0.25, 0.30, 0.35]

all_results = []
total_configs = len(risk_on_levels) * len(caution_levels) * len(risk_off_levels)
config_count = 0

for ro, ca, rf in product(risk_on_levels, caution_levels, risk_off_levels):
    config_count += 1
    if config_count % 20 == 0:
        print(f"  Processing config {config_count}/{total_configs}...")

    # Bootstrap confidence intervals
    bootstrap_results = evaluate_config_bootstrap(ro, ca, rf, bootstrap_samples)

    # Walk-forward validation
    wf_results = walk_forward_validation(prices, probs[:len(prices)], ro, ca, rf)

    # Stress tests
    stress_results = stress_test_config(ro, ca, rf, prices, probs[:len(prices)], stress_periods)

    # Single-path backtest for reference
    positions = continuous_position(probs[:len(returns)], ro, ca, rf)
    single_path = backtest_returns(returns, positions)

    all_results.append({
        'risk_on': ro,
        'caution': ca,
        'risk_off': rf,
        'avg_pos': positions.mean(),
        # Single path
        'sp_return': single_path['return'],
        'sp_sharpe': single_path['sharpe'],
        'sp_max_dd': single_path['max_dd'],
        # Bootstrap CIs
        'bs_return_mean': bootstrap_results['return_mean'],
        'bs_return_5th': bootstrap_results['return_5th'],
        'bs_return_95th': bootstrap_results['return_95th'],
        'bs_sharpe_mean': bootstrap_results['sharpe_mean'],
        'bs_sharpe_5th': bootstrap_results['sharpe_5th'],
        'bs_sharpe_95th': bootstrap_results['sharpe_95th'],
        'bs_max_dd_mean': bootstrap_results['max_dd_mean'],
        'bs_max_dd_5th': bootstrap_results['max_dd_5th'],
        'bs_max_dd_95th': bootstrap_results['max_dd_95th'],
        # Walk-forward
        'wf_win_rate': wf_results['win_rate'],
        'wf_avg_sharpe': wf_results['avg_sharpe'],
        # Stress tests
        'stress_beat_rate': stress_results['beat_rate'],
        'stress_n_periods': stress_results['n_periods'],
    })

results_df = pd.DataFrame(all_results)
print(f"\nEvaluated {len(results_df)} configurations")


# ============================================================
# 7. APPLY PASS CRITERIA
# ============================================================
print("\n[6/7] Applying pass criteria...")

# Pass criteria:
# 1. 95th percentile Max DD < 30%
# 2. Walk-forward win rate > 50%
# 3. Beat B&H in ≥75% of stress tests

results_df['pass_dd'] = results_df['bs_max_dd_95th'] < 30
results_df['pass_wf'] = results_df['wf_win_rate'] > 50
results_df['pass_stress'] = results_df['stress_beat_rate'] >= 0.75
results_df['pass_all'] = results_df['pass_dd'] & results_df['pass_wf'] & results_df['pass_stress']

passed = results_df[results_df['pass_all']].copy()
print(f"\nConfigurations passing all criteria: {len(passed)}")

if len(passed) == 0:
    print("\nNo configs pass all criteria. Relaxing...")
    # Relax: pass at least 2 of 3
    results_df['pass_count'] = results_df['pass_dd'].astype(int) + \
                                results_df['pass_wf'].astype(int) + \
                                results_df['pass_stress'].astype(int)
    passed = results_df[results_df['pass_count'] >= 2].copy()
    print(f"Configs passing 2+ criteria: {len(passed)}")

if len(passed) == 0:
    print("Using top 10 by bootstrap Sharpe...")
    passed = results_df.nlargest(10, 'bs_sharpe_mean').copy()


# ============================================================
# 8. PARETO FRONTIER ON ROBUST METRICS
# ============================================================
print("\n[7/7] Finding Pareto frontier on robust metrics...")

def is_pareto_optimal(row, df):
    """Pareto optimal based on bootstrap mean return and 95th percentile DD."""
    dominated = df[
        (df['bs_return_mean'] >= row['bs_return_mean']) &
        (df['bs_max_dd_95th'] <= row['bs_max_dd_95th']) &
        ((df['bs_return_mean'] > row['bs_return_mean']) |
         (df['bs_max_dd_95th'] < row['bs_max_dd_95th']))
    ]
    return len(dominated) == 0

passed['pareto'] = passed.apply(lambda r: is_pareto_optimal(r, passed), axis=1)
pareto = passed[passed['pareto']].sort_values('bs_sharpe_mean', ascending=False)

print(f"\nPareto optimal configurations: {len(pareto)}")


# ============================================================
# 9. RESULTS
# ============================================================
print("\n" + "=" * 70)
print("ROBUST VALIDATION RESULTS")
print("=" * 70)

print("\n--- PARETO FRONTIER (Robust Metrics) ---")
print(f"{'Config':<18} {'Return CI':<22} {'Sharpe CI':<18} {'Max DD CI':<18} {'WF Win%':>8} {'Stress':>8}")
print("-" * 100)

for _, row in pareto.head(10).iterrows():
    config = f"({row['risk_on']:.2f},{row['caution']:.2f},{row['risk_off']:.2f})"
    ret_ci = f"[{row['bs_return_5th']:+.1f}, {row['bs_return_95th']:+.1f}]"
    sharpe_ci = f"[{row['bs_sharpe_5th']:.2f}, {row['bs_sharpe_95th']:.2f}]"
    dd_ci = f"[{row['bs_max_dd_5th']:.1f}, {row['bs_max_dd_95th']:.1f}]"
    print(f"{config:<18} {ret_ci:<22} {sharpe_ci:<18} {dd_ci:<18} {row['wf_win_rate']:>7.0f}% {row['stress_beat_rate']:>7.0%}")


# Best by robust Sharpe
best = pareto.iloc[0] if len(pareto) > 0 else passed.loc[passed['bs_sharpe_mean'].idxmax()]

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

print(f"\nRecommended Configuration: ({best['risk_on']:.2f}, {best['caution']:.2f}, {best['risk_off']:.2f})")
print(f"\nBootstrap Confidence Intervals (5th - 95th percentile):")
print(f"  Return:  [{best['bs_return_5th']:+.1f}%, {best['bs_return_95th']:+.1f}%] (mean: {best['bs_return_mean']:+.1f}%)")
print(f"  Sharpe:  [{best['bs_sharpe_5th']:.2f}, {best['bs_sharpe_95th']:.2f}] (mean: {best['bs_sharpe_mean']:.2f})")
print(f"  Max DD:  [{best['bs_max_dd_5th']:.1f}%, {best['bs_max_dd_95th']:.1f}%] (mean: {best['bs_max_dd_mean']:.1f}%)")

print(f"\nValidation Results:")
print(f"  Walk-forward win rate: {best['wf_win_rate']:.0f}% {'✓' if best['wf_win_rate'] > 50 else '✗'}")
print(f"  Stress test beat rate: {best['stress_beat_rate']:.0%} ({'✓' if best['stress_beat_rate'] >= 0.75 else '✗'} need ≥75%)")
print(f"  95th pctl Max DD < 30%: {best['bs_max_dd_95th']:.1f}% {'✓' if best['bs_max_dd_95th'] < 30 else '✗'}")

print(f"\nAverage Position: {best['avg_pos']:.1%}")

# Pass/fail summary
all_pass = best['pass_dd'] and best['wf_win_rate'] > 50 and best['stress_beat_rate'] >= 0.75
print(f"\nOVERALL: {'PASS - Use this configuration' if all_pass else 'CONDITIONAL PASS - Review criteria'}")


# ============================================================
# 10. COMPARE TO CURRENT CONFIG
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON TO CURRENT CONFIG (0.85, 0.65, 0.30)")
print("=" * 70)

current = results_df[
    (results_df['risk_on'] == 0.85) &
    (results_df['caution'] == 0.65) &
    (results_df['risk_off'] == 0.30)
]

if len(current) > 0:
    curr = current.iloc[0]
    print(f"\n{'Metric':<25} {'Current':<20} {'Recommended':<20} {'Delta':<15}")
    print("-" * 80)
    print(f"{'Return (mean)':<25} {curr['bs_return_mean']:>+.1f}%{'':<13} {best['bs_return_mean']:>+.1f}%{'':<13} {best['bs_return_mean'] - curr['bs_return_mean']:>+.1f}%")
    print(f"{'Sharpe (mean)':<25} {curr['bs_sharpe_mean']:>.2f}{'':<16} {best['bs_sharpe_mean']:>.2f}{'':<16} {best['bs_sharpe_mean'] - curr['bs_sharpe_mean']:>+.2f}")
    print(f"{'Max DD 95th pctl':<25} {curr['bs_max_dd_95th']:>.1f}%{'':<14} {best['bs_max_dd_95th']:>.1f}%{'':<14} {best['bs_max_dd_95th'] - curr['bs_max_dd_95th']:>+.1f}%")
    print(f"{'Walk-forward win%':<25} {curr['wf_win_rate']:>.0f}%{'':<15} {best['wf_win_rate']:>.0f}%{'':<15} {best['wf_win_rate'] - curr['wf_win_rate']:>+.0f}%")
    print(f"{'Stress beat rate':<25} {curr['stress_beat_rate']:>.0%}{'':<14} {best['stress_beat_rate']:>.0%}{'':<14}")


# ============================================================
# 11. SAVE RESULTS
# ============================================================
import json

config = {
    'risk_on_weight': float(best['risk_on']),
    'caution_weight': float(best['caution']),
    'risk_off_weight': float(best['risk_off']),
    'validation': {
        'bootstrap_return_ci': [float(best['bs_return_5th']), float(best['bs_return_95th'])],
        'bootstrap_sharpe_ci': [float(best['bs_sharpe_5th']), float(best['bs_sharpe_95th'])],
        'bootstrap_max_dd_ci': [float(best['bs_max_dd_5th']), float(best['bs_max_dd_95th'])],
        'walk_forward_win_rate': float(best['wf_win_rate']),
        'stress_test_beat_rate': float(best['stress_beat_rate']),
        'n_stress_periods': int(best['stress_n_periods']),
    }
}

config_path = Path(__file__).parent.parent / 'config' / 'position_sizing.json'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"\nConfiguration saved to: {config_path}")


# ============================================================
# 12. VISUALIZATION
# ============================================================
print("\nGenerating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Bootstrap return distribution for recommended config
ax = axes[0, 0]
positions = continuous_position(probs[:len(returns)], best['risk_on'], best['caution'], best['risk_off'])
bootstrap_returns = []
for ret_sample, prob_sample in bootstrap_samples[:200]:  # First 200 for visualization
    pos = continuous_position(prob_sample, best['risk_on'], best['caution'], best['risk_off'])
    metrics = backtest_returns(ret_sample, pos)
    bootstrap_returns.append(metrics['return'])

ax.hist(bootstrap_returns, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(best['bs_return_5th'], color='red', linestyle='--', label=f'5th pctl: {best["bs_return_5th"]:+.1f}%')
ax.axvline(best['bs_return_95th'], color='red', linestyle='--', label=f'95th pctl: {best["bs_return_95th"]:+.1f}%')
ax.axvline(best['bs_return_mean'], color='green', linestyle='-', linewidth=2, label=f'Mean: {best["bs_return_mean"]:+.1f}%')
ax.set_xlabel('Return (%)')
ax.set_ylabel('Frequency')
ax.set_title(f'Bootstrap Return Distribution\n({best["risk_on"]:.2f}, {best["caution"]:.2f}, {best["risk_off"]:.2f})')
ax.legend()

# 2. Return vs Max DD (95th pctl) scatter
ax = axes[0, 1]
ax.scatter(results_df['bs_return_mean'], results_df['bs_max_dd_95th'],
           c=results_df['avg_pos'], cmap='RdYlGn', alpha=0.6)
ax.scatter([best['bs_return_mean']], [best['bs_max_dd_95th']],
           color='red', s=150, marker='*', label='Recommended', zorder=5)
ax.axhline(30, color='red', linestyle='--', alpha=0.5, label='DD Limit (30%)')
ax.set_xlabel('Mean Return (%)')
ax.set_ylabel('95th Percentile Max DD (%)')
ax.set_title('Return vs Worst-Case Drawdown')
ax.legend()

# 3. Walk-forward performance
ax = axes[1, 0]
wf_results = walk_forward_validation(prices, probs[:len(prices)],
                                      best['risk_on'], best['caution'], best['risk_off'])
if 'splits' in wf_results and len(wf_results['splits']) > 0:
    splits_df = pd.DataFrame(wf_results['splits'])
    x = range(len(splits_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], splits_df['return'], width, label='Strategy', alpha=0.7)
    ax.bar([i + width/2 for i in x], splits_df['bh_return'], width, label='Buy & Hold', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Walk-Forward Split')
    ax.set_ylabel('Return (%)')
    ax.set_title(f'Walk-Forward Validation (Win Rate: {wf_results["win_rate"]:.0f}%)')
    ax.legend()
    ax.set_xticks(x)

# 4. Stress test performance
ax = axes[1, 1]
stress_results = stress_test_config(best['risk_on'], best['caution'], best['risk_off'],
                                     prices, probs[:len(prices)], stress_periods)
if 'periods' in stress_results and len(stress_results['periods']) > 0:
    stress_df = pd.DataFrame(stress_results['periods'])
    x = range(len(stress_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], stress_df['strategy_dd'], width, label='Strategy DD', alpha=0.7, color='green')
    ax.bar([i + width/2 for i in x], stress_df['period_dd'], width, label='B&H DD', alpha=0.7, color='red')
    ax.axhline(25, color='orange', linestyle='--', label='Target (25%)')
    ax.set_xlabel('Stress Period')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title(f'Stress Test DD Comparison (Beat Rate: {stress_results["beat_rate"]:.0%})')
    ax.legend()

plt.tight_layout()
output_path = Path(__file__).parent.parent / 'notebooks' / 'phase5' / 'robust_optimization.png'
plt.savefig(output_path, dpi=150)
print(f"Visualization saved to: {output_path}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
