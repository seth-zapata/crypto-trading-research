"""
Phase 3 Regime Override Backtest
================================

On-chain signals (MVRV, SOPR) used as REGIME OVERRIDES, not ML features.

Key insight: On-chain metrics are valuable at extremes (cycle tops/bottoms),
not during neutral periods. They should override ML predictions only when
they have strong conviction.

Override Logic:
- MVRV < 1.0  → Force bullish (cycle bottom)
- MVRV > 3.7  → Force bearish (cycle top)
- SOPR < 0.95 → Force bullish (capitulation)
- SOPR > 1.05 → Force bearish (euphoria)
- Otherwise   → Trust ML prediction

Backtest Period: 2020-2024 (to capture actual cycle extremes)
"""

import asyncio
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("PHASE 3 REGIME OVERRIDE BACKTEST")
print("On-chain signals as REGIME OVERRIDES (not ML features)")
print("=" * 70)


# ============================================================
# REGIME OVERRIDE LOGIC
# ============================================================

def apply_regime_overrides(
    ml_signal: int,
    mvrv: float,
    sopr: float
) -> Tuple[int, str]:
    """
    Apply on-chain regime overrides to ML predictions.

    Returns:
        Tuple of (final_signal, override_reason)
        override_reason is 'none' if ML signal was used
    """
    # MVRV extremes - highest conviction overrides
    if mvrv < 1.0:
        return 1, 'mvrv_bottom'  # Force bullish - cycle bottom
    if mvrv > 3.7:
        return 0, 'mvrv_top'  # Force bearish - cycle top

    # SOPR extremes - secondary overrides
    if sopr < 0.95:
        return 1, 'sopr_capitulation'  # Force bullish - capitulation
    if sopr > 1.05:
        return 0, 'sopr_euphoria'  # Force bearish - euphoria

    # No extreme detected - trust ML
    return ml_signal, 'none'


# ============================================================
# 1. FETCH HISTORICAL ON-CHAIN DATA (2015-present)
# ============================================================
print("\n[1/7] Fetching historical on-chain data...")

import yaml
from data.ingestion.onchain import OnChainDataProvider, CoinMetricsProvider

config_path = Path(__file__).parent.parent / 'config' / 'api_keys.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)


async def fetch_all_onchain_history():
    """Fetch complete historical on-chain data."""

    # CoinMetrics MVRV (free tier - from 2015)
    print("   Fetching MVRV from CoinMetrics (2015-present)...")
    cm = CoinMetricsProvider()
    mvrv_df = await cm.fetch_mvrv('btc', start_date='2015-01-01')
    mvrv_df = mvrv_df.set_index('time')
    mvrv_df.index = mvrv_df.index.tz_localize(None)
    print(f"   MVRV: {len(mvrv_df)} rows ({mvrv_df.index.min().date()} to {mvrv_df.index.max().date()})")

    # Dune SOPR (from 2010)
    print("   Fetching SOPR from Dune Analytics (2010-present)...")
    provider = OnChainDataProvider(config['dune']['api_key'])
    sopr_df = await provider.fetch_sopr()
    sopr_df['date'] = pd.to_datetime(sopr_df['date'])
    sopr_df = sopr_df.set_index('date')
    sopr_df = sopr_df[['normalized_sopr']].rename(columns={'normalized_sopr': 'sopr'})
    print(f"   SOPR: {len(sopr_df)} rows ({sopr_df.index.min().date()} to {sopr_df.index.max().date()})")

    return mvrv_df, sopr_df


mvrv_history, sopr_history = asyncio.run(fetch_all_onchain_history())


# ============================================================
# 2. FETCH HISTORICAL PRICE DATA (2020-2024)
# ============================================================
print("\n[2/7] Fetching historical price data (2020-2024)...")

import yfinance as yf

# Use yfinance for longer history (Binance geo-blocked, Kraken limited)
print("   Fetching daily candles from Yahoo Finance (2020-present)...")
btc = yf.Ticker("BTC-USD")
df = btc.history(start="2020-01-01", end="2025-12-05", interval="1d")

# Standardize column names
df = df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})
df = df[['open', 'high', 'low', 'close', 'volume']]
df.index = df.index.tz_localize(None)

# Remove duplicates
df = df[~df.index.duplicated(keep='first')]

print(f"   Price data: {len(df)} daily candles")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")


# ============================================================
# 3. MERGE ON-CHAIN DATA WITH PRICES
# ============================================================
print("\n[3/7] Merging on-chain data with prices...")

# Align indices to daily dates
df['date'] = df.index.date
df['date'] = pd.to_datetime(df['date'])

# Prepare MVRV for merge
mvrv_daily = mvrv_history[['mvrv', 'mvrv_zscore']].copy()
mvrv_daily.index = pd.to_datetime(mvrv_daily.index.date)
mvrv_daily = mvrv_daily[~mvrv_daily.index.duplicated(keep='first')]

# Prepare SOPR for merge
sopr_daily = sopr_history[['sopr']].copy()
sopr_daily.index = pd.to_datetime(sopr_daily.index.date)
sopr_daily = sopr_daily[~sopr_daily.index.duplicated(keep='first')]

# Merge
df = df.merge(mvrv_daily, left_on='date', right_index=True, how='left')
df = df.merge(sopr_daily, left_on='date', right_index=True, how='left')

# Forward fill missing values
df['mvrv'] = df['mvrv'].ffill()
df['mvrv_zscore'] = df['mvrv_zscore'].ffill()
df['sopr'] = df['sopr'].ffill()

# Drop rows without on-chain data
df = df.dropna(subset=['mvrv', 'sopr'])

print(f"   Merged dataset: {len(df)} rows")
print(f"   MVRV range: {df['mvrv'].min():.2f} to {df['mvrv'].max():.2f}")
print(f"   SOPR range: {df['sopr'].min():.4f} to {df['sopr'].max():.4f}")

# Count extremes
mvrv_lows = (df['mvrv'] < 1.0).sum()
mvrv_highs = (df['mvrv'] > 3.7).sum()
sopr_caps = (df['sopr'] < 0.95).sum()
sopr_euphs = (df['sopr'] > 1.05).sum()

print(f"\n   Override conditions in data:")
print(f"   - MVRV < 1.0 (bottom): {mvrv_lows} days")
print(f"   - MVRV > 3.7 (top):    {mvrv_highs} days")
print(f"   - SOPR < 0.95 (cap):   {sopr_caps} days")
print(f"   - SOPR > 1.05 (euph):  {sopr_euphs} days")


# ============================================================
# 4. GENERATE TECHNICAL FEATURES (NO ON-CHAIN)
# ============================================================
print("\n[4/7] Generating technical features (NO on-chain in ML)...")

# Technical features ONLY - no on-chain metrics
df['return_1d'] = df['close'].pct_change()
df['return_5d'] = df['close'].pct_change(5)
df['return_20d'] = df['close'].pct_change(20)
df['sma_5'] = df['close'].rolling(5).mean()
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['volatility_20'] = df['return_1d'].rolling(20).std()

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, np.inf)
df['rsi_14'] = 100 - (100 / (1 + rs))

# MACD
ema12 = df['close'].ewm(span=12).mean()
ema26 = df['close'].ewm(span=26).mean()
df['macd'] = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9).mean()

# OBV
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

# Bollinger position
bb_mid = df['close'].rolling(20).mean()
bb_std = df['close'].rolling(20).std()
df['bb_position'] = (df['close'] - bb_mid) / (2 * bb_std)

# Volume ratio
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

# Target: next day return direction
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop warmup period and NaN
df = df.iloc[50:].dropna()

print(f"   Final dataset: {len(df)} samples")
print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")


# ============================================================
# 5. TRAIN BASELINE ML MODEL (technical features only)
# ============================================================
print("\n[5/7] Training baseline ML model...")

import lightgbm as lgb

# Feature columns - TECHNICAL ONLY (no on-chain!)
feature_cols = [
    'return_1d', 'return_5d', 'return_20d',
    'sma_5', 'sma_20', 'sma_50',
    'volatility_20', 'rsi_14', 'macd', 'macd_signal',
    'obv', 'bb_position', 'volume_ratio'
]

X = df[feature_cols]
y = df['target']

# Time series split (70/30)
split_idx = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_df = df.iloc[split_idx:].copy()

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Test period: {test_df.index[0].date()} to {test_df.index[-1].date()}")

# Train LightGBM
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=100)

# Get ML predictions
ml_probs = model.predict(X_test)
ml_preds = (ml_probs > 0.5).astype(int)
ml_accuracy = (ml_preds == y_test.values).mean()

print(f"   Baseline ML accuracy: {ml_accuracy:.1%}")


# ============================================================
# 6. APPLY REGIME OVERRIDES
# ============================================================
print("\n[6/7] Applying regime overrides...")

# Apply override logic to each prediction
override_preds = []
override_reasons = []

for i, (idx, row) in enumerate(test_df.iterrows()):
    ml_pred = ml_preds[i]
    mvrv = row['mvrv']
    sopr = row['sopr']

    final_pred, reason = apply_regime_overrides(ml_pred, mvrv, sopr)
    override_preds.append(final_pred)
    override_reasons.append(reason)

test_df['ml_pred'] = ml_preds
test_df['override_pred'] = override_preds
test_df['override_reason'] = override_reasons

# Count overrides
reason_counts = test_df['override_reason'].value_counts()
n_overrides = (test_df['override_reason'] != 'none').sum()
n_total = len(test_df)

print(f"   Total predictions: {n_total}")
print(f"   ML predictions used: {n_total - n_overrides} ({(n_total - n_overrides)/n_total:.1%})")
print(f"   Overrides applied: {n_overrides} ({n_overrides/n_total:.1%})")
print(f"\n   Override breakdown:")
for reason, count in reason_counts.items():
    if reason != 'none':
        print(f"      {reason}: {count}")

# Calculate override accuracy
override_accuracy = (test_df['override_pred'].values == y_test.values).mean()
print(f"\n   Override-enhanced accuracy: {override_accuracy:.1%}")
print(f"   Improvement: {(override_accuracy - ml_accuracy)*100:+.1f}%")


# ============================================================
# 7. BACKTEST COMPARISON
# ============================================================
print("\n[7/7] Running backtest comparison...")

from backtesting.engine import BacktestEngine, BacktestConfig, run_buy_and_hold

prices = test_df['close']

# Baseline signals (ML only)
baseline_signals = pd.Series(ml_preds, index=test_df.index)

# Override signals
override_signals = pd.Series(override_preds, index=test_df.index)

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    commission_rate=0.001,   # 0.1%
    slippage_rate=0.0005,    # 0.05%
    allow_short=False
)

engine = BacktestEngine(config)

print("   Running baseline backtest...")
baseline_result = engine.run(prices, baseline_signals)

print("   Running override-enhanced backtest...")
override_result = engine.run(prices, override_signals)

print("   Running buy-and-hold benchmark...")
bh_result = run_buy_and_hold(prices, initial_capital=10000.0)


# ============================================================
# ANALYZE BY REGIME
# ============================================================
print("\n" + "=" * 70)
print("PERFORMANCE BY REGIME")
print("=" * 70)

def analyze_by_regime(test_df: pd.DataFrame, prices: pd.Series) -> Dict:
    """Analyze performance separately for override vs non-override periods."""

    results = {}

    # Split into override and non-override periods
    override_mask = test_df['override_reason'] != 'none'

    # Non-override periods (ML only)
    ml_only_df = test_df[~override_mask]
    if len(ml_only_df) > 0:
        ml_correct = (ml_only_df['ml_pred'] == ml_only_df['target']).mean()
        results['ml_only'] = {
            'count': len(ml_only_df),
            'accuracy': ml_correct,
            'pct_of_total': len(ml_only_df) / len(test_df)
        }

    # Override periods (by type)
    for reason in test_df['override_reason'].unique():
        if reason == 'none':
            continue

        reason_df = test_df[test_df['override_reason'] == reason]
        if len(reason_df) > 0:
            correct = (reason_df['override_pred'] == reason_df['target']).mean()

            # Calculate average return during these periods
            reason_returns = []
            for idx, row in reason_df.iterrows():
                if row['override_pred'] == 1:  # Long
                    ret = row['target'] * 2 - 1  # +1 if correct, -1 if wrong
                else:  # Flat
                    ret = (1 - row['target']) * 2 - 1
                reason_returns.append(ret)

            results[reason] = {
                'count': len(reason_df),
                'accuracy': correct,
                'pct_of_total': len(reason_df) / len(test_df),
                'avg_signal_return': np.mean(reason_returns) if reason_returns else 0
            }

    return results

regime_analysis = analyze_by_regime(test_df, prices)

print("\n  Regime Performance:")
print("-" * 60)
for regime, stats in regime_analysis.items():
    print(f"\n  {regime.upper()}:")
    print(f"      Days: {stats['count']} ({stats['pct_of_total']:.1%} of test period)")
    print(f"      Accuracy: {stats['accuracy']:.1%}")


# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)

def print_metrics(name, result):
    m = result.metrics
    final_equity = result.equity_curve.iloc[-1]
    total_return = (final_equity / 10000 - 1) * 100
    print(f"\n{name}:")
    print(f"   Final Equity:    ${final_equity:,.2f}")
    print(f"   Total Return:    {total_return:+.2f}%")
    print(f"   Sharpe Ratio:    {m.sharpe_ratio:.2f}")
    print(f"   Max Drawdown:    {m.max_drawdown:.1%}")
    print(f"   Win Rate:        {m.win_rate:.1%}")
    print(f"   Num Trades:      {len(result.trades)}")
    return m.sharpe_ratio, total_return

bh_sharpe, bh_return = print_metrics("Buy & Hold", bh_result)
baseline_sharpe, baseline_return = print_metrics("Baseline ML (technical only)", baseline_result)
override_sharpe, override_return = print_metrics("ML + Regime Overrides", override_result)

# Calculate improvement
print("\n" + "-" * 70)
print("SHARPE RATIO COMPARISON")
print("-" * 70)

if baseline_sharpe != 0:
    improvement = ((override_sharpe - baseline_sharpe) / abs(baseline_sharpe)) * 100
else:
    improvement = float('inf') if override_sharpe > 0 else 0

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│ Strategy                         │ Sharpe │ Return │ vs Baseline  │
├────────────────────────────────────────────────────────────────────┤
│ Buy & Hold                       │ {bh_sharpe:>6.2f} │ {bh_return:>+6.1f}% │ (benchmark)  │
│ Baseline ML (technical only)     │ {baseline_sharpe:>6.2f} │ {baseline_return:>+6.1f}% │ (baseline)   │
│ ML + Regime Overrides            │ {override_sharpe:>6.2f} │ {override_return:>+6.1f}% │ {improvement:>+.1f}%        │
└────────────────────────────────────────────────────────────────────┘
""")

# On-chain data summary
print("ON-CHAIN DATA USED:")
print(f"   MVRV: {len(mvrv_history)} daily values (CoinMetrics 2015-present)")
print(f"   SOPR: {len(sopr_history)} daily values (Dune Analytics 2010-present)")
print(f"   Test period MVRV range: {test_df['mvrv'].min():.2f} - {test_df['mvrv'].max():.2f}")
print(f"   Test period SOPR range: {test_df['sopr'].min():.4f} - {test_df['sopr'].max():.4f}")

# Success criteria
target = 10
success = improvement >= target

print("\n" + "=" * 70)
if success:
    print(f"SUCCESS: Regime overrides improved Sharpe by {improvement:.1f}% (target: >= {target}%)")
else:
    print(f"Target not met: {improvement:.1f}% improvement (target: >= {target}%)")
print("=" * 70)

print(f"\nTest Period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
print(f"Test Samples: {len(test_df)} days")
print(f"Override Rate: {n_overrides}/{n_total} ({n_overrides/n_total:.1%})")
