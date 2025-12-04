"""
Phase 3 Backtest with REAL Historical On-Chain Data
====================================================

Uses actual historical MVRV and SOPR data, NOT simulated signals.

Data Sources:
- CoinMetrics: MVRV from 2015 (3,990+ daily rows)
- Dune Analytics: SOPR from 2010 (5,600+ daily rows)
"""

import asyncio
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.engine import BacktestEngine, BacktestConfig, run_buy_and_hold

print("=" * 70)
print("PHASE 3 BACKTEST WITH REAL ON-CHAIN DATA")
print("=" * 70)


# ============================================================
# 1. FETCH REAL HISTORICAL ON-CHAIN DATA
# ============================================================
print("\n[1/6] Fetching REAL historical on-chain data...")

import yaml
from data.ingestion.onchain import OnChainDataProvider, CoinMetricsProvider

with open(Path(__file__).parent.parent / 'config' / 'api_keys.yaml') as f:
    config = yaml.safe_load(f)

async def fetch_all_historical():
    """Fetch all historical on-chain data."""

    # CoinMetrics MVRV
    print("   Fetching MVRV from CoinMetrics (2015-present)...")
    cm = CoinMetricsProvider()
    mvrv_df = await cm.fetch_mvrv('btc', start_date='2015-01-01')
    mvrv_df = mvrv_df.set_index('time')
    mvrv_df.index = mvrv_df.index.tz_localize(None)  # Remove timezone
    print(f"   MVRV: {len(mvrv_df)} rows ({mvrv_df.index.min().date()} to {mvrv_df.index.max().date()})")

    # Dune SOPR
    print("   Fetching SOPR from Dune (2010-present)...")
    provider = OnChainDataProvider(config['dune']['api_key'])
    sopr_df = await provider.fetch_sopr()
    sopr_df['date'] = pd.to_datetime(sopr_df['date'])
    sopr_df = sopr_df.set_index('date')
    sopr_df = sopr_df[['normalized_sopr']].rename(columns={'normalized_sopr': 'sopr'})
    print(f"   SOPR: {len(sopr_df)} rows ({sopr_df.index.min().date()} to {sopr_df.index.max().date()})")

    return mvrv_df, sopr_df

mvrv_history, sopr_history = asyncio.run(fetch_all_historical())


# ============================================================
# 2. FETCH PRICE DATA
# ============================================================
print("\n[2/6] Fetching price data...")

import ccxt

exchange = ccxt.coinbase()
exchange.load_markets()

# Fetch 90 days of hourly BTC data (paginated)
all_ohlcv = []
since = exchange.parse8601((datetime.now() - timedelta(days=90)).isoformat())
end_time = exchange.milliseconds()

while since < end_time:
    ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', since=since, limit=300)
    if not ohlcv:
        break
    all_ohlcv.extend(ohlcv)
    since = ohlcv[-1][0] + 1
    if len(all_ohlcv) >= 2000:
        break

df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

print(f"   Price data: {len(df)} hourly candles")
print(f"   Date range: {df.index[0]} to {df.index[-1]}")


# ============================================================
# 3. MERGE ON-CHAIN DATA WITH PRICES
# ============================================================
print("\n[3/6] Merging on-chain data with prices...")

# Create daily date column for merging
df['date'] = df.index.date
df['date'] = pd.to_datetime(df['date'])

# Merge MVRV (daily data -> forward fill to hourly)
mvrv_daily = mvrv_history[['mvrv', 'mvrv_zscore']].copy()
mvrv_daily.index = pd.to_datetime(mvrv_daily.index.date)
df = df.merge(mvrv_daily, left_on='date', right_index=True, how='left')

# Merge SOPR (daily data -> forward fill to hourly)
sopr_daily = sopr_history[['sopr']].copy()
sopr_daily.index = pd.to_datetime(sopr_daily.index.date)
df = df.merge(sopr_daily, left_on='date', right_index=True, how='left')

# Forward fill any missing values
df['mvrv'] = df['mvrv'].ffill()
df['mvrv_zscore'] = df['mvrv_zscore'].ffill()
df['sopr'] = df['sopr'].ffill()

# Drop rows without on-chain data
df = df.dropna(subset=['mvrv', 'sopr'])

print(f"   Merged dataset: {len(df)} rows with on-chain data")
print(f"   MVRV range: {df['mvrv'].min():.2f} to {df['mvrv'].max():.2f}")
print(f"   SOPR range: {df['sopr'].min():.4f} to {df['sopr'].max():.4f}")


# ============================================================
# 4. GENERATE FEATURES & ON-CHAIN SIGNALS
# ============================================================
print("\n[4/6] Generating features and on-chain signals...")

# Technical features
df['return_1h'] = df['close'].pct_change()
df['return_4h'] = df['close'].pct_change(4)
df['return_24h'] = df['close'].pct_change(24)
df['sma_5'] = df['close'].rolling(5).mean()
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['volatility_20'] = df['return_1h'].rolling(20).std()

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

# ============================================================
# REAL ON-CHAIN SIGNALS (not simulated!)
# ============================================================

def mvrv_signal(mvrv):
    """Convert MVRV to signal based on research thresholds."""
    if mvrv < 0.8:
        return 1.0   # Strong buy (extreme undervaluation)
    elif mvrv < 1.0:
        return 0.5   # Buy (undervaluation)
    elif mvrv > 3.7:
        return -1.0  # Strong sell (extreme overvaluation)
    elif mvrv > 2.5:
        return -0.5  # Sell (overvaluation)
    else:
        return 0.0   # Neutral (fair value 1.0-2.5)

def sopr_signal(sopr):
    """Convert SOPR to signal based on research thresholds."""
    if sopr < 0.97:
        return 0.5   # Buy (capitulation - coins moving at loss)
    elif sopr > 1.03:
        return -0.3  # Sell (profit taking)
    else:
        return 0.0   # Neutral

# Apply REAL on-chain signals
df['mvrv_signal'] = df['mvrv'].apply(mvrv_signal)
df['sopr_signal'] = df['sopr'].apply(sopr_signal)

# Combined on-chain signal (weighted average)
df['onchain_signal'] = df['mvrv_signal'] * 0.6 + df['sopr_signal'] * 0.4

print(f"   On-chain signal range: {df['onchain_signal'].min():.2f} to {df['onchain_signal'].max():.2f}")
print(f"   MVRV signals: {(df['mvrv_signal'] != 0).sum()} non-neutral out of {len(df)}")
print(f"   SOPR signals: {(df['sopr_signal'] != 0).sum()} non-neutral out of {len(df)}")

# Target: next hour return direction
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop warmup period and NaN
df = df.iloc[50:].dropna()

print(f"   Final dataset: {len(df)} samples")


# ============================================================
# 5. TRAIN MODELS AND BACKTEST
# ============================================================
print("\n[5/6] Training models...")

import lightgbm as lgb

# Feature columns - BASELINE (technical only)
baseline_features = [
    'return_1h', 'return_4h', 'return_24h',
    'sma_5', 'sma_20', 'sma_50',
    'volatility_20', 'rsi_14', 'macd', 'macd_signal',
    'obv', 'bb_position', 'volume_ratio'
]

# Feature columns - ENHANCED (technical + on-chain)
enhanced_features = baseline_features + ['mvrv', 'mvrv_zscore', 'sopr', 'onchain_signal']

X_baseline = df[baseline_features]
X_enhanced = df[enhanced_features]
y = df['target']

# Time series split (70/30)
split_idx = int(len(df) * 0.7)
X_train_base, X_test_base = X_baseline.iloc[:split_idx], X_baseline.iloc[split_idx:]
X_train_enh, X_test_enh = X_enhanced.iloc[:split_idx], X_enhanced.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train LightGBM - Baseline
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'verbose': -1
}

train_data_base = lgb.Dataset(X_train_base, label=y_train)
model_baseline = lgb.train(params, train_data_base, num_boost_round=100)

baseline_probs = model_baseline.predict(X_test_base)
baseline_preds = (baseline_probs > 0.5).astype(int)
baseline_accuracy = (baseline_preds == y_test.values).mean()

print(f"   Baseline accuracy: {baseline_accuracy:.1%}")

# Train LightGBM - Enhanced with on-chain features
train_data_enh = lgb.Dataset(X_train_enh, label=y_train)
model_enhanced = lgb.train(params, train_data_enh, num_boost_round=100)

enhanced_probs = model_enhanced.predict(X_test_enh)
enhanced_preds = (enhanced_probs > 0.5).astype(int)
enhanced_accuracy = (enhanced_preds == y_test.values).mean()

print(f"   Enhanced accuracy: {enhanced_accuracy:.1%}")

# Feature importance for enhanced model
importance = pd.DataFrame({
    'feature': enhanced_features,
    'importance': model_enhanced.feature_importance()
}).sort_values('importance', ascending=False)

print("\n   Top features (enhanced model):")
for _, row in importance.head(5).iterrows():
    print(f"      {row['feature']}: {row['importance']}")


# ============================================================
# 6. RUN BACKTESTS
# ============================================================
print("\n[6/6] Running backtests...")

test_df = df.iloc[split_idx:].copy()
prices = test_df['close']

baseline_signals = pd.Series(baseline_preds, index=test_df.index)
enhanced_signals = pd.Series(enhanced_preds, index=test_df.index)

config = BacktestConfig(
    initial_capital=10000.0,
    commission_rate=0.001,
    slippage_rate=0.0005,
    allow_short=False
)

engine = BacktestEngine(config)

print("   Running baseline backtest...")
baseline_result = engine.run(prices, baseline_signals)

print("   Running enhanced backtest...")
enhanced_result = engine.run(prices, enhanced_signals)

print("   Running buy-and-hold benchmark...")
bh_result = run_buy_and_hold(prices, initial_capital=10000.0)


# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 70)
print("BACKTEST RESULTS (REAL ON-CHAIN DATA)")
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
    return m.sharpe_ratio

bh_sharpe = print_metrics("Buy & Hold", bh_result)
baseline_sharpe = print_metrics("Phase 2 Baseline (technical only)", baseline_result)
enhanced_sharpe = print_metrics("Phase 3 Enhanced (technical + on-chain)", enhanced_result)

# Calculate improvement
print("\n" + "-" * 70)
print("SHARPE RATIO COMPARISON")
print("-" * 70)

if baseline_sharpe != 0:
    improvement = ((enhanced_sharpe - baseline_sharpe) / abs(baseline_sharpe)) * 100
else:
    improvement = float('inf') if enhanced_sharpe > 0 else 0

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│ Strategy                         │ Sharpe │ vs Baseline           │
├────────────────────────────────────────────────────────────────────┤
│ Buy & Hold                       │ {bh_sharpe:>6.2f} │ (benchmark)           │
│ Phase 2: Baseline (technical)    │ {baseline_sharpe:>6.2f} │ (baseline)            │
│ Phase 3: Enhanced (+ on-chain)   │ {enhanced_sharpe:>6.2f} │ {improvement:>+6.1f}% improvement │
└────────────────────────────────────────────────────────────────────┘
""")

# On-chain data summary
print("ON-CHAIN DATA USED:")
print(f"   MVRV: {len(mvrv_history)} daily values (real CoinMetrics data)")
print(f"   SOPR: {len(sopr_history)} daily values (real Dune Analytics data)")
print(f"   Test period MVRV range: {test_df['mvrv'].min():.2f} - {test_df['mvrv'].max():.2f}")
print(f"   Test period SOPR range: {test_df['sopr'].min():.4f} - {test_df['sopr'].max():.4f}")

# Success criteria
target = 10
success = improvement >= target

print("\n" + "=" * 70)
if success:
    print(f"✓ SUCCESS: On-chain features improved Sharpe by {improvement:.1f}% (target: ≥{target}%)")
else:
    print(f"✗ Target not met: {improvement:.1f}% improvement (target: ≥{target}%)")
print("=" * 70)

print(f"\nTest Period: {test_df.index[0]} to {test_df.index[-1]}")
print(f"Test Samples: {len(test_df)}")
