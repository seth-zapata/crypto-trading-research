"""
Part 2: Add Funding Rates Feature
- Fetch from Binance API (free, no key needed)
- Add features: funding_rate, funding_7d_avg, funding_zscore
- Retrain best model with new features
- Re-run validation

Target: >35% return, <25% miss rate, <25% DD (95th pctl)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

np.random.seed(42)

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("PART 2: FUNDING RATES FEATURE")
print("=" * 70)

# ============================================================
# Step 1: Fetch Funding Rate Data from Binance
# ============================================================
print("\n[1/5] Fetching funding rate data from Binance...")

def fetch_funding_rates(symbol='BTCUSDT', start_date='2020-01-01', end_date=None):
    """
    Fetch historical funding rates from Binance Futures API.
    Free, no API key required.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    url = 'https://fapi.binance.com/fapi/v1/fundingRate'

    all_data = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'startTime': current_start,
            'endTime': end_ts,
            'limit': 1000
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Move to next batch
            last_time = data[-1]['fundingTime']
            current_start = last_time + 1

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"  Error fetching funding rates: {e}")
            break

    if not all_data:
        print("  WARNING: No funding rate data fetched")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df['fundingRate'] = df['fundingRate'].astype(float)
    df = df.set_index('fundingTime')

    # Resample to daily (take mean of 3 funding rates per day)
    df_daily = df['fundingRate'].resample('D').mean()

    return df_daily

# Fetch data
funding_rates = fetch_funding_rates(symbol='BTCUSDT', start_date='2020-01-01')

if len(funding_rates) == 0:
    print("  No funding rate data available - cannot proceed")
    sys.exit(1)

print(f"  Fetched {len(funding_rates)} days of funding rate data")
print(f"  Date range: {funding_rates.index[0]} to {funding_rates.index[-1]}")
print(f"  Mean funding rate: {funding_rates.mean()*100:.4f}%")
print(f"  Funding rate range: [{funding_rates.min()*100:.4f}%, {funding_rates.max()*100:.4f}%]")

# ============================================================
# Step 2: Create Funding Rate Features
# ============================================================
print("\n[2/5] Creating funding rate features...")

def create_funding_features(funding_rates):
    """
    Create funding rate features:
    - funding_rate: raw daily funding rate
    - funding_7d_avg: 7-day moving average
    - funding_zscore: z-score relative to 30-day mean/std
    """
    df = pd.DataFrame({'funding_rate': funding_rates})

    # 7-day moving average
    df['funding_7d_avg'] = df['funding_rate'].rolling(7).mean()

    # Z-score (30-day lookback)
    rolling_mean = df['funding_rate'].rolling(30).mean()
    rolling_std = df['funding_rate'].rolling(30).std()
    df['funding_zscore'] = (df['funding_rate'] - rolling_mean) / (rolling_std + 1e-8)

    # Forward fill NaN (from rolling windows)
    df = df.fillna(method='bfill')

    return df

funding_features = create_funding_features(funding_rates)
print(f"  Features created: {list(funding_features.columns)}")

# ============================================================
# Step 3: Merge with existing data
# ============================================================
print("\n[3/5] Merging funding features with GNN data...")

from data.ingestion.multi_asset import build_regime_detection_features
from models.predictors.regime_gnn import RegimeDetector, RegimeGNN

# Load base data
df_base, labels = build_regime_detection_features(
    start_date='2020-01-01',
    assets=['BTC', 'ETH', 'SOL']
)

# Merge funding features
funding_features.index = funding_features.index.tz_localize(None)
df_base.index = pd.to_datetime(df_base.index).tz_localize(None)

df_merged = df_base.join(funding_features, how='left')
df_merged = df_merged.dropna()

# Align labels
labels_aligned = labels.loc[df_merged.index]

print(f"  Original samples: {len(df_base)}")
print(f"  After merge: {len(df_merged)}")
print(f"  Features added: funding_rate, funding_7d_avg, funding_zscore")

# ============================================================
# Step 4: Train Enhanced GNN
# ============================================================
print("\n[4/5] Training GNN with funding rate features...")

# Modify node feature preparation to include funding rates
class EnhancedRegimeDetector(RegimeDetector):
    """Regime detector with funding rate features."""

    def _prepare_node_features(self, df, idx):
        """Add funding rate features to node features."""
        features = []

        for asset in self.assets:
            asset_features = []

            # Original features
            for period in [1, 5, 20]:
                col = f'{asset}_return_{period}d'
                if col in df.columns:
                    asset_features.append(df[col].iloc[idx])

            for window in [10, 20, 60]:
                col = f'{asset}_vol_{window}d'
                if col in df.columns:
                    asset_features.append(df[col].iloc[idx])

            if asset != 'BTC':
                col = f'{asset}_corr_to_BTC'
                if col in df.columns:
                    asset_features.append(df[col].iloc[idx])
            else:
                asset_features.append(1.0)

            # Add funding rate features (same for all assets, from BTC)
            if 'funding_rate' in df.columns:
                asset_features.append(df['funding_rate'].iloc[idx])
            if 'funding_7d_avg' in df.columns:
                asset_features.append(df['funding_7d_avg'].iloc[idx])
            if 'funding_zscore' in df.columns:
                asset_features.append(df['funding_zscore'].iloc[idx])

            features.append(asset_features)

        # Pad features to same length
        max_len = max(len(f) for f in features)
        features = [f + [0.0] * (max_len - len(f)) for f in features]

        import torch
        return torch.tensor(features, dtype=torch.float32)

# Prepare data
btc_returns = df_merged['BTC_return_1d'].values
split = int(len(df_merged) * 0.8)

detector = EnhancedRegimeDetector(assets=['BTC', 'ETH', 'SOL'], hidden_dim=64)
graphs, targets = detector.prepare_dataset(df_merged, labels_aligned)

train_graphs, val_graphs = graphs[:split], graphs[split:]
train_labels, val_labels = targets[:split], targets[split:]
train_returns = btc_returns[:split]
val_returns = btc_returns[split:]

print(f"  Train samples: {len(train_graphs)}")
print(f"  Val samples: {len(val_graphs)}")

# Train with best configuration from previous sweep
detector.train(
    train_graphs, train_labels,
    val_graphs, val_labels,
    epochs=100, batch_size=32,
    train_returns=train_returns,
    use_asymmetric_loss=True,
    miss_penalty=10.0  # Best from sweep
)

_, probs = detector.predict(val_graphs)

# ============================================================
# Step 5: Validation
# ============================================================
print("\n[5/5] Running validation...")

THRESHOLD = 0.40

def positions_from_probs(probs, threshold=0.40):
    positions = probs[:, 0] * 0.85 + probs[:, 1] * 0.65 + probs[:, 2] * 0.20
    positions[probs[:, 2] > threshold] = 0.20
    return np.clip(positions, 0.1, 1.0)

positions = positions_from_probs(probs, THRESHOLD)

# Miss rate
crash_mask = val_returns[:len(probs)] < -0.05
n_crashes = crash_mask.sum()
preds = probs.argmax(axis=1)
preds[probs[:, 2] > THRESHOLD] = 2
misses = ((preds == 0) & crash_mask).sum()
miss_rate = misses / n_crashes if n_crashes > 0 else 0

# Bootstrap
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

print("  Running bootstrap (1000 simulations)...")
bootstrap = block_bootstrap(val_returns[:len(positions)], positions)

# ============================================================
# Results
# ============================================================
print("\n" + "=" * 70)
print("RESULTS: GNN + FUNDING RATES")
print("=" * 70)

return_med = bootstrap['return'].median()
dd_95 = bootstrap['max_dd'].quantile(0.95)

print(f"\n  Avg Position: {positions.mean()*100:.1f}%")
print(f"  Return Median: {return_med:.1f}%")
print(f"  DD 95th pctl: {dd_95:.1f}%")
print(f"  Miss Rate: {miss_rate*100:.1f}%")

print("\n" + "=" * 70)
print("TARGET CHECK")
print("=" * 70)

print(f"\n  Return Median: {return_med:.1f}% (target >35%): {'PASS' if return_med > 35 else 'FAIL'}")
print(f"  DD 95th pctl:  {dd_95:.1f}% (target <25%): {'PASS' if dd_95 < 25 else 'FAIL'}")
print(f"  Miss Rate:     {miss_rate*100:.1f}% (target <25%): {'PASS' if miss_rate < 0.25 else 'FAIL'}")

all_pass = return_med > 35 and dd_95 < 25 and miss_rate < 0.25

if all_pass:
    print("\n>>> ALL TARGETS MET with Funding Rates <<<")
else:
    print("\n>>> Funding rates did not meet all targets <<<")
    print("Consider: Lower threshold, different features, or accept trade-offs")

# Compare with baseline
print("\n" + "=" * 70)
print("COMPARISON: With vs Without Funding Rates")
print("=" * 70)

print("\nNote: Run phase5_full_sweep.py baseline for comparison")
print(f"Current result with funding rates:")
print(f"  Position: {positions.mean()*100:.1f}%, Return: {return_med:.1f}%, DD: {dd_95:.1f}%, Miss: {miss_rate*100:.1f}%")
