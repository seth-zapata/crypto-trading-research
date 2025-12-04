"""
Phase 3 Signal Quality Analysis
===============================

Comprehensive analysis of ALL signal sources before Phase 4-6.

Goal: Determine if we have ANY alpha worth amplifying, or if we should
pivot to risk management focus.

Signal Sources Tested:
1. Technical indicators (baseline)
2. On-chain metrics (MVRV, SOPR)
3. Sentiment (FinBERT on Reddit)
"""

import asyncio
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("PHASE 3 SIGNAL QUALITY ANALYSIS")
print("Determining if we have alpha worth amplifying")
print("=" * 70)


# ============================================================
# 1. FETCH ALL DATA
# ============================================================
print("\n[1/6] Fetching data sources...")

import yaml
import yfinance as yf
from data.ingestion.onchain import OnChainDataProvider, CoinMetricsProvider

config_path = Path(__file__).parent.parent / 'config' / 'api_keys.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)


async def fetch_all_data():
    """Fetch price and on-chain data."""

    # Price data (2020-present)
    print("   Fetching BTC price data...")
    btc = yf.Ticker("BTC-USD")
    price_df = btc.history(start="2020-01-01", end="2025-12-05", interval="1d")
    price_df = price_df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    })
    price_df = price_df[['open', 'high', 'low', 'close', 'volume']]
    price_df.index = price_df.index.tz_localize(None)
    print(f"   Price data: {len(price_df)} days")

    # MVRV (CoinMetrics)
    print("   Fetching MVRV from CoinMetrics...")
    cm = CoinMetricsProvider()
    mvrv_df = await cm.fetch_mvrv('btc', start_date='2020-01-01')
    mvrv_df = mvrv_df.set_index('time')
    mvrv_df.index = mvrv_df.index.tz_localize(None)
    print(f"   MVRV data: {len(mvrv_df)} days")

    # SOPR (Dune)
    print("   Fetching SOPR from Dune...")
    provider = OnChainDataProvider(config['dune']['api_key'])
    sopr_df = await provider.fetch_sopr()
    sopr_df['date'] = pd.to_datetime(sopr_df['date'])
    sopr_df = sopr_df.set_index('date')
    sopr_df = sopr_df[['normalized_sopr']].rename(columns={'normalized_sopr': 'sopr'})
    print(f"   SOPR data: {len(sopr_df)} days")

    return price_df, mvrv_df, sopr_df


price_df, mvrv_df, sopr_df = asyncio.run(fetch_all_data())


# ============================================================
# 2. MERGE AND PREPARE DATA
# ============================================================
print("\n[2/6] Preparing merged dataset...")

df = price_df.copy()
df['date'] = df.index.date
df['date'] = pd.to_datetime(df['date'])

# Merge MVRV
mvrv_daily = mvrv_df[['mvrv', 'mvrv_zscore']].copy()
mvrv_daily.index = pd.to_datetime(mvrv_daily.index.date)
mvrv_daily = mvrv_daily[~mvrv_daily.index.duplicated(keep='first')]
df = df.merge(mvrv_daily, left_on='date', right_index=True, how='left')

# Merge SOPR
sopr_daily = sopr_df[['sopr']].copy()
sopr_daily.index = pd.to_datetime(sopr_daily.index.date)
sopr_daily = sopr_daily[~sopr_daily.index.duplicated(keep='first')]
df = df.merge(sopr_daily, left_on='date', right_index=True, how='left')

# Forward fill
df['mvrv'] = df['mvrv'].ffill()
df['mvrv_zscore'] = df['mvrv_zscore'].ffill()
df['sopr'] = df['sopr'].ffill()

# Calculate returns
df['return_1d'] = df['close'].pct_change()
df['return_5d'] = df['close'].pct_change(5)
df['return_20d'] = df['close'].pct_change(20)

# Next-day return (what we're trying to predict)
df['next_day_return'] = df['return_1d'].shift(-1)
df['next_day_up'] = (df['next_day_return'] > 0).astype(int)

# Technical indicators
df['rsi_14'] = 100 - (100 / (1 + df['close'].diff().clip(lower=0).rolling(14).mean() /
                              (-df['close'].diff().clip(upper=0)).rolling(14).mean().replace(0, np.inf)))
df['volatility_20'] = df['return_1d'].rolling(20).std()
df['sma_ratio'] = df['close'] / df['close'].rolling(20).mean()

# Drop NaN
df = df.dropna()

print(f"   Merged dataset: {len(df)} samples")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")


# ============================================================
# 3. TECHNICAL INDICATOR SIGNAL ANALYSIS
# ============================================================
print("\n[3/6] Analyzing technical indicator signals...")

def analyze_signal_quality(signal_series, next_return, signal_name):
    """Analyze predictive quality of a signal."""

    # Basic correlation
    corr = signal_series.corr(next_return)

    # Spearman (rank) correlation - more robust
    spearman_corr, spearman_p = stats.spearmanr(signal_series, next_return)

    # Information coefficient (rolling correlation stability)
    rolling_corr = signal_series.rolling(60).corr(next_return)
    ic_mean = rolling_corr.mean()
    ic_std = rolling_corr.std()
    ic_ratio = ic_mean / ic_std if ic_std > 0 else 0

    return {
        'name': signal_name,
        'pearson_corr': corr,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'ic_mean': ic_mean,
        'ic_ratio': ic_ratio
    }


tech_signals = {
    'RSI-14': df['rsi_14'],
    'Volatility-20': df['volatility_20'],
    'SMA Ratio': df['sma_ratio'],
    '1-Day Momentum': df['return_1d'],
    '5-Day Momentum': df['return_5d'],
    '20-Day Momentum': df['return_20d'],
}

print("\n   Technical Indicator Correlations with Next-Day Return:")
print("   " + "-" * 60)

tech_results = []
for name, signal in tech_signals.items():
    result = analyze_signal_quality(signal, df['next_day_return'], name)
    tech_results.append(result)
    sig = "***" if abs(result['pearson_corr']) > 0.05 else ""
    print(f"   {name:20s} | Pearson: {result['pearson_corr']:+.4f} | Spearman: {result['spearman_corr']:+.4f} {sig}")


# ============================================================
# 4. ON-CHAIN SIGNAL ANALYSIS
# ============================================================
print("\n[4/6] Analyzing on-chain signals...")

onchain_signals = {
    'MVRV': df['mvrv'],
    'MVRV Z-Score': df['mvrv_zscore'],
    'SOPR': df['sopr'],
}

print("\n   On-Chain Correlations with Next-Day Return:")
print("   " + "-" * 60)

onchain_results = []
for name, signal in onchain_signals.items():
    result = analyze_signal_quality(signal, df['next_day_return'], name)
    onchain_results.append(result)
    sig = "***" if abs(result['pearson_corr']) > 0.05 else ""
    print(f"   {name:20s} | Pearson: {result['pearson_corr']:+.4f} | Spearman: {result['spearman_corr']:+.4f} {sig}")

# Lag analysis for MVRV
print("\n   MVRV Lag Analysis (does it predict further ahead?):")
for lag in [1, 5, 10, 20, 40, 60]:
    future_return = df['close'].pct_change(lag).shift(-lag)
    corr = df['mvrv'].corr(future_return)
    print(f"   MVRV vs {lag}-day forward return: {corr:+.4f}")


# ============================================================
# 5. EXTREME SIGNAL ANALYSIS
# ============================================================
print("\n[5/6] Analyzing extreme signal accuracy...")

def analyze_extreme_accuracy(signal, next_return, next_up, name, high_thresh, low_thresh):
    """Analyze accuracy when signal is at extremes."""

    total = len(signal)

    # High extreme (bullish signal)
    high_mask = signal > high_thresh
    high_count = high_mask.sum()
    if high_count > 0:
        high_accuracy = next_up[high_mask].mean()
        high_avg_return = next_return[high_mask].mean()
    else:
        high_accuracy = np.nan
        high_avg_return = np.nan

    # Low extreme (bearish signal)
    low_mask = signal < low_thresh
    low_count = low_mask.sum()
    if low_count > 0:
        low_accuracy = (1 - next_up[low_mask]).mean()  # Accuracy of predicting DOWN
        low_avg_return = next_return[low_mask].mean()
    else:
        low_accuracy = np.nan
        low_avg_return = np.nan

    # Neutral
    neutral_mask = (signal >= low_thresh) & (signal <= high_thresh)
    neutral_count = neutral_mask.sum()
    neutral_avg_return = next_return[neutral_mask].mean() if neutral_count > 0 else np.nan

    return {
        'name': name,
        'high_thresh': high_thresh,
        'low_thresh': low_thresh,
        'high_count': high_count,
        'high_pct': high_count / total,
        'high_accuracy': high_accuracy,
        'high_avg_return': high_avg_return,
        'low_count': low_count,
        'low_pct': low_count / total,
        'low_accuracy': low_accuracy,
        'low_avg_return': low_avg_return,
        'neutral_count': neutral_count,
        'neutral_avg_return': neutral_avg_return
    }


# MVRV extremes
mvrv_extreme = analyze_extreme_accuracy(
    df['mvrv'], df['next_day_return'], df['next_day_up'],
    'MVRV', high_thresh=3.0, low_thresh=1.0
)

print(f"\n   MVRV Extreme Analysis:")
print(f"   High (>{mvrv_extreme['high_thresh']}): {mvrv_extreme['high_count']} days ({mvrv_extreme['high_pct']:.1%})")
print(f"      Bearish accuracy: {mvrv_extreme['high_accuracy']:.1%}" if not np.isnan(mvrv_extreme['high_accuracy']) else "      N/A")
print(f"      Avg next-day return: {mvrv_extreme['high_avg_return']:.3%}" if not np.isnan(mvrv_extreme['high_avg_return']) else "")
print(f"   Low (<{mvrv_extreme['low_thresh']}): {mvrv_extreme['low_count']} days ({mvrv_extreme['low_pct']:.1%})")
print(f"      Bullish accuracy: {mvrv_extreme['low_accuracy']:.1%}" if not np.isnan(mvrv_extreme['low_accuracy']) else "      N/A")
print(f"      Avg next-day return: {mvrv_extreme['low_avg_return']:.3%}" if not np.isnan(mvrv_extreme['low_avg_return']) else "")

# SOPR extremes
sopr_extreme = analyze_extreme_accuracy(
    df['sopr'], df['next_day_return'], df['next_day_up'],
    'SOPR', high_thresh=1.03, low_thresh=0.97
)

print(f"\n   SOPR Extreme Analysis:")
print(f"   High (>{sopr_extreme['high_thresh']}): {sopr_extreme['high_count']} days ({sopr_extreme['high_pct']:.1%})")
print(f"      Bearish accuracy: {sopr_extreme['high_accuracy']:.1%}" if not np.isnan(sopr_extreme['high_accuracy']) else "      N/A")
print(f"      Avg next-day return: {sopr_extreme['high_avg_return']:.3%}" if not np.isnan(sopr_extreme['high_avg_return']) else "")
print(f"   Low (<{sopr_extreme['low_thresh']}): {sopr_extreme['low_count']} days ({sopr_extreme['low_pct']:.1%})")
print(f"      Bullish accuracy: {sopr_extreme['low_accuracy']:.1%}" if not np.isnan(sopr_extreme['low_accuracy']) else "      N/A")
print(f"      Avg next-day return: {sopr_extreme['low_avg_return']:.3%}" if not np.isnan(sopr_extreme['low_avg_return']) else "")

# RSI extremes
rsi_extreme = analyze_extreme_accuracy(
    df['rsi_14'], df['next_day_return'], df['next_day_up'],
    'RSI-14', high_thresh=70, low_thresh=30
)

print(f"\n   RSI-14 Extreme Analysis:")
print(f"   Overbought (>{rsi_extreme['high_thresh']}): {rsi_extreme['high_count']} days ({rsi_extreme['high_pct']:.1%})")
print(f"      Bearish accuracy: {1-rsi_extreme['high_accuracy']:.1%}" if not np.isnan(rsi_extreme['high_accuracy']) else "      N/A")
print(f"      Avg next-day return: {rsi_extreme['high_avg_return']:.3%}" if not np.isnan(rsi_extreme['high_avg_return']) else "")
print(f"   Oversold (<{rsi_extreme['low_thresh']}): {rsi_extreme['low_count']} days ({rsi_extreme['low_pct']:.1%})")
print(f"      Bullish accuracy: {rsi_extreme['low_accuracy']:.1%}" if not np.isnan(rsi_extreme['low_accuracy']) else "      N/A")
print(f"      Avg next-day return: {rsi_extreme['low_avg_return']:.3%}" if not np.isnan(rsi_extreme['low_avg_return']) else "")


# ============================================================
# 6. SENTIMENT ANALYSIS (LIMITATION NOTED)
# ============================================================
print("\n[6/6] Sentiment Signal Analysis...")

print("""
   ⚠️  CRITICAL LIMITATION: Historical Sentiment Data Unavailable

   Reddit public JSON only provides RECENT posts (last ~1000).
   We cannot backtest sentiment on historical data without:
   1. Pushshift archive (deprecated)
   2. Reddit API with historical access (requires approval)
   3. Pre-collected sentiment dataset

   What we CAN do:
   - Test current sentiment correlation with recent returns
   - Use Fear & Greed Index as sentiment proxy (if available)
""")

# Try to use current sentiment as a point-in-time sample
from data.ingestion.reddit_sources import RedditPublicJSON
from data.ingestion.sentiment import FinBERTAnalyzer

async def get_current_sentiment():
    """Get current sentiment as a sample."""
    try:
        reddit = RedditPublicJSON()
        posts = await reddit.fetch_multiple_subreddits(
            ['Bitcoin', 'CryptoCurrency', 'BitcoinMarkets'],
            limit_per_sub=25
        )

        analyzer = FinBERTAnalyzer()

        scores = []
        for post in posts[:50]:
            result = analyzer.analyze(post['title'])
            if result.sentiment == 'positive':
                scores.append(result.positive_score)
            elif result.sentiment == 'negative':
                scores.append(-result.negative_score)
            else:
                scores.append(0)

        return np.mean(scores) if scores else 0, len(scores)
    except Exception as e:
        print(f"   Error fetching sentiment: {e}")
        return None, 0


current_sentiment, sample_size = asyncio.run(get_current_sentiment())

if current_sentiment is not None:
    print(f"\n   Current Sentiment Sample:")
    print(f"   Posts analyzed: {sample_size}")
    print(f"   Average sentiment score: {current_sentiment:+.3f}")
    print(f"   Interpretation: {'Bullish' if current_sentiment > 0.1 else 'Bearish' if current_sentiment < -0.1 else 'Neutral'}")
else:
    print("   Could not fetch current sentiment")


# ============================================================
# SUMMARY: SIGNAL QUALITY ASSESSMENT
# ============================================================
print("\n" + "=" * 70)
print("SIGNAL QUALITY SUMMARY")
print("=" * 70)

# Find best signals
all_results = tech_results + onchain_results
all_results_sorted = sorted(all_results, key=lambda x: abs(x['pearson_corr']), reverse=True)

print("\n   All Signals Ranked by |Correlation| with Next-Day Return:")
print("   " + "-" * 60)
for r in all_results_sorted:
    status = "WEAK" if abs(r['pearson_corr']) < 0.02 else "MARGINAL" if abs(r['pearson_corr']) < 0.05 else "USABLE"
    print(f"   {r['name']:20s} | {r['pearson_corr']:+.4f} | {status}")

# Best signal
best = all_results_sorted[0]
print(f"\n   Best Signal: {best['name']} (correlation: {best['pearson_corr']:+.4f})")

# Threshold for "usable" signal
usable_threshold = 0.03
usable_signals = [r for r in all_results if abs(r['pearson_corr']) >= usable_threshold]

print(f"\n   Signals with |correlation| >= {usable_threshold}: {len(usable_signals)}")
for r in usable_signals:
    print(f"      - {r['name']}: {r['pearson_corr']:+.4f}")


# ============================================================
# REALISTIC ASSESSMENT
# ============================================================
print("\n" + "=" * 70)
print("REALISTIC ASSESSMENT")
print("=" * 70)

# Calculate combined signal potential
best_corr = abs(best['pearson_corr'])

print(f"""
   HONEST FINDINGS:

   1. CORRELATION STRENGTH
      Best single signal: {best['name']} at {best['pearson_corr']:+.4f}
      This is {"VERY WEAK" if best_corr < 0.02 else "WEAK" if best_corr < 0.05 else "MARGINAL" if best_corr < 0.10 else "MODERATE"}

      For reference:
      - correlation < 0.02: Essentially noise
      - correlation 0.02-0.05: Weak but potentially useful with high volume
      - correlation 0.05-0.10: Marginal, could be amplified
      - correlation > 0.10: Meaningful signal (rare in financial data)

   2. EXTREME SIGNAL ACCURACY
      MVRV < 1.0 (cycle bottom): {mvrv_extreme['low_count']} days, {mvrv_extreme['low_accuracy']:.1%} bullish accuracy
      MVRV > 3.0 (cycle top): {mvrv_extreme['high_count']} days, accuracy N/A (too few samples)
      RSI < 30 (oversold): {rsi_extreme['low_count']} days, {rsi_extreme['low_accuracy']:.1%} bullish accuracy

   3. SIGNAL FREQUENCY
      On-chain extremes: {(mvrv_extreme['high_pct'] + mvrv_extreme['low_pct'])*100:.1f}% of days
      RSI extremes: {(rsi_extreme['high_pct'] + rsi_extreme['low_pct'])*100:.1f}% of days

   4. SENTIMENT
      Historical backtest: NOT POSSIBLE (no archived data)
      Current sample: {f"{current_sentiment:+.3f}" if current_sentiment else "N/A"}
""")


# ============================================================
# RECOMMENDATION
# ============================================================
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Decision logic
has_usable_signal = len(usable_signals) > 0
best_is_strong = best_corr >= 0.05
extreme_accuracy_good = (not np.isnan(mvrv_extreme['low_accuracy']) and mvrv_extreme['low_accuracy'] > 0.55) or \
                        (not np.isnan(rsi_extreme['low_accuracy']) and rsi_extreme['low_accuracy'] > 0.55)

if best_is_strong:
    recommendation = "A"
    rationale = f"Best signal ({best['name']}) has correlation {best['pearson_corr']:+.4f} >= 0.05 threshold"
elif has_usable_signal or extreme_accuracy_good:
    recommendation = "B"
    rationale = "Signals are weak but extreme accuracy shows potential for risk management"
else:
    recommendation = "C"
    rationale = "All signals too weak for daily trading; pivot to longer timeframe"

print(f"""
   RECOMMENDED PATH: {recommendation}

   Rationale: {rationale}
""")

if recommendation == "A":
    print("""
   PATH A: Continue to Phase 4-6 (Alpha Amplification)

   - Signals show edge worth amplifying
   - GNN can improve signal timing with cross-asset context
   - RL can optimize position sizing based on signal confidence
   - Target: Sharpe > 2.0 with controlled drawdowns
""")
elif recommendation == "B":
    print("""
   PATH B: Pivot to Risk Management Focus

   - Signals are weak for alpha generation
   - GNN/RL become DEFENSIVE tools:
     * Detect regime shifts to reduce exposure
     * Optimize position sizing (bet small when uncertain)
     * Target: Match BTC returns with 30-50% smaller drawdowns

   Success Metrics (revised):
   | Phase | Goal | Metric |
   |-------|------|--------|
   | GNN | Regime detection | Predict drawdowns 5+ days ahead |
   | RL | Dynamic sizing | Max drawdown < 20% vs BTC's 30% |
   | Ensemble | Variance reduction | Smoother equity curve |
""")
else:
    print("""
   PATH C: Pivot to Different Timeframe

   - Daily signals are noise
   - On-chain signals are CYCLE indicators (work on monthly scale)
   - Recommendation: Build REBALANCING system, not trading system

   Monthly Rebalancing Strategy:
   - MVRV < 1.0: 100% BTC allocation
   - MVRV 1.0-2.5: 60% BTC allocation
   - MVRV 2.5-3.5: 30% BTC allocation
   - MVRV > 3.5: 0% BTC (all stablecoins)

   This matches the signal frequency (cycle changes, not daily)
""")

print("=" * 70)

# Save results for RESULTS.md
results = {
    'best_signal': best['name'],
    'best_correlation': best['pearson_corr'],
    'usable_signals': len(usable_signals),
    'recommendation': recommendation,
    'mvrv_low_accuracy': mvrv_extreme['low_accuracy'],
    'rsi_oversold_accuracy': rsi_extreme['low_accuracy'],
}

print(f"\nAnalysis complete. Recommendation: PATH {recommendation}")
