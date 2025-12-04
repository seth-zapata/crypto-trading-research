# Notebooks Directory

This directory contains Jupyter notebooks for analysis, validation, and visualization at each phase of development.

## Directory Structure

```
notebooks/
├── README.md              # This file
├── phase1/                # Data Infrastructure
├── phase2/                # Baseline Model (LightGBM)
├── phase3/                # Alpha Sources (On-chain, Sentiment)
├── phase4/                # Advanced Models (LSTM, GNN, Ensemble)
├── phase5/                # Reinforcement Learning Agents
└── phase6/                # Production Systems
```

## Notebook Strategy

### Purpose

Notebooks serve three critical functions:

1. **Validation** - Prove that each phase's code works correctly
2. **Visibility** - Let you see what the system is doing and why
3. **Analysis** - Deep dive into performance, predictions, and behavior

### Notebooks Are NOT

- ❌ Production code (they're for analysis only)
- ❌ Required for day-to-day operation (dashboard handles that)
- ❌ Real-time monitoring tools

### Notebooks ARE

- ✅ Your window into "why" decisions were made
- ✅ Where you validate each phase before approval
- ✅ How you investigate when something seems wrong
- ✅ Deep analysis tools for weekly reviews

## Planned Notebooks by Phase

### Phase 1: Data Infrastructure
- `01_data_validation.ipynb` - Verify data quality, check for gaps
- `01b_feature_exploration.ipynb` - Visualize engineered features

### Phase 2: Baseline Model
- `02_lightgbm_training.ipynb` - Train and evaluate baseline
- `02b_feature_importance.ipynb` - Analyze which features matter
- `02c_backtest_analysis.ipynb` - Interactive backtest exploration

### Phase 3: Alpha Sources
- `03_onchain_signals.ipynb` - Validate MVRV, SOPR, netflows
- `03b_sentiment_analysis.ipynb` - Check RVS scores, bot filtering
- `03c_regime_detection.ipynb` - Verify regime classifier

### Phase 4: Advanced Models
- `04_model_comparison.ipynb` - LightGBM vs LSTM vs GNN
- `04b_ensemble_analysis.ipynb` - How ensemble combines signals
- `04c_meta_learning_demo.ipynb` - Few-shot adaptation tests

### Phase 5: RL Agents
- `05_rl_agent_behavior.ipynb` - Understand agent decisions
- `05b_agent_safety_tests.ipynb` - Verify constraints respected
- `05c_hierarchical_coordination.ipynb` - Strategic→Tactical→Execution flow

### Phase 6: Production
- `06_trade_analysis.ipynb` - Daily trade review
- `06b_performance_dashboard.ipynb` - Live performance metrics

### Ongoing Analysis
- `07_weekly_performance_review.ipynb` - Weekly comparison (Sundays)
- `08_model_drift_detection.ipynb` - Check if models are degrading

## Testing Notebooks

All notebooks must be tested before a phase is considered complete.

### Running Tests

```bash
# Execute a notebook and check for errors
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  notebooks/phase1/01_data_validation.ipynb \
  --output 01_data_validation_executed.ipynb

# Generate HTML report
jupyter nbconvert --to html \
  notebooks/phase1/01_data_validation_executed.ipynb \
  --output ../../reports/01_data_validation.html

# Validate outputs
python scripts/validate_notebook_outputs.py \
  notebooks/phase1/01_data_validation_executed.ipynb
```

### Validation Criteria

A notebook passes validation if:
- ✅ Executes without errors
- ✅ Produces output (proves it ran)
- ✅ Contains expected visualizations
- ✅ Numbers are realistic (no NaN, no impossible returns)

### Self-Healing Pattern

Notebooks are built to be resilient:

```python
# Example: Graceful handling of missing data
from pathlib import Path

DATA_FILE = Path('data/processed/features.csv')

if not DATA_FILE.exists():
    print("⚠️ Data file not found. Generating sample data...")
    df = generate_sample_data()
else:
    df = pd.read_csv(DATA_FILE)
```

## Viewing Results

### HTML Reports

After notebooks execute, HTML reports are saved to `reports/`:

```bash
ls reports/
# 01_data_validation.html
# 02_lightgbm_training.html
# ...
```

Open these in any browser to see results without running Jupyter.

### Interactive Exploration

For deeper analysis, open notebooks in JupyterLab:

```bash
jupyter lab
# Navigate to notebooks/phase1/
# Open and run cells interactively
```

## Best Practices

### When Creating Notebooks

1. **Clear markdown explanations** - Explain what each section does
2. **Progressive disclosure** - Start simple, add complexity
3. **Visualizations** - Use plots to make data tangible
4. **Error handling** - Gracefully handle missing data/services
5. **Reproducibility** - Set random seeds, document data sources

### When Reviewing Notebooks

1. **Run all cells** - Don't just read, execute
2. **Check outputs make sense** - Are numbers realistic?
3. **Look for warnings** - ⚠️ symbols indicate issues
4. **Compare to expectations** - Does performance match research?

## Common Issues

### Notebook Won't Execute

```bash
# Check if kernel is available
jupyter kernelspec list

# Ensure environment is activated
conda activate crypto-trading
```

### Missing Dependencies

If imports fail, the notebook should handle gracefully:
```python
try:
    import talib
except ImportError:
    print("⚠️ TA-Lib not installed. Using pandas_ta instead.")
    import pandas_ta as ta
```

### No Data Available

Notebooks should detect and report missing data:
```python
if df.empty:
    print("⚠️ No data available.")
    print("Run: python scripts/backfill_data.py --symbol BTC/USDT --days 7")
```
