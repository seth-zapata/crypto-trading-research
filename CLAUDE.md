# CLAUDE.md - Instructions for AI Implementation

**Last Updated:** December 2024  
**Target Model:** Claude Opus 4.5 via Claude Code  
**User Profile:** ML Engineer with 2.5 years experience, familiar with Python/PyTorch

---

## Project Overview

You are building a **production-grade cryptocurrency trading system** that uses state-of-the-art machine learning techniques validated by academic research (2020-2024). This is NOT a toy project - it will trade real money, so quality, safety, and robustness are paramount.

### What Makes This System Different

Unlike typical "AI trading bot" projects, this system is based on **validated research**:
- **On-chain analytics** (75-82% directional accuracy in academic studies)
- **Graph Neural Networks** for cross-asset modeling (2.2x Sharpe improvement)
- **Filtered sentiment analysis** (+291% returns in bear markets)
- **Hierarchical RL** (2.74 Sharpe vs 2.27 benchmark)

These aren't speculative techniques - they have peer-reviewed validation. Your job is to implement them correctly.

---

## Critical Principles

### 1. **Safety First**
- Never commit API keys or credentials to git
- Always use `.gitignore` for sensitive files
- Default to conservative risk parameters
- Paper trade extensively before live deployment
- Start with minimal capital ($1000-2000) when going live

### 2. **Research-Backed Only**
- Don't add "cool ideas" that lack validation
- Stick to techniques in PROJECT_SPEC.md
- If tempted to add something, ask the user first
- Simpler is better if performance is similar

### 3. **Incremental Development**
- Build in phases (see PROJECT_SPEC.md)
- Test each phase thoroughly before moving on
- Don't try to build everything at once
- Each commit should be a working state

### 4. **Fail Gracefully**
- Every external call (API, DB, file I/O) needs try-except
- Log errors with full context
- Have fallback behavior (cached data, skip iteration, alert)
- Never crash the main trading loop

---

## Git Workflow

### Initial Setup

```bash
# Initialize repository
git init
git branch -M main

# Create comprehensive .gitignore
cat > .gitignore << 'EOF'
# Credentials & Config
config/*_keys.yaml
config/exchanges.yaml
*.env
.env.*

# API Keys
**/api_key*.txt
**/secret*.txt

# Data & Models
data/raw/
data/processed/
models/saved/*.pth
models/saved/*.txt
models/saved/*.pkl
*.h5
*.weights

# Database
*.db
*.sqlite

# Logs
logs/
*.log
trading.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary
tmp/
temp/
*.tmp

# Backups
backups/
*.backup
*.bak
EOF

# Create initial commit
git add .
git commit -m "Initial commit: Project structure and specifications"
```

### Commit Message Standards

Use **conventional commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance (dependencies, config)

**Examples:**
```bash
git commit -m "feat(data): Add OHLCV ingestion from Binance
- Implement ExchangeDataIngester class
- Add rate limiting and retry logic
- Include historical backfill capability"

git commit -m "fix(risk): Correct Kelly criterion calculation
- Was using wrong formula for win/loss ratio
- Add unit tests for edge cases
- Refs #12"

git commit -m "docs(readme): Add installation instructions"
```

### Branching Strategy

```bash
# For each major phase:
git checkout -b phase-1-data-infrastructure
# ... work on phase 1 ...
git add .
git commit -m "feat(data): Complete Phase 1 data infrastructure"
git checkout main
git merge phase-1-data-infrastructure

# For experiments/features:
git checkout -b experiment-gnn-architecture
# ... test different approach ...
# If it works:
git checkout main
git merge experiment-gnn-architecture
# If it doesn't:
git checkout main
git branch -D experiment-gnn-architecture
```

### What to Commit

**DO commit:**
- All source code (.py files)
- Configuration templates (.yaml.example)
- Documentation (README.md, docs/)
- SQL schemas
- Test files
- Requirements/environment files
- Scripts for setup and deployment

**DON'T commit:**
- API keys or credentials
- Trained model files (too large, use model registry)
- Raw data files (reproducible via scripts)
- Logs or temporary files
- Virtual environments

### When to Push

```bash
# Push after completing a phase
git push origin main

# Push experimental branches for backup
git push origin experiment-name

# If working solo, pushing to GitHub serves as backup
# Consider private repository for trading systems
```

---

## MANDATORY: End-of-Phase Workflow

**CRITICAL: Notebooks are the GATE before committing. Do NOT commit/push code that hasn't been validated by working notebooks.**

### The Correct Order

```
1. Write production code (.py files)
2. Write pytest tests
3. Run pytest → fix until passing
4. Create/update phase notebooks
5. Execute notebooks → debug until passing  ← GATE
6. Validate notebook outputs
7. ONLY THEN: git add, commit, push
```

### Why Notebooks Gate the Commit

- Notebooks prove the code actually works end-to-end
- If a notebook fails, the code is broken - don't push broken code
- Notebooks catch integration issues pytest might miss
- User reviews HTML reports AFTER push, so code must already work

### Complete Phase Completion Workflow

```bash
# Step 1: Run pytest (must pass)
pytest tests/ -v
# If fails → fix code → rerun until pass

# Step 2: Execute ALL phase notebooks (must pass)
for nb in notebooks/phase1/*.ipynb; do
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    "$nb" --output "${nb%.ipynb}_executed.ipynb"

  # If this fails → DEBUG AND FIX → rerun
done

# Step 3: Validate notebook outputs
python scripts/validate_notebook_outputs.py notebooks/phase1/*_executed.ipynb
# Must show: "✅ All notebooks validated successfully!"

# Step 4: Generate HTML reports
for nb in notebooks/phase1/*_executed.ipynb; do
  jupyter nbconvert --to html "$nb" --output-dir reports/
done

# Step 5: ONLY NOW commit and push
git add .
git status  # Review what's being committed
git commit -m "feat(phase1): Complete data infrastructure

- Implemented ExchangeDataIngester
- Created TimescaleDB schema
- Built feature engineering pipeline
- All notebooks execute successfully
- pytest: 15/15 passing

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main

# Step 6: Verify push succeeded
git log --oneline -1
git status  # Should show "up to date with origin/main"
```

### If Notebooks Fail

**DO NOT COMMIT. Instead:**

1. Read the error message
2. Fix the underlying code (not just the notebook)
3. Re-run the notebook
4. Repeat until it passes
5. Then proceed to commit

**Example:**
```
❌ Notebook failed: KeyError 'mvrv_z_score'
   ↓
Fix: Update feature_engineering.py to include mvrv_z_score
   ↓
Re-run notebook
   ↓
✅ Notebook passes
   ↓
NOW commit both the .py fix AND the working notebook
```

### Checklist Before Every Push

- [ ] `pytest tests/` passes (all green)
- [ ] All phase notebooks execute without errors
- [ ] `validate_notebook_outputs.py` shows all passed
- [ ] HTML reports generated in `reports/`
- [ ] Commit message describes what was built
- [ ] `git push origin main` succeeds

---

## Code Generation Standards

### File Structure

Every Python file should follow this template:

```python
"""
Module Name: Brief description

Detailed description of what this module does, why it exists,
and how it fits into the larger system.

Author: Claude Opus 4.5
Date: YYYY-MM-DD
"""

import standard_library
import third_party
from local_module import LocalClass

# Constants
CONSTANT_NAME = value

# Module-level configuration
logger = logging.getLogger(__name__)


class MainClass:
    """
    Brief class description.
    
    Longer description explaining purpose, responsibilities,
    and key design decisions.
    
    Attributes:
        attr1: Description
        attr2: Description
    
    Example:
        >>> obj = MainClass(param=value)
        >>> result = obj.method()
    """
    
    def __init__(self, param: type):
        """Initialize with description of what setup does."""
        self.param = param
        
    def public_method(self, arg: type) -> return_type:
        """
        Public method description.
        
        Args:
            arg: Argument description with expected format/range
            
        Returns:
            Description of return value
            
        Raises:
            ValueError: When and why this is raised
            
        Example:
            >>> obj.public_method(arg)
            expected_output
        """
        try:
            result = self._private_helper(arg)
            return result
        except Exception as e:
            logger.error(f"Error in public_method: {e}", exc_info=True)
            raise
    
    def _private_helper(self, arg: type) -> return_type:
        """Private helper method description."""
        # Implementation
        pass


# Module-level functions
def utility_function(param: type) -> return_type:
    """Utility function description."""
    pass


# Entry point (if applicable)
if __name__ == "__main__":
    # Example usage or CLI
    pass
```

### Type Hints (REQUIRED)

```python
# Always include type hints
from typing import List, Dict, Tuple, Optional, Union

def process_data(
    data: pd.DataFrame,
    threshold: float = 0.5,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Function with complete type hints."""
    pass

# For complex types, use TypedDict
from typing import TypedDict

class TradeSignal(TypedDict):
    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_loss: float
```

### Error Handling Pattern

```python
# Always wrap external calls
def fetch_data_from_api(endpoint: str) -> Dict:
    """
    Fetch data from external API.
    
    Includes retry logic and comprehensive error handling.
    """
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched data from {endpoint}")
            return data
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("Max retries exceeded - using cached data")
                return self._load_cached_data()
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code}")
            raise
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise
```

### Logging Standards

```python
import logging
from pathlib import Path

# Setup (typically in __main__ or orchestrator)
def setup_logging(log_dir: Path = Path("logs")):
    """Configure logging for the entire system."""
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'trading.log'),
            logging.FileHandler(log_dir / 'errors.log', level=logging.ERROR),
            logging.StreamHandler()  # Console output
        ]
    )

# Usage in modules
logger = logging.getLogger(__name__)

# Log levels guide:
logger.debug("Detailed diagnostic info")  # Use sparingly
logger.info("Important system events")    # Normal operations
logger.warning("Unexpected but handled")   # Risk limit hit, fallback used
logger.error("Error requiring attention")  # Failed operations
logger.critical("System-threatening")      # Circuit breaker, shutdown needed
```

### Configuration Management

```python
# Use YAML for configuration
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_dir: Path = Path("config")) -> Dict[str, Any]:
    """
    Load all configuration files.
    
    Returns merged configuration dictionary.
    """
    config = {}
    
    # Load each config file
    for config_file in config_dir.glob("*.yaml"):
        if config_file.name.endswith('.example'):
            continue
            
        with open(config_file) as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    # Override with environment variables if present
    config = _apply_env_overrides(config)
    
    # Validate required keys
    _validate_config(config)
    
    return config

def _apply_env_overrides(config: Dict) -> Dict:
    """Replace ${ENV_VAR} placeholders with environment values."""
    import os
    import re
    
    def replace_env_vars(value):
        if isinstance(value, str):
            # Find ${VAR_NAME} patterns
            matches = re.findall(r'\$\{([^}]+)\}', value)
            for match in matches:
                env_value = os.getenv(match)
                if env_value:
                    value = value.replace(f'${{{match}}}', env_value)
        elif isinstance(value, dict):
            return {k: replace_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_env_vars(item) for item in value]
        return value
    
    return replace_env_vars(config)
```

---

## Testing Strategy

### Unit Tests (pytest)

```python
# tests/test_feature_engineering.py

import pytest
import pandas as pd
import numpy as np
from data.processing.features import FeatureEngineer

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'time': dates,
        'open': 50000 + np.random.randn(100) * 100,
        'high': 50100 + np.random.randn(100) * 100,
        'low': 49900 + np.random.randn(100) * 100,
        'close': 50000 + np.random.randn(100) * 100,
        'volume': 1000 + np.random.randn(100) * 50
    })
    
    return data

def test_feature_engineer_generates_all_features(sample_ohlcv_data):
    """Test that feature engineer generates expected features."""
    engineer = FeatureEngineer()
    
    result = engineer.generate_all_features(sample_ohlcv_data)
    
    # Check that key features exist
    assert 'return_1' in result.columns
    assert 'volatility_20' in result.columns
    assert 'rsi_14' in result.columns
    
    # Check no NaN in recent data (after warmup)
    assert result.iloc[-10:].isna().sum().sum() == 0

def test_feature_engineer_handles_missing_data():
    """Test graceful handling of missing data."""
    # Create data with NaN
    data = pd.DataFrame({
        'close': [100, np.nan, 102, 103, np.nan]
    })
    
    engineer = FeatureEngineer()
    result = engineer._add_price_features(data)
    
    # Should handle NaN gracefully
    assert result is not None
    assert len(result) == len(data)

def test_feature_engineer_consistency():
    """Test that same input produces same output."""
    data = sample_ohlcv_data()
    engineer = FeatureEngineer()
    
    result1 = engineer.generate_all_features(data.copy())
    result2 = engineer.generate_all_features(data.copy())
    
    pd.testing.assert_frame_equal(result1, result2)
```

### Integration Tests

```python
# tests/integration/test_data_pipeline.py

import pytest
from data.ingestion.exchanges import ExchangeDataIngester
from data.processing.features import FeatureEngineer
from data.storage.timeseries_db import TimeSeriesDB

@pytest.mark.integration
def test_full_data_pipeline():
    """Test complete data flow: fetch → process → store → retrieve."""
    
    # 1. Fetch data (using test mode)
    ingester = ExchangeDataIngester({'test_mode': True})
    raw_data = ingester.fetch_ohlcv_historical(
        'binance',
        'BTC/USDT',
        '1h',
        start='2024-01-01',
        end='2024-01-02'
    )
    
    assert len(raw_data) > 0
    
    # 2. Generate features
    engineer = FeatureEngineer()
    features = engineer.generate_all_features(raw_data)
    
    assert 'return_1' in features.columns
    
    # 3. Store in database
    db = TimeSeriesDB('postgresql://test_user@localhost/test_db')
    db.store_features(features, table='test_features')
    
    # 4. Retrieve and verify
    retrieved = db.fetch_features(
        table='test_features',
        symbol='BTC/USDT',
        start='2024-01-01',
        end='2024-01-02'
    )
    
    assert len(retrieved) == len(features)
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_feature_engineering.py

# Run only integration tests
pytest -m integration

# Run excluding slow tests
pytest -m "not slow"
```

---

## Decision-Making Guidelines

### When to Ask the User

Ask when:
1. **Trade-offs without clear winner**
   - "Should we use GNN (more complex, potentially better) or just LSTM (simpler, faster)?"
   - "Glassnode costs $29/month. Use it or stick with free CryptoQuant?"

2. **Risk parameters**
   - "Max drawdown: 15% or 20%?"
   - "Position sizing: full Kelly or half Kelly?"

3. **Something seems wrong**
   - "Backtest shows 10x returns - this looks like overfitting. Investigate or proceed?"
   - "Model accuracy dropped from 65% to 48%. Retrain or debug?"

4. **Security/safety concerns**
   - "API key needs more permissions. Proceed?"
   - "Ready to deploy to production. Confirm?"

5. **Costs money**
   - Before making paid API calls (except cheap historical data fetches)
   - Before subscribing to data services

### Safe Assumptions (Don't Ask)

Proceed without asking when:
1. **Implementation details**
   - How to structure a class
   - Which Python library to use (if multiple work)
   - Directory structure within specified guidelines

2. **Standard practices**
   - Using pytest for tests
   - Using .gitignore for sensitive files
   - Logging errors vs warnings

3. **Bug fixes**
   - Off-by-one errors
   - Missing imports
   - Type mismatches

4. **Documentation**
   - Adding docstrings
   - Creating examples
   - Writing comments

5. **Safe defaults**
   - Conservative risk parameters
   - Paper trading before live
   - More logging rather than less

### Complexity Trade-offs

**Prefer simpler solutions when:**
- Performance difference < 5%
- Maintenance burden is much higher
- Dependencies multiply
- Debugging becomes harder

**Example:**
```
Option A: Custom GNN architecture (2 days to implement, 3% better)
Option B: Pre-trained TimeGNN from PyG (2 hours, 3% worse)

Choose: Option B initially, revisit if bottleneck
```

**Prefer complex solutions when:**
- Performance difference > 20%
- Validated by research
- Core to the system
- Scales better long-term

**Example:**
```
Option A: Simple moving average crossover
Option B: Ensemble (LightGBM + LSTM + GNN) with research backing

Choose: Option B (this is why we're building this system)
```

---

## Phase-by-Phase Implementation

### Week 1: Foundation

**Goals:**
- Database setup and working
- Can fetch data from Coinbase (using `coinbaseadvanced` class)
- Can store OHLCV in TimescaleDB
- Basic feature engineering

**Critical Implementation Notes:**
- Use `ccxt.coinbaseadvanced()` for Coinbase API (NOT `coinbase` or `coinbaseexchange`)
- Coinbase API secret is EC private key in PEM format - .env stores literal `\n` text that must be converted:
  ```python
  # .env files store \n as two characters, not actual newlines
  # Must convert before passing to ccxt:
  api_secret = os.getenv('COINBASE_API_SECRET').replace('\\n', '\n')
  ```

**Validation:**
```bash
python tests/test_data_pipeline.py
# Should pass without errors

python scripts/fetch_btc_data.py --days 7
# Should populate database with 7 days of BTC/USDT data
```

**Git milestone:**
```bash
git tag -a v0.1-data-foundation -m "Phase 1: Data infrastructure complete"
git push origin v0.1-data-foundation
```

### Week 2: Baseline Model

**Goals:**
- LightGBM training pipeline
- Backtesting framework
- Walk-forward validation
- First performance metrics

**Validation:**
- Backtest Sharpe > 1.0 (with realistic transaction costs)
- Win rate > 50%
- Model doesn't just predict "always up"

**Git milestone:**
```bash
git tag -a v0.2-baseline-model -m "Phase 2: LightGBM baseline working"
```

### Week 3: Alpha Sources

**Goals:**
- On-chain data integration (MVRV, SOPR, netflows)
- Sentiment analysis (Twitter + Reddit)
- Regime classification
- Combined signal generation

**Validation:**
- Regime classifier identifies known bull/bear periods correctly
- Sentiment RVS score makes intuitive sense
- Combined signals improve Sharpe by 10%+

### Week 4-5: Advanced Models & RL

**Goals:**
- LSTM model (optional, if time permits)
- GNN cross-asset (optional)
- Hierarchical RL agents
- Ensemble system

**Validation:**
- Each model adds marginal value
- Ensemble beats best individual model
- RL agents learn sensible policies

### Week 6: Production Ready

**Goals:**
- Risk management enforcement
- Order execution system
- Monitoring dashboard
- Paper trading mode

**Validation:**
- Paper trade for 1 week minimum
- All trades logged correctly
- Risk limits enforced
- No crashes or data loss

---

## Common Pitfalls to Avoid

### 1. **Look-Ahead Bias**
```python
# BAD - Uses future data
df['signal'] = (df['close'].shift(-1) > df['close']).astype(int)

# GOOD - Only uses past data
df['signal'] = (df['close'] > df['close'].shift(1)).astype(int)
```

### 2. **Overfitting**
```python
# BAD - Optimizing on all data
best_params = optimize_params(data)

# GOOD - Time series split
train, test = data[:split], data[split:]
best_params = optimize_params(train)
validate_performance(test, best_params)
```

### 3. **Ignoring Transaction Costs**
```python
# BAD - Assumes free trading
profit = (exit_price - entry_price) * quantity

# GOOD - Includes spread + commission
entry_cost = quantity * entry_price * (1 + commission)
exit_value = quantity * exit_price * (1 - commission)
profit = exit_value - entry_cost
```

### 4. **Not Handling Missing Data**
```python
# BAD - Will crash on NaN
df['return'] = df['close'].pct_change()
model.fit(df)

# GOOD - Handle NaN explicitly
df['return'] = df['close'].pct_change()
df = df.dropna()  # Or forward-fill, depending on context
assert df.isna().sum().sum() == 0, "Still have NaN values"
model.fit(df)
```

### 5. **Insufficient Error Handling**
```python
# BAD - Will crash if API fails
data = exchange.fetch_ohlcv(symbol)

# GOOD - Graceful degradation
try:
    data = exchange.fetch_ohlcv(symbol)
except ccxt.NetworkError:
    logger.warning("Network error - using cached data")
    data = load_from_cache(symbol)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    data = None
```

---

## Performance Optimization

### Profile Before Optimizing

```python
# Use line_profiler
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(expensive_function)
lp.run('expensive_function(data)')
lp.print_stats()

# Or use cProfile
import cProfile
cProfile.run('main_function()')
```

### Common Optimizations

**1. Vectorize with NumPy/Pandas**
```python
# BAD - Python loops
for i in range(len(df)):
    df.loc[i, 'return'] = (df.loc[i, 'close'] - df.loc[i-1, 'close']) / df.loc[i-1, 'close']

# GOOD - Vectorized
df['return'] = df['close'].pct_change()
```

**2. Use Appropriate Data Structures**
```python
# BAD - List for lookups
prices = [...]
if target in prices:  # O(n)

# GOOD - Set for lookups
prices = set([...])
if target in prices:  # O(1)
```

**3. Cache Expensive Operations**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_indicators(symbol: str, period: int) -> pd.DataFrame:
    # Expensive calculation
    pass
```

**4. Use Batch Processing**
```python
# BAD - One at a time
for symbol in symbols:
    features = engineer.generate_all_features(load_data(symbol))
    store_features(features)

# GOOD - Batch process
all_data = [load_data(s) for s in symbols]
all_features = [engineer.generate_all_features(d) for d in all_data]
store_features_batch(all_features)
```

---

## Security Checklist

Before deploying:

- [ ] No API keys in code
- [ ] All secrets in environment variables or config files (gitignored)
- [ ] `.gitignore` includes all sensitive patterns
- [ ] Database password not hardcoded
- [ ] API rate limits respected
- [ ] Input validation on all user inputs
- [ ] SQL queries use parameterization (no string concatenation)
- [ ] Error messages don't leak sensitive info
- [ ] Logging doesn't include API keys or passwords
- [ ] File permissions set correctly on config files

---

## Notebook Testing Requirements

**CRITICAL: You MUST test every notebook you create before declaring a phase complete.**

**EVEN MORE CRITICAL: If a notebook fails, DEBUG AND FIX IT YOURSELF. Do NOT ask the user for help unless you've exhausted all options.**

### Autonomous Testing and Debugging Workflow

After creating a notebook, follow this loop until it works:

1. **Execute the notebook**
2. **If it fails → DEBUG AND FIX IT IMMEDIATELY**
3. **Re-test**
4. **Repeat until it passes**
5. **Only then report completion**

### Testing Loop (Fully Autonomous)

```bash
# Loop: Test → Fix → Retest → Fix → ... → Success
while true; do
  # Attempt execution
  jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    notebooks/phase1/01_data_validation.ipynb \
    --output 01_data_validation_executed.ipynb
  
  if [ $? -eq 0 ]; then
    echo "✓ Notebook executed successfully"
    break
  else
    echo "✗ Notebook failed - analyzing error and fixing..."
    # Analyze the error, fix the notebook, try again
  fi
done

# Generate HTML report
jupyter nbconvert --to html \
  notebooks/phase1/01_data_validation_executed.ipynb \
  --output ../../reports/01_data_validation.html

# Validate outputs
python scripts/validate_notebook_outputs.py \
  notebooks/phase1/01_data_validation_executed.ipynb
```

### Common Errors and AUTONOMOUS Fixes

**DO NOT ASK THE USER. FIX IT YOURSELF.**

#### Error 1: Missing Import
```
Error: ImportError: No module named 'talib'
```

**Your Action (automatic):**
1. Add to `requirements.txt`: `TA-Lib==0.4.28`
2. Add to `environment.yml`: `- ta-lib=0.4.28`
3. Note in commit: "Added missing talib dependency"
4. Re-run notebook test
5. If still fails, try alternative library (e.g., pandas_ta)

**DO NOT ASK:** "Should I add talib to requirements?"  
**JUST DO IT.**

#### Error 2: File Not Found
```
Error: FileNotFoundError: data/ohlcv.csv not found
```

**Your Action (automatic):**
1. Check if data exists: `ls data/`
2. If missing, generate sample data for the notebook
3. Or update notebook to check existence first:
   ```python
   if not Path('data/ohlcv.csv').exists():
       print("⚠️ No data found. Run: python scripts/backfill_data.py")
   ```
4. Re-run notebook test

**DO NOT ASK:** "There's no data, should I generate sample data?"  
**JUST FIX IT.**

#### Error 3: Database Connection Failed
```
Error: psycopg2.OperationalError: could not connect to server
```

**Your Action (automatic):**
1. Add connection check at top of notebook:
   ```python
   try:
       conn = psycopg2.connect(...)
   except psycopg2.OperationalError:
       print("⚠️ Database not running. Start with: docker-compose up -d")
       print("Using sample data for demonstration...")
       # Use mock/sample data instead
   ```
2. Re-run notebook test

**DO NOT ASK:** "Database isn't running, what should I do?"  
**HANDLE IT GRACEFULLY.**

#### Error 4: Missing Column
```
Error: KeyError: 'feature_123' not found in DataFrame
```

**Your Action (automatic):**
1. Check what features actually exist
2. Update notebook to use existing features
3. Or generate the missing feature in the notebook
4. Re-run notebook test

**DO NOT ASK:** "This feature is missing, should I remove it?"  
**FIX THE NOTEBOOK.**

#### Error 5: Empty DataFrame
```
Error: ValueError: cannot plot empty DataFrame
```

**Your Action (automatic):**
1. Add check before plotting:
   ```python
   if df.empty:
       print("⚠️ No data available. This notebook requires backfilled data.")
   else:
       df.plot()
   ```
2. Re-run notebook test

**DO NOT ASK:** "The DataFrame is empty, should I handle this?"  
**YES, HANDLE IT.**

### Autonomous Debugging Strategy

When a notebook fails, follow this systematic approach:

**Step 1: Read the error carefully**
```python
# Error tells you exactly what's wrong
# ImportError → missing library
# KeyError → missing column/key
# FileNotFoundError → missing file
# ConnectionError → service not running
```

**Step 2: Determine the fix**
- Missing dependency? → Add to requirements.txt
- Missing data? → Generate sample or add graceful handling
- Wrong path? → Fix the path
- Service down? → Add fallback behavior

**Step 3: Implement the fix**
- Edit the notebook
- Or edit requirements.txt
- Or create missing files
- Whatever is needed to make it work

**Step 4: Re-test immediately**
```bash
jupyter nbconvert --execute notebooks/phase1/01_data_validation.ipynb
```

**Step 5: Repeat until success**
- If still fails, analyze NEW error
- Fix that one
- Re-test
- Continue until notebook executes cleanly

### When to Actually Ask the User

**ONLY ask the user if:**

1. **Fundamental design decision needed**
   ```
   "The notebook needs real exchange data to work. Options:
   A) Use paper trading API (free but requires signup)
   B) Use sample CSV data (immediate but not real-time)
   Which do you prefer?"
   ```

2. **Costs money**
   ```
   "To make this notebook work with live data, we need Glassnode API ($29/mo).
   Should I use free CryptoQuant instead?"
   ```

3. **Security concern**
   ```
   "This notebook would need your actual API key to test. 
   Should I use mock data instead?"
   ```

4. **Genuinely stuck after multiple attempts**
   ```
   "I've tried 5 different approaches to fix this notebook error:
   1. Added missing imports
   2. Switched to alternative library
   3. Used mock data
   4. Simplified the plotting code
   5. Checked all paths
   
   The error persists: [specific error]
   I need guidance on how to proceed."
   ```

**DO NOT ASK about:**
- ❌ "Should I fix this error?" → YES, FIX IT
- ❌ "The notebook failed, what should I do?" → DEBUG IT
- ❌ "Is it okay to add this library?" → IF NEEDED, ADD IT
- ❌ "Can I use sample data?" → YES, USE IT
- ❌ "Should I add error handling?" → YES, ADD IT

### Notebook Self-Healing Pattern

Build notebooks to be resilient from the start:

```python
# GOOD: Self-healing notebook pattern

import sys
from pathlib import Path

# 1. Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 2. Try imports with fallbacks
try:
    import talib
    USE_TALIB = True
except ImportError:
    print("⚠️ TA-Lib not installed. Using pandas alternatives.")
    import pandas_ta as ta
    USE_TALIB = False

# 3. Check data exists
DATA_FILE = PROJECT_ROOT / 'data' / 'processed' / 'features.csv'
if not DATA_FILE.exists():
    print(f"⚠️ Data file not found: {DATA_FILE}")
    print("Generating sample data for demonstration...")
    # Generate sample data
    df = generate_sample_data()
else:
    df = pd.read_csv(DATA_FILE)

# 4. Check database connection
try:
    conn = psycopg2.connect(...)
    print("✓ Database connected")
except psycopg2.OperationalError:
    print("⚠️ Database not available. Using cached data.")
    df = pd.read_csv(CACHE_FILE)

# 5. Graceful degradation
if df.empty:
    print("⚠️ No data available.")
    print("Run: python scripts/backfill_data.py --symbol BTC/USDT --days 7")
else:
    # Do analysis
    df.plot()
```

This way, notebooks rarely fail - they adapt to what's available.

### Validation Script

Create `scripts/validate_notebook_outputs.py`:

```python
"""
Validate that notebooks executed successfully and produced expected outputs.
"""

import json
import sys
from pathlib import Path

def validate_notebook(notebook_path: Path) -> bool:
    """
    Check that notebook executed and has expected outputs.
    
    Returns True if validation passes.
    """
    with open(notebook_path) as f:
        nb = json.load(f)
    
    errors = []
    has_plots = False
    has_output = False
    
    for cell in nb['cells']:
        # Check for execution errors
        if cell['cell_type'] == 'code':
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'error':
                    errors.append(output.get('evalue', 'Unknown error'))
                
                # Check for plots (image/png in outputs)
                if 'data' in output and 'image/png' in output['data']:
                    has_plots = True
                
                # Check for text output
                if output.get('output_type') in ['stream', 'execute_result']:
                    has_output = True
    
    # Validation checks
    if errors:
        print(f"❌ FAILED: Notebook has errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    if not has_output:
        print(f"⚠️  WARNING: Notebook has no output (did it run?)")
        return False
    
    if not has_plots:
        print(f"⚠️  WARNING: Notebook has no plots (expected visualizations)")
        # Not a failure, but flag it
    
    print(f"✅ PASSED: Notebook executed successfully")
    if has_plots:
        print(f"  - Contains {sum(1 for _ in nb['cells'])} plots")
    print(f"  - No errors detected")
    
    return True

if __name__ == "__main__":
    notebook_path = Path(sys.argv[1])
    success = validate_notebook(notebook_path)
    sys.exit(0 if success else 1)
```

### MANDATORY: Visual Verification of Notebook Charts

**CRITICAL: You MUST visually inspect all chart/image outputs in notebooks, not just check that they executed.**

After executing notebooks, extract and view all generated images:

```python
# Extract images from executed notebook
import json
import base64
from pathlib import Path

def extract_and_verify_images(notebook_path: str) -> None:
    """Extract all images from notebook for visual verification."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell.get('outputs'):
            for j, output in enumerate(cell['outputs']):
                data = output.get('data', {})
                if 'image/png' in data:
                    img_path = f"/tmp/nb_chart_{i}_{j}.png"
                    with open(img_path, 'wb') as img_file:
                        img_file.write(base64.b64decode(data['image/png']))
                    print(f"Saved: {img_path}")

# Then use the Read tool to view each image file
```

**What to verify in each chart:**

1. **Price Charts:**
   - Price range matches reported min/max values
   - No gaps or discontinuities (unless expected)
   - Trend direction looks realistic
   - No flat lines or obvious data errors

2. **Feature Distribution Histograms:**
   - Returns: Should be centered around 0, roughly normal
   - RSI: Must be bounded [0, 100]
   - Volume ratio: Should be centered around 1.0
   - Volatility: Should be right-skewed (volatility clusters)

3. **Correlation Heatmaps:**
   - Diagonal should be 1.0 (self-correlation)
   - No impossible values (outside [-1, 1])
   - Related features should show expected correlations

4. **Time Series Plots:**
   - X-axis dates should match expected range
   - No impossible values (negative prices, etc.)
   - Patterns should look realistic for the asset

**Example verification workflow:**
```bash
# 1. Execute notebook
jupyter nbconvert --execute notebook.ipynb

# 2. Extract images
python -c "from scripts.extract_images import extract_and_verify_images; extract_and_verify_images('notebook.ipynb')"

# 3. View each image using the Read tool
# Read /tmp/nb_chart_*.png files

# 4. Verify each chart looks correct before committing
```

**DO NOT commit if charts show:**
- Empty plots
- Clearly wrong data (e.g., BTC at $500 in 2024)
- Missing labels or titles
- Truncated or cut-off visualizations
- Error messages in plot area

### Phase Completion Checklist

Before marking ANY phase as complete, verify:

**Production Code:**
- [ ] All Python files have docstrings
- [ ] All functions have type hints
- [ ] pytest tests pass
- [ ] Code follows style guide

**Analysis Notebooks:**
- [ ] Notebooks execute without errors (`nbconvert --execute`)
- [ ] Expected visualizations are present
- [ ] **VISUALLY VERIFIED all chart outputs look correct** ← NEW
- [ ] Numbers/metrics are realistic (no NaN, no 10000% returns)
- [ ] Markdown cells explain what's happening
- [ ] HTML export successful (proves images render)

**Validation:**
```bash
# Run this before declaring phase complete
pytest tests/                              # Unit tests pass
jupyter nbconvert --execute notebooks/phase1/*.ipynb  # Notebooks run
python scripts/validate_notebook_outputs.py notebooks/phase1/*.ipynb  # Outputs valid
```

### Common Notebook Issues to Check

**1. Missing Data**
```python
# BAD - Will crash if data missing
df = fetch_data()
df.plot()

# GOOD - Handle missing data
df = fetch_data()
if df is None or len(df) == 0:
    print("⚠️ No data available. Run backfill script first.")
else:
    df.plot()
```

**2. Hardcoded Paths**
```python
# BAD - Won't work on user's machine
data = pd.read_csv('/home/claude/data.csv')

# GOOD - Relative paths
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
data = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'features.csv')
```

**3. Missing Imports**
```python
# Make sure all imports are at the top
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
```

**4. Plots Not Displaying**
```python
# Always include this for notebooks
%matplotlib inline

# And explicit show() for complex plots
fig, ax = plt.subplots()
ax.plot(data)
plt.show()
```

**5. Long-Running Cells**
```python
# Add progress indicators for slow operations
from tqdm.auto import tqdm

for symbol in tqdm(symbols, desc="Processing symbols"):
    process(symbol)
```

### Reporting Test Results

When you complete a phase, report:

```
Phase 1 Complete!

✅ Production Code:
  - data/ingestion/exchanges.py
  - data/storage/timeseries_db.py
  - data/processing/features.py
  - All pytest tests passing (15/15)

✅ Analysis Notebooks:
  - notebooks/phase1/01_data_validation.ipynb
    • Executed successfully
    • Generated 4 plots (OHLCV, volume, feature distributions)
    • Shows 2,160 rows of BTC/USDT data
    • HTML report: reports/01_data_validation.html
    
  - notebooks/phase1/01b_feature_exploration.ipynb
    • Executed successfully  
    • Generated 8 plots (feature importance, correlations)
    • Shows 127 features generated
    • HTML report: reports/01b_feature_exploration.html

✅ Validation:
  - Data quality check: PASSED
  - Database query test: PASSED
  - Feature engineering test: PASSED

📊 User Actions:
  1. Review HTML reports in reports/ directory
  2. Open notebooks in Jupyter if you want to explore interactively
  3. Approve Phase 2 if satisfied

Ready to proceed to Phase 2?
```

---

## Final Checklist Before Live Trading

- [ ] Backtested on 1+ year of data
- [ ] Walk-forward validation shows consistent performance
- [ ] Paper traded successfully for 2+ weeks
- [ ] All risk limits tested and working
- [ ] Circuit breakers trigger appropriately
- [ ] Execution costs match expectations
- [ ] Database backup strategy in place
- [ ] Monitoring and alerting configured
- [ ] Have emergency stop procedure
- [ ] Starting with minimal capital ($1000-2000)
- [ ] User has reviewed all core logic
- [ ] Documentation is complete
- [ ] All notebooks execute without errors
- [ ] HTML reports generated for all phases

---

## Getting Help

If you encounter something outside these guidelines:

1. **Check PROJECT_SPEC.md** - Maybe it's detailed there
2. **Look for similar code** - Pattern might exist elsewhere in project
3. **Check research summary** - Academic backing might clarify
4. **Ask the user** - When in doubt, especially for:
   - Risk/safety decisions
   - Performance trade-offs
   - Costs money
   - Significant architecture changes

---

## Remember

You're building a system that will handle real money. Quality and safety matter more than speed. Take the time to:

- Write tests
- Handle errors properly
- Log thoroughly
- Validate assumptions
- Start conservative

The goal isn't to build the fanciest system - it's to build a **reliable, profitable system** based on validated research.

Good luck! 🚀

---

**Document Version:** 1.0  
**Last Reviewed:** December 2024  
**Next Review:** After Phase 3 completion
