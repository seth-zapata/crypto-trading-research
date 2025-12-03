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
- Can fetch data from Binance
- Can store OHLCV in TimescaleDB
- Basic feature engineering

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
