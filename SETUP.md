# Setup Guide

Complete installation and configuration guide for the Crypto Trading System.

## Prerequisites

- **OS**: Linux (Ubuntu 22.04+), macOS 13+, or Windows 11 with WSL2
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 250GB+ SSD
- **Python**: 3.11+
- **Conda**: Miniconda or Anaconda

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/seth-zapata/crypto-trading-research.git
cd crypto-trading-research

# 2. Create conda environment
conda env create -f environment.yml
conda activate crypto-trading

# 3. Configure exchange credentials
cp config/exchanges.yaml.example config/exchanges.yaml
# Edit config/exchanges.yaml with your API keys

# 4. Install and start PostgreSQL + TimescaleDB
# (See detailed instructions below)

# 5. Start Redis
redis-server --daemonize yes

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ccxt; print(f'CCXT: {ccxt.__version__}')"
pytest tests/ -v --tb=short
```

## Detailed Installation

### Step 1: Python Environment

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create and activate environment
conda env create -f environment.yml
conda activate crypto-trading

# Verify Python version
python --version  # Should be 3.11.x
```

### Step 2: PostgreSQL + TimescaleDB

TimescaleDB is a PostgreSQL extension optimized for time-series data.

#### Ubuntu/Debian

```bash
# Add TimescaleDB repository
sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
sudo apt update

# Install PostgreSQL 15 and TimescaleDB
sudo apt install -y postgresql-15 timescaledb-2-postgresql-15

# Configure TimescaleDB
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE USER crypto_trader WITH PASSWORD 'your_secure_password_here';
CREATE DATABASE crypto_trading OWNER crypto_trader;
\c crypto_trading
CREATE EXTENSION IF NOT EXISTS timescaledb;
EOF
```

#### macOS

```bash
# Install via Homebrew
brew install postgresql@15
brew install timescaledb

# Initialize PostgreSQL
initdb /usr/local/var/postgres
brew services start postgresql@15

# Configure TimescaleDB
timescaledb-tune --quiet --yes

# Create database
psql postgres << EOF
CREATE USER crypto_trader WITH PASSWORD 'your_secure_password_here';
CREATE DATABASE crypto_trading OWNER crypto_trader;
\c crypto_trading
CREATE EXTENSION IF NOT EXISTS timescaledb;
EOF
```

#### Verify Installation

```bash
psql -h localhost -U crypto_trader -d crypto_trading -c "SELECT version();"
psql -h localhost -U crypto_trader -d crypto_trading -c "SELECT * FROM pg_extension WHERE extname = 'timescaledb';"
```

### Step 3: Redis

Redis is used for caching and real-time data.

#### Ubuntu/Debian

```bash
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Verify
redis-cli ping  # Should return PONG
```

#### macOS

```bash
brew install redis
brew services start redis

# Verify
redis-cli ping  # Should return PONG
```

### Step 4: Configure Credentials

**CRITICAL: Never commit credentials to git!**

```bash
# Copy template
cp config/exchanges.yaml.example config/exchanges.yaml

# Edit with your credentials
nano config/exchanges.yaml
```

#### Option A: Direct Configuration (Less Secure)

Edit `config/exchanges.yaml` directly with your API keys.

#### Option B: Environment Variables (Recommended)

```bash
# Create .env file (already in .gitignore)
cat > .env << 'EOF'
# Exchange API Keys
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET_API_KEY=your_testnet_key
BINANCE_TESTNET_API_SECRET=your_testnet_secret

# Database
DATABASE_URL=postgresql://crypto_trader:your_password@localhost:5432/crypto_trading
REDIS_URL=redis://localhost:6379

# Optional: Data Providers
# GLASSNODE_API_KEY=your_key
# TWITTER_API_KEY=your_key
# REDDIT_CLIENT_ID=your_id
# REDDIT_CLIENT_SECRET=your_secret
EOF

# Secure the file
chmod 600 .env
```

Then in `config/exchanges.yaml`, use environment variable placeholders:

```yaml
binance:
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}
```

### Step 5: Exchange Setup (Binance)

1. Create a Binance account at https://www.binance.com
2. Complete KYC verification
3. Go to API Management
4. Create new API key with these permissions:
   - Enable Reading
   - Enable Spot Trading
   - **Disable** Withdrawals (security)
   - **Disable** Futures (not needed initially)
5. Whitelist your IP address
6. Save API key and secret securely

For paper trading, also create testnet credentials:
- Testnet: https://testnet.binance.vision

### Step 6: GPU Support (Optional)

If you have an NVIDIA GPU:

```bash
# Check CUDA compatibility
nvidia-smi

# Install CUDA toolkit via conda
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 7: TA-Lib (Optional)

Technical Analysis Library requires C library installation:

#### Ubuntu/Debian

```bash
sudo apt-get install -y libta-lib-dev
pip install TA-Lib
```

#### macOS

```bash
brew install ta-lib
pip install TA-Lib
```

Note: The project uses `pandas-ta` as a pure Python alternative that doesn't require TA-Lib.

## Verification

Run the verification script to ensure everything is working:

```bash
# Run all tests
pytest tests/ -v

# Check database connection
python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://crypto_trader:password@localhost:5432/crypto_trading')
conn = engine.connect()
print('Database connection: OK')
conn.close()
"

# Check Redis
python -c "
import redis
r = redis.Redis(host='localhost', port=6379)
r.ping()
print('Redis connection: OK')
"

# Check exchange API (read-only)
python -c "
import ccxt
exchange = ccxt.binance({'enableRateLimit': True})
ticker = exchange.fetch_ticker('BTC/USDT')
print(f'BTC/USDT: \${ticker[\"last\"]:,.2f}')
print('Exchange API: OK')
"
```

## Directory Structure

After setup, your directory should look like:

```
crypto-trading-research/
├── .git/                      # Git repository
├── .gitignore                 # Security patterns
├── .env                       # Environment variables (not in git)
├── environment.yml            # Conda environment
├── SETUP.md                   # This file
├── CLAUDE.md                  # AI implementation guide
├── PROJECT_SPEC.md            # Technical specification
├── crypto_trading_research.md # Research summary
│
├── config/
│   ├── exchanges.yaml         # Your credentials (not in git)
│   ├── exchanges.yaml.example # Template (in git)
│   ├── models.yaml            # Model hyperparameters
│   ├── risk.yaml              # Risk management
│   └── strategies.yaml        # Strategy settings
│
├── data/
│   ├── ingestion/             # Data fetching modules
│   ├── processing/            # Feature engineering
│   └── storage/               # Database interfaces
│
├── models/
│   ├── predictors/            # ML models
│   ├── meta_learning/         # MAML wrapper
│   └── saved/                 # Trained models (not in git)
│
├── agents/                    # RL agents
├── risk/                      # Risk management
├── execution/                 # Order execution
├── backtesting/               # Backtesting framework
├── monitoring/                # Performance monitoring
├── scripts/                   # Utility scripts
├── sql/                       # Database schemas
├── tests/                     # Test suite
├── notebooks/                 # Analysis notebooks
└── logs/                      # Runtime logs (not in git)
```

## Common Issues

### Issue: TimescaleDB extension not found

```bash
# Make sure you're connected to the right database
psql -d crypto_trading -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

### Issue: PyTorch Geometric installation fails

```bash
# Install dependencies separately
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric
```

### Issue: Permission denied for config files

```bash
chmod 600 config/exchanges.yaml
chmod 600 .env
```

### Issue: Redis connection refused

```bash
# Check if Redis is running
redis-cli ping

# Start Redis if not running
redis-server --daemonize yes
```

## Next Steps

After setup is complete:

1. **Phase 1**: Run data backfill
   ```bash
   python scripts/backfill_data.py --symbols BTC/USDT ETH/USDT --days 90
   ```

2. **Phase 2**: Train baseline model
   ```bash
   python scripts/train_models.py --model lightgbm
   ```

3. **Paper Trading**: Start paper trading
   ```bash
   python orchestrator.py --mode paper
   ```

See PROJECT_SPEC.md for full implementation timeline.

## Support

- GitHub Issues: https://github.com/seth-zapata/crypto-trading-research/issues
- See CLAUDE.md for implementation guidelines
