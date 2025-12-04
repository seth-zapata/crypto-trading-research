-- =============================================================================
-- Crypto Trading System - TimescaleDB Schema
-- =============================================================================
-- This schema is auto-executed on first docker-compose up via the
-- /docker-entrypoint-initdb.d mount.
--
-- Author: Claude Opus 4.5
-- Date: 2024-12-03
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- OHLCV (Price) Data
-- =============================================================================
-- Core price data table - one row per symbol per timestamp
-- Using TimescaleDB hypertable for efficient time-series queries

CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      VARCHAR(20)     NOT NULL,
    exchange    VARCHAR(20)     NOT NULL DEFAULT 'coinbase',
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,

    -- Composite primary key for upserts
    PRIMARY KEY (time, symbol, exchange)
);

-- Convert to hypertable (TimescaleDB magic for time-series)
-- chunk_time_interval = 7 days (good for hourly data)
SELECT create_hypertable(
    'ohlcv',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Index for fast lookups by symbol
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time
    ON ohlcv (symbol, time DESC);

-- Index for exchange-specific queries
CREATE INDEX IF NOT EXISTS idx_ohlcv_exchange_symbol
    ON ohlcv (exchange, symbol, time DESC);

-- =============================================================================
-- Features (Computed Indicators)
-- =============================================================================
-- Stores computed features for ML models
-- Separate from OHLCV to allow different feature versions

CREATE TABLE IF NOT EXISTS features (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          VARCHAR(20)     NOT NULL,
    feature_version VARCHAR(10)     NOT NULL DEFAULT 'v1',

    -- Price-based features
    return_1h       DOUBLE PRECISION,
    return_4h       DOUBLE PRECISION,
    return_24h      DOUBLE PRECISION,
    log_return_1h   DOUBLE PRECISION,

    -- Moving averages
    sma_5           DOUBLE PRECISION,
    sma_10          DOUBLE PRECISION,
    sma_20          DOUBLE PRECISION,
    sma_50          DOUBLE PRECISION,
    ema_12          DOUBLE PRECISION,
    ema_26          DOUBLE PRECISION,

    -- Volatility
    volatility_20   DOUBLE PRECISION,
    atr_14          DOUBLE PRECISION,
    bb_upper        DOUBLE PRECISION,
    bb_lower        DOUBLE PRECISION,
    bb_width        DOUBLE PRECISION,

    -- Momentum
    rsi_14          DOUBLE PRECISION,
    macd            DOUBLE PRECISION,
    macd_signal     DOUBLE PRECISION,
    macd_hist       DOUBLE PRECISION,

    -- Volume
    volume_sma_20   DOUBLE PRECISION,
    volume_ratio    DOUBLE PRECISION,
    obv             DOUBLE PRECISION,

    -- Composite primary key
    PRIMARY KEY (time, symbol, feature_version)
);

-- Convert to hypertable
SELECT create_hypertable(
    'features',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Index for feature lookups
CREATE INDEX IF NOT EXISTS idx_features_symbol_time
    ON features (symbol, time DESC);

-- =============================================================================
-- Trades (Execution Log)
-- =============================================================================
-- Records all trades for backtesting validation and live trading

CREATE TABLE IF NOT EXISTS trades (
    id              SERIAL          PRIMARY KEY,
    time            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(20)     NOT NULL,
    exchange        VARCHAR(20)     NOT NULL,
    side            VARCHAR(10)     NOT NULL,  -- 'buy' or 'sell'
    order_type      VARCHAR(20)     NOT NULL,  -- 'market', 'limit'
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    cost            DOUBLE PRECISION NOT NULL,  -- quantity * price
    fee             DOUBLE PRECISION DEFAULT 0,
    fee_currency    VARCHAR(10),
    order_id        VARCHAR(100),   -- Exchange order ID
    status          VARCHAR(20)     NOT NULL DEFAULT 'filled',
    mode            VARCHAR(20)     NOT NULL DEFAULT 'backtest',  -- 'backtest', 'paper', 'live'
    strategy        VARCHAR(50),
    notes           TEXT
);

-- Index for trade analysis
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
    ON trades (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_trades_mode
    ON trades (mode, time DESC);

-- =============================================================================
-- Signals (Model Predictions)
-- =============================================================================
-- Stores model predictions for analysis and debugging

CREATE TABLE IF NOT EXISTS signals (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          VARCHAR(20)     NOT NULL,
    model_name      VARCHAR(50)     NOT NULL,
    model_version   VARCHAR(20)     NOT NULL DEFAULT 'v1',

    -- Prediction
    direction       VARCHAR(10),    -- 'up', 'down', 'neutral'
    confidence      DOUBLE PRECISION,
    predicted_return DOUBLE PRECISION,

    -- Actual outcome (filled in later)
    actual_return   DOUBLE PRECISION,
    correct         BOOLEAN,

    -- Metadata
    features_used   JSONB,

    PRIMARY KEY (time, symbol, model_name, model_version)
);

-- Convert to hypertable
SELECT create_hypertable(
    'signals',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- System State
-- =============================================================================
-- Tracks system state for recovery and monitoring

CREATE TABLE IF NOT EXISTS system_state (
    key             VARCHAR(100)    PRIMARY KEY,
    value           JSONB           NOT NULL,
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Data Quality Log
-- =============================================================================
-- Tracks data quality issues for monitoring

CREATE TABLE IF NOT EXISTS data_quality_log (
    id              SERIAL          PRIMARY KEY,
    time            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    check_type      VARCHAR(50)     NOT NULL,  -- 'missing_data', 'outlier', 'stale'
    symbol          VARCHAR(20),
    severity        VARCHAR(20)     NOT NULL,  -- 'info', 'warning', 'error'
    message         TEXT            NOT NULL,
    details         JSONB
);

CREATE INDEX IF NOT EXISTS idx_data_quality_time
    ON data_quality_log (time DESC);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to get latest price for a symbol
CREATE OR REPLACE FUNCTION get_latest_price(p_symbol VARCHAR)
RETURNS TABLE (
    time TIMESTAMPTZ,
    close DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT o.time, o.close
    FROM ohlcv o
    WHERE o.symbol = p_symbol
    ORDER BY o.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to get OHLCV for a time range
CREATE OR REPLACE FUNCTION get_ohlcv_range(
    p_symbol VARCHAR,
    p_start TIMESTAMPTZ,
    p_end TIMESTAMPTZ
)
RETURNS TABLE (
    time TIMESTAMPTZ,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT o.time, o.open, o.high, o.low, o.close, o.volume
    FROM ohlcv o
    WHERE o.symbol = p_symbol
      AND o.time >= p_start
      AND o.time <= p_end
    ORDER BY o.time ASC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Grants (if using separate application user)
-- =============================================================================
-- Uncomment if you create a separate app user:
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO crypto_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO crypto_app;

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Record schema version
INSERT INTO system_state (key, value)
VALUES ('schema_version', '{"version": "1.0", "created": "2024-12-03"}')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();

-- Log schema creation
INSERT INTO data_quality_log (check_type, severity, message)
VALUES ('schema_init', 'info', 'Database schema initialized successfully');
